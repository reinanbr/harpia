//! Parity with PyTorch/transformers on errei's `errei-cita-v1` NLI cross-encoder.
//!
//! Opt-in: `HARPIA_ERREI_DIR=~/Desktop/std/plugins/python/errei cargo test -p harpia-local --release -- --ignored`
//! Golden logits were produced once with transformers (see `docs/parity.md`); the texts are
//! read from errei's data in place because SciFact is CC BY-NC 2.0.

use harpia_local::{DType, Device, SequenceClassifier};
use std::time::Instant;

#[test]
#[ignore = "parity: needs HARPIA_ERREI_DIR"]
fn errei_cita_v1_matches_transformers() {
    let Ok(errei) = std::env::var("HARPIA_ERREI_DIR") else {
        eprintln!("HARPIA_ERREI_DIR not set; skipping");
        return;
    };
    let errei = std::path::PathBuf::from(errei);
    let golden: serde_json::Value = serde_json::from_str(include_str!("golden/errei_cita_v1.json")).unwrap();
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(errei.join("treino/dados/pares_val.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let t0 = Instant::now();
    let clf = SequenceClassifier::load_with(&errei.join("treino/modelos/errei-cita-v1"), &Device::Cpu, DType::F32, 192)
        .unwrap();
    let load = t0.elapsed();
    let labels: Vec<String> =
        golden["labels"].as_array().unwrap().iter().map(|l| l.as_str().unwrap().to_string()).collect();
    assert_eq!(harpia_core::traits::PairClassifier::labels(&clf), labels.as_slice());
    let items = golden["items"].as_array().unwrap();
    let texts: Vec<(String, String)> = items
        .iter()
        .map(|it| {
            let r = &rows[it["line"].as_u64().unwrap() as usize - 1];
            (r["b"].as_str().unwrap().to_string(), r["a"].as_str().unwrap().to_string())
        })
        .collect();
    for (it, (p, h)) in items.iter().zip(&texts) {
        if let Some(ids) = it["input_ids"].as_array() {
            let expected: Vec<u32> = ids.iter().map(|x| x.as_u64().unwrap() as u32).collect();
            assert_eq!(clf.encode_pair(p, h).unwrap(), expected, "tokenization of line {}", it["line"]);
        }
    }
    let pairs: Vec<(&str, &str)> = texts.iter().map(|(p, h)| (p.as_str(), h.as_str())).collect();
    let t1 = Instant::now();
    let logits = clf.pair_logits(&pairs).unwrap();
    let infer = t1.elapsed();
    let tol = golden["tolerance"].as_f64().unwrap() as f32;
    let mut max_diff = 0f32;
    let mut same_argmax = 0;
    for (it, got) in items.iter().zip(&logits) {
        let want: Vec<f32> = it["logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        for (a, b) in got.iter().zip(&want) {
            max_diff = max_diff.max((a - b).abs());
        }
        let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        same_argmax += usize::from(am(got) == am(&want));
    }
    eprintln!(
        "errei-cita-v1: load {:.2}s, {} pairs in {:.2}s ({:.1} pairs/s), max |Δlogit| = {max_diff:.2e}, argmax agreement {same_argmax}/{}",
        load.as_secs_f64(),
        pairs.len(),
        infer.as_secs_f64(),
        pairs.len() as f64 / infer.as_secs_f64(),
        items.len()
    );
    assert!(max_diff <= tol, "max |Δlogit| {max_diff} > {tol}");
    assert_eq!(same_argmax, items.len());
}
