//! Parity with transformers on Qwen2.5-0.5B-Instruct: chat template, tokenization, logits and
//! greedy decoding. Opt-in: `HARPIA_QWEN_DIR=<snapshot dir> cargo test -p harpia-local --release --test parity_qwen -- --ignored`
//! (download with `harpia download Qwen/Qwen2.5-0.5B-Instruct --online`).

use harpia_core::traits::{GenerationParams, Tokenizer};
use harpia_core::types::{Message, Role};
use harpia_local::{CausalLm, DType, Device, ModelDir, TextGenerator};
use std::time::Instant;

#[test]
#[ignore = "parity: needs HARPIA_QWEN_DIR"]
fn qwen2_5_matches_transformers() {
    let Ok(dir) = std::env::var("HARPIA_QWEN_DIR") else { return };
    let dir = std::path::PathBuf::from(dir);
    let g: serde_json::Value = serde_json::from_str(include_str!("golden/qwen2_5_0_5b_instruct.json")).unwrap();
    let messages: Vec<Message> = g["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| Message::new(Role::parse(m["role"].as_str().unwrap()).unwrap(), m["content"].as_str().unwrap()))
        .collect();
    let t0 = Instant::now();
    let lm = TextGenerator::load(&dir).unwrap();
    let load = t0.elapsed();
    let rendered = lm.render_chat(&messages).unwrap();
    assert_eq!(rendered, g["rendered"].as_str().unwrap(), "chat template rendering");
    let ids = lm.tokenizer().encode(&rendered, false).unwrap().ids;
    let want_ids: Vec<u32> = g["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    assert_eq!(ids, want_ids, "tokenization");

    let md = ModelDir::open(&dir).unwrap();
    let mut causal =
        CausalLm::from_var_builder(&md, md.var_builder(DType::F32, &Device::Cpu).unwrap(), DType::F32, &Device::Cpu)
            .unwrap();
    let logits = causal.logits_all(&ids).unwrap().to_vec2::<f32>().unwrap();
    let mut max_diff = 0f32;
    for (pos, (tids, tvals)) in
        g["top_ids"].as_array().unwrap().iter().zip(g["top_logits"].as_array().unwrap()).enumerate()
    {
        for (i, v) in tids.as_array().unwrap().iter().zip(tvals.as_array().unwrap()) {
            max_diff = max_diff.max((logits[pos][i.as_u64().unwrap() as usize] - v.as_f64().unwrap() as f32).abs());
        }
    }
    let t1 = Instant::now();
    // transformers applies generation_config.json's repetition_penalty (1.1) even when greedy
    let defaults = lm.default_params();
    assert_eq!(defaults.repetition_penalty, 1.1);
    let params = GenerationParams {
        max_new_tokens: 20,
        temperature: 0.0,
        repetition_penalty: defaults.repetition_penalty,
        ..Default::default()
    };
    let out = lm.generate_with(&ids, &params, None, |_| {}).unwrap();
    let secs = t1.elapsed().as_secs_f64();
    let want_gen: Vec<u32> = g["greedy_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    eprintln!(
        "qwen2.5-0.5b: load {:.2}s, max |Δlogit| over top-50 x {} positions = {max_diff:.2e}, greedy {:.1} tok/s, text {:?}",
        load.as_secs_f64(),
        ids.len(),
        out.token_ids.len() as f64 / secs,
        out.text
    );
    assert!(max_diff <= g["tolerance"].as_f64().unwrap() as f32, "max |Δlogit| {max_diff}");
    assert_eq!(out.token_ids.as_slice(), &want_gen[..out.token_ids.len()], "greedy decoding");
    assert_eq!(out.text.as_str(), g["greedy_text"].as_str().unwrap());
}
