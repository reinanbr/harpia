//! Training benchmarks: one LoRA SFT run of a few steps (step time and tokens/s).
//!
//! Uses the tiny random-weight Qwen2 from `harpia_local::testing` by default (plumbing
//! overhead only). Set `HARPIA_BENCH_DECODER` to a real Hugging Face decoder directory (e.g.
//! Qwen2.5-0.5B) to measure a real workload, and record hardware and commit in
//! `docs/benchmarks.md`.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use harpia_core::traits::Tokenizer;
use harpia_data::pack::TokenizedSample;
use harpia_local::{TextGenerator, testing};
use harpia_train::{LoraConfig, LoraTrainConfig, train_lora_sft};
use std::hint::black_box;

const STEPS: usize = 4;

fn lora_sft(c: &mut Criterion) {
    let tmp = tempfile::tempdir().unwrap();
    let (base, kind) = match std::env::var_os("HARPIA_BENCH_DECODER") {
        Some(p) => (std::path::PathBuf::from(p), "real"),
        None => {
            let d = tmp.path().join("base");
            testing::qwen2(&d).unwrap();
            (d, "tiny")
        }
    };
    let lm = TextGenerator::load(&base).unwrap();
    let text = "the harpy eagle eats sloths in the forest ".repeat(8);
    let ids = lm.tokenizer().encode(&text, false).unwrap().ids;
    let tokens = ids.len();
    let loss_mask = vec![true; tokens];
    let samples = vec![TokenizedSample { ids, loss_mask }];
    drop(lm);

    let cfg = LoraTrainConfig {
        lora: LoraConfig { r: 16, alpha: 32.0, ..Default::default() },
        epochs: STEPS,
        lr: 1e-3,
        warmup_ratio: 0.0,
        ..Default::default()
    };
    let mut g = c.benchmark_group(format!("train_{kind}"));
    g.sample_size(10);
    g.throughput(Throughput::Elements((tokens * STEPS) as u64));
    g.bench_function(format!("lora_sft_{STEPS}_steps_{tokens}_tokens"), |b| {
        b.iter(|| {
            let out = tmp.path().join("out");
            let _ = std::fs::remove_dir_all(&out);
            black_box(train_lora_sft(&base, &out, &cfg, &samples, |_, _, _| {}).unwrap())
        })
    });
    g.finish();
}

criterion_group!(benches, lora_sft);
criterion_main!(benches);
