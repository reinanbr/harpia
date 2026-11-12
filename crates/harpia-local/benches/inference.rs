//! Local inference benchmarks.
//!
//! By default the models are the tiny random-weight ones from `harpia_local::testing`, which
//! measure plumbing overhead only. Point the environment variables at real Hugging Face model
//! directories to measure real workloads (and record hardware and commit in
//! `docs/benchmarks.md`):
//!
//! - `HARPIA_BENCH_DECODER`: decoder LLM (e.g. Qwen2.5-0.5B-Instruct) → time to first token and
//!   32-token seeded sampling;
//! - `HARPIA_BENCH_ENCODER`: sequence classifier / cross-encoder (e.g. an XLM-R NLI model) →
//!   pairs/s at batch 32;
//! - `HARPIA_BENCH_EMBEDDER`: sentence-embedding encoder (e.g. e5-small) → embeddings/s at
//!   batch 32.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use harpia_core::traits::{Embedder, GenerationParams, Model, PairClassifier};
use harpia_local::{EncoderEmbedder, SequenceClassifier, TextGenerator, testing};
use std::hint::black_box;
use std::path::PathBuf;

/// The model directory from `var`, or a tiny model written by `make` into a temp dir.
fn model_dir(
    var: &str,
    tmp: &tempfile::TempDir,
    make: fn(&std::path::Path) -> harpia_core::Result<()>,
) -> (PathBuf, &'static str) {
    match std::env::var_os(var) {
        Some(p) => (PathBuf::from(p), "real"),
        None => {
            let dir = tmp.path().join(var);
            make(&dir).expect("tiny model");
            (dir, "tiny")
        }
    }
}

const PROMPT: &str = "the harpy eagle eats sloths in the forest";

fn decoder(c: &mut Criterion) {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, kind) = model_dir("HARPIA_BENCH_DECODER", &tmp, testing::qwen2);
    let lm = TextGenerator::load(&dir).unwrap();
    let mut g = c.benchmark_group(format!("decoder_{kind}"));
    g.sample_size(10);
    let first = GenerationParams { max_new_tokens: 1, ..Default::default() };
    g.bench_function("time_to_first_token", |b| b.iter(|| black_box(lm.generate(PROMPT, &first).unwrap())));
    let params =
        GenerationParams { max_new_tokens: 32, temperature: 0.8, top_p: 0.95, seed: Some(0), ..Default::default() };
    let produced = lm.generate(PROMPT, &params).unwrap().token_ids.len().max(1);
    g.throughput(Throughput::Elements(produced as u64));
    g.bench_function(format!("sampled_{produced}_tokens"), |b| {
        b.iter(|| black_box(lm.generate(PROMPT, &params).unwrap()))
    });
    g.finish();
}

fn encoders(c: &mut Criterion) {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, kind) = model_dir("HARPIA_BENCH_ENCODER", &tmp, |d| {
        testing::xlm_roberta_classifier(d, &["entailment", "neutral", "contradiction"])
    });
    let clf = SequenceClassifier::load(&dir).unwrap().with_batch_size(32);
    let premise = "The harpy eagle lives in tropical lowland rainforests and hunts sloths and monkeys in the canopy.";
    let pairs: Vec<(&str, &str)> = (0..32)
        .map(|i| {
            (premise, if i % 2 == 0 { "The harpy eagle eats sloths." } else { "The harpy eagle lives in Europe." })
        })
        .collect();
    let mut g = c.benchmark_group(format!("encoder_{kind}"));
    g.sample_size(10);
    g.throughput(Throughput::Elements(pairs.len() as u64));
    g.bench_function("classify_pairs_batch32", |b| b.iter(|| black_box(clf.classify_pairs(&pairs).unwrap())));
    g.finish();

    let (dir, kind) = model_dir("HARPIA_BENCH_EMBEDDER", &tmp, |d| testing::xlm_roberta_classifier(d, &["a", "b"]));
    let emb = EncoderEmbedder::load(&dir).unwrap();
    let texts: Vec<&str> = (0..32).map(|i| if i % 2 == 0 { premise } else { "the harpy eagle eats sloths" }).collect();
    let mut g = c.benchmark_group(format!("embedder_{kind}"));
    g.sample_size(10);
    g.throughput(Throughput::Elements(texts.len() as u64));
    g.bench_function("embed_batch32", |b| b.iter(|| black_box(emb.embed(&texts).unwrap())));
    g.finish();
}

criterion_group!(benches, decoder, encoders);
criterion_main!(benches);
