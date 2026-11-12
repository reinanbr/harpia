//! Data and retrieval benchmarks.
//!
//! - `data/curate`: normalize + quality filter + PII redaction over JSONL-like rows (MiB/s).
//! - `data/minhash_dedup`: MinHash + LSH near-duplicate detection (MiB/s).
//! - `data/template_mask_pack`: chat template rendering with assistant spans, loss mask and
//!   best-fit packing to 4096 (tokens/s). Token ids are bytes here so the figure isolates
//!   Harpia's own work; real tokenizers add their own cost.
//! - `retrieval/{flat,hnsw}_query`: top-10 queries over `HARPIA_BENCH_VECTORS` (default
//!   100 000) random 128-d vectors.
//!
//! Inputs are generated deterministically (fixed seeds).

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use harpia_core::traits::ChatTemplate;
use harpia_core::{Message, Span};
use harpia_data::curate::{Pipeline, Stage};
use harpia_data::dedup::{LshParams, near_duplicates};
use harpia_data::filter::{QualityFilter, QualityThresholds};
use harpia_data::pack::{PackOptions, PackStrategy, TokenizedSample, pack};
use harpia_data::template::{CHATML, JinjaChatTemplate, loss_mask};
use harpia_data::vector::{FlatIndex, HnswIndex, HnswParams, Metric};
use harpia_text::Lang;
use harpia_text::normalize::NormalizeOptions;
use serde_json::{Value, json};
use std::hint::black_box;

/// Tiny deterministic PRNG (SplitMix64) so the bench needs no extra dependencies.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

const WORDS: &[&str] = &[
    "the",
    "harpy",
    "eagle",
    "hunts",
    "sloths",
    "and",
    "monkeys",
    "in",
    "tall",
    "emergent",
    "trees",
    "of",
    "lowland",
    "rainforest",
    "where",
    "it",
    "builds",
    "large",
    "nests",
    "from",
    "branches",
    "females",
    "are",
    "bigger",
    "than",
    "males",
    "deforestation",
    "threatens",
    "its",
    "habitat",
    "across",
    "brazil",
    "panama",
    "researchers",
    "track",
    "birds",
    "with",
    "radio",
    "collars",
    "every",
    "season",
    "while",
    "local",
    "guides",
    "report",
    "sightings",
];

fn sentence(rng: &mut Rng, words: usize) -> String {
    let mut s: String =
        (0..words).map(|_| WORDS[(rng.next() % WORDS.len() as u64) as usize]).collect::<Vec<_>>().join(" ");
    if let Some(first) = s.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    s.push('.');
    s
}

fn document(rng: &mut Rng) -> String {
    let n = 4 + (rng.next() % 8) as usize;
    let mut doc = (0..n)
        .map(|_| {
            let len = 8 + (rng.next() % 12) as usize;
            sentence(rng, len)
        })
        .collect::<Vec<_>>()
        .join(" ");
    if rng.next().is_multiple_of(10) {
        doc.push_str(" Contact field.team@example.org or +55 61 99999-1234 for the data.");
    }
    doc
}

fn rows(n: usize) -> Vec<Value> {
    let mut rng = Rng(7);
    (0..n).map(|i| json!({"id": i, "text": document(&mut rng)})).collect()
}

fn bytes_of(rows: &[Value]) -> u64 {
    rows.iter().map(|r| r["text"].as_str().map_or(0, str::len) as u64).sum()
}

fn curation(c: &mut Criterion) {
    let input = rows(2000);
    let pipe = Pipeline::new("text")
        .stage(Stage::Normalize(NormalizeOptions { collapse_whitespace: true, ..Default::default() }))
        .stage(Stage::Filter(Box::new(QualityFilter::new(
            QualityThresholds { min_words: Some(20), ..QualityThresholds::default() },
            Lang::En.data(),
        ))))
        .stage(Stage::RedactPii(Default::default()));
    let mut g = c.benchmark_group("data");
    g.throughput(Throughput::Bytes(bytes_of(&input)));
    g.bench_function("curate", |b| {
        b.iter_batched(|| input.clone(), |rows| black_box(pipe.run(rows).unwrap()), BatchSize::LargeInput)
    });

    // near-duplicates: every 5th document is a lightly edited copy of the previous one
    let mut texts: Vec<String> = input.iter().map(|r| r["text"].as_str().unwrap_or("").to_string()).collect();
    for i in (5..texts.len()).step_by(5) {
        texts[i] = texts[i - 1].replacen("the", "a", 1);
    }
    let total: u64 = texts.iter().map(|t| t.len() as u64).sum();
    g.throughput(Throughput::Bytes(total));
    g.bench_function("minhash_dedup", |b| {
        b.iter(|| black_box(near_duplicates(&texts, LshParams { shingle: 5, ..LshParams::for_threshold(0.8) })))
    });

    // chat template + loss mask + pack
    let tmpl = JinjaChatTemplate::new(CHATML, "", "<|im_end|>").unwrap();
    let mut rng = Rng(11);
    let chats: Vec<Vec<Message>> = (0..500)
        .map(|_| {
            vec![
                Message::system("You are a field biologist."),
                Message::user(sentence(&mut rng, 12)),
                Message::assistant(document(&mut rng)),
            ]
        })
        .collect();
    let byte_tokens: u64 = chats.iter().map(|m| tmpl.render(m, false).unwrap().len() as u64).sum();
    g.throughput(Throughput::Elements(byte_tokens));
    g.bench_function("template_mask_pack", |b| {
        b.iter(|| {
            let samples: Vec<TokenizedSample> = chats
                .iter()
                .map(|m| {
                    let r = tmpl.render_with_spans(m).unwrap();
                    let offsets: Vec<Span> = (0..r.text.len()).map(|i| Span::new(i, i + 1)).collect();
                    TokenizedSample {
                        ids: r.text.bytes().map(u32::from).collect(),
                        loss_mask: loss_mask(&offsets, &r.assistant_spans),
                    }
                })
                .collect();
            let opts = PackOptions { strategy: PackStrategy::BestFitDecreasing, ..PackOptions::new(4096, 0) };
            black_box(pack(samples, opts).unwrap())
        })
    });
    g.finish();
}

fn retrieval(c: &mut Criterion) {
    let n: usize = std::env::var("HARPIA_BENCH_VECTORS").ok().and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let dim = 128;
    let mut rng = Rng(3);
    let mut vec = || (0..dim).map(|_| rng.unit()).collect::<Vec<f32>>();
    let data: Vec<Vec<f32>> = (0..n).map(|_| vec()).collect();
    let queries: Vec<Vec<f32>> = (0..64).map(|_| vec()).collect();

    let mut flat = FlatIndex::new(dim, Metric::Cosine);
    let mut hnsw = HnswIndex::new(dim, Metric::Cosine, HnswParams::default());
    let t0 = std::time::Instant::now();
    for (i, v) in data.iter().enumerate() {
        flat.add(&i.to_string(), v).unwrap();
    }
    let flat_build = t0.elapsed();
    let t0 = std::time::Instant::now();
    for (i, v) in data.iter().enumerate() {
        hnsw.add(&i.to_string(), v).unwrap();
    }
    eprintln!("built {n} x {dim}: flat {flat_build:.2?}, hnsw {:.2?}", t0.elapsed());

    let mut g = c.benchmark_group("retrieval");
    g.throughput(Throughput::Elements(queries.len() as u64));
    g.bench_function(format!("flat_query_{n}"), |b| {
        b.iter(|| {
            for q in &queries {
                black_box(flat.search(q, 10, None).unwrap());
            }
        })
    });
    g.bench_function(format!("hnsw_query_{n}"), |b| {
        b.iter(|| {
            for q in &queries {
                black_box(hnsw.search(q, 10, None).unwrap());
            }
        })
    });
    g.finish();
}

criterion_group!(benches, curation, retrieval);
criterion_main!(benches);
