//! Benchmarks for segmentation, statistics, shingling, MinHash and fuzzy matching.
//!
//! Run with `cargo bench -p harpia-text`. Compare against the Python references described in
//! `docs/benchmarks.md` on the same machine; do not publish numbers measured elsewhere.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use harpia_text::{Lang, fuzzy, segment, similarity, stats};
use std::hint::black_box;

fn corpus() -> String {
    let para = "O gavião-real é uma das maiores aves de rapina do mundo. Vive nas florestas tropicais \
                da América Central e do Sul, onde caça preguiças e macacos no dossel. Segundo Silva et al. \
                (2020), a espécie depende de grandes áreas de floresta contínua, e o desmatamento reduz \
                drasticamente os locais de nidificação disponíveis. ";
    para.repeat(400) // ~150 KB, the size of a 20-file LaTeX article
}

fn bench(c: &mut Criterion) {
    let text = corpus();
    let pt = Lang::PtBr.data();
    let mut g = c.benchmark_group("text");
    g.throughput(Throughput::Bytes(text.len() as u64));
    g.bench_function("words", |b| b.iter(|| segment::words(black_box(&text)).count()));
    g.bench_function("sentences_pt", |b| b.iter(|| segment::sentences(black_box(&text), &pt).count()));
    g.bench_function("signals", |b| b.iter(|| stats::signals(black_box(&text), &pt)));
    g.bench_function("flesch_pt", |b| b.iter(|| stats::flesch(black_box(&text), &pt)));
    g.bench_function("mattr_100", |b| b.iter(|| stats::mattr(black_box(&text), 100)));
    g.bench_function("shingles_k5", |b| b.iter(|| similarity::shingle_set(black_box(&text), 5).len()));
    let mh = similarity::MinHasher::new(128, 1);
    g.bench_function("minhash_128", |b| b.iter(|| mh.signature_of(black_box(&text), 5)));
    g.finish();

    let page = &text[..8000];
    let quote = "a espécie depende de grandes áreas de floresta contínua";
    c.bench_function("fuzzy/partial_ratio_8k", |b| b.iter(|| fuzzy::partial_ratio(black_box(quote), black_box(page))));
    c.bench_function("fuzzy/token_set_ratio", |b| {
        b.iter(|| {
            fuzzy::token_set_ratio(black_box("harpy eagle forest canopy"), black_box("the harpy eagle of the forest"))
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
