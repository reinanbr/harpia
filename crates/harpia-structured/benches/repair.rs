//! Benchmarks for JSON extraction, repair and constrained-decoding checks.

use criterion::{Criterion, criterion_group, criterion_main};
use harpia_structured::constraint::{Constraint, JsonSchemaConstraint};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    let valid = serde_json::to_string(&serde_json::json!({
        "items": (0..200).map(|i| serde_json::json!({"id": i, "label": "supports", "score": 0.5, "tags": ["a", "b"]})).collect::<Vec<_>>()
    }))
    .unwrap();
    let broken = format!(
        "Sure, here it is:\n```json\n{}",
        valid.replace("\"label\"", "label").replace("\"supports\"", "'supports'").trim_end_matches("]}")
    );
    c.bench_function("structured/extract_valid", |b| b.iter(|| harpia_structured::extract_json(black_box(&valid))));
    c.bench_function("structured/repair_broken", |b| b.iter(|| harpia_structured::repair::repair(black_box(&broken))));
    let schema = serde_json::json!({"type": "object", "properties": {"label": {"enum": ["supports", "contradicts", "neutral"]}, "score": {"type": "number"}}, "required": ["label", "score"]});
    let cons = JsonSchemaConstraint::new(&schema).unwrap();
    let prefix = r#"{"label": "supports", "score": 0.87"#;
    c.bench_function("structured/schema_prefix_check", |b| b.iter(|| cons.check(black_box(prefix))));
}

criterion_group!(benches, bench);
criterion_main!(benches);
