//! End-to-end LLM data preparation on an in-memory toy corpus:
//! clean → quality filter → PII redaction → near-dedup → decontaminate → chat format →
//! seeded split → dataset card.
//!
//! `cargo run -p harpia --example llm_data_pipeline`

use harpia::data::card::{DatasetCard, SplitStats};
use harpia::data::curate::{Pipeline, Stage};
use harpia::data::decontam::Decontaminator;
use harpia::data::dedup::{LshParams, near_duplicates};
use harpia::data::filter::{QualityFilter, QualityThresholds};
use harpia::data::format::{Format, from_chat_sample, to_chat_sample};
use harpia::data::split::{Ratios, random_split};
use harpia::text::Lang;
use harpia::text::normalize::NormalizeOptions;
use serde_json::{Value, json};

fn main() -> harpia::Result<()> {
    let raw = vec![
        json!({"instruction": "What do harpy eagles eat?", "output": "Harpy eagles mainly eat sloths and monkeys that they catch in the forest canopy, plus some birds and reptiles."}),
        json!({"instruction": "What do harpy eagles eat?", "output": "Harpy eagles mainly eat sloths and monkeys that they catch in the forest canopy, plus some birds and reptiles!"}),
        json!({"instruction": "Email me", "output": "Sure, write to someone@example.org and I will forward the field notes about the nests."}),
        json!({"instruction": "x", "output": "ok"}),
        json!({"instruction": "Capital of Brazil?", "output": "The capital of Brazil is Brasília, which was inaugurated in 1960 as a planned city."}),
    ];

    // 1. curation on the answer field
    let pipe = Pipeline::new("output")
        .stage(Stage::Normalize(NormalizeOptions { collapse_whitespace: true, ..Default::default() }))
        .stage(Stage::Filter(Box::new(QualityFilter::new(
            QualityThresholds { min_words: Some(5), ..QualityThresholds::default() },
            Lang::En.data(),
        ))))
        .stage(Stage::RedactPii(Default::default()));
    let (rows, report) = pipe.run(raw)?;

    // 2. near-duplicate removal
    let texts: Vec<&str> = rows.iter().map(|r| r["output"].as_str().unwrap_or("")).collect();
    let keep = near_duplicates(&texts, LshParams { shingle: 3, ..LshParams::for_threshold(0.7) });
    let rows: Vec<Value> = rows.into_iter().zip(&keep).filter(|(_, k)| **k).map(|(r, _)| r).collect();

    // 3. decontamination against a toy benchmark
    let mut decon = Decontaminator::new(6);
    decon.add_benchmark("toy-geo", ["Q: The capital of Brazil is Brasília, which was inaugurated in 1960. True?"]);
    let rows: Vec<Value> =
        rows.into_iter().filter(|r| decon.check(r["output"].as_str().unwrap_or(""), 1).is_none()).collect();

    // 4. chat format + split
    let chats: Vec<Value> = rows
        .iter()
        .map(|r| to_chat_sample(r, Format::Alpaca).and_then(|s| from_chat_sample(&s, Format::Messages)))
        .collect::<harpia::Result<_>>()?;
    let split = random_split(chats.len(), Ratios::new(1.0, 0.0, 0.0)?, 42);

    // 5. card
    let mut card = DatasetCard::new("harpy-toy", "Toy instruction data about harpy eagles.");
    card.license = Some("CC0-1.0".into());
    card.curation = Some(report);
    card.splits.insert("train".into(), SplitStats { rows: split.train.len() as u64, tokens: None });
    card.extra.insert("near_duplicates_removed".into(), json!(keep.iter().filter(|k| !**k).count()));
    println!("{}", serde_json::to_string_pretty(&chats)?);
    println!("{}", card.to_markdown());
    Ok(())
}
