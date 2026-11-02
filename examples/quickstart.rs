//! The README quick start, compiled: curate rows, render a chat template, parse LLM JSON.
//!
//! `cargo run -p harpia --example quickstart`

use harpia::core::traits::ChatTemplate;
use harpia::core::types::Message;
use harpia::data::curate::{Pipeline, Stage};
use harpia::data::filter::{QualityFilter, QualityThresholds};
use harpia::data::template::{CHATML, JinjaChatTemplate};
use harpia::text::Lang;

#[derive(serde::Deserialize, Debug)]
#[allow(dead_code)]
struct Verdict {
    label: String,
    score: f64,
}

fn main() -> harpia::Result<()> {
    // 1. curate raw text
    let rows = vec![
        serde_json::json!({"text": "Contact ana@example.com about the harpy eagle survey results."}),
        serde_json::json!({"text": "ok"}),
    ];
    let pipe = Pipeline::new("text")
        .stage(Stage::RedactPii(Default::default()))
        .stage(Stage::Filter(Box::new(QualityFilter::new(QualityThresholds::minimal(), Lang::En.data()))))
        .stage(Stage::ExactDedup);
    let (clean, report) = pipe.run(rows)?;
    println!("kept {} of {}: {}", report.output, report.input, clean[0]["text"]);

    // 2. render chat data with the model's template and find the tokens to train on
    let template = JinjaChatTemplate::new(CHATML, "", "<|im_end|>")?;
    let rendered = template.render_with_spans(&[Message::user("Hi"), Message::assistant("Hello!")])?;
    println!("trained span: {:?}", rendered.assistant_spans[0].slice(&rendered.text));

    // 3. get JSON out of a chatty model reply
    let v: Verdict = harpia::structured::parse_llm_json("Sure! {label: 'supports', score: 0.9,}")?;
    println!("{v:?}");
    Ok(())
}
