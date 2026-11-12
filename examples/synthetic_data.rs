//! Synthetic data generation with schema-constrained output against a mocked provider:
//! render a prompt per row, call the provider through the batch runner, parse the JSON reply,
//! attach provenance, and resume from the cache on a second run.
//!
//! Swap the mock for `OpenAiCompatible`, `Anthropic`, `Ollama` or a local `TextGenerator`; the
//! code is the same because they all implement `Provider`.
//!
//! `cargo run -p harpia --example synthetic_data --features client,eval`

use harpia::client::{BatchRunner, SynthTemplate, synthesize};
use harpia::eval::testing::FnProvider;
use serde_json::{Value, json};

const TEMPLATE: &str = r#"
model = "mock-writer"
system = "You write short exam questions as JSON."
user = "Write one {{ level }} question about {{ topic }}."
output_field = "generated"
params = { temperature = 0.7, seed = 42, max_new_tokens = 128 }

[schema]
type = "object"
required = ["question", "answer"]
properties = { question = { type = "string" }, answer = { type = "string" } }
"#;

fn main() -> harpia::Result<()> {
    let template = SynthTemplate::from_toml(TEMPLATE)?;
    let rows = vec![
        json!({"topic": "harpy eagle diet", "level": "easy"}),
        json!({"topic": "harpy eagle habitat", "level": "medium"}),
        json!({"topic": "harpy eagle conservation", "level": "hard"}),
    ];

    // The mock answers like a real model would: JSON wrapped in prose and a code fence, which
    // harpia-structured extracts and parses.
    let provider = FnProvider::new(|prompt: &str| {
        let topic = prompt.split(" about ").nth(1).unwrap_or("it").trim_end_matches('.');
        format!(
            "Sure! Here it is:\n```json\n{{\"question\": \"What do you know about {topic}?\", \"answer\": \"See the field guide.\"}}\n```"
        )
    });

    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("synth.jsonl");
    let runner = BatchRunner::new(4).requests_per_minute(600.0);

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let (out, report) = rt.block_on(synthesize(&provider, &template, &rows, &runner, &cache))?;
    println!("first run: {} generated, {} from cache", report.completed, report.from_cache);
    for row in &out {
        println!("{}", serde_json::to_string(row)?);
    }

    // Interrupted jobs resume: rows already in the cache are not sent again.
    let (again, report) = rt.block_on(synthesize(&provider, &template, &rows, &runner, &cache))?;
    println!("second run: {} generated, {} from cache", report.completed, report.from_cache);
    assert_eq!(again, out);

    let questions: Vec<&Value> = out.iter().map(|r| &r["generated"]["question"]).collect();
    assert!(questions.iter().all(|q| q.is_string()));
    Ok(())
}
