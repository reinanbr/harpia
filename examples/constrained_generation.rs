//! Local generation with constrained decoding: a JSON Schema, a fixed set of choices and a
//! regex. The constraint masks the logits at every step, so the output is valid by construction
//! even from a model that has never seen JSON.
//!
//! The model here is a random-weight Qwen2 in a temporary directory. Its tokenizer is word-level
//! with a few dozen words (`{ } " : , label supports contradicts yes no ...`), so the
//! constraints below are spelled with those words; with a real model directory any schema,
//! choice list or regex works the same way.
//!
//! `cargo run -p harpia --release --example constrained_generation --features local`

use harpia::core::traits::{ChatRequest, GenerationParams, Provider};
use harpia::core::types::Message;
use harpia::local::{TextGenerator, testing};
use harpia::structured::constraint::{Choice, Constraint, JsonSchemaConstraint, PrefixStatus, RegexConstraint};
use serde_json::json;

fn main() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    testing::qwen2(dir.path())?;
    let lm = TextGenerator::load(dir.path())?;

    let schema = json!({
        "type": "object",
        "properties": {"label": {"enum": ["supports", "contradicts"]}},
        "required": ["label"],
        "additionalProperties": false
    });
    let params = GenerationParams { max_new_tokens: 64, temperature: 0.8, seed: Some(7), ..Default::default() };

    let js = JsonSchemaConstraint::new(&schema)?;
    let g = lm.generate_constrained("the harpy eagle", &params, &js)?;
    println!("json schema -> {}  ({:?}, {} tokens)", g.text, js.check(&g.text), g.token_ids.len());
    if js.check(&g.text) == PrefixStatus::Complete {
        let v: serde_json::Value = serde_json::from_str(&g.text)?;
        println!("  parsed label = {}", v["label"]);
    }

    let label = Choice::new(["supports", "contradicts"]);
    let g = lm.generate_constrained("label", &GenerationParams { max_new_tokens: 8, ..Default::default() }, &label)?;
    println!("choice      -> {:?}", g.text);

    let answers = RegexConstraint::new("(yes|no)(,(yes|no)){2}")?;
    let g =
        lm.generate_constrained("a bird", &GenerationParams { max_new_tokens: 5, ..Default::default() }, &answers)?;
    println!("regex       -> {:?}  ({:?})", g.text, answers.check(&g.text));

    // Token log-probabilities come with every generation (used by eval and perplexity).
    println!("logprobs    -> {:?}", g.logprobs.iter().map(|l| format!("{l:.2}")).collect::<Vec<_>>());

    // The same model behind the `Provider` trait: chat template + response schema.
    let req = ChatRequest {
        model: "local".into(),
        messages: vec![Message::user("hello")],
        params,
        response_schema: Some(schema),
        tools: vec![],
    };
    let rt = tokio::runtime::Builder::new_current_thread().build()?;
    let resp = rt.block_on(lm.chat(&req))?;
    println!("chat        -> {}", resp.message.content);
    Ok(())
}
