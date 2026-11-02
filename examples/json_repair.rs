//! Repairing and streaming LLM JSON, and checking a constrained-decoding prefix.
//!
//! `cargo run -p harpia --example json_repair`

use harpia::structured::IncrementalJson;
use harpia::structured::constraint::{Constraint, JsonSchemaConstraint};
use harpia::structured::repair::repair_to_string;
use harpia::structured::toolcall::parse_tool_calls;

fn main() -> harpia::Result<()> {
    let messy = "Of course! Here is the result:\n```json\n{'label': 'supports', confidence: .92, evidence: [\"p. 3\",], // done\n";
    println!("repaired: {}", repair_to_string(messy)?);

    let mut stream = IncrementalJson::new();
    for chunk in ["{\"answer\": \"Bras", "ília\", \"year\": 19", "60}"] {
        stream.push(chunk);
        println!("partial: {:?} (complete: {})", stream.current(), stream.is_complete());
    }

    let schema = serde_json::json!({
        "type": "object",
        "properties": {"label": {"enum": ["supports", "contradicts", "neutral"]}},
        "required": ["label"], "additionalProperties": false
    });
    let c = JsonSchemaConstraint::new(&schema)?;
    for prefix in [r#"{"label": "sup"#, r#"{"label": "maybe"#, r#"{"label": "neutral"}"#] {
        println!("{prefix:<24} → {:?}", c.check(prefix));
    }

    let calls = parse_tool_calls(r#"<tool_call>{"name": "search", "arguments": {"q": "harpia harpyja"}}</tool_call>"#);
    println!("tool call: {} {}", calls[0].name, calls[0].arguments);
    Ok(())
}
