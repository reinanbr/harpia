//! Parse tool/function calls emitted as text by models.
//!
//! Recognized formats (all may appear multiple times in one output):
//! - Hermes/Qwen tags: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`;
//! - OpenAI: `{"tool_calls": [{"id": ..., "function": {"name": ..., "arguments": "<json string>"}}]}`
//!   or a bare `{"name": ..., "arguments": ...}` / `{"function": {...}}`;
//! - Anthropic: `{"type": "tool_use", "id": ..., "name": ..., "input": {...}}`.

use harpia_core::ToolCall;
use serde_json::Value;

/// Extract every tool call found in `text`.
///
/// ```
/// use harpia_structured::toolcall::parse_tool_calls;
/// let out = r#"Let me check. <tool_call>{"name": "search", "arguments": {"q": "harpia"}}</tool_call>"#;
/// let calls = parse_tool_calls(out);
/// assert_eq!(calls[0].name, "search");
/// assert_eq!(calls[0].arguments["q"], "harpia");
/// ```
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let mut rest = text;
    let mut tagged = false;
    while let Some(open) = rest.find("<tool_call>") {
        tagged = true;
        let body = &rest[open + "<tool_call>".len()..];
        let end = body.find("</tool_call>").unwrap_or(body.len());
        if let Ok(v) = crate::repair::repair(&body[..end]) {
            collect(&v, &mut out);
        }
        rest = &body[end..];
    }
    if !tagged {
        for js in crate::extract::extract_all(text) {
            if let Ok(v) = serde_json::from_str::<Value>(js) {
                collect(&v, &mut out);
            }
        }
    }
    out
}

fn arguments(v: Option<&Value>) -> Value {
    match v {
        // OpenAI encodes arguments as a JSON string
        Some(Value::String(s)) => crate::repair::repair(s).unwrap_or(Value::String(s.clone())),
        Some(v) => v.clone(),
        None => Value::Object(Default::default()),
    }
}

fn collect(v: &Value, out: &mut Vec<ToolCall>) {
    match v {
        Value::Array(items) => items.iter().for_each(|i| collect(i, out)),
        Value::Object(o) => {
            if let Some(calls) = o.get("tool_calls") {
                return collect(calls, out);
            }
            if let Some(Value::Object(f)) = o.get("function") {
                if let Some(name) = f.get("name").and_then(Value::as_str) {
                    out.push(ToolCall {
                        id: o.get("id").and_then(Value::as_str).map(String::from),
                        name: name.to_owned(),
                        arguments: arguments(f.get("arguments").or(f.get("parameters"))),
                    });
                }
                return;
            }
            if let Some(name) = o.get("name").and_then(Value::as_str) {
                let args = o.get("arguments").or(o.get("input")).or(o.get("parameters"));
                let is_call = args.is_some() || o.get("type").and_then(Value::as_str) == Some("tool_use");
                if is_call {
                    out.push(ToolCall {
                        id: o.get("id").and_then(Value::as_str).map(String::from),
                        name: name.to_owned(),
                        arguments: arguments(args),
                    });
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_and_anthropic_formats() {
        let oa = r#"{"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}}]}"#;
        let c = parse_tool_calls(oa);
        assert_eq!((c[0].id.as_deref(), c[0].name.as_str(), c[0].arguments["x"].as_i64()), (Some("c1"), "f", Some(1)));
        let an = r#"[{"type": "tool_use", "id": "t1", "name": "g", "input": {"y": true}}]"#;
        let c = parse_tool_calls(an);
        assert_eq!(c[0].name, "g");
        assert_eq!(c[0].arguments["y"], true);
        assert!(parse_tool_calls(r#"{"name": "just data"}"#).is_empty());
    }

    #[test]
    fn multiple_hermes_tags() {
        let t = "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>\n<tool_call>{'name': 'b', 'arguments': {'k': 2}}</tool_call>";
        let names: Vec<String> = parse_tool_calls(t).into_iter().map(|c| c.name).collect();
        assert_eq!(names, ["a", "b"]);
    }
}
