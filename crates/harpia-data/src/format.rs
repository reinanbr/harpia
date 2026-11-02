//! Converters between common chat/instruction dataset schemas.

use harpia_core::types::{ChatSample, Message, Metadata, PreferencePair, Role};
use harpia_core::{HarpiaError, Result};
use serde_json::{Value, json};

/// A dataset row schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Format {
    /// OpenAI style: `{"messages": [{"role", "content"}]}`.
    Messages,
    /// ShareGPT: `{"conversations": [{"from": "human"|"gpt"|"system", "value"}]}`.
    ShareGpt,
    /// Alpaca: `{"instruction", "input"?, "output", "system"?}`.
    Alpaca,
    /// `{"prompt", "completion"}`.
    PromptCompletion,
}

impl Format {
    /// Parse `messages`, `sharegpt`, `alpaca`, `prompt-completion`.
    pub fn parse(s: &str) -> Result<Format> {
        match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "messages" | "openai" | "chatml" => Ok(Format::Messages),
            "sharegpt" | "conversations" => Ok(Format::ShareGpt),
            "alpaca" => Ok(Format::Alpaca),
            "prompt-completion" | "completion" => Ok(Format::PromptCompletion),
            other => Err(HarpiaError::invalid(format!("unknown dataset format `{other}`"))),
        }
    }

    /// Guess the format of a row from its keys.
    pub fn detect(row: &Value) -> Option<Format> {
        let o = row.as_object()?;
        if o.contains_key("messages") {
            Some(Format::Messages)
        } else if o.contains_key("conversations") {
            Some(Format::ShareGpt)
        } else if o.contains_key("instruction") && o.contains_key("output") {
            Some(Format::Alpaca)
        } else if o.contains_key("prompt") && o.contains_key("completion") {
            Some(Format::PromptCompletion)
        } else {
            None
        }
    }
}

fn str_field<'a>(row: &'a Value, k: &str) -> Option<&'a str> {
    row.get(k).and_then(Value::as_str)
}

fn need<'a>(row: &'a Value, k: &str) -> Result<&'a str> {
    str_field(row, k).ok_or_else(|| HarpiaError::Validation(format!("missing string field `{k}`")))
}

fn parse_messages(list: &Value, role_key: &str, content_key: &str) -> Result<Vec<Message>> {
    let arr = list.as_array().ok_or_else(|| HarpiaError::Validation("messages must be an array".into()))?;
    arr.iter()
        .map(|m| {
            let role_s = m
                .get(role_key)
                .and_then(Value::as_str)
                .ok_or_else(|| HarpiaError::Validation(format!("message without `{role_key}`")))?;
            let role =
                Role::parse(role_s).ok_or_else(|| HarpiaError::Validation(format!("unknown role `{role_s}`")))?;
            let content = match m.get(content_key) {
                Some(Value::String(s)) => s.clone(),
                // OpenAI multimodal content parts: keep the text parts
                Some(Value::Array(parts)) => {
                    parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n")
                }
                Some(Value::Null) | None => String::new(),
                Some(other) => other.to_string(),
            };
            let mut msg = Message::new(role, content);
            if let Some(Value::Array(calls)) = m.get("tool_calls") {
                msg.tool_calls = calls.iter().filter_map(|c| serde_json::from_value(c.clone()).ok()).collect();
            }
            msg.tool_call_id = m.get("tool_call_id").and_then(Value::as_str).map(String::from);
            Ok(msg)
        })
        .collect()
}

fn metadata(row: &Value, skip: &[&str]) -> Metadata {
    row.as_object()
        .map(|o| o.iter().filter(|(k, _)| !skip.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

/// Convert a row into a [`ChatSample`]. Unknown extra fields are kept as metadata.
///
/// ```
/// use harpia_data::format::{to_chat_sample, Format};
/// let row = serde_json::json!({"conversations": [{"from": "human", "value": "Oi"}, {"from": "gpt", "value": "Olá!"}], "source": "x"});
/// let s = to_chat_sample(&row, Format::ShareGpt).unwrap();
/// assert_eq!(s.messages[1].content, "Olá!");
/// assert_eq!(s.metadata["source"], "x");
/// ```
pub fn to_chat_sample(row: &Value, format: Format) -> Result<ChatSample> {
    match format {
        Format::Messages => Ok(ChatSample {
            messages: parse_messages(row.get("messages").unwrap_or(&Value::Null), "role", "content")?,
            metadata: metadata(row, &["messages"]),
        }),
        Format::ShareGpt => Ok(ChatSample {
            messages: parse_messages(row.get("conversations").unwrap_or(&Value::Null), "from", "value")?,
            metadata: metadata(row, &["conversations"]),
        }),
        Format::Alpaca => {
            let mut messages = Vec::new();
            if let Some(sys) = str_field(row, "system").filter(|s| !s.is_empty()) {
                messages.push(Message::system(sys));
            }
            let instruction = need(row, "instruction")?;
            let user = match str_field(row, "input").filter(|s| !s.trim().is_empty()) {
                Some(input) => format!("{instruction}\n\n{input}"),
                None => instruction.to_string(),
            };
            messages.push(Message::user(user));
            messages.push(Message::assistant(need(row, "output")?));
            Ok(ChatSample { messages, metadata: metadata(row, &["system", "instruction", "input", "output"]) })
        }
        Format::PromptCompletion => Ok(ChatSample {
            messages: vec![Message::user(need(row, "prompt")?), Message::assistant(need(row, "completion")?)],
            metadata: metadata(row, &["prompt", "completion"]),
        }),
    }
}

/// Serialize a [`ChatSample`] into `format`.
pub fn from_chat_sample(sample: &ChatSample, format: Format) -> Result<Value> {
    let mut out = match format {
        Format::Messages => json!({ "messages": sample.messages }),
        Format::ShareGpt => {
            let conv: Vec<Value> = sample
                .messages
                .iter()
                .map(|m| {
                    let from = match m.role {
                        Role::User => "human",
                        Role::Assistant => "gpt",
                        Role::System => "system",
                        Role::Tool => "tool",
                    };
                    json!({"from": from, "value": m.content})
                })
                .collect();
            json!({ "conversations": conv })
        }
        Format::Alpaca | Format::PromptCompletion => {
            let sys: Vec<&Message> = sample.messages.iter().filter(|m| m.role == Role::System).collect();
            let rest: Vec<&Message> = sample.messages.iter().filter(|m| m.role != Role::System).collect();
            if rest.len() != 2 || rest[0].role != Role::User || rest[1].role != Role::Assistant {
                return Err(HarpiaError::Validation(
                    "only single-turn user/assistant samples convert to alpaca or prompt-completion".into(),
                ));
            }
            if format == Format::Alpaca {
                let mut v = json!({"instruction": rest[0].content, "input": "", "output": rest[1].content});
                if let Some(s) = sys.first() {
                    v["system"] = json!(s.content);
                }
                v
            } else {
                json!({"prompt": rest[0].content, "completion": rest[1].content})
            }
        }
    };
    if let Some(o) = out.as_object_mut() {
        for (k, v) in &sample.metadata {
            o.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    Ok(out)
}

/// Parse a preference row: `prompt` (string or message list), `chosen` and `rejected`
/// (strings, or message lists whose last assistant turn is used).
pub fn to_preference(row: &Value) -> Result<PreferencePair> {
    let prompt = match row.get("prompt") {
        Some(Value::String(s)) => vec![Message::user(s.clone())],
        Some(v @ Value::Array(_)) => parse_messages(v, "role", "content")?,
        _ => return Err(HarpiaError::Validation("missing `prompt`".into())),
    };
    let answer = |k: &str| -> Result<String> {
        match row.get(k) {
            Some(Value::String(s)) => Ok(s.clone()),
            Some(v @ Value::Array(_)) => parse_messages(v, "role", "content")?
                .into_iter()
                .rev()
                .find(|m| m.role == Role::Assistant)
                .map(|m| m.content)
                .ok_or_else(|| HarpiaError::Validation(format!("`{k}` has no assistant message"))),
            _ => Err(HarpiaError::Validation(format!("missing `{k}`"))),
        }
    };
    Ok(PreferencePair {
        prompt,
        chosen: answer("chosen")?,
        rejected: answer("rejected")?,
        metadata: metadata(row, &["prompt", "chosen", "rejected"]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpaca_roundtrip() {
        let row = json!({"instruction": "Translate", "input": "cat", "output": "gato"});
        assert_eq!(Format::detect(&row), Some(Format::Alpaca));
        let s = to_chat_sample(&row, Format::Alpaca).unwrap();
        assert_eq!(s.messages[0].content, "Translate\n\ncat");
        let back = from_chat_sample(&s, Format::PromptCompletion).unwrap();
        assert_eq!(back["completion"], "gato");
        let sg = from_chat_sample(&s, Format::ShareGpt).unwrap();
        assert_eq!(sg["conversations"][0]["from"], "human");
    }

    #[test]
    fn preference_rows() {
        let row = json!({"prompt": "2+2?", "chosen": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}], "rejected": "5"});
        let p = to_preference(&row).unwrap();
        assert_eq!((p.chosen.as_str(), p.rejected.as_str()), ("4", "5"));
        assert!(to_preference(&json!({"prompt": "x"})).is_err());
    }
}
