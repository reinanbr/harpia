//! Structured output on top of any [`Provider`]: native JSON-schema mode, then extraction,
//! repair, full JSON Schema validation and corrective retries.

use harpia_core::traits::{ChatRequest, ChatResponse, Provider};
use harpia_core::types::Message;
use harpia_core::{HarpiaError, Result};
use harpia_structured::schema::{JsonSchema, schema_for, validate};
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Make a schema acceptable to strict structured-output modes: drop `$schema`, and add
/// `"additionalProperties": false` to every object schema that does not set it.
///
/// ```
/// let s = harpia_client::strict_schema(serde_json::json!({"$schema": "x", "type": "object", "properties": {"a": {"type": "object"}}}));
/// assert_eq!(s["additionalProperties"], false);
/// assert_eq!(s["properties"]["a"]["additionalProperties"], false);
/// assert!(s.get("$schema").is_none());
/// ```
pub fn strict_schema(mut schema: Value) -> Value {
    fn walk(v: &mut Value) {
        match v {
            Value::Object(o) => {
                if o.get("type").and_then(Value::as_str) == Some("object") || o.contains_key("properties") {
                    o.entry("additionalProperties").or_insert(Value::Bool(false));
                }
                for (_, child) in o.iter_mut() {
                    walk(child);
                }
            }
            Value::Array(a) => a.iter_mut().for_each(walk),
            _ => {}
        }
    }
    if let Some(o) = schema.as_object_mut() {
        o.remove("$schema");
    }
    walk(&mut schema);
    schema
}

/// Ask for JSON matching `schema` and deserialize it into `T`.
///
/// The schema is sent in the provider's native structured-output mode. The reply is then
/// extracted/repaired ([`harpia_structured::parse_llm_json`]) and validated with full JSON
/// Schema. On failure the model is shown the error and asked again, up to `max_retries` times.
/// The returned response carries the usage summed over all attempts.
pub async fn chat_json<P, T>(
    provider: &P,
    request: &ChatRequest,
    schema: &Value,
    max_retries: u32,
) -> Result<(T, ChatResponse)>
where
    P: Provider,
    T: DeserializeOwned,
{
    let mut req = request.clone();
    req.response_schema = Some(schema.clone());
    let mut total = harpia_core::TokenUsage::default();
    let mut last_err = String::new();
    for _ in 0..=max_retries {
        let mut resp = provider.chat(&req).await?;
        total += resp.usage;
        let attempt: Result<T> = harpia_structured::parse_llm_json::<Value>(&resp.message.content).and_then(|v| {
            validate(&v, schema)?;
            Ok(serde_json::from_value(v)?)
        });
        match attempt {
            Ok(value) => {
                resp.usage = total;
                return Ok((value, resp));
            }
            Err(e) => {
                last_err = e.to_string();
                req.messages.push(Message::assistant(resp.message.content.clone()));
                req.messages.push(Message::user(format!(
                    "Your previous reply did not match the required JSON schema: {last_err}. Reply with only the corrected JSON."
                )));
            }
        }
    }
    Err(HarpiaError::Validation(format!("no valid JSON after {} attempt(s): {last_err}", max_retries + 1)))
}

/// [`chat_json`] with the schema derived from `T` (made strict with [`strict_schema`]).
pub async fn chat_typed<P, T>(provider: &P, request: &ChatRequest, max_retries: u32) -> Result<(T, ChatResponse)>
where
    P: Provider,
    T: DeserializeOwned + JsonSchema,
{
    let schema = strict_schema(schema_for::<T>());
    chat_json(provider, request, &schema, max_retries).await
}
