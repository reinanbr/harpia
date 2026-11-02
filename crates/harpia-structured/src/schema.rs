//! JSON Schema validation and schema derivation from Rust types (feature `schema`).

use harpia_core::{HarpiaError, Result};
use serde_json::Value;

pub use schemars::JsonSchema;

/// Validate `instance` against `schema` (full JSON Schema, via the `jsonschema` crate).
/// The error lists every violation with its JSON pointer.
///
/// ```
/// use serde_json::json;
/// let schema = json!({"type": "object", "required": ["a"]});
/// assert!(harpia_structured::schema::validate(&json!({"a": 1}), &schema).is_ok());
/// assert!(harpia_structured::schema::validate(&json!({}), &schema).is_err());
/// ```
pub fn validate(instance: &Value, schema: &Value) -> Result<()> {
    let validator =
        jsonschema::validator_for(schema).map_err(|e| HarpiaError::invalid(format!("invalid JSON Schema: {e}")))?;
    let errors: Vec<String> =
        validator.iter_errors(instance).map(|e| format!("{}: {}", e.instance_path(), e)).collect();
    if errors.is_empty() { Ok(()) } else { Err(HarpiaError::Validation(errors.join("; "))) }
}

/// JSON Schema of a Rust type deriving [`JsonSchema`], for prompts and tool definitions.
///
/// ```
/// #[derive(schemars::JsonSchema, serde::Deserialize)]
/// struct Answer { label: String, confidence: f64 }
/// let s = harpia_structured::schema::schema_for::<Answer>();
/// assert_eq!(s["properties"]["confidence"]["type"], "number");
/// ```
pub fn schema_for<T: JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).expect("schemars output is valid JSON")
}

/// Parse LLM output into `T` and validate it against `T`'s derived schema.
pub fn parse_validated<T: JsonSchema + serde::de::DeserializeOwned>(text: &str) -> Result<T> {
    let schema = schema_for::<T>();
    let value: Value = crate::parse_llm_json(text)?;
    validate(&value, &schema)?;
    Ok(serde_json::from_value(value)?)
}
