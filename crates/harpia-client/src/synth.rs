//! Synthetic data generation: render a prompt template per dataset row, call a provider,
//! attach the output plus provenance to the row. Resumable through the batch cache.

use crate::batch::{BatchReport, BatchRunner};
use harpia_core::traits::{ChatRequest, ChatResponse, GenerationParams, Provider};
use harpia_core::types::Message;
use harpia_core::{HarpiaError, Result};
use harpia_data::JsonlCache;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

/// A generation recipe.
///
/// ```
/// let t = harpia_client::SynthTemplate::from_toml(r#"
/// model = "claude-opus-5-5"
/// system = "You write exam questions."
/// user = "Write one question about: {{ topic }}"
/// output_field = "question"
/// "#).unwrap();
/// let req = t.render(&serde_json::json!({"topic": "harpy eagles"})).unwrap();
/// assert_eq!(req.messages[1].content, "Write one question about: harpy eagles");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynthTemplate {
    /// Model name at the provider.
    pub model: String,
    /// Optional system prompt (a Jinja template over the row).
    #[serde(default)]
    pub system: Option<String>,
    /// User prompt (a Jinja template over the row's fields).
    pub user: String,
    /// JSON Schema for structured output; the parsed value is stored instead of text.
    #[serde(default)]
    pub schema: Option<Value>,
    /// Field that receives the output.
    #[serde(default = "default_field")]
    pub output_field: String,
    /// Decoding parameters.
    #[serde(default)]
    pub params: GenerationParams,
}

fn default_field() -> String {
    "output".into()
}

impl SynthTemplate {
    /// Parse a TOML recipe.
    pub fn from_toml(text: &str) -> Result<Self> {
        toml_from_str(text)
    }

    fn render_str(template: &str, row: &Value) -> Result<String> {
        let mut env = minijinja::Environment::new();
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
        env.render_str(template, minijinja::Value::from(minijinja::value::Serde(row)))
            .map_err(|e| HarpiaError::Template(format!("{e:#}")))
    }

    /// The request for one row (missing template variables are errors).
    pub fn render(&self, row: &Value) -> Result<ChatRequest> {
        let mut messages = Vec::new();
        if let Some(s) = &self.system {
            messages.push(Message::system(Self::render_str(s, row)?));
        }
        messages.push(Message::user(Self::render_str(&self.user, row)?));
        Ok(ChatRequest {
            model: self.model.clone(),
            messages,
            params: self.params.clone(),
            response_schema: self.schema.clone().map(crate::strict_schema),
            tools: vec![],
        })
    }
}

fn toml_from_str<T: serde::de::DeserializeOwned>(text: &str) -> Result<T> {
    toml::from_str(text).map_err(|e| HarpiaError::invalid(format!("synth template: {e}")))
}

/// Stable cache key of a request: model + rendered messages + schema.
pub fn request_key(req: &ChatRequest) -> String {
    let payload =
        json!({"model": req.model, "messages": req.messages, "schema": req.response_schema, "seed": req.params.seed});
    format!("{:016x}", xxhash_rust::xxh3::xxh3_64(payload.to_string().as_bytes()))
}

/// Generate for every row. Results are cached in `cache_path` (JSONL), so re-running after an
/// interruption only calls the provider for missing rows. Each output row is the input row plus
/// `output_field` and a `_provenance` object (model, provider, prompt hash, seed, usage).
pub async fn synthesize<P: Provider>(
    provider: &P,
    template: &SynthTemplate,
    rows: &[Value],
    runner: &BatchRunner,
    cache_path: &Path,
) -> Result<(Vec<Value>, BatchReport)> {
    let mut items = Vec::with_capacity(rows.len());
    let mut keys = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let req = template.render(row).map_err(|e| HarpiaError::Record {
            path: None,
            line: i as u64 + 1,
            message: e.to_string(),
        })?;
        let key = request_key(&req);
        keys.push(key.clone());
        items.push((key, req));
    }
    let mut cache: JsonlCache<String, ChatResponse> = JsonlCache::open(cache_path)?;
    let report = runner.run(provider, items, &mut cache).await?;
    let mut out = Vec::new();
    for (row, key) in rows.iter().zip(&keys) {
        let Some(resp) = cache.get(key) else { continue };
        let mut r = row.clone();
        let value = match &template.schema {
            Some(_) => harpia_structured::parse_llm_json::<Value>(&resp.message.content)
                .unwrap_or_else(|_| Value::String(resp.message.content.clone())),
            None => Value::String(resp.message.content.clone()),
        };
        if let Some(o) = r.as_object_mut() {
            o.insert(template.output_field.clone(), value);
            o.insert(
                "_provenance".into(),
                json!({
                    "provider": provider.name(), "model": resp.model, "prompt_hash": key,
                    "seed": template.params.seed, "usage": resp.usage, "finish_reason": resp.finish_reason,
                }),
            );
        }
        out.push(r);
    }
    Ok((out, report))
}
