//! Anthropic Messages API over raw HTTP (Rust has no official Anthropic SDK).
//!
//! Sends `anthropic-version: 2026-10-21`. On that API version sampling parameters
//! (`temperature`, `top_p`, `top_k`) are rejected, so they are **not** forwarded; depth and cost
//! are controlled with `effort` (see [`Anthropic::effort`]) and `max_tokens`.

use crate::common::{Secret, chunk_stream, host_of, json_ok, status_error};
use harpia_core::traits::{ChatChunk, ChatRequest, ChatResponse, ChatStream, FinishReason, Provider};
use harpia_core::types::{Message, Role, TokenUsage, ToolCall};
use harpia_core::{HarpiaError, Result};
use harpia_http::{HttpClient, Request};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

/// API version sent in the `anthropic-version` header.
pub const API_VERSION: &str = "2026-10-21";

/// Model used in Harpia's examples and docs.
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";

/// Anthropic provider.
///
/// ```no_run
/// use harpia_client::Anthropic;
/// use harpia_core::traits::{ChatRequest, Provider};
/// use harpia_core::types::Message;
/// use harpia_core::NetworkPolicy;
/// use harpia_http::{HttpClient, HttpConfig};
/// use std::sync::Arc;
/// # async fn run() -> harpia_core::Result<()> {
/// let http = Arc::new(HttpClient::new(HttpConfig::new(NetworkPolicy::Online))?);
/// let claude = Anthropic::from_env(http)?; // ANTHROPIC_API_KEY
/// let req = ChatRequest { model: "claude-opus-5-5".into(), messages: vec![Message::user("Hello")], ..Default::default() };
/// let reply = claude.chat(&req).await?;
/// println!("{}", reply.message.content);
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct Anthropic {
    http: Arc<HttpClient>,
    base_url: String,
    api_key: Secret,
    /// `output_config.effort` (`low`, `medium`, `high`, `xhigh`, `max`); `None` uses the model
    /// default.
    pub effort: Option<String>,
    /// Values for the `anthropic-beta` header.
    pub betas: Vec<String>,
}

impl Anthropic {
    /// Provider with an explicit key.
    pub fn new(http: Arc<HttpClient>, api_key: Secret) -> Self {
        Self { http, base_url: "https://api.anthropic.com".into(), api_key, effort: None, betas: vec![] }
    }

    /// Provider reading `ANTHROPIC_API_KEY`.
    pub fn from_env(http: Arc<HttpClient>) -> Result<Self> {
        let key = Secret::from_env("ANTHROPIC_API_KEY")
            .ok_or_else(|| HarpiaError::invalid("ANTHROPIC_API_KEY is not set"))?;
        Ok(Self::new(http, key))
    }

    /// Override the base URL (gateways, tests).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    fn request(&self, body: &Value) -> Request {
        let mut r = Request::post_json(format!("{}/v1/messages", self.base_url), body)
            .header("x-api-key", self.api_key.expose())
            .header("anthropic-version", API_VERSION);
        if !self.betas.is_empty() {
            r = r.header("anthropic-beta", self.betas.join(","));
        }
        r
    }

    /// The request body for `/v1/messages`.
    pub fn body(&self, req: &ChatRequest, stream: bool) -> Value {
        let system: Vec<&str> =
            req.messages.iter().filter(|m| m.role == Role::System).map(|m| m.content.as_str()).collect();
        let mut messages: Vec<Value> = Vec::new();
        for m in req.messages.iter().filter(|m| m.role != Role::System) {
            match m.role {
                Role::User => messages.push(json!({"role": "user", "content": m.content})),
                Role::Assistant => {
                    let mut blocks = Vec::new();
                    if !m.content.is_empty() {
                        blocks.push(json!({"type": "text", "text": m.content}));
                    }
                    for (i, c) in m.tool_calls.iter().enumerate() {
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": c.id.clone().unwrap_or_else(|| format!("toolu_{i}")),
                            "name": c.name,
                            "input": c.arguments,
                        }));
                    }
                    messages.push(json!({"role": "assistant", "content": blocks}));
                }
                Role::Tool => {
                    let block = json!({"type": "tool_result", "tool_use_id": m.tool_call_id.clone().unwrap_or_default(), "content": m.content});
                    // all results of one assistant turn go in a single user message
                    let merged = messages.last_mut().filter(|last| {
                        last["role"] == "user"
                            && last["content"].as_array().is_some_and(|a| a.iter().all(|b| b["type"] == "tool_result"))
                    });
                    match merged {
                        Some(last) => last["content"].as_array_mut().expect("array").push(block),
                        None => messages.push(json!({"role": "user", "content": [block]})),
                    }
                }
                Role::System => unreachable!("filtered"),
            }
        }
        let mut b = json!({"model": req.model, "max_tokens": req.params.max_new_tokens, "messages": messages});
        let o = b.as_object_mut().expect("object");
        if !system.is_empty() {
            o.insert("system".into(), json!(system.join("\n\n")));
        }
        if !req.params.stop.is_empty() {
            o.insert("stop_sequences".into(), json!(req.params.stop));
        }
        let mut output_config = serde_json::Map::new();
        if let Some(e) = &self.effort {
            output_config.insert("effort".into(), json!(e));
        }
        if let Some(schema) = &req.response_schema {
            output_config.insert("format".into(), json!({"type": "json_schema", "schema": schema}));
        }
        if !output_config.is_empty() {
            o.insert("output_config".into(), Value::Object(output_config));
        }
        if !req.tools.is_empty() {
            let tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.parameters}))
                .collect();
            o.insert("tools".into(), json!(tools));
        }
        if stream {
            o.insert("stream".into(), json!(true));
        }
        b
    }
}

pub(crate) fn finish_reason(s: Option<&str>) -> FinishReason {
    match s {
        Some("max_tokens") | Some("model_context_window_exceeded") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("refusal") => FinishReason::ContentFilter,
        _ => FinishReason::Stop, // end_turn, stop_sequence, pause_turn
    }
}

/// `input_tokens` in [`TokenUsage`] is the whole prompt: uncached input plus cache writes
/// plus cache reads; `cached_input_tokens` is the cache-read part.
fn usage(u: &Value) -> TokenUsage {
    let read = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
    let write = u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    TokenUsage {
        input_tokens: u["input_tokens"].as_u64().unwrap_or(0) + read + write,
        output_tokens: u["output_tokens"].as_u64().unwrap_or(0),
        cached_input_tokens: read,
    }
}

/// Parse a non-streaming Messages response. Thinking blocks are skipped.
pub fn parse_response(v: &Value, model: &str) -> Result<ChatResponse> {
    let blocks = v["content"].as_array().ok_or_else(|| HarpiaError::Validation("message without content".into()))?;
    let mut message = Message::assistant(String::new());
    for b in blocks {
        match b["type"].as_str() {
            Some("text") => message.content.push_str(b["text"].as_str().unwrap_or("")),
            Some("tool_use") => message.tool_calls.push(ToolCall {
                id: b["id"].as_str().map(String::from),
                name: b["name"].as_str().unwrap_or("").to_string(),
                arguments: b["input"].clone(),
            }),
            _ => {}
        }
    }
    Ok(ChatResponse {
        model: v["model"].as_str().unwrap_or(model).to_string(),
        message,
        finish_reason: finish_reason(v["stop_reason"].as_str()),
        usage: usage(&v["usage"]),
    })
}

#[derive(Default)]
struct StreamAcc {
    tools: BTreeMap<u64, (Option<String>, String, String)>,
    usage: TokenUsage,
    finish: Option<FinishReason>,
    done: bool,
}

impl StreamAcc {
    fn final_chunk(&mut self) -> Vec<ChatChunk> {
        if self.done {
            return vec![];
        }
        self.done = true;
        let tool_calls = std::mem::take(&mut self.tools)
            .into_values()
            .map(|(id, name, json)| ToolCall {
                id,
                name,
                arguments: serde_json::from_str(if json.is_empty() { "{}" } else { &json })
                    .or_else(|_| harpia_structured::repair::repair(&json))
                    .unwrap_or(Value::String(json)),
            })
            .collect();
        vec![ChatChunk {
            delta: String::new(),
            tool_calls,
            usage: Some(self.usage),
            finish_reason: Some(self.finish.unwrap_or(FinishReason::Stop)),
        }]
    }
}

impl Provider for Anthropic {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let r = self.http.send(self.request(&self.body(request, false))).await?;
        parse_response(&json_ok(&host_of(&self.base_url), &r)?, &request.model)
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let host = host_of(&self.base_url);
        let mut resp = self.http.send_streaming(self.request(&self.body(request, true))).await?;
        if !(200..300).contains(&resp.status) {
            let mut body = Vec::new();
            while let Some(Ok(b)) = futures_util::StreamExt::next(&mut resp.body).await {
                body.extend_from_slice(&b);
            }
            return Err(status_error(&host, resp.status, &body));
        }
        Ok(chunk_stream(
            resp.body,
            true,
            StreamAcc::default(),
            move |acc, ev| {
                if ev.data.is_empty() {
                    return Ok(vec![]);
                }
                let v: Value = serde_json::from_str(&ev.data)?;
                match v["type"].as_str().unwrap_or_default() {
                    "message_start" => {
                        let u = usage(&v["message"]["usage"]);
                        acc.usage.input_tokens = u.input_tokens;
                        acc.usage.cached_input_tokens = u.cached_input_tokens;
                        acc.usage.output_tokens = u.output_tokens;
                        Ok(vec![])
                    }
                    "content_block_start" => {
                        let cb = &v["content_block"];
                        if cb["type"] == "tool_use" {
                            acc.tools.insert(
                                v["index"].as_u64().unwrap_or(0),
                                (
                                    cb["id"].as_str().map(String::from),
                                    cb["name"].as_str().unwrap_or("").to_string(),
                                    String::new(),
                                ),
                            );
                        }
                        Ok(vec![])
                    }
                    "content_block_delta" => {
                        let d = &v["delta"];
                        match d["type"].as_str() {
                            Some("text_delta") => Ok(vec![ChatChunk {
                                delta: d["text"].as_str().unwrap_or("").to_string(),
                                ..Default::default()
                            }]),
                            Some("input_json_delta") => {
                                if let Some(t) = acc.tools.get_mut(&v["index"].as_u64().unwrap_or(0)) {
                                    t.2.push_str(d["partial_json"].as_str().unwrap_or(""));
                                }
                                Ok(vec![])
                            }
                            _ => Ok(vec![]), // thinking_delta, signature_delta, ...
                        }
                    }
                    "message_delta" => {
                        if let Some(s) = v["delta"]["stop_reason"].as_str() {
                            acc.finish = Some(finish_reason(Some(s)));
                        }
                        if let Some(o) = v["usage"]["output_tokens"].as_u64() {
                            acc.usage.output_tokens = o;
                        }
                        Ok(vec![])
                    }
                    "message_stop" => Ok(acc.final_chunk()),
                    "error" => Err(HarpiaError::Remote {
                        host: host.clone(),
                        message: v["error"]["message"].as_str().unwrap_or("stream error").to_string(),
                    }),
                    _ => Ok(vec![]), // ping, content_block_stop
                }
            },
            StreamAcc::final_chunk,
        ))
    }
}
