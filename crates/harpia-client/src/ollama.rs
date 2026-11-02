//! Ollama native API (`/api/chat`, `/api/embed`). Works over plain `http://`, so it needs no
//! TLS and stays pure Rust.

use crate::common::{chunk_stream, host_of, json_ok, status_error};
use harpia_core::traits::{ChatChunk, ChatRequest, ChatResponse, ChatStream, FinishReason, Provider};
use harpia_core::types::{Message, TokenUsage, ToolCall};
use harpia_core::{HarpiaError, Result};
use harpia_http::{HttpClient, Request};
use serde_json::{Value, json};
use std::sync::Arc;

/// Ollama provider.
#[derive(Debug, Clone)]
pub struct Ollama {
    http: Arc<HttpClient>,
    base_url: String,
}

impl Ollama {
    /// Provider for `base_url` (default server: `http://localhost:11434`).
    pub fn new(http: Arc<HttpClient>, base_url: impl Into<String>) -> Self {
        Self { http, base_url: base_url.into().trim_end_matches('/').to_string() }
    }

    /// Provider for the default local server.
    pub fn local(http: Arc<HttpClient>) -> Self {
        Self::new(http, "http://localhost:11434")
    }

    /// The request body for `/api/chat`.
    pub fn body(&self, req: &ChatRequest, stream: bool) -> Value {
        let messages: Vec<Value> = req
            .messages
            .iter()
            .map(|m| {
                let mut v = json!({"role": m.role.as_str(), "content": m.content});
                if !m.tool_calls.is_empty() {
                    v["tool_calls"] = json!(
                        m.tool_calls
                            .iter()
                            .map(|c| json!({"function": {"name": c.name, "arguments": c.arguments}}))
                            .collect::<Vec<_>>()
                    );
                }
                v
            })
            .collect();
        let p = &req.params;
        let mut options = json!({"num_predict": p.max_new_tokens, "temperature": p.temperature});
        let o = options.as_object_mut().expect("object");
        if p.top_k > 0 {
            o.insert("top_k".into(), json!(p.top_k));
        }
        if p.top_p < 1.0 {
            o.insert("top_p".into(), json!(p.top_p));
        }
        if p.min_p > 0.0 {
            o.insert("min_p".into(), json!(p.min_p));
        }
        if p.repetition_penalty != 1.0 {
            o.insert("repeat_penalty".into(), json!(p.repetition_penalty));
        }
        if let Some(s) = p.seed {
            o.insert("seed".into(), json!(s));
        }
        if !p.stop.is_empty() {
            o.insert("stop".into(), json!(p.stop));
        }
        let mut b = json!({"model": req.model, "messages": messages, "stream": stream, "options": options});
        if let Some(schema) = &req.response_schema {
            b["format"] = schema.clone();
        }
        if !req.tools.is_empty() {
            b["tools"] = json!(
                req.tools
                    .iter()
                    .map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}))
                    .collect::<Vec<_>>()
            );
        }
        b
    }

    /// Embed texts with `/api/embed`.
    pub async fn embed(&self, model: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let r = self
            .http
            .send(Request::post_json(format!("{}/api/embed", self.base_url), &json!({"model": model, "input": texts})))
            .await?;
        let v = json_ok(&host_of(&self.base_url), &r)?;
        Ok(v["embeddings"]
            .as_array()
            .ok_or_else(|| HarpiaError::Remote {
                host: host_of(&self.base_url),
                message: "no embeddings in response".into(),
            })?
            .iter()
            .map(|e| {
                e.as_array().map(|a| a.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()).unwrap_or_default()
            })
            .collect())
    }
}

fn tool_calls(msg: &Value) -> Vec<ToolCall> {
    msg["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| ToolCall {
            id: None,
            name: c["function"]["name"].as_str().unwrap_or("").to_string(),
            arguments: c["function"]["arguments"].clone(),
        })
        .collect()
}

fn finish(v: &Value, has_tools: bool) -> FinishReason {
    if has_tools {
        return FinishReason::ToolCalls;
    }
    match v["done_reason"].as_str() {
        Some("length") => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

fn usage(v: &Value) -> TokenUsage {
    TokenUsage {
        input_tokens: v["prompt_eval_count"].as_u64().unwrap_or(0),
        output_tokens: v["eval_count"].as_u64().unwrap_or(0),
        cached_input_tokens: 0,
    }
}

impl Provider for Ollama {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let r = self
            .http
            .send(Request::post_json(format!("{}/api/chat", self.base_url), &self.body(request, false)))
            .await?;
        let v = json_ok(&host_of(&self.base_url), &r)?;
        let mut message = Message::assistant(v["message"]["content"].as_str().unwrap_or(""));
        message.tool_calls = tool_calls(&v["message"]);
        let fr = finish(&v, !message.tool_calls.is_empty());
        Ok(ChatResponse {
            model: v["model"].as_str().unwrap_or(&request.model).to_string(),
            message,
            finish_reason: fr,
            usage: usage(&v),
        })
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let host = host_of(&self.base_url);
        let mut resp = self
            .http
            .send_streaming(Request::post_json(format!("{}/api/chat", self.base_url), &self.body(request, true)))
            .await?;
        if !(200..300).contains(&resp.status) {
            let mut body = Vec::new();
            while let Some(Ok(b)) = futures_util::StreamExt::next(&mut resp.body).await {
                body.extend_from_slice(&b);
            }
            return Err(status_error(&host, resp.status, &body));
        }
        Ok(chunk_stream(
            resp.body,
            false,
            Vec::<ToolCall>::new(),
            |calls, ev| {
                let v: Value = serde_json::from_str(&ev.data)?;
                if let Some(e) = v["error"].as_str() {
                    return Err(HarpiaError::Remote { host: "ollama".into(), message: e.to_string() });
                }
                calls.extend(tool_calls(&v["message"]));
                let delta = v["message"]["content"].as_str().unwrap_or("").to_string();
                if v["done"].as_bool() == Some(true) {
                    let tc = std::mem::take(calls);
                    let fr = finish(&v, !tc.is_empty());
                    return Ok(vec![ChatChunk {
                        delta,
                        tool_calls: tc,
                        usage: Some(usage(&v)),
                        finish_reason: Some(fr),
                    }]);
                }
                Ok(if delta.is_empty() { vec![] } else { vec![ChatChunk { delta, ..Default::default() }] })
            },
            |_| vec![],
        ))
    }
}
