//! OpenAI Chat Completions API and compatible servers.

use crate::common::{Secret, chunk_stream, host_of, json_ok, status_error};
use harpia_core::traits::{ChatChunk, ChatRequest, ChatResponse, ChatStream, FinishReason, Provider};
use harpia_core::types::{Message, TokenUsage, ToolCall};
use harpia_core::{HarpiaError, Result};
use harpia_http::{HttpClient, Request};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

/// OpenAI-compatible provider (`/chat/completions`, `/embeddings`).
///
/// ```no_run
/// use harpia_client::OpenAiCompatible;
/// use harpia_http::{HttpClient, HttpConfig};
/// use harpia_core::NetworkPolicy;
/// use std::sync::Arc;
/// let http = Arc::new(HttpClient::new(HttpConfig::new(NetworkPolicy::Online)).unwrap());
/// // a local vLLM / llama.cpp server needs no key
/// let local = OpenAiCompatible::new(http.clone(), "http://localhost:8000/v1", None);
/// let openai = OpenAiCompatible::openai(http).unwrap(); // reads OPENAI_API_KEY
/// ```
#[derive(Debug, Clone)]
pub struct OpenAiCompatible {
    http: Arc<HttpClient>,
    base_url: String,
    api_key: Option<Secret>,
    /// Also send `top_k`, `min_p` and `repetition_penalty` (understood by vLLM and
    /// llama.cpp; rejected by OpenAI). Default `false`.
    pub extended_sampling: bool,
    /// Extra headers (e.g. `HTTP-Referer` for OpenRouter).
    pub extra_headers: Vec<(String, String)>,
}

impl OpenAiCompatible {
    /// Provider for `base_url` (e.g. `https://api.openai.com/v1`).
    pub fn new(http: Arc<HttpClient>, base_url: impl Into<String>, api_key: Option<Secret>) -> Self {
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            extended_sampling: false,
            extra_headers: vec![],
        }
    }

    /// OpenAI itself, with `OPENAI_API_KEY` from the environment.
    pub fn openai(http: Arc<HttpClient>) -> Result<Self> {
        let key =
            Secret::from_env("OPENAI_API_KEY").ok_or_else(|| HarpiaError::invalid("OPENAI_API_KEY is not set"))?;
        Ok(Self::new(http, "https://api.openai.com/v1", Some(key)))
    }

    fn request(&self, path: &str, body: &Value) -> Request {
        let mut r = Request::post_json(format!("{}{path}", self.base_url), body);
        if let Some(k) = &self.api_key {
            r = r.header("authorization", format!("Bearer {}", k.expose()));
        }
        for (k, v) in &self.extra_headers {
            r = r.header(k.clone(), v.clone());
        }
        r
    }

    /// The request body for `/chat/completions`.
    pub fn body(&self, req: &ChatRequest, stream: bool) -> Value {
        let messages: Vec<Value> = req.messages.iter().map(message_json).collect();
        let p = &req.params;
        let mut b = json!({ "model": req.model, "messages": messages, "max_tokens": p.max_new_tokens, "temperature": p.temperature });
        let o = b.as_object_mut().expect("object");
        if p.top_p < 1.0 {
            o.insert("top_p".into(), json!(p.top_p));
        }
        if !p.stop.is_empty() {
            o.insert("stop".into(), json!(p.stop));
        }
        if let Some(s) = p.seed {
            o.insert("seed".into(), json!(s));
        }
        if p.frequency_penalty != 0.0 {
            o.insert("frequency_penalty".into(), json!(p.frequency_penalty));
        }
        if p.presence_penalty != 0.0 {
            o.insert("presence_penalty".into(), json!(p.presence_penalty));
        }
        if self.extended_sampling {
            if p.top_k > 0 {
                o.insert("top_k".into(), json!(p.top_k));
            }
            if p.min_p > 0.0 {
                o.insert("min_p".into(), json!(p.min_p));
            }
            if p.repetition_penalty != 1.0 {
                o.insert("repetition_penalty".into(), json!(p.repetition_penalty));
            }
        }
        if let Some(schema) = &req.response_schema {
            o.insert(
                "response_format".into(),
                json!({"type": "json_schema", "json_schema": {"name": "response", "schema": schema, "strict": true}}),
            );
        }
        if !req.tools.is_empty() {
            let tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}))
                .collect();
            o.insert("tools".into(), json!(tools));
        }
        if stream {
            o.insert("stream".into(), json!(true));
            o.insert("stream_options".into(), json!({"include_usage": true}));
        }
        b
    }

    /// Embed texts with `/embeddings`.
    pub async fn embed(&self, model: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let r = self.http.send(self.request("/embeddings", &json!({"model": model, "input": texts}))).await?;
        let v = json_ok(&host_of(&self.base_url), &r)?;
        let mut data: Vec<(u64, Vec<f32>)> = v["data"]
            .as_array()
            .ok_or_else(|| HarpiaError::Remote {
                host: host_of(&self.base_url),
                message: "embeddings response without data".into(),
            })?
            .iter()
            .map(|d| {
                let idx = d["index"].as_u64().unwrap_or(0);
                let emb = d["embedding"]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_f64).map(|x| x as f32).collect())
                    .unwrap_or_default();
                (idx, emb)
            })
            .collect();
        data.sort_by_key(|(i, _)| *i);
        Ok(data.into_iter().map(|(_, e)| e).collect())
    }
}

fn message_json(m: &Message) -> Value {
    let mut v = json!({"role": m.role.as_str(), "content": m.content});
    if !m.tool_calls.is_empty() {
        let calls: Vec<Value> = m
            .tool_calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({
                    "id": c.id.clone().unwrap_or_else(|| format!("call_{i}")),
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments.to_string()},
                })
            })
            .collect();
        v["tool_calls"] = json!(calls);
    }
    if let Some(id) = &m.tool_call_id {
        v["tool_call_id"] = json!(id);
    }
    if let Some(n) = &m.name {
        v["name"] = json!(n);
    }
    v
}

fn parse_arguments(s: &str) -> Value {
    serde_json::from_str(s).or_else(|_| harpia_structured::repair::repair(s)).unwrap_or(Value::String(s.to_string()))
}

pub(crate) fn finish_reason(s: Option<&str>) -> FinishReason {
    match s {
        Some("length") => FinishReason::Length,
        Some("tool_calls") | Some("function_call") => FinishReason::ToolCalls,
        Some("content_filter") => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

fn usage(u: &Value) -> TokenUsage {
    TokenUsage {
        input_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
        output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
        cached_input_tokens: u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
    }
}

/// Parse a non-streaming completion.
pub fn parse_response(v: &Value, model: &str) -> Result<ChatResponse> {
    let choice = v["choices"].get(0).ok_or_else(|| HarpiaError::Validation("completion without choices".into()))?;
    let msg = &choice["message"];
    let mut message = Message::assistant(msg["content"].as_str().unwrap_or(""));
    if let Some(calls) = msg["tool_calls"].as_array() {
        message.tool_calls = calls
            .iter()
            .map(|c| ToolCall {
                id: c["id"].as_str().map(String::from),
                name: c["function"]["name"].as_str().unwrap_or("").to_string(),
                arguments: parse_arguments(c["function"]["arguments"].as_str().unwrap_or("{}")),
            })
            .collect();
    }
    Ok(ChatResponse {
        model: v["model"].as_str().unwrap_or(model).to_string(),
        message,
        finish_reason: finish_reason(choice["finish_reason"].as_str()),
        usage: usage(&v["usage"]),
    })
}

#[derive(Default)]
struct StreamAcc {
    tools: BTreeMap<u64, (Option<String>, String, String)>,
    finish: Option<FinishReason>,
    usage: Option<TokenUsage>,
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
            .map(|(id, name, args)| ToolCall {
                id,
                name,
                arguments: parse_arguments(if args.is_empty() { "{}" } else { &args }),
            })
            .collect();
        vec![ChatChunk {
            delta: String::new(),
            tool_calls,
            usage: self.usage,
            finish_reason: Some(self.finish.unwrap_or(FinishReason::Stop)),
        }]
    }
}

impl Provider for OpenAiCompatible {
    fn name(&self) -> &str {
        "openai-compatible"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let r = self.http.send(self.request("/chat/completions", &self.body(request, false))).await?;
        parse_response(&json_ok(&host_of(&self.base_url), &r)?, &request.model)
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let host = host_of(&self.base_url);
        let mut resp = self.http.send_streaming(self.request("/chat/completions", &self.body(request, true))).await?;
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
            |acc, ev| {
                if ev.data.trim() == "[DONE]" {
                    return Ok(acc.final_chunk());
                }
                let v: Value = serde_json::from_str(&ev.data)?;
                if let Some(err) = v.get("error") {
                    return Err(HarpiaError::Remote { host: "stream".into(), message: err.to_string() });
                }
                if v["usage"].is_object() {
                    acc.usage = Some(usage(&v["usage"]));
                }
                let mut out = Vec::new();
                if let Some(choice) = v["choices"].get(0) {
                    let delta = &choice["delta"];
                    if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
                        out.push(ChatChunk { delta: text.to_string(), ..Default::default() });
                    }
                    for tc in delta["tool_calls"].as_array().into_iter().flatten() {
                        let e = acc.tools.entry(tc["index"].as_u64().unwrap_or(0)).or_default();
                        if let Some(id) = tc["id"].as_str() {
                            e.0 = Some(id.to_string());
                        }
                        if let Some(n) = tc["function"]["name"].as_str() {
                            e.1.push_str(n);
                        }
                        if let Some(a) = tc["function"]["arguments"].as_str() {
                            e.2.push_str(a);
                        }
                    }
                    if let Some(f) = choice["finish_reason"].as_str() {
                        acc.finish = Some(finish_reason(Some(f)));
                    }
                }
                Ok(out)
            },
            StreamAcc::final_chunk,
        ))
    }
}
