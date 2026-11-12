//! `harpia serve --http`: an OpenAI-compatible local server (feature `server`).
//!
//! Endpoints: `POST /v1/chat/completions` (with SSE streaming), `POST /v1/completions`,
//! `POST /v1/embeddings`, `POST /v1/rerank`, `GET /v1/models`, `GET /health`, and
//! `POST /rpc` (the stdio JSON-RPC methods, without model paths). Only models registered on the
//! command line can be used. Inference runs on blocking threads; each model serves one request
//! at a time.

use super::engine::{
    Engine, RpcError, chat_ids, code, constraint, document_texts, gen_params, parse_messages, rank, run, str_list,
    to_reply,
};
use axum::Router;
use axum::extract::rejection::JsonRejection;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, extract::DefaultBodyLimit};
use harpia::core::TokenUsage;
use harpia::core::traits::FinishReason;
use harpia::core::traits::{Embedder, Reranker, Tokenizer};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

struct AppState {
    engine: Arc<Engine>,
    api_key: Option<String>,
    next_id: AtomicU64,
}

impl AppState {
    fn id(&self, prefix: &str) -> String {
        format!("{prefix}-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

type Shared = Arc<AppState>;

/// Bind `addr` and serve until Ctrl-C. Prints `harpia: listening on http://<addr>` to stderr
/// once the socket is bound.
pub fn serve(engine: Engine, addr: SocketAddr, api_key: Option<String>) -> harpia::core::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        if !local.ip().is_loopback() && api_key.is_none() {
            eprintln!("harpia: warning: listening on a non-loopback address without an API key");
        }
        eprintln!("harpia: listening on http://{local}");
        let state = Arc::new(AppState { engine: Arc::new(engine), api_key, next_id: AtomicU64::new(1) });
        axum::serve(listener, router(state))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

fn router(state: Shared) -> Router {
    let api = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/rpc", post(rpc))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth));
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .merge(api)
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(state)
}

// ---------------------------------------------------------------- errors and auth

struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, kind: "invalid_request_error", message: message.into() }
    }
}

impl From<RpcError> for ApiError {
    fn from(e: RpcError) -> Self {
        let (status, kind) = match e.code {
            code::INVALID_PARAMS | code::NO_JSON => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            code::NOT_AVAILABLE => (StatusCode::NOT_FOUND, "not_found_error"),
            code::OFFLINE => (StatusCode::FORBIDDEN, "network_disabled"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };
        Self { status, kind, message: e.message }
    }
}

impl From<harpia::core::HarpiaError> for ApiError {
    fn from(e: harpia::core::HarpiaError) -> Self {
        RpcError::from(e).into()
    }
}

impl From<JsonRejection> for ApiError {
    fn from(e: JsonRejection) -> Self {
        Self::bad(e.body_text())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({"error": {"message": self.message, "type": self.kind, "param": null, "code": null}});
        (self.status, Json(body)).into_response()
    }
}

type ApiResult = Result<Response, ApiError>;

/// Byte-wise comparison that does not stop at the first difference.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn auth(State(s): State<Shared>, headers: HeaderMap, req: Request, next: Next) -> Response {
    if let Some(key) = &s.api_key {
        let given = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if !same(given.as_bytes(), key.as_bytes()) {
            return ApiError {
                status: StatusCode::UNAUTHORIZED,
                kind: "authentication_error",
                message: "missing or invalid API key".into(),
            }
            .into_response();
        }
    }
    next.run(req).await
}

// ---------------------------------------------------------------- helpers

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn finish(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        _ => "stop",
    }
}

fn usage(u: &TokenUsage) -> Value {
    json!({"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens, "total_tokens": u.total()})
}

fn model_name(body: &Value) -> Result<Option<&str>, ApiError> {
    match body.get("model") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(ApiError::bad("`model` must be a string")),
    }
}

fn wants_stream(body: &Value) -> bool {
    body.get("stream").and_then(Value::as_bool).unwrap_or(false)
}

fn include_usage(body: &Value) -> bool {
    body.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false)
}

fn check_n(body: &Value) -> Result<(), ApiError> {
    match body.get("n").and_then(Value::as_u64) {
        Some(n) if n > 1 => Err(ApiError::bad("n > 1 is not supported")),
        _ => Ok(()),
    }
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, ApiError> + Send + 'static) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(f).await.map_err(|e| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        kind: "server_error",
        message: format!("worker failed: {e}"),
    })?
}

/// SSE response fed by a channel of JSON payloads; `[DONE]` is sent when the sender closes.
fn sse(rx: mpsc::UnboundedReceiver<Value>) -> Response {
    let stream = futures_util::stream::unfold((rx, false), |(mut rx, done)| async move {
        if done {
            return None;
        }
        match rx.recv().await {
            Some(v) => Some((Ok::<_, Infallible>(Event::default().data(v.to_string())), (rx, false))),
            None => Some((Ok(Event::default().data("[DONE]")), (rx, true))),
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

// ---------------------------------------------------------------- endpoints

async fn models(State(s): State<Shared>) -> Json<Value> {
    let e = &s.engine;
    let mut data = Vec::new();
    for (kind, names) in [
        ("decoder", e.decoders.names()),
        ("embedder", e.embedders.names()),
        ("reranker", e.rerankers.names()),
        ("classifier", e.classifiers.names()),
        ("translator", e.translators.names()),
    ] {
        for n in names {
            data.push(json!({"id": n, "object": "model", "created": 0, "owned_by": "harpia", "kind": kind}));
        }
    }
    Json(json!({"object": "list", "data": data}))
}

async fn chat_completions(State(s): State<Shared>, body: Result<Json<Value>, JsonRejection>) -> ApiResult {
    let Json(body) = body?;
    check_n(&body)?;
    let (name, lm) = s.engine.decoders.resolve(model_name(&body)?, false)?;
    let messages = parse_messages(body.get("messages").ok_or_else(|| ApiError::bad("missing `messages`"))?)?;
    let ids = chat_ids(&lm, &messages)?;
    let params = gen_params(&body, lm.default_params())?;
    let wants_tools = body.get("tools").and_then(Value::as_array).is_some_and(|t| !t.is_empty());
    let id = s.id("chatcmpl");
    let created = now();
    if !wants_stream(&body) {
        let g = blocking(move || {
            let c = constraint(&body)?;
            Ok(run(&lm, &ids, &params, c.as_deref(), false, &mut |_| {})?)
        })
        .await?;
        let reply = to_reply(g, wants_tools);
        let mut message = json!({"role": "assistant", "content": reply.message.content});
        if !reply.message.tool_calls.is_empty() {
            message["tool_calls"] = openai_tool_calls(&reply.message.tool_calls);
        }
        return Ok(Json(json!({
            "id": id, "object": "chat.completion", "created": created, "model": name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish(reply.finish_reason), "logprobs": null}],
            "usage": usage(&reply.usage),
        }))
        .into_response());
    }
    let c = constraint(&body)?;
    let with_usage = include_usage(&body);
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::task::spawn_blocking(move || {
        let chunk = |delta: Value, fin: Option<&str>| {
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": name,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": fin, "logprobs": null}]})
        };
        let _ = tx.send(chunk(json!({"role": "assistant", "content": ""}), None));
        let res = run(&lm, &ids, &params, c.as_deref(), true, &mut |d| {
            let _ = tx.send(chunk(json!({"content": d}), None));
        });
        match res {
            Ok(g) => {
                let reply = to_reply(g, wants_tools);
                let delta = if reply.message.tool_calls.is_empty() {
                    json!({})
                } else {
                    let mut calls = openai_tool_calls(&reply.message.tool_calls);
                    for (i, c) in calls.as_array_mut().into_iter().flatten().enumerate() {
                        c["index"] = json!(i);
                    }
                    json!({"tool_calls": calls})
                };
                let _ = tx.send(chunk(delta, Some(finish(reply.finish_reason))));
                if with_usage {
                    let _ = tx.send(json!({"id": id, "object": "chat.completion.chunk", "created": created,
                        "model": name, "choices": [], "usage": usage(&reply.usage)}));
                }
            }
            Err(e) => {
                let _ = tx.send(json!({"error": {"message": e.message, "type": "server_error"}}));
            }
        }
    });
    Ok(sse(rx))
}

fn openai_tool_calls(calls: &[harpia::core::types::ToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({
                    "id": c.id.clone().unwrap_or_else(|| format!("call_{i}")),
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments.to_string()},
                })
            })
            .collect(),
    )
}

async fn completions(State(s): State<Shared>, body: Result<Json<Value>, JsonRejection>) -> ApiResult {
    let Json(body) = body?;
    check_n(&body)?;
    if body.get("echo").and_then(Value::as_bool).unwrap_or(false) {
        return Err(ApiError::bad("echo is not supported"));
    }
    let (name, lm) = s.engine.decoders.resolve(model_name(&body)?, false)?;
    let prompts = str_list(&body, &["prompt"])?;
    if prompts.is_empty() {
        return Err(ApiError::bad("`prompt` is empty"));
    }
    let params = gen_params(&body, lm.default_params())?;
    let id = s.id("cmpl");
    let created = now();
    if !wants_stream(&body) {
        let out = blocking(move || {
            let c = constraint(&body)?;
            let mut choices = Vec::new();
            let mut total = TokenUsage::default();
            for (i, p) in prompts.iter().enumerate() {
                let ids = lm.tokenizer().encode(p, true)?.ids;
                let g = run(&lm, &ids, &params, c.as_deref(), false, &mut |_| {})?;
                total.input_tokens += g.usage.input_tokens;
                total.output_tokens += g.usage.output_tokens;
                choices.push(
                    json!({"index": i, "text": g.text, "finish_reason": finish(g.finish_reason), "logprobs": null}),
                );
            }
            Ok((choices, total))
        })
        .await?;
        return Ok(Json(json!({
            "id": id, "object": "text_completion", "created": created, "model": name,
            "choices": out.0, "usage": usage(&out.1),
        }))
        .into_response());
    }
    if prompts.len() != 1 {
        return Err(ApiError::bad("streaming supports a single prompt"));
    }
    let c = constraint(&body)?;
    let with_usage = include_usage(&body);
    let ids = lm.tokenizer().encode(&prompts[0], true)?.ids;
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::task::spawn_blocking(move || {
        let chunk = |text: &str, fin: Option<&str>| {
            json!({"id": id, "object": "text_completion", "created": created, "model": name,
                   "choices": [{"index": 0, "text": text, "finish_reason": fin, "logprobs": null}]})
        };
        match run(&lm, &ids, &params, c.as_deref(), true, &mut |d| {
            let _ = tx.send(chunk(d, None));
        }) {
            Ok(g) => {
                let mut last = chunk("", Some(finish(g.finish_reason)));
                if with_usage {
                    last["usage"] = usage(&g.usage);
                }
                let _ = tx.send(last);
            }
            Err(e) => {
                let _ = tx.send(json!({"error": {"message": e.message, "type": "server_error"}}));
            }
        }
    });
    Ok(sse(rx))
}

async fn embeddings(State(s): State<Shared>, body: Result<Json<Value>, JsonRejection>) -> ApiResult {
    let Json(body) = body?;
    let (name, emb) = s.engine.embedders.resolve(model_name(&body)?, false)?;
    let texts = str_list(&body, &["input"]).map_err(|_| {
        ApiError::bad("`input` must be a string or an array of strings (token arrays are not supported)")
    })?;
    let base64 = match body.get("encoding_format").and_then(Value::as_str) {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => return Err(ApiError::bad(format!("unsupported encoding_format `{other}`"))),
    };
    if body.get("dimensions").is_some_and(|d| d.as_u64() != Some(emb.dim() as u64)) {
        return Err(ApiError::bad(format!("this model only produces {}-dimensional embeddings", emb.dim())));
    }
    let vecs = blocking(move || {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Ok(emb.embed(&refs)?)
    })
    .await?;
    let data: Vec<Value> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let embedding = if base64 { json!(base64_f32(v)) } else { json!(v) };
            json!({"object": "embedding", "index": i, "embedding": embedding})
        })
        .collect();
    // token counts are not exposed by the embedder; report zeros rather than guess
    Ok(Json(json!({"object": "list", "data": data, "model": name, "usage": {"prompt_tokens": 0, "total_tokens": 0}}))
        .into_response())
}

/// Little-endian f32 bytes in standard base64, as OpenAI returns for `encoding_format: base64`.
fn base64_f32(v: &[f32]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n =
            (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    out
}

async fn rerank(State(s): State<Shared>, body: Result<Json<Value>, JsonRejection>) -> ApiResult {
    let Json(body) = body?;
    let (name, rr) = s.engine.rerankers.resolve(model_name(&body)?, false)?;
    let query = body.get("query").and_then(Value::as_str).ok_or_else(|| ApiError::bad("missing `query`"))?.to_string();
    let docs = document_texts(body.get("documents").ok_or_else(|| ApiError::bad("missing `documents`"))?)?;
    let top_n = body.get("top_n").and_then(Value::as_u64).map(|n| n as usize);
    let return_docs = body.get("return_documents").and_then(Value::as_bool).unwrap_or(false);
    let (scores, docs) = blocking(move || {
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let s = rr.rerank(&query, &refs)?;
        Ok((s, docs))
    })
    .await?;
    let results: Vec<Value> = rank(&scores, top_n)
        .into_iter()
        .map(|r| {
            let i = r["index"].as_u64().unwrap_or(0) as usize;
            let mut o = json!({"index": i, "relevance_score": r["score"]});
            if return_docs {
                o["document"] = json!({"text": docs[i]});
            }
            o
        })
        .collect();
    Ok(Json(json!({"id": s.id("rerank"), "model": name, "results": results})).into_response())
}

async fn rpc(State(s): State<Shared>, body: Result<Json<Value>, JsonRejection>) -> ApiResult {
    let Json(req) = body?;
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let Some(method) = req.get("method").and_then(Value::as_str).map(String::from) else {
        return Ok(Json(json!({"jsonrpc": "2.0", "id": id, "error": {"code": code::INVALID_REQUEST, "message": "missing `method`"}})).into_response());
    };
    if method == "shutdown" {
        return Ok(Json(json!({"jsonrpc": "2.0", "id": id, "error": {"code": code::METHOD_NOT_FOUND, "message": "shutdown is only available over stdio"}})).into_response());
    }
    let params = req.get("params").cloned().unwrap_or(Value::Null);
    let engine = s.engine.clone();
    let handle = tokio::runtime::Handle::current();
    let result = blocking(move || Ok(handle.block_on(engine.call(&method, &params, false, &mut |_| {})))).await?;
    let reply = match result {
        Ok(v) => json!({"jsonrpc": "2.0", "id": id, "result": v}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e.to_json()}),
    };
    Ok(Json(reply).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_reference() {
        // 1.0f32 = 00 00 80 3f
        assert_eq!(base64_f32(&[1.0]), "AACAPw==");
        assert_eq!(base64_f32(&[1.0, -2.0]), "AACAPwAAAMA=");
        assert!(same(b"abc", b"abc") && !same(b"abc", b"abd") && !same(b"ab", b"abc"));
    }
}
