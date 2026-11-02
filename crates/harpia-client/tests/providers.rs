//! Providers against a local mock server: no real network.

use harpia_client::{
    Anthropic, BatchRunner, Ollama, OpenAiCompatible, Secret, UsageTracker, WithMiddleware, chat_json, collect_stream,
};
use harpia_core::traits::{ChatRequest, FinishReason, Provider, ToolSpec};
use harpia_core::types::{Message, ToolCall};
use harpia_core::{HarpiaError, NetworkPolicy};
use harpia_http::{HttpClient, HttpConfig};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn http() -> Arc<HttpClient> {
    Arc::new(HttpClient::new(HttpConfig::new(NetworkPolicy::Online).retries(1, Duration::from_millis(1))).unwrap())
}

fn req(model: &str, text: &str) -> ChatRequest {
    ChatRequest {
        model: model.into(),
        messages: vec![Message::system("Be brief."), Message::user(text)],
        ..Default::default()
    }
}

fn sse(events: &[Value]) -> String {
    events.iter().map(|e| format!("data: {e}\n\n")).collect()
}

async fn last_body(server: &MockServer) -> Value {
    let reqs = server.received_requests().await.unwrap();
    serde_json::from_slice(&reqs.last().unwrap().body).unwrap()
}

// ------------------------------------------------------------------ OpenAI-compatible

#[tokio::test]
async fn openai_chat_with_tool_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .and(body_partial_json(json!({"model": "m", "temperature": 0.0, "max_tokens": 256})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "m-2026",
            "choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": null,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{\"q\": \"harpia\"}"}}]}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4}}
        })))
        .mount(&server)
        .await;
    let p = OpenAiCompatible::new(http(), format!("{}/v1", server.uri()), Some(Secret::new("sk-test")));
    let mut r = req("m", "find harpia");
    r.tools = vec![ToolSpec {
        name: "search".into(),
        description: "web search".into(),
        parameters: json!({"type": "object"}),
    }];
    let resp = p.chat(&r).await.unwrap();
    assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
    assert_eq!(resp.message.tool_calls[0].arguments["q"], "harpia");
    assert_eq!((resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.cached_input_tokens), (12, 5, 4));
    let body = last_body(&server).await;
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["tools"][0]["function"]["name"], "search");
}

#[tokio::test]
async fn openai_stream_text_tools_usage() {
    let server = MockServer::start().await;
    let body = sse(&[
        json!({"choices": [{"delta": {"content": "Har"}}]}),
        json!({"choices": [{"delta": {"content": "pia"}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{\"a\":"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": " 1}"}}]}, "finish_reason": "tool_calls"}]}),
        json!({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
    ]) + "data: [DONE]\n\n";
    Mock::given(path("/chat/completions"))
        .and(body_partial_json(json!({"stream": true, "stream_options": {"include_usage": true}})))
        .respond_with(
            ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(body),
        )
        .mount(&server)
        .await;
    let p = OpenAiCompatible::new(http(), server.uri(), None);
    let all = collect_stream(p.chat_stream(&req("m", "x")).await.unwrap()).await.unwrap();
    assert_eq!(all.delta, "Harpia");
    assert_eq!(all.tool_calls, vec![ToolCall { id: Some("c1".into()), name: "f".into(), arguments: json!({"a": 1}) }]);
    assert_eq!(all.usage.unwrap().output_tokens, 2);
    assert_eq!(all.finish_reason, Some(FinishReason::ToolCalls));
}

#[tokio::test]
async fn openai_embeddings_in_index_order() {
    let server = MockServer::start().await;
    Mock::given(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [
            {"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]})))
        .mount(&server)
        .await;
    let p = OpenAiCompatible::new(http(), server.uri(), None);
    assert_eq!(p.embed("e", &["a", "b"]).await.unwrap(), vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
}

// ------------------------------------------------------------------ Anthropic

#[tokio::test]
async fn anthropic_chat_headers_body_and_parsing() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant-test"))
        .and(header("anthropic-version", "2026-10-21"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "claude-opus-5-5", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Manaus"}}],
            "usage": {"input_tokens": 10, "output_tokens": 7, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5}
        })))
        .mount(&server)
        .await;
    let mut p = Anthropic::new(http(), Secret::new("sk-ant-test")).with_base_url(server.uri());
    p.effort = Some("low".into());
    let mut r = req("claude-opus-5-5", "weather?");
    r.params.temperature = 0.3;
    r.params.stop = vec!["END".into()];
    r.response_schema = Some(json!({"type": "object", "additionalProperties": false}));
    let resp = p.chat(&r).await.unwrap();
    assert_eq!(resp.message.content, "Checking.");
    assert_eq!(resp.message.tool_calls[0].arguments["city"], "Manaus");
    assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
    assert_eq!((resp.usage.input_tokens, resp.usage.cached_input_tokens), (115, 100));
    let body = last_body(&server).await;
    assert_eq!(body["system"], "Be brief.");
    assert_eq!(body["stop_sequences"][0], "END");
    assert_eq!(body["output_config"]["effort"], "low");
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    for k in ["temperature", "top_p", "top_k"] {
        assert!(body.get(k).is_none(), "{k} must not be sent on API version 2026-10-21");
    }
}

#[test]
fn anthropic_tool_results_share_one_user_message() {
    let p = Anthropic::new(http(), Secret::new("k"));
    let mut a = Message::assistant("");
    a.tool_calls = vec![
        ToolCall { id: Some("t1".into()), name: "f".into(), arguments: json!({}) },
        ToolCall { id: Some("t2".into()), name: "g".into(), arguments: json!({}) },
    ];
    let r = ChatRequest {
        model: "claude-opus-5-5".into(),
        messages: vec![Message::user("go"), a, Message::tool("t1", "r1"), Message::tool("t2", "r2")],
        ..Default::default()
    };
    let b = p.body(&r, false);
    let msgs = b["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[1]["content"][1]["type"], "tool_use");
    assert_eq!(msgs[2]["content"].as_array().unwrap().len(), 2);
    assert_eq!(msgs[2]["content"][1]["tool_use_id"], "t2");
}

#[tokio::test]
async fn anthropic_stream_events() {
    let server = MockServer::start().await;
    let ev = |name: &str, data: Value| format!("event: {name}\ndata: {data}\n\n");
    let body = [
        ev("message_start", json!({"type": "message_start", "message": {"usage": {"input_tokens": 9, "output_tokens": 1}}})),
        ev("content_block_start", json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})),
        ev("content_block_delta", json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Olá"}})),
        ev("ping", json!({"type": "ping"})),
        ev("content_block_start", json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "f"}})),
        ev("content_block_delta", json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"x\": "}})),
        ev("content_block_delta", json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "2}"}})),
        ev("message_delta", json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}})),
        ev("message_stop", json!({"type": "message_stop"})),
    ]
    .concat();
    Mock::given(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;
    let p = Anthropic::new(http(), Secret::new("k")).with_base_url(server.uri());
    let all = collect_stream(p.chat_stream(&req("claude-opus-5-5", "x")).await.unwrap()).await.unwrap();
    assert_eq!(all.delta, "Olá");
    assert_eq!(all.tool_calls[0].arguments, json!({"x": 2}));
    let u = all.usage.unwrap();
    assert_eq!((u.input_tokens, u.output_tokens), (9, 12));
    assert_eq!(all.finish_reason, Some(FinishReason::ToolCalls));
}

#[tokio::test]
async fn api_errors_are_readable_and_secret_free() {
    let server = MockServer::start().await;
    Mock::given(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(
            json!({"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}),
        ))
        .mount(&server)
        .await;
    let p = Anthropic::new(http(), Secret::new("sk-ant-SECRET")).with_base_url(server.uri());
    let err = p.chat(&req("claude-opus-5-5", "x")).await.unwrap_err().to_string();
    assert!(err.contains("HTTP 401: invalid x-api-key"), "{err}");
    assert!(!err.contains("SECRET"));
    assert!(!format!("{p:?}").contains("SECRET"));
}

// ------------------------------------------------------------------ Ollama

#[tokio::test]
async fn ollama_chat_stream_and_embed() {
    let server = MockServer::start().await;
    Mock::given(path("/api/chat"))
        .and(body_partial_json(json!({"stream": true, "options": {"top_k": 40}})))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            "{\"message\":{\"content\":\"Bom \"},\"done\":false}\n{\"message\":{\"content\":\"dia\"},\"done\":false}\n{\"message\":{\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":4,\"eval_count\":2}\n",
        ))
        .mount(&server)
        .await;
    Mock::given(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embeddings": [[0.5, 0.5]]})))
        .mount(&server)
        .await;
    let p = Ollama::new(http(), server.uri());
    let mut r = req("llama3.2", "oi");
    r.params.top_k = 40;
    let all = collect_stream(p.chat_stream(&r).await.unwrap()).await.unwrap();
    assert_eq!(all.delta, "Bom dia");
    assert_eq!(all.usage.unwrap().input_tokens, 4);
    assert_eq!(p.embed("nomic-embed-text", &["x"]).await.unwrap(), vec![vec![0.5, 0.5]]);
}

// ------------------------------------------------------------------ helpers

#[tokio::test]
async fn chat_json_retries_once_with_the_validation_error() {
    let server = MockServer::start().await;
    let reply = |text: &str| {
        ResponseTemplate::new(200).set_body_json(json!({"choices": [{"finish_reason": "stop", "message": {"content": text}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
    };
    Mock::given(path("/chat/completions"))
        .respond_with(reply("{\"label\": \"maybe\"}"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(path("/chat/completions"))
        .respond_with(reply("```json\n{\"label\": \"supports\"}\n```"))
        .mount(&server)
        .await;
    let p = OpenAiCompatible::new(http(), server.uri(), None);
    let schema = json!({"type": "object", "properties": {"label": {"enum": ["supports", "contradicts"]}}, "required": ["label"], "additionalProperties": false});
    #[derive(serde::Deserialize)]
    struct Out {
        label: String,
    }
    let (out, resp): (Out, _) = chat_json(&p, &req("m", "classify"), &schema, 1).await.unwrap();
    assert_eq!(out.label, "supports");
    assert_eq!(resp.usage.input_tokens, 2, "usage summed over attempts");
    let second = last_body(&server).await;
    assert_eq!(second["response_format"]["type"], "json_schema");
    assert!(
        second["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap().contains("did not match")
    );
}

#[tokio::test]
async fn batch_runner_resumes_from_cache_and_tracks_usage() {
    let server = MockServer::start().await;
    Mock::given(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}})))
        .expect(3)
        .mount(&server)
        .await;
    let tracker = Arc::new(UsageTracker::default());
    let p = WithMiddleware::new(OpenAiCompatible::new(http(), server.uri(), None)).layer(tracker.clone());
    let items: Vec<(String, ChatRequest)> = (0..3).map(|i| (format!("row-{i}"), req("m", &format!("q{i}")))).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.jsonl");
    let runner = BatchRunner::new(2).requests_per_minute(6000.0);
    let mut cache = harpia_data::JsonlCache::open(&path).unwrap();
    let r1 = runner.run(&p, items.clone(), &mut cache).await.unwrap();
    assert_eq!((r1.completed, r1.from_cache, r1.usage.total()), (3, 0, 9));
    let mut cache = harpia_data::JsonlCache::open(&path).unwrap();
    let r2 = runner.run(&p, items, &mut cache).await.unwrap();
    assert_eq!((r2.completed, r2.from_cache), (0, 3));
    assert_eq!(tracker.totals().0, 3);
}

#[tokio::test]
async fn offline_policy_blocks_every_provider() {
    let offline = Arc::new(HttpClient::new(HttpConfig::default()).unwrap());
    let r = req("m", "x");
    let errs = [
        OpenAiCompatible::new(offline.clone(), "http://127.0.0.1:9/v1", None).chat(&r).await.unwrap_err(),
        Anthropic::new(offline.clone(), Secret::new("k")).chat(&r).await.unwrap_err(),
        Ollama::local(offline.clone()).chat(&r).await.unwrap_err(),
    ];
    assert!(errs.iter().all(|e| matches!(e, HarpiaError::NetworkDisabled { .. })));
    assert_eq!(offline.requests_sent(), 0);
}

#[tokio::test]
async fn synthesize_rows_with_provenance() {
    let server = MockServer::start().await;
    Mock::given(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": "{\"q\": \"What do harpy eagles eat?\"}"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})))
        .expect(2)
        .mount(&server)
        .await;
    let p = OpenAiCompatible::new(http(), server.uri(), None);
    let t = harpia_client::SynthTemplate::from_toml(
        "model = \"m\"\nuser = \"Question about {{ topic }}\"\noutput_field = \"gen\"\nschema = { type = \"object\", properties = { q = { type = \"string\" } }, required = [\"q\"] }\n",
    )
    .unwrap();
    let rows = vec![json!({"topic": "diet"}), json!({"topic": "nests"})];
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("synth.jsonl");
    let (out, rep) = harpia_client::synthesize(&p, &t, &rows, &BatchRunner::new(2), &cache).await.unwrap();
    assert_eq!(rep.completed, 2);
    assert_eq!(out[0]["gen"]["q"], "What do harpy eagles eat?");
    assert_eq!(out[1]["_provenance"]["provider"], "openai-compatible");
    assert_eq!(last_body(&server).await["response_format"]["json_schema"]["schema"]["additionalProperties"], false);
    let (again, rep2) = harpia_client::synthesize(&p, &t, &rows, &BatchRunner::new(2), &cache).await.unwrap();
    assert_eq!((again.len(), rep2.from_cache), (2, 2));
    assert!(t.render(&json!({"other": 1})).is_err(), "missing variables are errors");
}
