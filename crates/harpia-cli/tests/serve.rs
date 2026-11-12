//! End-to-end tests of `harpia serve --stdio` and `harpia serve --http` with tiny random models.

use harpia::core::Document;
use harpia::data::chunk::RecursiveChunker;
use harpia::data::rag::{RagConfig, RagPipeline};
use harpia::local::{EncoderEmbedder, testing};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

struct Models {
    _dir: tempfile::TempDir,
    lm: PathBuf,
    nli: PathBuf,
    mt: PathBuf,
    rag: PathBuf,
}

fn models() -> Models {
    let dir = tempfile::tempdir().unwrap();
    let lm = dir.path().join("lm");
    testing::qwen2(&lm).unwrap();
    let nli = dir.path().join("nli");
    testing::xlm_roberta_classifier(&nli, &["entailment", "neutral", "contradiction"]).unwrap();
    let mt = dir.path().join("mt");
    testing::marian(&mt).unwrap();
    let rag = dir.path().join("rag");
    let emb = EncoderEmbedder::load(&nli).unwrap();
    let mut pipe = RagPipeline::new(Box::new(RecursiveChunker::new(200)), &emb, RagConfig::default());
    pipe.add_documents(&[
        Document::new("d1", "the harpy eagle eats sloths in the forest"),
        Document::new("d2", "hello world"),
    ])
    .unwrap();
    pipe.save(&rag).unwrap();
    Models { _dir: dir, lm, nli, mt, rag }
}

fn model_args(m: &Models) -> Vec<String> {
    let s = |p: &Path| p.display().to_string();
    vec![
        "--decoder".into(),
        format!("tiny={}", s(&m.lm)),
        "--embedder".into(),
        s(&m.nli),
        "--classifier".into(),
        s(&m.nli),
        "--reranker".into(),
        s(&m.nli),
        "--translator".into(),
        s(&m.mt),
        "--rag".into(),
        format!("docs={}", s(&m.rag)),
    ]
}

fn bin() -> PathBuf {
    assert_cmd::cargo::cargo_bin("harpia")
}

// ---------------------------------------------------------------- stdio

#[test]
fn stdio_serves_every_method() {
    let m = models();
    let calls = [
        json!({"jsonrpc": "2.0", "id": 1, "method": "info"}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "generate", "params": {"prompt": "the bird", "choices": ["yes", "no"]}}),
        json!({"jsonrpc": "2.0", "id": 3, "method": "chat", "params": {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 3, "stream": true}}),
        json!({"jsonrpc": "2.0", "id": 4, "method": "embed", "params": {"texts": ["the harpy eagle", "a bird"]}}),
        json!({"jsonrpc": "2.0", "id": 5, "method": "rerank", "params": {"query": "eagle", "documents": ["the harpy eagle", {"text": "hello"}], "top_n": 1}}),
        json!({"jsonrpc": "2.0", "id": 6, "method": "classify", "params": {"pairs": [["the eagle eats sloths", "the eagle eats"]]}}),
        json!({"jsonrpc": "2.0", "id": 7, "method": "translate", "params": {"texts": ["the harpy eagle"], "beams": 2}}),
        json!({"jsonrpc": "2.0", "id": 8, "method": "retrieve", "params": {"query": "harpy eagle", "k": 1, "budget": 50}}),
        json!({"jsonrpc": "2.0", "id": 9, "method": "data.filter", "params": {"texts": ["hi", "The harpy eagle is a very large bird of prey that lives in the forest."], "thresholds": {"min_words": 5}}}),
        json!({"jsonrpc": "2.0", "id": 10, "method": "structured.repair", "params": {"text": "Sure: {a: 'x',}"}}),
        json!({"jsonrpc": "2.0", "id": 11, "method": "verify.claim", "params": {"claim": "The harpy eagle lives in Brazil."}}),
        json!({"jsonrpc": "2.0", "id": 12, "method": "generate", "params": {"prompt": "x", "model": "missing"}}),
        json!({"jsonrpc": "2.0", "id": 13, "method": "generate", "params": {"prompt": "x", "model": m.lm}}),
        json!({"jsonrpc": "2.0", "id": 14, "method": "shutdown"}),
    ];
    let input: String = calls.iter().map(|c| format!("{c}\n")).collect();
    let out = assert_cmd::Command::new(bin())
        .args(["serve", "--stdio"])
        .args(model_args(&m))
        .write_stdin(input)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let lines: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let deltas: Vec<&Value> = lines.iter().filter(|l| l["method"] == "stream").collect();
    let replies: Vec<&Value> = lines.iter().filter(|l| l.get("id").is_some()).collect();
    assert_eq!(replies.len(), calls.len(), "{lines:#?}");
    let r = |id: u64| replies.iter().find(|r| r["id"] == id).copied().unwrap();
    for id in 1..=11 {
        assert!(r(id).get("error").is_none(), "call {id} failed: {}", r(id));
    }

    assert_eq!(r(1)["result"]["models"]["decoders"], json!(["tiny"]));
    assert_eq!(r(1)["result"]["network"], "offline");
    assert!(["yes", "no"].contains(&r(2)["result"]["text"].as_str().unwrap()));
    // the streamed deltas add up to the final text
    let chat = &r(3)["result"];
    assert!(deltas.iter().all(|d| d["params"]["id"] == 3));
    let joined: String = deltas.iter().map(|d| d["params"]["delta"].as_str().unwrap()).collect();
    assert_eq!(joined, chat["message"]["content"].as_str().unwrap());
    assert_eq!(chat["message"]["role"], "assistant");
    assert_eq!(r(4)["result"]["embeddings"].as_array().unwrap().len(), 2);
    assert_eq!(r(5)["result"]["results"].as_array().unwrap().len(), 1);
    assert_eq!(r(6)["result"]["results"][0]["scores"].as_object().unwrap().len(), 3);
    assert_eq!(r(7)["result"]["translations"].as_array().unwrap().len(), 1);
    let hits = &r(8)["result"]["hits"];
    assert_eq!(hits.as_array().unwrap().len(), 1);
    assert!(r(8)["result"]["context"]["text"].as_str().unwrap().starts_with("[1]"));
    assert_eq!(r(9)["result"]["kept"], 1);
    assert_eq!(r(9)["result"]["decisions"][0]["reason"], "min_words");
    assert_eq!(r(10)["result"]["value"], json!({"a": "x"}));
    // offline: verification is skipped and nothing is disclosed
    assert_eq!(r(11)["result"]["verification"]["status"], "skipped");
    assert_eq!(r(11)["result"]["disclosure"], json!([]));
    assert_eq!(r(12)["error"]["code"], -32003);
    // stdio clients may name a model directory directly
    assert!(r(13).get("result").is_some(), "{}", r(13));
}

// ---------------------------------------------------------------- http

struct Server {
    child: Child,
    addr: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_http(m: &Models, key: Option<&str>) -> Server {
    let mut cmd = Command::new(bin());
    cmd.args(["serve", "--http", "--bind", "127.0.0.1:0"]).args(model_args(m));
    if let Some(k) = key {
        cmd.env("HARPIA_TEST_KEY", k).args(["--api-key-env", "HARPIA_TEST_KEY"]);
    }
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    let addr = loop {
        line.clear();
        assert!(err.read_line(&mut line).unwrap() > 0, "server exited before listening");
        if let Some(a) = line.trim().strip_prefix("harpia: listening on http://") {
            break a.to_string();
        }
    };
    // keep draining stderr so the server never blocks on a full pipe
    std::thread::spawn(move || std::io::copy(&mut err, &mut std::io::sink()));
    Server { child, addr }
}

/// Minimal HTTP/1.1 client: (status, decoded body).
fn request(s: &Server, method: &str, path: &str, body: Option<&Value>, key: Option<&str>) -> (u16, String) {
    let mut c = TcpStream::connect(&s.addr).unwrap();
    let body = body.map(Value::to_string).unwrap_or_default();
    let auth = key.map(|k| format!("Authorization: Bearer {k}\r\n")).unwrap_or_default();
    write!(
        c,
        "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{auth}\r\n{body}",
        s.addr,
        body.len()
    )
    .unwrap();
    let mut raw = Vec::new();
    c.read_to_end(&mut raw).unwrap();
    let raw = String::from_utf8(raw).unwrap();
    let (head, rest) = raw.split_once("\r\n\r\n").unwrap();
    let status: u16 = head.split(' ').nth(1).unwrap().parse().unwrap();
    let chunked = head.to_ascii_lowercase().contains("transfer-encoding: chunked");
    (status, if chunked { dechunk(rest) } else { rest.to_string() })
}

fn dechunk(mut s: &str) -> String {
    let mut out = String::new();
    while let Some((size, rest)) = s.split_once("\r\n") {
        let n = usize::from_str_radix(size.trim(), 16).unwrap();
        if n == 0 {
            break;
        }
        out.push_str(&rest[..n]);
        s = &rest[n + 2..];
    }
    out
}

fn json_body(r: (u16, String)) -> (u16, Value) {
    (r.0, serde_json::from_str(&r.1).unwrap_or_else(|e| panic!("{e}: {}", r.1)))
}

fn sse_events(body: &str) -> Vec<String> {
    body.lines().filter_map(|l| l.strip_prefix("data: ")).map(String::from).collect()
}

#[test]
fn http_is_openai_compatible() {
    let m = models();
    let s = spawn_http(&m, Some("sekret"));
    let key = Some("sekret");

    assert_eq!(request(&s, "GET", "/health", None, None).0, 200);
    assert_eq!(request(&s, "GET", "/v1/models", None, None).0, 401);
    assert_eq!(request(&s, "GET", "/v1/models", None, Some("wrong")).0, 401);
    let (st, models) = json_body(request(&s, "GET", "/v1/models", None, key));
    assert_eq!(st, 200);
    assert_eq!(models["data"][0]["id"], "tiny");

    // chat, constrained to a JSON schema through response_format
    let schema = json!({"type": "object", "properties": {"label": {"enum": ["yes", "no"]}}, "required": ["label"]});
    let (st, v) = json_body(request(
        &s,
        "POST",
        "/v1/chat/completions",
        Some(&json!({
            "model": "tiny", "max_tokens": 30, "seed": 1,
            "messages": [{"role": "system", "content": "answer"}, {"role": "user", "content": [{"type": "text", "text": "is the harpy eagle a bird"}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": schema}},
        })),
        key,
    ));
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    if v["choices"][0]["finish_reason"] == "stop" {
        let parsed: Value = serde_json::from_str(content).unwrap();
        assert!(parsed["label"] == "yes" || parsed["label"] == "no");
    }
    let u = &v["usage"];
    assert_eq!(u["total_tokens"], u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap());

    // streaming chat: role chunk, content chunks, finish chunk, usage chunk, [DONE]
    let (st, body) = request(
        &s,
        "POST",
        "/v1/chat/completions",
        Some(&json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4, "stream": true,
                     "stream_options": {"include_usage": true}})),
        key,
    );
    assert_eq!(st, 200);
    let events = sse_events(&body);
    assert_eq!(events.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = events[..events.len() - 1].iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(chunks.iter().all(|c| c["object"] == "chat.completion.chunk"));
    let fin = &chunks[chunks.len() - 2];
    assert!(fin["choices"][0]["finish_reason"].is_string());
    assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"].as_u64().unwrap() as usize, chunks.len() - 3);

    // completions with two prompts
    let (st, v) = json_body(request(
        &s,
        "POST",
        "/v1/completions",
        Some(&json!({"prompt": ["the bird", "the harpy"], "max_tokens": 2})),
        key,
    ));
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["choices"].as_array().unwrap().len(), 2);
    assert_eq!(v["choices"][1]["index"], 1);

    // embeddings as floats and as base64
    let (st, v) =
        json_body(request(&s, "POST", "/v1/embeddings", Some(&json!({"input": ["a bird", "the forest"]})), key));
    assert_eq!(st, 200, "{v}");
    let dim = v["data"][0]["embedding"].as_array().unwrap().len();
    let (_, b) = json_body(request(
        &s,
        "POST",
        "/v1/embeddings",
        Some(&json!({"input": "a bird", "encoding_format": "base64"})),
        key,
    ));
    assert_eq!(b["data"][0]["embedding"].as_str().unwrap().len(), (dim * 4).div_ceil(3) * 4);

    // rerank
    let (st, v) = json_body(request(
        &s,
        "POST",
        "/v1/rerank",
        Some(
            &json!({"query": "eagle", "documents": ["the harpy eagle", "hello world", "a bird"], "top_n": 2, "return_documents": true}),
        ),
        key,
    ));
    assert_eq!(st, 200, "{v}");
    let res = v["results"].as_array().unwrap();
    assert_eq!(res.len(), 2);
    assert!(res[0]["relevance_score"].as_f64() >= res[1]["relevance_score"].as_f64());
    assert!(res[0]["document"]["text"].is_string());

    // JSON-RPC over HTTP; model paths are refused
    let (_, v) = json_body(request(
        &s,
        "POST",
        "/rpc",
        Some(&json!({"jsonrpc": "2.0", "id": 1, "method": "retrieve", "params": {"query": "eagle"}})),
        key,
    ));
    assert!(v["result"]["hits"].is_array(), "{v}");
    let (_, v) = json_body(request(
        &s,
        "POST",
        "/rpc",
        Some(&json!({"jsonrpc": "2.0", "id": 2, "method": "generate", "params": {"prompt": "x", "model": m.lm}})),
        key,
    ));
    assert_eq!(v["error"]["code"], -32003);

    // errors use OpenAI's shape
    let (st, v) = json_body(request(
        &s,
        "POST",
        "/v1/chat/completions",
        Some(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})),
        key,
    ));
    assert_eq!(st, 404);
    assert!(v["error"]["message"].as_str().unwrap().contains("tiny"));
    let (st, v) = json_body(request(&s, "POST", "/v1/chat/completions", Some(&json!({"messages": "nope"})), key));
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "invalid_request_error");
}
