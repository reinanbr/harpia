//! `harpia serve --stdio`: newline-delimited JSON-RPC 2.0 on stdin/stdout.
//!
//! One request (or batch) per line; one response (or batch) per line, in request order.
//! Requests without an `id` are notifications and get no response. With `"stream": true`,
//! `generate` and `chat` first emit `stream` notifications carrying text deltas. Logs go to
//! stderr only, so stdout stays machine-readable.

use super::engine::{Engine, RpcError, code};
use serde_json::{Value, json};
use std::io::{BufRead, Write};

/// Serve until EOF on stdin or a `shutdown` call.
pub fn serve(engine: &Engine, input: impl BufRead, mut output: impl Write) -> harpia::core::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let (reply, stop) = rt.block_on(handle_line(engine, &line, &mut output));
        if let Some(r) = reply {
            write_line(&mut output, &r)?;
        }
        if stop {
            break;
        }
    }
    Ok(())
}

fn write_line(out: &mut impl Write, v: &Value) -> std::io::Result<()> {
    serde_json::to_writer(&mut *out, v)?;
    out.write_all(b"\n")?;
    out.flush()
}

/// Handle one input line; returns the reply (if any) and whether to stop serving.
async fn handle_line(engine: &Engine, line: &str, out: &mut impl Write) -> (Option<Value>, bool) {
    let msg: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return (Some(error_reply(Value::Null, &RpcError::new(code::PARSE, e.to_string()))), false),
    };
    match msg {
        Value::Array(items) if items.is_empty() => {
            (Some(error_reply(Value::Null, &RpcError::new(code::INVALID_REQUEST, "empty batch"))), false)
        }
        Value::Array(items) => {
            let mut replies = Vec::new();
            let mut stop = false;
            for item in &items {
                let (r, s) = handle_one(engine, item, out).await;
                replies.extend(r);
                stop |= s;
            }
            ((!replies.is_empty()).then_some(Value::Array(replies)), stop)
        }
        other => handle_one(engine, &other, out).await,
    }
}

async fn handle_one(engine: &Engine, req: &Value, out: &mut impl Write) -> (Option<Value>, bool) {
    let id = req.get("id").cloned();
    let invalid = |m: &str| {
        (Some(error_reply(id.clone().unwrap_or(Value::Null), &RpcError::new(code::INVALID_REQUEST, m))), false)
    };
    if !req.is_object() {
        return invalid("a request must be an object");
    }
    if req.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return invalid("`jsonrpc` must be \"2.0\"");
    }
    let Some(method) = req.get("method").and_then(Value::as_str) else {
        return invalid("missing `method`");
    };
    if id.as_ref().is_some_and(|i| !(i.is_string() || i.is_number() || i.is_null())) {
        return invalid("`id` must be a string, number or null");
    }
    let params = req.get("params").cloned().unwrap_or(Value::Null);
    let stream_id = id.clone();
    let mut on_delta = |d: &str| {
        if let Some(sid) = &stream_id {
            // a failed write here also fails the final reply; nothing more to do
            let _ = write_line(out, &json!({"jsonrpc": "2.0", "method": "stream", "params": {"id": sid, "delta": d}}));
        }
    };
    let result = engine.call(method, &params, true, &mut on_delta).await;
    let stop = method == "shutdown" && result.is_ok();
    let Some(id) = id else {
        return (None, stop);
    };
    let reply = match result {
        Ok(v) => json!({"jsonrpc": "2.0", "id": id, "result": v}),
        Err(e) => error_reply(id, &e),
    };
    (Some(reply), stop)
}

fn error_reply(id: Value, e: &RpcError) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": e.to_json()})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::engine::EngineConfig;

    fn run(input: &str) -> Vec<Value> {
        let engine = Engine::new(EngineConfig { lang: "en".into(), ..Default::default() }).unwrap();
        let mut out = Vec::new();
        serve(&engine, input.as_bytes(), &mut out).unwrap();
        String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    #[test]
    fn protocol_errors_and_batches() {
        let r = run(concat!(
            "not json\n",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"nope\"}\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"info\"}\n",
            "[{\"jsonrpc\":\"2.0\",\"id\":\"a\",\"method\":\"structured.repair\",\"params\":{\"text\":\"{a: 1,}\"}},",
            "{\"jsonrpc\":\"2.0\",\"id\":\"b\",\"method\":\"generate\",\"params\":{\"prompt\":\"x\"}}]\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"shutdown\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"info\"}\n",
        ));
        assert_eq!(r.len(), 4, "{r:?}");
        assert_eq!(r[0]["error"]["code"], code::PARSE);
        assert_eq!(r[1]["error"]["code"], code::METHOD_NOT_FOUND);
        assert_eq!(r[2][0]["result"]["value"], json!({"a": 1}));
        assert_eq!(r[2][1]["error"]["code"], code::NOT_AVAILABLE);
        assert_eq!(r[3]["result"], Value::Null);
    }
}
