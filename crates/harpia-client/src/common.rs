//! Shared plumbing: secrets, error mapping, SSE/NDJSON decoding into chunk streams.

use futures_util::StreamExt;
use harpia_core::traits::{ChatChunk, ChatStream};
use harpia_core::{HarpiaError, Result};
use harpia_http::{ByteStream, Response};
use std::collections::VecDeque;

/// A secret string (API key) that never prints.
#[derive(Clone, Default)]
pub struct Secret(String);

impl Secret {
    /// Wrap a secret.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Read from an environment variable.
    pub fn from_env(var: &str) -> Option<Self> {
        std::env::var(var).ok().filter(|v| !v.trim().is_empty()).map(Self)
    }

    /// The secret value (only for building request headers).
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_empty() { "Secret(<empty>)" } else { "Secret(***)" })
    }
}

/// Host of a URL, for error messages.
pub(crate) fn host_of(url: &str) -> String {
    url.split("://").nth(1).and_then(|r| r.split(['/', '?']).next()).unwrap_or(url).to_string()
}

/// Error for a non-2xx response, using the provider's error message when present.
pub(crate) fn status_error(host: &str, status: u16, body: &[u8]) -> HarpiaError {
    let msg = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| {
                    e.get("message").and_then(|m| m.as_str()).map(String::from).or_else(|| e.as_str().map(String::from))
                })
                .or_else(|| v.get("message").and_then(|m| m.as_str()).map(String::from))
        })
        .unwrap_or_else(|| String::from_utf8_lossy(&body[..body.len().min(300)]).into_owned());
    HarpiaError::Remote { host: host.to_string(), message: format!("HTTP {status}: {msg}") }
}

/// Parse a successful JSON response or map the error.
pub(crate) fn json_ok(host: &str, r: &Response) -> Result<serde_json::Value> {
    if !r.is_success() {
        return Err(status_error(host, r.status, &r.body));
    }
    r.json()
}

/// One server-sent event.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SseEvent {
    /// `event:` field, if any.
    pub event: Option<String>,
    /// Concatenated `data:` lines.
    pub data: String,
}

/// Incremental SSE decoder (handles events split across chunks and multi-byte UTF-8).
#[derive(Debug, Default)]
pub struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    /// Feed bytes; returns complete events.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        loop {
            let text_end = find_event_end(&self.buf);
            let Some((end, sep)) = text_end else { break };
            let raw: Vec<u8> = self.buf.drain(..end + sep).collect();
            if let Some(ev) = parse_event(&String::from_utf8_lossy(&raw[..end])) {
                out.push(ev);
            }
        }
        out
    }

    /// Flush a trailing event without a final blank line.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let rest = std::mem::take(&mut self.buf);
        parse_event(&String::from_utf8_lossy(&rest)).into_iter().collect()
    }
}

fn find_event_end(buf: &[u8]) -> Option<(usize, usize)> {
    for i in 0..buf.len() {
        if buf[i..].starts_with(b"\n\n") {
            return Some((i, 2));
        }
        if buf[i..].starts_with(b"\r\n\r\n") {
            return Some((i, 4));
        }
    }
    None
}

fn parse_event(block: &str) -> Option<SseEvent> {
    let mut ev = SseEvent::default();
    let mut has_data = false;
    for line in block.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let (field, value) =
            line.split_once(':').map(|(f, v)| (f, v.strip_prefix(' ').unwrap_or(v))).unwrap_or((line, ""));
        match field {
            "event" => ev.event = Some(value.to_string()),
            "data" => {
                if has_data {
                    ev.data.push('\n');
                }
                ev.data.push_str(value);
                has_data = true;
            }
            _ => {}
        }
    }
    (has_data || ev.event.is_some()).then_some(ev)
}

struct StreamState<S, F, G> {
    body: ByteStream,
    framing: Framing,
    out: VecDeque<Result<ChatChunk>>,
    state: S,
    on_event: F,
    on_end: G,
    ended: bool,
}

enum Framing {
    Sse(SseParser),
    Ndjson(Vec<u8>),
}

/// Build a [`ChatStream`] from a byte stream. `on_event` maps each SSE event (or NDJSON line,
/// given as an event with no name) to zero or more chunks; `on_end` may emit final chunks.
pub(crate) fn chunk_stream<S, F, G>(body: ByteStream, sse: bool, state: S, on_event: F, on_end: G) -> ChatStream
where
    S: Send + 'static,
    F: FnMut(&mut S, SseEvent) -> Result<Vec<ChatChunk>> + Send + 'static,
    G: FnMut(&mut S) -> Vec<ChatChunk> + Send + 'static,
{
    let st = StreamState {
        body,
        framing: if sse { Framing::Sse(SseParser::default()) } else { Framing::Ndjson(Vec::new()) },
        out: VecDeque::new(),
        state,
        on_event,
        on_end,
        ended: false,
    };
    Box::pin(futures_util::stream::unfold(st, |mut st| async move {
        loop {
            if let Some(item) = st.out.pop_front() {
                return Some((item, st));
            }
            if st.ended {
                return None;
            }
            let events: Vec<SseEvent> = match st.body.next().await {
                Some(Ok(bytes)) => match &mut st.framing {
                    Framing::Sse(p) => p.push(&bytes),
                    Framing::Ndjson(buf) => {
                        buf.extend_from_slice(&bytes);
                        let mut evs = Vec::new();
                        while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                            let line: Vec<u8> = buf.drain(..=pos).collect();
                            let s = String::from_utf8_lossy(&line).trim().to_string();
                            if !s.is_empty() {
                                evs.push(SseEvent { event: None, data: s });
                            }
                        }
                        evs
                    }
                },
                Some(Err(e)) => {
                    st.out.push_back(Err(e));
                    st.ended = true;
                    continue;
                }
                None => {
                    let evs = match &mut st.framing {
                        Framing::Sse(p) => p.finish(),
                        Framing::Ndjson(buf) => {
                            let s = String::from_utf8_lossy(buf).trim().to_string();
                            buf.clear();
                            if s.is_empty() { vec![] } else { vec![SseEvent { event: None, data: s }] }
                        }
                    };
                    for ev in evs {
                        match (st.on_event)(&mut st.state, ev) {
                            Ok(c) => st.out.extend(c.into_iter().map(Ok)),
                            Err(e) => st.out.push_back(Err(e)),
                        }
                    }
                    let fin = (st.on_end)(&mut st.state);
                    st.out.extend(fin.into_iter().map(Ok));
                    st.ended = true;
                    continue;
                }
            };
            for ev in events {
                match (st.on_event)(&mut st.state, ev) {
                    Ok(c) => st.out.extend(c.into_iter().map(Ok)),
                    Err(e) => {
                        st.out.push_back(Err(e));
                        st.ended = true;
                        break;
                    }
                }
            }
        }
    }))
}

/// Collect a stream into the full text, final usage and finish reason.
pub async fn collect_stream(mut s: ChatStream) -> Result<ChatChunk> {
    let mut acc = ChatChunk::default();
    while let Some(c) = s.next().await {
        let c = c?;
        acc.delta.push_str(&c.delta);
        acc.tool_calls.extend(c.tool_calls);
        if c.usage.is_some() {
            acc.usage = c.usage;
        }
        if c.finish_reason.is_some() {
            acc.finish_reason = c.finish_reason;
        }
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_split_across_chunks() {
        let mut p = SseParser::default();
        assert!(p.push(b"event: a\ndata: {\"x\":").is_empty());
        let evs = p.push(b" 1}\n\n: comment\n\ndata: l1\ndata: l2\r\n\r\ndata: tail");
        assert_eq!(evs[0], SseEvent { event: Some("a".into()), data: "{\"x\": 1}".into() });
        assert_eq!(evs[1].data, "l1\nl2");
        assert_eq!(p.finish()[0].data, "tail");
    }

    #[test]
    fn secrets_do_not_print() {
        assert_eq!(format!("{:?}", Secret::new("sk-123")), "Secret(***)");
        assert_eq!(host_of("https://api.x.com/v1/chat"), "api.x.com");
        let e = status_error(
            "h",
            401,
            br#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
        );
        assert_eq!(e.to_string(), "remote error from h: HTTP 401: invalid x-api-key");
    }
}
