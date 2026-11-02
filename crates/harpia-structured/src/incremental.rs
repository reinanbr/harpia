//! Best-effort parsing of JSON while it streams in.

use serde_json::Value;

/// Accumulates streamed text and exposes the best-effort value parsed so far.
///
/// Incomplete strings are shown as far as they arrived, unfinished containers are closed and
/// keys without a value yet are omitted. Each call to [`current`](Self::current) re-parses the
/// buffer (linear in its size); call it at the rate you need to refresh a UI, not per byte.
///
/// ```
/// use harpia_structured::IncrementalJson;
/// let mut inc = IncrementalJson::new();
/// inc.push(r#"{"title": "Harp"#);
/// assert_eq!(inc.current().unwrap()["title"], "Harp");
/// inc.push(r#"ia", "tags": ["rust""#);
/// assert_eq!(inc.current().unwrap(), serde_json::json!({"title": "Harpia", "tags": ["rust"]}));
/// assert!(!inc.is_complete());
/// inc.push("]}");
/// assert!(inc.is_complete());
/// ```
#[derive(Debug, Clone, Default)]
pub struct IncrementalJson {
    buf: String,
}

impl IncrementalJson {
    /// Empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a chunk.
    pub fn push(&mut self, chunk: &str) {
        self.buf.push_str(chunk);
    }

    /// Text received so far.
    pub fn text(&self) -> &str {
        &self.buf
    }

    /// Best-effort value so far (`None` before anything JSON-like arrived).
    pub fn current(&self) -> Option<Value> {
        crate::repair::repair(&self.buf).ok()
    }

    /// `true` once the buffer holds a complete, strictly valid JSON value.
    pub fn is_complete(&self) -> bool {
        crate::extract::extract_json(&self.buf).is_some()
    }
}
