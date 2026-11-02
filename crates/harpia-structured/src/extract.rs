//! Locate JSON inside chatty LLM output.

use harpia_core::{HarpiaError, Result};
use serde::de::DeserializeOwned;

/// Content of the first Markdown code fence (```` ``` ```` or `~~~`, any info string), or the
/// whole text when there is none. An unterminated fence returns everything after it.
pub fn strip_fences(text: &str) -> &str {
    for fence in ["```", "~~~"] {
        if let Some(open) = text.find(fence) {
            let after = &text[open + fence.len()..];
            // skip the info string (e.g. "json") up to the end of the line
            let body_start = after.find('\n').map(|i| i + 1).unwrap_or(after.len());
            let body = &after[body_start..];
            return match body.find(fence) {
                Some(close) => &body[..close],
                None => body,
            };
        }
    }
    text
}

/// End (exclusive byte index) of the balanced JSON value starting at `start` (`{` or `[`),
/// respecting strings and escapes. `None` if it never closes.
fn balanced_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (off, &b) in bytes[start..].iter().enumerate() {
        if in_str {
            match (esc, b) {
                (true, _) => esc = false,
                (false, b'\\') => esc = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + off + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// All strictly valid JSON objects/arrays in `text`, in order (outermost only).
pub fn extract_all(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(rel) = text[i..].find(['{', '[']) {
        let start = i + rel;
        match balanced_end(text, start) {
            Some(end) if serde_json::from_str::<serde_json::Value>(&text[start..end]).is_ok() => {
                out.push(&text[start..end]);
                i = end;
            }
            _ => i = start + 1,
        }
    }
    out
}

/// The first strictly valid JSON object/array in `text` (searching inside a code fence first).
///
/// ```
/// use harpia_structured::extract_json;
/// let out = "Sure! Here you go:\n```json\n{\"ok\": true}\n```\nAnything else?";
/// assert_eq!(extract_json(out), Some("{\"ok\": true}"));
/// ```
pub fn extract_json(text: &str) -> Option<&str> {
    let fenced = strip_fences(text);
    extract_all(fenced).into_iter().next().or_else(|| extract_all(text).into_iter().next())
}

/// Deserialize `T` from LLM output: strict extraction first, then [`repair`](crate::repair).
///
/// ```
/// #[derive(serde::Deserialize, Debug, PartialEq)]
/// struct Answer { label: String, confidence: f64 }
/// let a: Answer = harpia_structured::parse_llm_json("The answer: {label: 'yes', confidence: 0.9").unwrap();
/// assert_eq!(a, Answer { label: "yes".into(), confidence: 0.9 });
/// ```
pub fn parse_llm_json<T: DeserializeOwned>(text: &str) -> Result<T> {
    if let Some(s) = extract_json(text)
        && let Ok(v) = serde_json::from_str(s)
    {
        return Ok(v);
    }
    let value = crate::repair::repair(text)?;
    serde_json::from_value(value)
        .map_err(|e| HarpiaError::Validation(format!("repaired JSON does not match the target type: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fences_and_prose() {
        assert_eq!(strip_fences("a\n```\nX\n```"), "X\n");
        assert_eq!(strip_fences("no fence"), "no fence");
        assert_eq!(extract_json("x [1, 2] y {\"a\": \"}\"}"), Some("[1, 2]"));
        assert_eq!(extract_all("{bad} {\"a\": \"}\"} [3]"), vec!["{\"a\": \"}\"}", "[3]"]);
        assert_eq!(extract_json("nothing here"), None);
    }
}
