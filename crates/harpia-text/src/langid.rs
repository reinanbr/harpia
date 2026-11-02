//! Language identification (pure Rust, via `whatlang`).

/// Detected language.
#[derive(Debug, Clone, PartialEq)]
pub struct Detected {
    /// ISO 639-3 code (e.g. `"por"`, `"eng"`).
    pub code: &'static str,
    /// English name of the language.
    pub name: &'static str,
    /// Detector confidence in `0.0..=1.0`.
    pub confidence: f64,
    /// Whether the detector considers the result reliable.
    pub reliable: bool,
}

/// Detect the language of `text`; `None` if undetectable (e.g. empty or symbols only).
///
/// ```
/// let d = harpia_text::langid::detect("Este é um texto escrito em português do Brasil.").unwrap();
/// assert_eq!(d.code, "por");
/// ```
pub fn detect(text: &str) -> Option<Detected> {
    let info = whatlang::detect(text)?;
    Some(Detected {
        code: info.lang().code(),
        name: info.lang().eng_name(),
        confidence: info.confidence(),
        reliable: info.is_reliable(),
    })
}

/// Map an ISO 639-3 code to the bundled [`Lang`](crate::Lang), when available.
pub fn bundled(code: &str) -> Option<crate::Lang> {
    match code {
        "eng" => Some(crate::Lang::En),
        "por" => Some(crate::Lang::PtBr),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_english() {
        let d = detect("The quick brown fox jumps over the lazy dog near the river bank.").unwrap();
        assert_eq!(d.code, "eng");
        assert_eq!(bundled(d.code), Some(crate::Lang::En));
        assert!(detect("").is_none());
    }
}
