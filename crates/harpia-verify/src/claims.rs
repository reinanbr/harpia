//! Claim extraction and query building.
//!
//! Claims are sentences by default (byte spans into the answer). Queries default to
//! [`QueryMode::KeywordsOnly`]: content words only, so the full text of an answer or a private
//! document is never sent to a third-party search API unless the caller opts in with
//! [`QueryMode::FullText`].

use harpia_core::Span;
use harpia_text::lang::LangData;
use serde::{Deserialize, Serialize};

/// What is sent to search APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
    /// Up to `max_keywords` content words (no stopwords), in order of appearance.
    #[default]
    KeywordsOnly,
    /// The whole claim text (opt-in).
    FullText,
}

/// A claim to check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    /// Claim text.
    pub text: String,
    /// Byte span in the source answer, when the claim was split from one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
}

/// Split `answer` into sentence claims, skipping very short fragments and questions.
///
/// ```
/// use harpia_text::lang::Lang;
/// let c = harpia_verify::claims::split_claims("The harpy eagle lives in Brazil. Really? Yes.", &Lang::En.data(), 3);
/// assert_eq!(c.len(), 1);
/// assert_eq!(c[0].text, "The harpy eagle lives in Brazil.");
/// ```
pub fn split_claims(answer: &str, lang: &LangData, min_words: usize) -> Vec<Claim> {
    harpia_text::segment::sentences(answer, lang)
        .filter(|s| {
            let t = s.slice(answer).trim();
            !t.ends_with('?') && harpia_text::segment::word_count(t) >= min_words
        })
        .map(|s| Claim { text: s.slice(answer).trim().to_string(), span: Some(s) })
        .collect()
}

/// Content words of `text` (lowercase, no stopwords, at least 3 characters or containing a
/// digit), unique, in order.
pub fn keywords(text: &str, lang: &LangData, max: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for s in harpia_text::segment::words(text) {
        let w = s.slice(text);
        let lower = w.to_lowercase();
        let has_digit = w.chars().any(|c| c.is_ascii_digit());
        if (lower.chars().count() < 3 && !has_digit) || lang.is_stopword(&lower) || out.contains(&lower) {
            continue;
        }
        out.push(lower);
        if out.len() >= max {
            break;
        }
    }
    out
}

/// The query string for `claim` under `mode`.
pub fn build_query(claim: &str, mode: QueryMode, lang: &LangData, max_keywords: usize) -> String {
    match mode {
        QueryMode::FullText => claim.split_whitespace().collect::<Vec<_>>().join(" "),
        QueryMode::KeywordsOnly => keywords(claim, lang, max_keywords).join(" "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harpia_text::lang::Lang;

    #[test]
    fn keyword_queries_drop_stopwords() {
        let en = Lang::En.data();
        assert_eq!(
            keywords("The harpy eagle is the largest eagle of the Americas in 2024", &en, 10),
            ["harpy", "eagle", "largest", "americas", "2024"]
        );
        assert_eq!(build_query("The harpy eagle   hunts", QueryMode::FullText, &en, 3), "The harpy eagle hunts");
        assert_eq!(build_query("The harpy eagle hunts sloths", QueryMode::KeywordsOnly, &en, 2), "harpy eagle");
        let pt = Lang::PtBr.data();
        assert_eq!(keywords("A harpia é a maior águia do Brasil", &pt, 10), ["harpia", "maior", "águia", "brasil"]);
    }
}
