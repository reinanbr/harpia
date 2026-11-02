//! Heuristic quality filters implementing [`DataFilter`], with presets and TOML configuration.

use harpia_core::traits::{DataFilter, FilterDecision};
use harpia_core::{HarpiaError, Result};
use harpia_text::stats::{duplicate_ngram_fraction, signals, top_ngram_fraction};
use harpia_text::{Lang, LangData};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Thresholds for [`QualityFilter`]. `None` disables a check.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QualityThresholds {
    /// Minimum number of characters.
    pub min_chars: Option<usize>,
    /// Maximum number of characters.
    pub max_chars: Option<usize>,
    /// Minimum number of words.
    pub min_words: Option<usize>,
    /// Maximum number of words.
    pub max_words: Option<usize>,
    /// Mean word length range (chars).
    pub mean_word_len: Option<(f64, f64)>,
    /// Minimum fraction of alphabetic chars among non-whitespace.
    pub min_alpha_ratio: Option<f64>,
    /// Maximum fraction of symbol chars among non-whitespace.
    pub max_symbol_ratio: Option<f64>,
    /// Maximum fraction of lines starting with a bullet.
    pub max_bullet_line_ratio: Option<f64>,
    /// Maximum fraction of lines ending with an ellipsis.
    pub max_ellipsis_line_ratio: Option<f64>,
    /// Maximum fraction of duplicated lines.
    pub max_duplicate_line_ratio: Option<f64>,
    /// Minimum fraction of stopwords among words (catches word salad and code/tables).
    pub min_stopword_ratio: Option<f64>,
    /// Maximum fraction covered by the most frequent 2-, 3- and 4-gram.
    pub max_top_ngram: Option<[f64; 3]>,
    /// Maximum fraction covered by duplicated 5- to 10-grams (one threshold per n).
    pub max_duplicate_ngram: Option<[f64; 6]>,
}

impl QualityThresholds {
    /// Thresholds in the spirit of Gopher's quality rules (Rae et al., 2021).
    pub fn gopher_like() -> Self {
        Self {
            min_words: Some(50),
            max_words: Some(100_000),
            mean_word_len: Some((3.0, 10.0)),
            min_alpha_ratio: Some(0.8),
            max_symbol_ratio: Some(0.1),
            max_bullet_line_ratio: Some(0.9),
            max_ellipsis_line_ratio: Some(0.3),
            max_duplicate_line_ratio: Some(0.3),
            min_stopword_ratio: Some(0.05),
            max_top_ngram: Some([0.20, 0.18, 0.16]),
            max_duplicate_ngram: Some([0.15, 0.14, 0.13, 0.12, 0.11, 0.10]),
            ..Self::default()
        }
    }

    /// Lighter rules in the spirit of C4's line-level heuristics, applied document-wide.
    pub fn c4_like() -> Self {
        Self {
            min_words: Some(25),
            min_alpha_ratio: Some(0.6),
            max_duplicate_line_ratio: Some(0.5),
            max_ellipsis_line_ratio: Some(0.5),
            ..Self::default()
        }
    }

    /// Only drop empty and near-empty texts.
    pub fn minimal() -> Self {
        Self { min_chars: Some(1), min_words: Some(3), ..Self::default() }
    }

    /// Preset by name: `gopher-like`, `c4-like`, `minimal`.
    pub fn preset(name: &str) -> Result<Self> {
        match name {
            "gopher-like" | "gopher" => Ok(Self::gopher_like()),
            "c4-like" | "c4" => Ok(Self::c4_like()),
            "minimal" => Ok(Self::minimal()),
            other => Err(HarpiaError::invalid(format!(
                "unknown filter preset `{other}` (use gopher-like, c4-like or minimal)"
            ))),
        }
    }

    /// Parse thresholds from TOML.
    ///
    /// ```
    /// use harpia_data::filter::QualityThresholds;
    /// let t = QualityThresholds::from_toml("min_words = 10\nmax_symbol_ratio = 0.2").unwrap();
    /// assert_eq!(t.min_words, Some(10));
    /// ```
    pub fn from_toml(text: &str) -> Result<Self> {
        toml::from_str(text).map_err(|e| HarpiaError::invalid(format!("filter config: {e}")))
    }
}

/// Applies [`QualityThresholds`]; drop reasons name the failed check (`"min_words"`, ...).
#[derive(Debug, Clone)]
pub struct QualityFilter {
    name: String,
    t: QualityThresholds,
    lang: Arc<LangData>,
}

impl QualityFilter {
    /// New filter with word lists of `lang` (stopwords).
    pub fn new(thresholds: QualityThresholds, lang: Arc<LangData>) -> Self {
        Self { name: "quality".into(), t: thresholds, lang }
    }

    /// Rename (for reports).
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl DataFilter for QualityFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn decide(&self, text: &str) -> FilterDecision {
        let t = &self.t;
        let s = signals(text, &self.lang);
        let drop = |r: &str| FilterDecision::Drop(r.to_string());
        if t.min_chars.is_some_and(|m| s.chars < m) {
            return drop("min_chars");
        }
        if t.max_chars.is_some_and(|m| s.chars > m) {
            return drop("max_chars");
        }
        if t.min_words.is_some_and(|m| s.words < m) {
            return drop("min_words");
        }
        if t.max_words.is_some_and(|m| s.words > m) {
            return drop("max_words");
        }
        if t.mean_word_len.is_some_and(|(lo, hi)| s.mean_word_len < lo || s.mean_word_len > hi) {
            return drop("mean_word_len");
        }
        if t.min_alpha_ratio.is_some_and(|m| s.alpha_ratio < m) {
            return drop("min_alpha_ratio");
        }
        if t.max_symbol_ratio.is_some_and(|m| s.symbol_ratio > m) {
            return drop("max_symbol_ratio");
        }
        if t.max_bullet_line_ratio.is_some_and(|m| s.bullet_line_ratio > m) {
            return drop("max_bullet_line_ratio");
        }
        if t.max_ellipsis_line_ratio.is_some_and(|m| s.ellipsis_line_ratio > m) {
            return drop("max_ellipsis_line_ratio");
        }
        if t.max_duplicate_line_ratio.is_some_and(|m| s.duplicate_line_ratio > m) {
            return drop("max_duplicate_line_ratio");
        }
        if t.min_stopword_ratio.is_some_and(|m| s.stopword_ratio < m) {
            return drop("min_stopword_ratio");
        }
        if let Some(th) = t.max_top_ngram {
            for (i, max) in th.iter().enumerate() {
                if top_ngram_fraction(text, i + 2) > *max {
                    return drop(&format!("top_{}gram", i + 2));
                }
            }
        }
        if let Some(th) = t.max_duplicate_ngram {
            for (i, max) in th.iter().enumerate() {
                if duplicate_ngram_fraction(text, i + 5) > *max {
                    return drop(&format!("duplicate_{}gram", i + 5));
                }
            }
        }
        FilterDecision::Keep
    }
}

/// Keeps texts detected as one of the allowed languages (ISO 639-3 codes, e.g. `"por"`).
#[cfg(feature = "langid")]
#[derive(Debug, Clone)]
pub struct LanguageFilter {
    allowed: Vec<String>,
    min_confidence: f64,
}

#[cfg(feature = "langid")]
impl LanguageFilter {
    /// New filter.
    pub fn new<I, S>(allowed: I, min_confidence: f64) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self { allowed: allowed.into_iter().map(Into::into).collect(), min_confidence }
    }
}

#[cfg(feature = "langid")]
impl DataFilter for LanguageFilter {
    fn name(&self) -> &str {
        "language"
    }

    fn decide(&self, text: &str) -> FilterDecision {
        match harpia_text::langid::detect(text) {
            Some(d) if self.allowed.iter().any(|a| a == d.code) && d.confidence >= self.min_confidence => {
                FilterDecision::Keep
            }
            Some(d) => FilterDecision::Drop(format!("lang_{}", d.code)),
            None => FilterDecision::Drop("lang_unknown".into()),
        }
    }
}

/// A filter from a closure.
pub struct FnFilter<F> {
    name: String,
    f: F,
}

impl<F: Fn(&str) -> FilterDecision + Send + Sync> FnFilter<F> {
    /// Wrap `f` under `name`.
    pub fn new(name: impl Into<String>, f: F) -> Self {
        Self { name: name.into(), f }
    }
}

impl<F: Fn(&str) -> FilterDecision + Send + Sync> DataFilter for FnFilter<F> {
    fn name(&self) -> &str {
        &self.name
    }
    fn decide(&self, text: &str) -> FilterDecision {
        (self.f)(text)
    }
}

/// Default language data for filters.
pub fn default_lang() -> Arc<LangData> {
    Lang::En.data()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gopher_drops_spam_and_keeps_prose() {
        let f = QualityFilter::new(QualityThresholds::gopher_like(), Lang::En.data());
        let prose = "The harpy eagle is one of the largest birds of prey in the world. It lives in the \
                     tropical forests of the Americas, where it hunts sloths and monkeys in the canopy. \
                     Because it needs large areas of intact forest, it is threatened by deforestation, and \
                     conservation programs in Brazil and Panama work to protect its nesting trees.";
        assert_eq!(f.decide(prose), FilterDecision::Keep);
        assert_eq!(f.decide("buy now"), FilterDecision::Drop("min_words".into()));
        let spam = "buy cheap pills now ".repeat(30);
        assert!(matches!(f.decide(&spam), FilterDecision::Drop(r) if r.contains("gram") || r == "min_stopword_ratio"));
    }

    #[test]
    fn presets_by_name() {
        assert!(QualityThresholds::preset("c4-like").is_ok());
        assert!(QualityThresholds::preset("nope").is_err());
        assert!(QualityThresholds::from_toml("unknown = 1").is_err());
    }
}
