//! Fast multi-phrase matching (Aho-Corasick), case- and accent-insensitive, with spans in the
//! original text and whole-word boundaries.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use harpia_core::{HarpiaError, Result, Span};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

/// A folded (lowercase, accent-stripped) copy of a text with a map back to original offsets.
#[derive(Debug, Clone)]
pub struct Folded {
    /// Folded text.
    pub text: String,
    /// For each byte of `text`, the start byte of the originating char in the original.
    starts: Vec<usize>,
    /// For each byte of `text`, the end byte of the originating char in the original.
    ends: Vec<usize>,
}

impl Folded {
    /// Fold `original`.
    pub fn new(original: &str) -> Self {
        let mut text = String::with_capacity(original.len());
        let mut starts = Vec::with_capacity(original.len());
        let mut ends = Vec::with_capacity(original.len());
        for (i, c) in original.char_indices() {
            let end = i + c.len_utf8();
            for l in c.to_lowercase() {
                for d in std::iter::once(l).nfkd().filter(|d| !is_combining_mark(*d)) {
                    let before = text.len();
                    text.push(d);
                    for _ in before..text.len() {
                        starts.push(i);
                        ends.push(end);
                    }
                }
            }
        }
        Self { text, starts, ends }
    }

    /// Map a span of the folded text back to the original text.
    pub fn original_span(&self, folded: Span) -> Span {
        if folded.is_empty() {
            let p = self.starts.get(folded.start).copied().unwrap_or_else(|| self.ends.last().copied().unwrap_or(0));
            return Span::new(p, p);
        }
        Span::new(self.starts[folded.start], self.ends[folded.end - 1])
    }
}

/// Fold a string (lowercase + strip accents + NFKD compatibility folding).
pub fn fold(s: &str) -> String {
    Folded::new(s).text
}

/// A match of [`PhraseMatcher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhraseMatch {
    /// Index of the phrase in the list given to [`PhraseMatcher::new`].
    pub phrase: usize,
    /// Span in the original text.
    pub span: Span,
}

/// Finds occurrences of many phrases at once, leftmost-longest, whole words only by default.
///
/// ```
/// use harpia_text::phrases::PhraseMatcher;
/// let m = PhraseMatcher::new(["vale ressaltar", "é importante notar"]).unwrap();
/// let t = "Vale Ressaltar que... É importante notar isso.";
/// let found: Vec<&str> = m.find_iter(t).map(|x| x.span.slice(t)).collect();
/// assert_eq!(found, ["Vale Ressaltar", "É importante notar"]);
/// ```
#[derive(Debug, Clone)]
pub struct PhraseMatcher {
    ac: AhoCorasick,
    whole_words: bool,
}

impl PhraseMatcher {
    /// Build from phrases (folded internally).
    pub fn new<I, S>(phrases: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let pats: Vec<String> = phrases.into_iter().map(|p| fold(p.as_ref().trim())).collect();
        if pats.iter().any(String::is_empty) {
            return Err(HarpiaError::invalid("empty phrase"));
        }
        let ac = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&pats)
            .map_err(|e| HarpiaError::invalid(e.to_string()))?;
        Ok(Self { ac, whole_words: true })
    }

    /// Allow matches inside words.
    pub fn allow_partial_words(mut self) -> Self {
        self.whole_words = false;
        self
    }

    /// All matches in `text`.
    pub fn find_iter(&self, text: &str) -> std::vec::IntoIter<PhraseMatch> {
        let folded = Folded::new(text);
        let ft = &folded.text;
        let free = |c: Option<char>| !c.is_some_and(char::is_alphanumeric);
        let mut out = Vec::new();
        for m in self.ac.find_iter(ft) {
            if self.whole_words && !(free(ft[..m.start()].chars().next_back()) && free(ft[m.end()..].chars().next())) {
                continue;
            }
            out.push(PhraseMatch {
                phrase: m.pattern().as_usize(),
                span: folded.original_span(Span::new(m.start(), m.end())),
            });
        }
        out.into_iter()
    }

    /// `true` if any phrase occurs.
    pub fn is_match(&self, text: &str) -> bool {
        self.find_iter(text).next().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folded_maps_back() {
        let t = "AÇÃO ﬁnal";
        let f = Folded::new(t);
        assert_eq!(f.text, "acao final");
        let i = f.text.find("final").unwrap();
        assert_eq!(f.original_span(Span::new(i, i + 5)).slice(t), "ﬁnal");
    }

    #[test]
    fn whole_words() {
        let m = PhraseMatcher::new(["cat"]).unwrap();
        assert_eq!(m.find_iter("concatenate cat").count(), 1);
        assert_eq!(m.clone().allow_partial_words().find_iter("concatenate cat").count(), 2);
        assert!(PhraseMatcher::new([" "]).is_err());
    }
}
