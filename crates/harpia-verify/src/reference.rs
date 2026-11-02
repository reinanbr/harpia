//! Reference verification: does a cited work exist, and do its metadata match?
//!
//! Lookup by DOI (Crossref, then OpenAlex) or arXiv id first, then bibliographic search; each
//! candidate is compared with a weighted fuzzy match (title 0.6, first-author surname 0.25,
//! year ±1 0.15; venue only reported). Outcomes:
//! `Confirmed` (title ≥ 0.9 and score ≥ threshold, no divergences), `Divergent { fields }` (found
//! with differing metadata, or only a partial match), `NotFound` (sources answered, nothing
//! matched; **not** proof that the work does not exist), `Unavailable` (no source answered).
//! Retraction and correction notices are always surfaced.
//!
//! [`check_quote`] locates a quoted passage in a work's full text pages with `partial_ratio`.

use crate::adapters::SourceAdapter;
use crate::fetch::{Disclosure, Fetcher};
use crate::work::{Lookup, Work, WorkId, clean_doi, valid_arxiv, valid_doi};
use harpia_text::fuzzy::{partial_ratio, token_set_ratio};
use serde::{Deserialize, Serialize};

/// A reference to check.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ReferenceQuery {
    /// Citation key or row id.
    pub key: String,
    /// Entry type (`article`, `book`, `techreport`...), used to soften "not found".
    #[serde(default)]
    pub kind: String,
    /// Title.
    pub title: String,
    /// Author family names in order.
    #[serde(default)]
    pub authors: Vec<String>,
    /// Year.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub year: Option<i32>,
    /// Journal / venue.
    #[serde(default)]
    pub venue: String,
    /// DOI.
    #[serde(default)]
    pub doi: String,
    /// arXiv id.
    #[serde(default)]
    pub arxiv: String,
    /// ISBN.
    #[serde(default)]
    pub isbn: String,
}

/// Outcome of a reference check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum ReferenceStatus {
    /// Found with matching metadata.
    Confirmed,
    /// Found (or partially matched) with differing fields.
    Divergent {
        /// Human-readable descriptions of the differing fields.
        fields: Vec<String>,
    },
    /// Sources answered but nothing matched (not proof of non-existence).
    NotFound,
    /// No source could be consulted.
    Unavailable,
    /// Not attempted (offline).
    Skipped {
        /// Why.
        reason: String,
    },
}

/// Component scores of a match.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct MatchScores {
    /// Weighted score.
    pub score: f64,
    /// Title similarity (`token_set_ratio` on normalized text).
    pub title: f64,
    /// First-author score.
    pub author: f64,
    /// Year score.
    pub year: f64,
}

/// Result of a reference check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceResult {
    /// Citation key.
    pub key: String,
    /// Status.
    #[serde(flatten)]
    pub status: ReferenceStatus,
    /// Confidence in the status.
    pub confidence: f64,
    /// Best candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched: Option<Work>,
    /// Scores of the best candidate.
    pub scores: MatchScores,
    /// Retraction reported by a source.
    pub retracted: bool,
    /// Correction notices.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corrections: Vec<String>,
    /// Supporting notes (sources consulted, DOI problems, closest candidate...).
    pub notes: Vec<String>,
    /// Sources consulted, in order.
    pub consulted: Vec<String>,
    /// Entry type rarely indexed by scholarly APIs (reports, theses, books without DOI).
    pub weakly_indexed: bool,
}

/// Settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceConfig {
    /// Minimum weighted score for `Confirmed` (with title ≥ 0.9).
    pub threshold: f64,
    /// Source ids tried for DOI lookups, in order.
    pub doi_sources: Vec<String>,
    /// Source ids tried for bibliographic search, in order.
    pub search_sources: Vec<String>,
    /// Source id for arXiv lookups.
    pub arxiv_source: String,
    /// Results per search.
    pub rows: usize,
}

impl Default for ReferenceConfig {
    fn default() -> Self {
        Self {
            threshold: 0.85,
            doi_sources: vec!["crossref".into(), "openalex".into()],
            search_sources: vec!["crossref".into(), "openalex".into()],
            arxiv_source: "arxiv".into(),
            rows: 5,
        }
    }
}

/// Normalization used for matching: accents stripped, lowercase, markup and punctuation removed.
pub fn norm(s: &str) -> String {
    let no_tags = crate::adapters::strip_tags(s);
    let plain = harpia_text::normalize::strip_accents(&no_tags).to_lowercase();
    plain
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn ratio(a: &str, b: &str) -> f64 {
    let (a, b) = (norm(a), norm(b));
    if a.is_empty() || b.is_empty() { 0.0 } else { token_set_ratio(&a, &b) }
}

/// Compare a reference with a candidate (errei's weights). Returns scores and divergences.
pub fn compare(r: &ReferenceQuery, c: &Work) -> (MatchScores, Vec<String>) {
    let title = ratio(&r.title, &c.title);
    let mut div = Vec::new();
    let author = match (r.authors.first(), c.authors.is_empty()) {
        (Some(first), false) => {
            let mine = norm(first);
            let theirs: Vec<String> = c.authors.iter().map(|a| norm(&a.family)).collect();
            let s = if theirs.first() == Some(&mine) {
                1.0
            } else if theirs.contains(&mine) {
                0.8
            } else {
                theirs.iter().take(3).map(|t| ratio(&mine, t)).fold(0.0, f64::max) * 0.8
            };
            if s < 0.8 {
                div.push(format!("first author: '{first}' vs '{}'", c.authors[0].family));
            }
            s
        }
        _ => 0.5,
    };
    let year = match (r.year, c.year) {
        (Some(a), Some(b)) => {
            let d = (a - b).abs();
            if d >= 1 {
                div.push(format!("year: {a} vs {b}"));
            }
            match d {
                0 => 1.0,
                1 => 0.7,
                _ => 0.0,
            }
        }
        _ => 0.5,
    };
    if !r.venue.is_empty() && !c.venue.is_empty() && ratio(&r.venue, &c.venue) < 0.6 {
        div.push(format!("venue: '{}' vs '{}'", r.venue, c.venue));
    }
    (MatchScores { score: 0.6 * title + 0.25 * author + 0.15 * year, title, author, year }, div)
}

const WEAKLY_INDEXED: &[&str] = &[
    "techreport",
    "report",
    "phdthesis",
    "mastersthesis",
    "thesis",
    "misc",
    "unpublished",
    "booklet",
    "manual",
    "book",
    "inbook",
];

fn notfound_confidence(best: f64, weak: bool) -> f64 {
    let c = (0.5 + (0.85 - best)).clamp(0.3, 0.95);
    ((c * if weak { 0.5 } else { 1.0 }) * 100.0).round() / 100.0
}

/// Checks references against the registry's scholarly sources.
pub struct ReferenceVerifier<'a> {
    fetcher: &'a Fetcher,
    adapters: Vec<Box<dyn SourceAdapter>>,
    cfg: ReferenceConfig,
}

impl std::fmt::Debug for ReferenceVerifier<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceVerifier").field("cfg", &self.cfg).finish_non_exhaustive()
    }
}

impl<'a> ReferenceVerifier<'a> {
    /// New verifier with the built-in adapters.
    pub fn new(fetcher: &'a Fetcher, cfg: ReferenceConfig) -> Self {
        Self { fetcher, adapters: crate::adapters::builtin(), cfg }
    }

    /// Add (or replace, by name) an adapter.
    pub fn with_adapter(mut self, adapter: Box<dyn SourceAdapter>) -> Self {
        self.adapters.retain(|a| a.name() != adapter.name());
        self.adapters.push(adapter);
        self
    }

    fn source(&self, id: &str) -> Option<(&crate::registry::Source, &dyn SourceAdapter)> {
        let s = self.fetcher.registry().get(id).filter(|s| s.enabled && s.unavailable_reason().is_none())?;
        let a = self.adapters.iter().find(|a| a.name() == s.adapter)?;
        Some((s, a.as_ref()))
    }

    /// Everything sent so far.
    pub fn disclosure(&self) -> Disclosure {
        self.fetcher.disclosure()
    }

    /// Check one reference.
    pub async fn verify(&self, r: &ReferenceQuery) -> ReferenceResult {
        let mut res = ReferenceResult {
            key: r.key.clone(),
            status: ReferenceStatus::Unavailable,
            confidence: 0.0,
            matched: None,
            scores: MatchScores::default(),
            retracted: false,
            corrections: Vec::new(),
            notes: Vec::new(),
            consulted: Vec::new(),
            weakly_indexed: WEAKLY_INDEXED.contains(&r.kind.to_ascii_lowercase().as_str()) && r.doi.trim().is_empty(),
        };
        if let Some(reason) = self.fetcher.skip_reason() {
            res.status = ReferenceStatus::Skipped { reason };
            return res;
        }
        if r.title.trim().is_empty() {
            res.status = ReferenceStatus::NotFound;
            res.notes.push("the reference has no title".into());
            return res;
        }
        let mut answered = 0usize;
        let mut unavailable = false;
        let mut best: Option<(MatchScores, Vec<String>, Work)> = None;
        let mut doi_problem = String::new();
        let doi = if valid_doi(&r.doi) { clean_doi(&r.doi) } else { String::new() };
        if !doi.is_empty() {
            let mut found = false;
            for id in &self.cfg.doi_sources {
                let Some((s, a)) = self.source(id) else { continue };
                res.consulted.push(s.id.clone());
                match a.lookup(self.fetcher, s, &WorkId::Doi(doi.clone())).await {
                    Lookup::Unavailable(e) => {
                        unavailable = true;
                        tracing::debug!(source = %s.id, "lookup unavailable: {e}");
                    }
                    Lookup::NotFound => answered += 1,
                    Lookup::Found(w) => {
                        answered += 1;
                        let (m, d) = compare(r, &w);
                        if m.title >= 0.8 {
                            best = Some((m, d, *w));
                        } else {
                            doi_problem = format!("the DOI {doi} belongs to another work: '{}'", w.title);
                        }
                        found = true;
                        break;
                    }
                }
            }
            if !found && answered > 0 {
                doi_problem = format!("the DOI {doi} was not found ({})", self.cfg.doi_sources.join("/"));
            }
        }
        if best.is_none()
            && valid_arxiv(&r.arxiv)
            && let Some((s, a)) = self.source(&self.cfg.arxiv_source)
        {
            res.consulted.push(s.id.clone());
            match a.lookup(self.fetcher, s, &WorkId::Arxiv(r.arxiv.clone())).await {
                Lookup::Unavailable(_) => unavailable = true,
                Lookup::NotFound => answered += 1,
                Lookup::Found(w) => {
                    answered += 1;
                    let (m, d) = compare(r, &w);
                    if m.title >= 0.8 {
                        best = Some((m, d, *w));
                    }
                }
            }
        }
        if best.is_none() {
            let mut q = r.title.clone();
            if let Some(a) = r.authors.first() {
                q.push(' ');
                q.push_str(a);
            }
            if let Some(y) = r.year {
                q.push_str(&format!(" {y}"));
            }
            for id in &self.cfg.search_sources {
                let Some((s, a)) = self.source(id) else { continue };
                res.consulted.push(s.id.clone());
                match a.search(self.fetcher, s, &q, self.cfg.rows).await {
                    Err(_) => unavailable = true,
                    Ok(cands) => {
                        answered += 1;
                        for c in cands {
                            let (m, d) = compare(r, &c);
                            if best.as_ref().is_none_or(|b| m.score > b.0.score) {
                                best = Some((m, d, c));
                            }
                        }
                    }
                }
                if best.as_ref().is_some_and(|b| b.0.score >= self.cfg.threshold && b.0.title >= 0.9) {
                    break;
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        res.consulted.retain(|s| seen.insert(s.clone()));
        if let Some((m, divergences, w)) = best.clone() {
            res.scores = m;
            res.retracted = w.retracted == Some(true);
            res.corrections = w.corrections.clone();
            if res.retracted {
                res.notes.push(format!("{} reports a retraction", w.source));
            }
            if m.title >= 0.9 && m.score >= self.cfg.threshold {
                let mut fields = divergences;
                if !doi_problem.is_empty() {
                    fields.insert(0, doi_problem);
                }
                res.status =
                    if fields.is_empty() { ReferenceStatus::Confirmed } else { ReferenceStatus::Divergent { fields } };
                res.confidence = (m.score * 100.0).round() / 100.0;
                res.notes.push(format!(
                    "title {:.0}%, author {:.0}%, year {:.0}%",
                    m.title * 100.0,
                    m.author * 100.0,
                    m.year * 100.0
                ));
                res.matched = Some(w);
                return res;
            }
            if m.score >= 0.7 && m.title >= 0.75 {
                let mut fields = vec![format!("only a partial match (title {:.0}%)", m.title * 100.0)];
                if !doi_problem.is_empty() {
                    fields.insert(0, doi_problem);
                }
                fields.extend(divergences);
                res.status = ReferenceStatus::Divergent { fields };
                res.confidence = (m.score * 100.0).round() / 100.0;
                res.matched = Some(w);
                return res;
            }
        }
        if answered == 0 {
            res.status = ReferenceStatus::Unavailable;
            res.notes.push("no source answered".into());
            return res;
        }
        res.status = ReferenceStatus::NotFound;
        res.notes.push(format!("sources consulted: {}", res.consulted.join(", ")));
        if !doi_problem.is_empty() {
            res.notes.push(doi_problem);
        }
        if let Some((m, _, w)) = &best {
            res.notes.push(format!("closest candidate: '{}' (similarity {:.0}%)", w.title, m.score * 100.0));
        }
        if unavailable {
            res.notes.push("some source was unavailable; the result is incomplete".into());
        }
        res.confidence = notfound_confidence(best.map_or(0.0, |b| b.0.score), res.weakly_indexed);
        res
    }
}

/// Result of locating a quote in full text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuoteCheck {
    /// Whether the quote was found (best similarity ≥ 0.9).
    pub found: bool,
    /// Best `partial_ratio` similarity.
    pub score: f64,
    /// 1-based page with the best match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    /// Whether that page lies within the cited page range (when one was given). PDF page numbers
    /// can differ from printed page numbers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_matches: Option<bool>,
}

/// Locate `quote` in `pages` (full text per page). `cited` is the cited page range.
///
/// ```
/// use harpia_verify::reference::check_quote;
/// let pages = ["Intro text.".to_string(), "As shown, readability formulas apply to textbooks.".to_string()];
/// let q = check_quote("readability formulas apply to textbooks", &pages, Some((2, 2)));
/// assert!(q.found && q.page == Some(2) && q.page_matches == Some(true));
/// ```
pub fn check_quote(quote: &str, pages: &[String], cited: Option<(usize, usize)>) -> QuoteCheck {
    let q = norm(quote);
    let mut best = (0.0, None);
    if !q.is_empty() {
        for (i, p) in pages.iter().enumerate() {
            let s = partial_ratio(&q, &norm(p));
            if s > best.0 {
                best = (s, Some(i + 1));
            }
        }
    }
    let found = best.0 >= 0.9;
    QuoteCheck {
        found,
        score: best.0,
        page: best.1.filter(|_| found),
        page_matches: if found { cited.zip(best.1).map(|((a, b), p)| a <= p && p <= b) } else { None },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work::Author;

    fn work(title: &str, family: &str, year: i32) -> Work {
        Work {
            title: title.into(),
            authors: vec![Author { family: family.into(), full: String::new() }],
            year: Some(year),
            ..Work::default()
        }
    }

    #[test]
    fn compare_weights() {
        let r = ReferenceQuery {
            title: "Deep learning".into(),
            authors: vec!["LeCun".into()],
            year: Some(2015),
            venue: "Nature".into(),
            ..Default::default()
        };
        let (m, d) = compare(&r, &work("Deep Learning", "LeCun", 2015));
        assert!((m.score - 1.0).abs() < 1e-9 && d.is_empty());
        let (m, d) = compare(&r, &work("Deep learning", "Bengio", 2016));
        assert!(d.iter().any(|x| x.starts_with("first author")) && d.iter().any(|x| x.starts_with("year")));
        assert!((m.year - 0.7).abs() < 1e-9);
        assert_eq!(norm("Detecção <i>quântica</i>, já!"), "deteccao quantica ja");
        assert_eq!(notfound_confidence(0.0, false), 0.95);
        assert_eq!(notfound_confidence(0.0, true), 0.48);
    }

    #[test]
    fn quote_not_found() {
        let q = check_quote("completely different sentence here", &["nothing like it".into()], None);
        assert!(!q.found && q.page.is_none());
    }
}
