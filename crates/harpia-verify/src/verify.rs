//! Claim verification against trusted sources.
//!
//! claim → query (keywords by default) → search every enabled text source → passages (BM25 over
//! the sentences of each returned text) → NLI [`PairClassifier`] on each top sentence and its
//! 2-sentence window (premise = passage, hypothesis = claim) → calibrated decision per window →
//! per-source verdict → explicit cross-source aggregation:
//!
//! - the most trusted tier with a decisive verdict (supported or contradicted) decides;
//! - a contradiction is never hidden by support from a lower tier (it decides if it is the most
//!   trusted decisive verdict; otherwise it is reported as disagreement);
//! - conflicting verdicts inside the deciding tier give `NotEnoughInfo` with `disagreement`
//!   set: disagreement is reported, not averaged.
//!
//! The result is an indicator with evidence, never a verdict on truth. `NotEnoughInfo` is the
//! normal outcome when sources do not cover the point, and "not found" does not mean false.

use crate::adapters::SourceAdapter;
use crate::claims::{Claim, QueryMode, build_query, split_claims};
use crate::fetch::{Disclosure, Fetcher};
use crate::work::Work;
use harpia_core::traits::PairClassifier;
use harpia_core::{Result, Scores, Span};
use harpia_eval::calibration::{Calibrated, DecisionRule, Thresholds};
use harpia_text::bm25::{Bm25, Bm25Params};
use harpia_text::lang::{Lang, LangData};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Outcome of checking one claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum Status {
    /// Trusted evidence entails the claim.
    Supported,
    /// Trusted evidence contradicts the claim.
    Contradicted,
    /// Sources answered but do not settle the point (the usual outcome).
    NotEnoughInfo,
    /// No source could be consulted (network errors, frozen-store misses).
    Unavailable,
    /// Not attempted (e.g. offline).
    Skipped {
        /// Why.
        reason: String,
    },
}

/// One piece of evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// Source id.
    pub source: String,
    /// Source tier.
    pub tier: u8,
    /// Document URL.
    pub url: String,
    /// Document title.
    pub title: String,
    /// Passage (one sentence or a 2-sentence window).
    pub passage: String,
    /// Byte span of the passage in the document text.
    pub span: Span,
    /// Decided label for this passage.
    pub label: String,
    /// Classifier scores.
    pub scores: Scores,
    /// Lexical (BM25) score of the passage's first sentence against the claim.
    pub retrieval_score: f32,
    /// When the document was retrieved.
    pub retrieved_at: String,
    /// SHA-256 of the response the document came from.
    pub content_hash: String,
    /// Content license, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

/// Verdict of one source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceVerdict {
    /// Source id.
    pub source: String,
    /// Tier.
    pub tier: u8,
    /// Status from this source alone.
    pub status: Status,
    /// Probability of the decided label (best passage).
    pub confidence: f32,
    /// Documents examined.
    pub documents: usize,
    /// Error, when the source was unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Result for one claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verification {
    /// The claim.
    pub claim: String,
    /// Aggregated status.
    #[serde(flatten)]
    pub status: Status,
    /// Confidence in the status (probability of the deciding passage; 0 when not decisive).
    pub confidence: f32,
    /// Best evidence, decisive passages first.
    pub evidence: Vec<Evidence>,
    /// Per-source verdicts.
    pub per_source: Vec<SourceVerdict>,
    /// Sources disagree (reported, never averaged away).
    pub disagreement: bool,
    /// Queries sent.
    pub queries: Vec<String>,
    /// Notes about LLM assistance used for this claim (empty when none).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub llm_assistance: Vec<String>,
}

/// Label names of the NLI classifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NliLabels {
    /// Label meaning "supported".
    pub entailment: String,
    /// Label meaning "contradicted".
    pub contradiction: String,
    /// Label meaning "neither".
    pub neutral: String,
}

impl Default for NliLabels {
    fn default() -> Self {
        Self { entailment: "entailment".into(), contradiction: "contradiction".into(), neutral: "neutral".into() }
    }
}

/// Verifier settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifyConfig {
    /// What is sent to search APIs.
    pub query_mode: QueryMode,
    /// Keywords per query.
    pub max_keywords: usize,
    /// Results requested from each source.
    pub results_per_source: usize,
    /// Top sentences (by BM25) examined per document.
    pub passages_per_document: usize,
    /// Evidence items kept in the result.
    pub max_evidence: usize,
    /// Language of claims and evidence (`en`, `pt-br`).
    pub lang: String,
    /// Classifier labels.
    pub labels: NliLabels,
    /// Decision rule and thresholds (default: NLI rule at 0.5).
    pub decision: Calibrated,
    /// Restrict to these source ids (default: every enabled source with a text adapter).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<String>>,
    /// Minimum words for a sentence to count as a claim when splitting answers.
    pub min_claim_words: usize,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        let labels = NliLabels::default();
        let rule = DecisionRule::nli(&labels.entailment, &labels.contradiction, &labels.neutral);
        Self {
            query_mode: QueryMode::KeywordsOnly,
            max_keywords: 8,
            results_per_source: 5,
            passages_per_document: 4,
            max_evidence: 5,
            lang: "en".into(),
            decision: Calibrated::fixed(rule.clone(), Thresholds::uniform(&rule, 0.5)),
            labels,
            sources: None,
            min_claim_words: 4,
        }
    }
}

/// Checks claims against the sources of a [`Fetcher`]'s registry.
pub struct ClaimVerifier<'a> {
    fetcher: &'a Fetcher,
    adapters: Vec<Box<dyn SourceAdapter>>,
    classifier: &'a dyn PairClassifier,
    cfg: VerifyConfig,
    lang: Arc<LangData>,
}

impl std::fmt::Debug for ClaimVerifier<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaimVerifier").field("cfg", &self.cfg).finish_non_exhaustive()
    }
}

/// Per-claim result inside an answer report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimResult {
    /// Byte span of the claim in the answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
    /// Verification.
    pub verification: Verification,
}

/// Grounding report for a whole answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnswerReport {
    /// Per-claim results.
    pub claims: Vec<ClaimResult>,
    /// Counts per status name.
    pub counts: std::collections::BTreeMap<String, usize>,
    /// Everything sent to third parties.
    pub disclosure: Disclosure,
}

/// Machine-friendly status name.
pub fn status_name(s: &Status) -> &'static str {
    match s {
        Status::Supported => "supported",
        Status::Contradicted => "contradicted",
        Status::NotEnoughInfo => "not_enough_info",
        Status::Unavailable => "unavailable",
        Status::Skipped { .. } => "skipped",
    }
}

impl<'a> ClaimVerifier<'a> {
    /// New verifier with the built-in adapters compiled into this build. Panics on an unknown
    /// `cfg.lang`.
    pub fn new(fetcher: &'a Fetcher, classifier: &'a dyn PairClassifier, cfg: VerifyConfig) -> Self {
        let lang = Lang::parse(&cfg.lang).map(Lang::data).expect("supported verification language");
        Self { fetcher, adapters: crate::adapters::builtin(), classifier, cfg, lang }
    }

    /// Add (or replace, by name) an adapter.
    pub fn with_adapter(mut self, adapter: Box<dyn SourceAdapter>) -> Self {
        self.adapters.retain(|a| a.name() != adapter.name());
        self.adapters.push(adapter);
        self
    }

    /// Settings.
    pub fn config(&self) -> &VerifyConfig {
        &self.cfg
    }

    fn skipped(&self, claim: &str, reason: String) -> Verification {
        Verification {
            claim: claim.into(),
            status: Status::Skipped { reason },
            confidence: 0.0,
            evidence: Vec::new(),
            per_source: Vec::new(),
            disagreement: false,
            queries: Vec::new(),
            llm_assistance: Vec::new(),
        }
    }

    /// Check one claim. `extra_queries` (e.g. LLM rewrites) are searched in addition to the
    /// built query.
    pub async fn verify_claim(&self, claim: &str, extra_queries: &[String]) -> Verification {
        if let Some(reason) = self.fetcher.skip_reason() {
            return self.skipped(claim, reason);
        }
        let mut queries = vec![build_query(claim, self.cfg.query_mode, &self.lang, self.cfg.max_keywords)];
        queries.extend(extra_queries.iter().filter(|q| !q.trim().is_empty()).cloned());
        queries.retain(|q| !q.trim().is_empty());
        queries.dedup();
        if queries.is_empty() {
            return Verification { status: Status::NotEnoughInfo, ..self.skipped(claim, String::new()) };
        }
        let mut per_source = Vec::new();
        let mut evidence: Vec<Evidence> = Vec::new();
        for source in &self.fetcher.registry().sources {
            if !source.enabled || self.cfg.sources.as_ref().is_some_and(|ids| !ids.contains(&source.id)) {
                continue;
            }
            let Some(adapter) = self.adapters.iter().find(|a| a.name() == source.adapter) else { continue };
            if !adapter.provides_text() {
                continue;
            }
            if let Some(reason) = source.unavailable_reason() {
                per_source.push(SourceVerdict {
                    source: source.id.clone(),
                    tier: source.tier,
                    status: Status::Unavailable,
                    confidence: 0.0,
                    documents: 0,
                    error: Some(reason),
                });
                continue;
            }
            let mut works: Vec<Work> = Vec::new();
            let mut error = None;
            for q in &queries {
                match adapter.search(self.fetcher, source, q, self.cfg.results_per_source).await {
                    Ok(ws) => {
                        for w in ws {
                            if !works.iter().any(|x| x.url == w.url && x.title == w.title) {
                                works.push(w);
                            }
                        }
                    }
                    Err(e) => error = Some(e.to_string()),
                }
            }
            if works.is_empty() && error.is_some() {
                per_source.push(SourceVerdict {
                    source: source.id.clone(),
                    tier: source.tier,
                    status: Status::Unavailable,
                    confidence: 0.0,
                    documents: 0,
                    error,
                });
                continue;
            }
            let (verdict, mut ev) = match self.judge_source(claim, source.id.as_str(), source.tier, &works) {
                Ok(x) => x,
                Err(e) => {
                    per_source.push(SourceVerdict {
                        source: source.id.clone(),
                        tier: source.tier,
                        status: Status::Unavailable,
                        confidence: 0.0,
                        documents: works.len(),
                        error: Some(format!("classifier: {e}")),
                    });
                    continue;
                }
            };
            evidence.append(&mut ev);
            per_source.push(verdict);
        }
        let (status, confidence, disagreement) = aggregate(&per_source);
        let decisive = |e: &Evidence| e.label == self.cfg.labels.entailment || e.label == self.cfg.labels.contradiction;
        evidence.sort_by(|a, b| {
            decisive(b).cmp(&decisive(a)).then(a.tier.cmp(&b.tier)).then(label_prob(b).total_cmp(&label_prob(a)))
        });
        evidence.truncate(self.cfg.max_evidence);
        Verification {
            claim: claim.into(),
            status,
            confidence,
            evidence,
            per_source,
            disagreement,
            queries,
            llm_assistance: Vec::new(),
        }
    }

    fn judge_source(
        &self,
        claim: &str,
        source: &str,
        tier: u8,
        works: &[Work],
    ) -> Result<(SourceVerdict, Vec<Evidence>)> {
        let mut evidence = Vec::new();
        let mut best: Vec<(String, f32)> = Vec::new();
        for w in works.iter().filter(|w| !w.text.trim().is_empty()) {
            let sents: Vec<Span> = harpia_text::segment::sentences(&w.text, &self.lang).collect();
            if sents.is_empty() {
                continue;
            }
            let bm = Bm25::new(sents.iter().map(|s| s.slice(&w.text)), &self.lang, Bm25Params::default());
            let mut ranked: Vec<(usize, f64)> = bm.scores(claim).into_iter().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut windows: Vec<(Span, f32)> = Vec::new();
            for &(i, score) in ranked.iter().take(self.cfg.passages_per_document) {
                windows.push((sents[i], score as f32));
                if let Some(next) = sents.get(i + 1) {
                    windows.push((Span::new(sents[i].start, next.end), score as f32));
                }
            }
            windows.dedup();
            let pairs: Vec<(&str, &str)> = windows.iter().map(|(s, _)| (s.slice(&w.text), claim)).collect();
            let scores = self.classifier.classify_pairs(&pairs)?;
            for ((span, rs), sc) in windows.iter().zip(scores) {
                for (l, p) in sc.iter() {
                    match best.iter_mut().find(|b| b.0 == l) {
                        Some(b) => b.1 = b.1.max(p),
                        None => best.push((l.to_string(), p)),
                    }
                }
                let label = self.cfg.decision.decide(&sc).to_string();
                evidence.push(Evidence {
                    source: source.into(),
                    tier,
                    url: w.url.clone(),
                    title: w.title.clone(),
                    passage: span.slice(&w.text).to_string(),
                    span: *span,
                    label,
                    scores: sc,
                    retrieval_score: *rs,
                    retrieved_at: w.retrieved_at.clone(),
                    content_hash: w.content_hash.clone(),
                    license: w.license.clone(),
                });
            }
        }
        let agg = Scores::new(best);
        let label = self.cfg.decision.decide(&agg).to_string();
        let status = if evidence.is_empty() {
            Status::NotEnoughInfo
        } else if label == self.cfg.labels.entailment {
            Status::Supported
        } else if label == self.cfg.labels.contradiction {
            Status::Contradicted
        } else {
            Status::NotEnoughInfo
        };
        let confidence = match status {
            Status::Supported | Status::Contradicted => agg.get(&label).unwrap_or(0.0),
            _ => 0.0,
        };
        // Keep only this source's most informative passages.
        evidence.sort_by(|a, b| label_prob(b).total_cmp(&label_prob(a)));
        evidence.truncate(self.cfg.max_evidence);
        Ok((
            SourceVerdict { source: source.into(), tier, status, confidence, documents: works.len(), error: None },
            evidence,
        ))
    }

    /// Split `answer` into claims and check each one.
    pub async fn verify_answer(&self, answer: &str) -> AnswerReport {
        let claims = split_claims(answer, &self.lang, self.cfg.min_claim_words);
        self.verify_claims(&claims).await
    }

    /// Check a list of claims (e.g. from LLM decomposition).
    pub async fn verify_claims(&self, claims: &[Claim]) -> AnswerReport {
        let mut out = Vec::with_capacity(claims.len());
        for c in claims {
            out.push(ClaimResult { span: c.span, verification: self.verify_claim(&c.text, &[]).await });
        }
        let mut counts = std::collections::BTreeMap::new();
        for c in &out {
            *counts.entry(status_name(&c.verification.status).to_string()).or_insert(0) += 1;
        }
        AnswerReport { claims: out, counts, disclosure: self.fetcher.disclosure() }
    }
}

fn label_prob(e: &Evidence) -> f32 {
    e.scores.get(&e.label).unwrap_or(0.0)
}

/// Cross-source aggregation (see the module docs). Returns (status, confidence, disagreement).
pub fn aggregate(per_source: &[SourceVerdict]) -> (Status, f32, bool) {
    let answered: Vec<&SourceVerdict> = per_source.iter().filter(|v| v.status != Status::Unavailable).collect();
    if answered.is_empty() {
        return (Status::Unavailable, 0.0, false);
    }
    let decisive: Vec<&SourceVerdict> =
        answered.iter().copied().filter(|v| matches!(v.status, Status::Supported | Status::Contradicted)).collect();
    let supported = decisive.iter().any(|v| v.status == Status::Supported);
    let contradicted = decisive.iter().any(|v| v.status == Status::Contradicted);
    let disagreement = supported && contradicted;
    let Some(top_tier) = decisive.iter().map(|v| v.tier).min() else {
        return (Status::NotEnoughInfo, 0.0, false);
    };
    let top: Vec<&&SourceVerdict> = decisive.iter().filter(|v| v.tier == top_tier).collect();
    let top_s = top
        .iter()
        .filter(|v| v.status == Status::Supported)
        .map(|v| v.confidence)
        .fold(None, |a: Option<f32>, c| Some(a.map_or(c, |a| a.max(c))));
    let top_c = top
        .iter()
        .filter(|v| v.status == Status::Contradicted)
        .map(|v| v.confidence)
        .fold(None, |a: Option<f32>, c| Some(a.map_or(c, |a| a.max(c))));
    match (top_s, top_c) {
        (Some(_), Some(_)) => (Status::NotEnoughInfo, 0.0, true),
        (Some(c), None) => (Status::Supported, c, disagreement),
        (None, Some(c)) => (Status::Contradicted, c, disagreement),
        (None, None) => (Status::NotEnoughInfo, 0.0, disagreement),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(source: &str, tier: u8, status: Status, confidence: f32) -> SourceVerdict {
        SourceVerdict { source: source.into(), tier, status, confidence, documents: 1, error: None }
    }

    #[test]
    fn aggregation_respects_tiers_and_reports_disagreement() {
        // Tier-1 contradiction is not hidden by tier-3 support.
        let (s, c, d) = aggregate(&[v("a", 3, Status::Supported, 0.99), v("b", 1, Status::Contradicted, 0.7)]);
        assert_eq!((s, c, d), (Status::Contradicted, 0.7, true));
        // Same tier conflict: reported, not averaged.
        let (s, _, d) = aggregate(&[v("a", 1, Status::Supported, 0.9), v("b", 1, Status::Contradicted, 0.8)]);
        assert_eq!((s, d), (Status::NotEnoughInfo, true));
        // Lower-tier contradiction under tier-1 support: support decides, disagreement flagged.
        let (s, _, d) = aggregate(&[v("a", 1, Status::Supported, 0.9), v("b", 2, Status::Contradicted, 0.95)]);
        assert_eq!((s, d), (Status::Supported, true));
        assert_eq!(aggregate(&[v("a", 1, Status::NotEnoughInfo, 0.0)]).0, Status::NotEnoughInfo);
        assert_eq!(aggregate(&[v("a", 1, Status::Unavailable, 0.0)]).0, Status::Unavailable);
        assert_eq!(aggregate(&[]).0, Status::Unavailable);
    }
}
