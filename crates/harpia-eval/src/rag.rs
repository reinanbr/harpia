//! RAG evaluation: retrieval metrics (recall@k, precision@k, MRR, nDCG), context precision and
//! recall, answer faithfulness against retrieved context and answer relevance.
//!
//! Faithfulness splits the answer into sentences and checks each one against the context with a
//! [`PairClassifier`] (premise = context window, hypothesis = answer sentence) using
//! [`best_of_windows`]; a sentence counts as supported when the calibrated rule says so. This is
//! an indicator, not proof: NLI models degrade out of domain.

use crate::calibration::{Calibrated, best_of_windows};
use harpia_core::traits::{Embedder, PairClassifier};
use harpia_core::{Result, Span};
use harpia_text::lang::LangData;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Fraction of relevant ids that appear in the top `k` retrieved.
///
/// ```
/// use harpia_eval::rag::recall_at_k;
/// assert_eq!(recall_at_k(&["a", "x", "b"], &["a", "b"], 2), 0.5);
/// ```
pub fn recall_at_k<S: AsRef<str>>(retrieved: &[S], relevant: &[S], k: usize) -> f64 {
    let rel: HashSet<&str> = relevant.iter().map(AsRef::as_ref).collect();
    if rel.is_empty() {
        return 0.0;
    }
    let hits = top_unique(retrieved, k).filter(|r| rel.contains(r)).count();
    hits as f64 / rel.len() as f64
}

/// Fraction of the top `k` retrieved that are relevant.
pub fn precision_at_k<S: AsRef<str>>(retrieved: &[S], relevant: &[S], k: usize) -> f64 {
    let rel: HashSet<&str> = relevant.iter().map(AsRef::as_ref).collect();
    let top: Vec<&str> = top_unique(retrieved, k).collect();
    if top.is_empty() {
        return 0.0;
    }
    top.iter().filter(|r| rel.contains(*r)).count() as f64 / top.len() as f64
}

fn top_unique<S: AsRef<str>>(retrieved: &[S], k: usize) -> impl Iterator<Item = &str> {
    let mut seen = HashSet::new();
    retrieved.iter().map(AsRef::as_ref).filter(move |r| seen.insert(*r)).take(k)
}

/// Reciprocal rank of the first relevant id (0 when none is retrieved).
pub fn reciprocal_rank<S: AsRef<str>>(retrieved: &[S], relevant: &[S]) -> f64 {
    let rel: HashSet<&str> = relevant.iter().map(AsRef::as_ref).collect();
    retrieved.iter().position(|r| rel.contains(r.as_ref())).map_or(0.0, |i| 1.0 / (i as f64 + 1.0))
}

/// nDCG@k with graded relevance (`gains[id]`, missing = 0) and the `2^rel - 1` gain.
///
/// ```
/// use harpia_eval::rag::ndcg_at_k;
/// use std::collections::HashMap;
/// let gains = HashMap::from([("a".to_string(), 1.0)]);
/// assert_eq!(ndcg_at_k(&["a", "b"], &gains, 10), 1.0);
/// assert!((ndcg_at_k(&["b", "a"], &gains, 10) - 1.0 / 3f64.log2()).abs() < 1e-12);
/// ```
pub fn ndcg_at_k<S: AsRef<str>>(retrieved: &[S], gains: &std::collections::HashMap<String, f64>, k: usize) -> f64 {
    let dcg = |rels: &mut dyn Iterator<Item = f64>| -> f64 {
        rels.take(k).enumerate().map(|(i, g)| (2f64.powf(g) - 1.0) / (i as f64 + 2.0).log2()).sum()
    };
    let actual = dcg(&mut top_unique(retrieved, k).map(|r| gains.get(r).copied().unwrap_or(0.0)));
    let mut ideal: Vec<f64> = gains.values().copied().filter(|g| *g > 0.0).collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let best = dcg(&mut ideal.into_iter());
    if best == 0.0 { 0.0 } else { actual / best }
}

/// One RAG evaluation example.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RagExample {
    /// User question.
    pub question: String,
    /// Retrieved ids, best first.
    pub retrieved: Vec<String>,
    /// Ids judged relevant.
    pub relevant: Vec<String>,
    /// Retrieved context passages (in retrieval order), for faithfulness.
    #[serde(default)]
    pub contexts: Vec<String>,
    /// Generated answer.
    #[serde(default)]
    pub answer: String,
    /// Reference answer, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
}

/// Retrieval figures averaged over examples.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RetrievalReport {
    /// Cut-off.
    pub k: usize,
    /// Mean recall@k.
    pub recall: f64,
    /// Mean precision@k (context precision).
    pub precision: f64,
    /// Mean reciprocal rank.
    pub mrr: f64,
    /// Mean nDCG@k (binary relevance).
    pub ndcg: f64,
    /// Examples.
    pub n: usize,
}

/// Retrieval metrics over a set of examples (binary relevance).
pub fn retrieval_report(examples: &[RagExample], k: usize) -> RetrievalReport {
    let n = examples.len();
    if n == 0 {
        return RetrievalReport { k, ..Default::default() };
    }
    let mut r = RetrievalReport { k, n, ..Default::default() };
    for e in examples {
        r.recall += recall_at_k(&e.retrieved, &e.relevant, k);
        r.precision += precision_at_k(&e.retrieved, &e.relevant, k);
        r.mrr += reciprocal_rank(&e.retrieved, &e.relevant);
        let gains = e.relevant.iter().map(|id| (id.clone(), 1.0)).collect();
        r.ndcg += ndcg_at_k(&e.retrieved, &gains, k);
    }
    for v in [&mut r.recall, &mut r.precision, &mut r.mrr, &mut r.ndcg] {
        *v /= n as f64;
    }
    r
}

/// Verdict for one answer sentence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentenceSupport {
    /// Byte span of the sentence in the answer.
    pub span: Span,
    /// Decided label.
    pub label: String,
    /// Index of the context passage holding the best evidence for that label.
    pub context: Option<usize>,
    /// Byte span of the evidence window inside that passage.
    pub evidence: Option<Span>,
}

/// Faithfulness of one answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Faithfulness {
    /// Fraction of answer sentences decided as `supported_label`.
    pub score: f64,
    /// Per-sentence detail.
    pub sentences: Vec<SentenceSupport>,
}

/// Check every sentence of `answer` against `contexts`.
///
/// For each sentence, every context passage is scanned with [`best_of_windows`]; per label the
/// best window over all passages is kept, then `rule` decides. `supported_label` is the label that
/// counts as grounded (e.g. `"entailment"`).
pub fn faithfulness(
    classifier: &dyn PairClassifier,
    rule: &Calibrated,
    supported_label: &str,
    answer: &str,
    contexts: &[String],
    lang: &LangData,
) -> Result<Faithfulness> {
    let sents: Vec<Span> = harpia_text::segment::sentences(answer, lang).collect();
    let mut out = Vec::with_capacity(sents.len());
    for span in &sents {
        let claim = span.slice(answer);
        let mut best: Vec<(String, f32, usize, Span)> = Vec::new();
        for (ci, ctx) in contexts.iter().enumerate() {
            let cs: Vec<Span> = harpia_text::segment::sentences(ctx, lang).collect();
            let Some(w) = best_of_windows(classifier, claim, ctx, &cs)? else { continue };
            for (l, p) in w.scores.iter() {
                let ev = w.evidence[l];
                match best.iter_mut().find(|b| b.0 == l) {
                    Some(b) if p > b.1 => *b = (l.to_string(), p, ci, ev),
                    Some(_) => {}
                    None => best.push((l.to_string(), p, ci, ev)),
                }
            }
        }
        let scores = harpia_core::Scores::new(best.iter().map(|b| (b.0.clone(), b.1)).collect());
        let label = rule.decide(&scores).to_string();
        let found = best.iter().find(|b| b.0 == label);
        out.push(SentenceSupport { span: *span, context: found.map(|b| b.2), evidence: found.map(|b| b.3), label });
    }
    let supported = out.iter().filter(|s| s.label == supported_label).count();
    Ok(Faithfulness { score: if out.is_empty() { 0.0 } else { supported as f64 / out.len() as f64 }, sentences: out })
}

/// Cosine similarity between the question and answer embeddings, mapped to `0.0..=1.0`.
pub fn answer_relevance(embedder: &dyn Embedder, question: &str, answer: &str) -> Result<f64> {
    let v = embedder.embed(&[question, answer])?;
    Ok(f64::from((cosine(&v[0], &v[1]) + 1.0) / 2.0))
}

/// Context recall against a reference answer: fraction of reference sentences whose content
/// words appear (at least `min_overlap` of them) in some context passage. A lexical proxy that
/// needs no model.
pub fn context_recall(reference: &str, contexts: &[String], lang: &LangData, min_overlap: f64) -> f64 {
    let sents: Vec<Span> = harpia_text::segment::sentences(reference, lang).collect();
    if sents.is_empty() {
        return 0.0;
    }
    let ctx_words: Vec<HashSet<String>> = contexts.iter().map(|c| content_words(c, lang)).collect();
    let covered = sents
        .iter()
        .filter(|s| {
            let w = content_words(s.slice(reference), lang);
            !w.is_empty()
                && ctx_words
                    .iter()
                    .any(|c| w.iter().filter(|x| c.contains(*x)).count() as f64 / w.len() as f64 >= min_overlap)
        })
        .count();
    covered as f64 / sents.len() as f64
}

fn content_words(text: &str, lang: &LangData) -> HashSet<String> {
    harpia_text::segment::alpha_words(text)
        .map(|s| s.slice(text).to_lowercase())
        .filter(|w| !lang.is_stopword(w))
        .collect()
}

/// Cosine similarity of two vectors (0 when either is zero).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{DecisionRule, Thresholds};
    use harpia_core::{Label, Scores};
    use harpia_text::lang::Lang;

    #[test]
    fn retrieval_metrics() {
        let e = RagExample {
            retrieved: vec!["x".into(), "a".into(), "b".into()],
            relevant: vec!["a".into(), "b".into()],
            ..Default::default()
        };
        let r = retrieval_report(&[e], 2);
        assert_eq!((r.recall, r.precision, r.mrr), (0.5, 0.5, 0.5));
        let ideal = 1.0 + 1.0 / 3f64.log2();
        assert!((r.ndcg - (1.0 / 3f64.log2()) / ideal).abs() < 1e-12);
        assert_eq!(recall_at_k(&["a", "a"], &["a", "b"], 2), 0.5, "duplicates count once");
        assert_eq!(precision_at_k::<&str>(&[], &["a"], 3), 0.0);
    }

    struct Overlap;
    impl PairClassifier for Overlap {
        fn labels(&self) -> &[Label] {
            &[]
        }
        fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Scores>> {
            Ok(pairs
                .iter()
                .map(|(p, h)| {
                    let hw: Vec<&str> = h.split_whitespace().collect();
                    let hit = hw.iter().filter(|w| p.contains(*w)).count() as f32 / hw.len().max(1) as f32;
                    Scores::new(vec![("entailment".into(), hit), ("contradiction".into(), 0.0)])
                })
                .collect())
        }
    }

    #[test]
    fn faithfulness_flags_unsupported_sentences() {
        let en = Lang::En.data();
        let rule = DecisionRule::nli("entailment", "contradiction", "neutral");
        let cal = Calibrated::fixed(rule.clone(), Thresholds::uniform(&rule, 0.8));
        let contexts = vec!["Unrelated text. Harpy eagles live in Brazil.".to_string()];
        let f =
            faithfulness(&Overlap, &cal, "entailment", "Harpy eagles live in Brazil. Penguins fly.", &contexts, &en)
                .unwrap();
        assert_eq!(f.score, 0.5);
        assert_eq!(f.sentences[0].label, "entailment");
        assert_eq!(f.sentences[0].context, Some(0));
        assert_eq!(f.sentences[0].evidence.unwrap().slice(&contexts[0]), "Harpy eagles live in Brazil.");
        assert_eq!(f.sentences[1].label, "neutral");
        assert!((context_recall("Harpy eagles live in Brazil.", &contexts, &en, 0.9) - 1.0).abs() < 1e-12);
        assert_eq!(context_recall("Penguins swim.", &contexts, &en, 0.5), 0.0);
    }
}
