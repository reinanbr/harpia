//! Threshold calibration for classifiers, with decision rules as data.
//!
//! The workflow is deliberately one-way: thresholds are tuned with [`grid_search`] on a
//! [`ValidationSet`] and come back as a [`Calibrated`] value; a [`TestSet`] can only be passed to
//! [`Calibrated::evaluate`], which applies the thresholds **frozen**. There is no API that tunes on
//! a `TestSet`, and evaluating on the very data used for tuning is refused (content hashes are
//! compared), so "tuning on test" has to be done on purpose.
//!
//! [`best_of_windows`] scores a hypothesis against every sentence and every 2-sentence window of
//! a passage and keeps, per label, the best window as evidence (the NLI pattern used for claim
//! verification).

use crate::classification::ClassificationReport;
use harpia_core::traits::PairClassifier;
use harpia_core::{HarpiaError, Result, Scores, Span};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// One step of a decision rule: pick `label` when its score reaches the label's threshold and is
/// at least the score of every label in `must_beat`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleStep {
    /// Label decided by this step.
    pub label: String,
    /// Labels whose scores this label must match or exceed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub must_beat: Vec<String>,
}

/// An ordered decision rule over label scores, with a fallback label.
///
/// ```
/// use harpia_eval::calibration::{DecisionRule, Thresholds};
/// use harpia_core::Scores;
/// // "contradicts" wins only above its threshold and when it beats "supports".
/// let rule: DecisionRule = serde_json::from_str(r#"{
///   "steps": [{"label": "contradicts", "must_beat": ["supports"]}, {"label": "supports"}],
///   "fallback": "neutral"}"#).unwrap();
/// let t = Thresholds::uniform(&rule, 0.5);
/// let s = Scores::new(vec![("supports".into(), 0.6), ("contradicts".into(), 0.55), ("neutral".into(), 0.1)]);
/// assert_eq!(rule.decide(&t, &s), "supports"); // 0.55 passes 0.5 but does not beat 0.6
/// let s = Scores::new(vec![("supports".into(), 0.3), ("contradicts".into(), 0.55)]);
/// assert_eq!(rule.decide(&t, &s), "contradicts");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRule {
    /// Steps, tried in order.
    pub steps: Vec<RuleStep>,
    /// Label when no step fires.
    pub fallback: String,
}

impl DecisionRule {
    /// The usual NLI rule: `contradiction` (must beat `entailment`), then `entailment`, else
    /// `neutral`, with the given label names.
    pub fn nli(entailment: &str, contradiction: &str, neutral: &str) -> Self {
        Self {
            steps: vec![
                RuleStep { label: contradiction.into(), must_beat: vec![entailment.into()] },
                RuleStep { label: entailment.into(), must_beat: Vec::new() },
            ],
            fallback: neutral.into(),
        }
    }

    /// Plain argmax-with-thresholds: each label must reach its threshold and beat all others.
    pub fn argmax(labels: &[&str], fallback: &str) -> Self {
        Self {
            steps: labels
                .iter()
                .map(|l| RuleStep {
                    label: l.to_string(),
                    must_beat: labels.iter().filter(|o| *o != l).map(|o| o.to_string()).collect(),
                })
                .collect(),
            fallback: fallback.into(),
        }
    }

    /// Labels that carry a threshold (one per step).
    pub fn thresholded_labels(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.label.as_str()).collect()
    }

    /// Every label the rule can output.
    pub fn output_labels(&self) -> Vec<&str> {
        let mut v = self.thresholded_labels();
        if !v.contains(&self.fallback.as_str()) {
            v.push(&self.fallback);
        }
        v
    }

    /// Apply the rule. Missing scores count as 0.
    pub fn decide<'a>(&'a self, thresholds: &Thresholds, scores: &Scores) -> &'a str {
        let get = |l: &str| scores.get(l).unwrap_or(0.0);
        for step in &self.steps {
            let s = get(&step.label);
            let t = thresholds.0.get(&step.label).copied().unwrap_or(0.5);
            if s >= t && step.must_beat.iter().all(|o| s >= get(o)) {
                return &step.label;
            }
        }
        &self.fallback
    }
}

/// Per-label thresholds.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Thresholds(pub BTreeMap<String, f32>);

impl Thresholds {
    /// The same threshold for every thresholded label of `rule`.
    pub fn uniform(rule: &DecisionRule, t: f32) -> Self {
        Self(rule.thresholded_labels().into_iter().map(|l| (l.to_string(), t)).collect())
    }
}

/// Examples with classifier scores and gold labels.
#[derive(Debug, Clone, PartialEq, Default)]
struct Labeled {
    scores: Vec<Scores>,
    gold: Vec<String>,
}

impl Labeled {
    fn hash(&self) -> String {
        let mut h = xxhash_rust::xxh3::Xxh3::new();
        for (s, g) in self.scores.iter().zip(&self.gold) {
            for (l, p) in s.iter() {
                h.update(l.as_bytes());
                h.update(&p.to_le_bytes());
            }
            h.update(g.as_bytes());
            h.update(&[0xff]);
        }
        format!("{:016x}", h.digest())
    }
}

/// Validation data: the only input accepted by [`grid_search`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValidationSet(Labeled);

/// Test data: only accepted by [`Calibrated::evaluate`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TestSet(Labeled);

macro_rules! labeled_set {
    ($t:ident) => {
        impl $t {
            /// From parallel vectors of scores and gold labels.
            pub fn new(scores: Vec<Scores>, gold: Vec<String>) -> Result<Self> {
                if scores.len() != gold.len() {
                    return Err(HarpiaError::invalid(format!(
                        "{} scores but {} gold labels",
                        scores.len(),
                        gold.len()
                    )));
                }
                Ok(Self(Labeled { scores, gold }))
            }
            /// Number of examples.
            pub fn len(&self) -> usize {
                self.0.gold.len()
            }
            /// `true` when empty.
            pub fn is_empty(&self) -> bool {
                self.0.gold.is_empty()
            }
        }
    };
}
labeled_set!(ValidationSet);
labeled_set!(TestSet);

/// What [`grid_search`] maximizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Macro F1 over the rule's output labels.
    #[default]
    MacroF1,
    /// Accuracy.
    Accuracy,
    /// Support-weighted F1.
    WeightedF1,
}

impl Objective {
    fn value(self, r: &ClassificationReport) -> f64 {
        match self {
            Self::MacroF1 => r.macro_f1,
            Self::Accuracy => r.accuracy,
            Self::WeightedF1 => r.weighted_f1,
        }
    }
}

/// Thresholds fitted on validation data, ready to be applied frozen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibrated {
    /// The decision rule.
    pub rule: DecisionRule,
    /// Fitted thresholds.
    pub thresholds: Thresholds,
    /// Objective that was maximized.
    pub objective: Objective,
    /// Objective value on the validation set.
    pub validation_score: f64,
    /// Validation examples used.
    pub validation_n: usize,
    /// Content hash of the validation set (used to refuse evaluating on it as "test").
    pub validation_hash: String,
}

impl Calibrated {
    /// Thresholds chosen by hand (no fitting), e.g. a model's shipped defaults.
    pub fn fixed(rule: DecisionRule, thresholds: Thresholds) -> Self {
        Self {
            rule,
            thresholds,
            objective: Objective::MacroF1,
            validation_score: f64::NAN,
            validation_n: 0,
            validation_hash: String::new(),
        }
    }

    /// Decide one example.
    pub fn decide(&self, scores: &Scores) -> &str {
        self.rule.decide(&self.thresholds, scores)
    }

    /// Apply the frozen thresholds to the test set and report.
    pub fn evaluate(&self, test: &TestSet) -> Result<ClassificationReport> {
        if !self.validation_hash.is_empty() && test.0.hash() == self.validation_hash {
            return Err(HarpiaError::invalid("the test set is identical to the validation set used for calibration"));
        }
        let pred: Vec<&str> = test.0.scores.iter().map(|s| self.decide(s)).collect();
        let gold: Vec<&str> = test.0.gold.iter().map(String::as_str).collect();
        Ok(ClassificationReport::new(&gold, &pred, &self.rule.output_labels()))
    }

    /// Save as pretty JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, serde_json::to_vec_pretty(self)?).map_err(|e| HarpiaError::io(path, e))
    }

    /// Load from JSON.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Ok(serde_json::from_slice(&std::fs::read(path).map_err(|e| HarpiaError::io(path, e))?)?)
    }
}

/// Thresholds `0.05, 0.10, ..., 0.95`.
pub fn default_grid() -> Vec<f32> {
    (1..20).map(|i| i as f32 * 0.05).collect()
}

/// Exhaustive search over `grid` for every thresholded label of `rule` (the cartesian product,
/// capped at one million combinations). Ties keep the first combination in grid order, so results
/// are deterministic.
pub fn grid_search(rule: &DecisionRule, val: &ValidationSet, grid: &[f32], objective: Objective) -> Result<Calibrated> {
    let labels: Vec<String> = rule.thresholded_labels().into_iter().map(String::from).collect();
    if grid.is_empty() || val.is_empty() {
        return Err(HarpiaError::invalid("grid search needs a non-empty grid and validation set"));
    }
    let combos = grid.len().checked_pow(labels.len() as u32).filter(|&c| c <= 1_000_000);
    let Some(combos) = combos else {
        return Err(HarpiaError::invalid("too many threshold combinations; use a coarser grid"));
    };
    let gold: Vec<&str> = val.0.gold.iter().map(String::as_str).collect();
    let outputs = rule.output_labels();
    let mut best: Option<(f64, Thresholds)> = None;
    for c in 0..combos {
        let mut rest = c;
        let mut t = Thresholds::default();
        for l in &labels {
            t.0.insert(l.clone(), grid[rest % grid.len()]);
            rest /= grid.len();
        }
        let pred: Vec<&str> = val.0.scores.iter().map(|s| rule.decide(&t, s)).collect();
        let v = objective.value(&ClassificationReport::new(&gold, &pred, &outputs));
        if best.as_ref().is_none_or(|(b, _)| v > *b) {
            best = Some((v, t));
        }
    }
    let (validation_score, thresholds) = best.expect("at least one combination");
    Ok(Calibrated {
        rule: rule.clone(),
        thresholds,
        objective,
        validation_score,
        validation_n: val.len(),
        validation_hash: val.0.hash(),
    })
}

/// Best window per label, from [`best_of_windows`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowScores {
    /// Per label, the maximum probability over all windows.
    pub scores: Scores,
    /// Per label, the byte span (in the passage) of the window that produced the maximum.
    pub evidence: BTreeMap<String, Span>,
    /// Number of windows scored.
    pub windows: usize,
}

/// Single sentences plus every pair of consecutive sentences, as byte spans of `text`.
pub fn sentence_windows(sentences: &[Span]) -> Vec<Span> {
    let mut w: Vec<Span> = sentences.to_vec();
    w.extend(sentences.windows(2).map(|p| Span::new(p[0].start, p[1].end)));
    w
}

/// Score `hypothesis` against each window of `passage` (premise = window) and keep the best
/// window per label. `sentences` are byte spans of `passage` (e.g. from
/// `harpia_text::segment::sentences`). Returns `None` when there are no sentences.
pub fn best_of_windows(
    classifier: &dyn PairClassifier,
    hypothesis: &str,
    passage: &str,
    sentences: &[Span],
) -> Result<Option<WindowScores>> {
    let windows = sentence_windows(sentences);
    if windows.is_empty() {
        return Ok(None);
    }
    let pairs: Vec<(&str, &str)> = windows.iter().map(|w| (w.slice(passage), hypothesis)).collect();
    let scores = classifier.classify_pairs(&pairs)?;
    let mut best: Vec<(String, f32, Span)> = Vec::new();
    for (w, s) in windows.iter().zip(&scores) {
        for (l, p) in s.iter() {
            match best.iter_mut().find(|b| b.0 == l) {
                Some(b) if p > b.1 => {
                    b.1 = p;
                    b.2 = *w;
                }
                Some(_) => {}
                None => best.push((l.to_string(), p, *w)),
            }
        }
    }
    Ok(Some(WindowScores {
        scores: Scores::new(best.iter().map(|b| (b.0.clone(), b.1)).collect()),
        evidence: best.into_iter().map(|b| (b.0, b.2)).collect(),
        windows: windows.len(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use harpia_core::Label;

    fn sc(s: f32, c: f32) -> Scores {
        Scores::new(vec![("s".into(), s), ("c".into(), c), ("n".into(), (1.0 - s - c).max(0.0))])
    }

    #[test]
    fn errei_rule_semantics() {
        let rule = DecisionRule::nli("s", "c", "n");
        let t = Thresholds(BTreeMap::from([("s".into(), 0.6), ("c".into(), 0.4)]));
        assert_eq!(rule.decide(&t, &sc(0.3, 0.45)), "c");
        // c reaches its threshold but does not beat s, and s is below its own threshold.
        assert_eq!(rule.decide(&t, &sc(0.5, 0.45)), "n");
        assert_eq!(rule.decide(&t, &sc(0.7, 0.1)), "s");
    }

    #[test]
    fn grid_search_then_frozen_test() {
        // Gold "s" when s > 0.7, "c" when c > 0.3, else "n": best thresholds sit near those.
        let mut vs = Vec::new();
        let mut vg = Vec::new();
        for i in 0..40 {
            let s = (i as f32 + 0.5) / 40.0; // never exactly on a grid point
            let c = if i % 3 == 0 { 0.35 + (i as f32) / 200.0 } else { 0.1 };
            let gold = if c > 0.3 && c >= s {
                "c"
            } else if s > 0.7 {
                "s"
            } else {
                "n"
            };
            vs.push(sc(s, c));
            vg.push(gold.to_string());
        }
        let val = ValidationSet::new(vs.clone(), vg.clone()).unwrap();
        let rule = DecisionRule::nli("s", "c", "n");
        let cal = grid_search(&rule, &val, &default_grid(), Objective::MacroF1).unwrap();
        assert_eq!(cal.validation_score, 1.0, "{cal:?}");
        // Same content as test is refused; different data is evaluated with frozen thresholds.
        assert!(cal.evaluate(&TestSet::new(vs.clone(), vg.clone()).unwrap()).is_err());
        let test = TestSet::new(vs[..20].to_vec(), vg[..20].to_vec()).unwrap();
        let report = cal.evaluate(&test).unwrap();
        assert_eq!(report.accuracy, 1.0);
        // Round trip through JSON keeps the decision.
        let dir = tempfile::tempdir().unwrap();
        cal.save(dir.path().join("t.json")).unwrap();
        assert_eq!(Calibrated::load(dir.path().join("t.json")).unwrap(), cal);
    }

    struct Keyword;
    impl PairClassifier for Keyword {
        fn labels(&self) -> &[Label] {
            &[]
        }
        fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Scores>> {
            Ok(pairs
                .iter()
                .map(|(p, _)| {
                    let s = if p.contains("eagle") { 0.9 } else { 0.1 };
                    let c = if p.contains("not") { 0.8 } else { 0.05 };
                    Scores::new(vec![("s".into(), s), ("c".into(), c)])
                })
                .collect())
        }
    }

    #[test]
    fn windows_keep_best_evidence() {
        let text = "Owls hunt at night. The harpy eagle is big. It is not small.";
        let sents = [Span::new(0, 19), Span::new(20, 43), Span::new(44, 60)];
        assert_eq!(sentence_windows(&sents).len(), 5);
        let w = best_of_windows(&Keyword, "harpies are large", text, &sents).unwrap().unwrap();
        assert_eq!(w.windows, 5);
        assert_eq!(w.scores.get("s"), Some(0.9));
        assert_eq!(w.evidence["s"].slice(text), "The harpy eagle is big.");
        assert_eq!(w.evidence["c"].slice(text), "It is not small.");
        assert!(best_of_windows(&Keyword, "x", "", &[]).unwrap().is_none());
    }
}
