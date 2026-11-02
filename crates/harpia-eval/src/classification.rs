//! Classification metrics: accuracy, per-class precision/recall/F1/support, macro and weighted
//! F1, and the confusion matrix.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Per-class figures.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ClassStats {
    /// Precision.
    pub precision: f64,
    /// Recall.
    pub recall: f64,
    /// F1.
    pub f1: f64,
    /// Number of gold examples of this class.
    pub support: usize,
}

/// Full classification report.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ClassificationReport {
    /// Labels in matrix order.
    pub labels: Vec<String>,
    /// Number of examples.
    pub n: usize,
    /// Accuracy.
    pub accuracy: f64,
    /// Per-class figures, keyed by label.
    pub per_class: BTreeMap<String, ClassStats>,
    /// Unweighted mean F1 over labels.
    pub macro_f1: f64,
    /// Support-weighted mean F1.
    pub weighted_f1: f64,
    /// `confusion[gold][predicted]`, indices into `labels`.
    pub confusion: Vec<Vec<usize>>,
}

impl ClassificationReport {
    /// Build a report. `labels` fixes the order (and includes classes that never occur); labels
    /// seen in the data but missing from `labels` are appended in sorted order.
    ///
    /// ```
    /// use harpia_eval::classification::ClassificationReport;
    /// let r = ClassificationReport::new(&["a", "b", "a", "b"], &["a", "b", "b", "b"], &["a", "b"]);
    /// assert_eq!(r.accuracy, 0.75);
    /// assert_eq!(r.confusion, vec![vec![1, 1], vec![0, 2]]);
    /// ```
    pub fn new<S: AsRef<str>>(gold: &[S], predicted: &[S], labels: &[&str]) -> Self {
        assert_eq!(gold.len(), predicted.len(), "gold and predicted lengths differ");
        let mut order: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
        let mut extra: Vec<String> =
            gold.iter().chain(predicted).map(|s| s.as_ref().to_string()).filter(|l| !order.contains(l)).collect();
        extra.sort();
        extra.dedup();
        order.extend(extra);
        let idx = |l: &str| order.iter().position(|x| x == l).expect("label indexed");
        let k = order.len();
        let mut confusion = vec![vec![0usize; k]; k];
        for (g, p) in gold.iter().zip(predicted) {
            confusion[idx(g.as_ref())][idx(p.as_ref())] += 1;
        }
        let n = gold.len();
        let correct: usize = (0..k).map(|i| confusion[i][i]).sum();
        let mut per_class = BTreeMap::new();
        let (mut f1_sum, mut weighted) = (0.0, 0.0);
        for (i, label) in order.iter().enumerate() {
            let tp = confusion[i][i];
            let support: usize = confusion[i].iter().sum();
            let predicted_n: usize = (0..k).map(|g| confusion[g][i]).sum();
            let precision = if predicted_n == 0 { 0.0 } else { tp as f64 / predicted_n as f64 };
            let recall = if support == 0 { 0.0 } else { tp as f64 / support as f64 };
            let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
            f1_sum += f1;
            weighted += f1 * support as f64;
            per_class.insert(label.clone(), ClassStats { precision, recall, f1, support });
        }
        Self {
            labels: order,
            n,
            accuracy: if n == 0 { 0.0 } else { correct as f64 / n as f64 },
            per_class,
            macro_f1: if k == 0 { 0.0 } else { f1_sum / k as f64 },
            weighted_f1: if n == 0 { 0.0 } else { weighted / n as f64 },
            confusion,
        }
    }

    /// Plain-text table (labels, precision, recall, F1, support, then the summary rows).
    pub fn to_text(&self) -> String {
        let w = self.labels.iter().map(String::len).max().unwrap_or(5).max(12);
        let mut s = format!("{:<w$} {:>9} {:>9} {:>9} {:>9}\n", "", "precision", "recall", "f1", "support");
        for l in &self.labels {
            let c = &self.per_class[l];
            s += &format!("{l:<w$} {:>9.4} {:>9.4} {:>9.4} {:>9}\n", c.precision, c.recall, c.f1, c.support);
        }
        s += &format!("{:<w$} {:>29.4} {:>9}\n", "accuracy", self.accuracy, self.n);
        s += &format!("{:<w$} {:>29.4} {:>9}\n", "macro f1", self.macro_f1, self.n);
        s += &format!("{:<w$} {:>29.4} {:>9}\n", "weighted f1", self.weighted_f1, self.n);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_matches_hand_computation() {
        let gold = ["s", "s", "c", "n", "n", "n"];
        let pred = ["s", "n", "c", "n", "s", "n"];
        let r = ClassificationReport::new(&gold, &pred, &["s", "c", "n"]);
        assert_eq!(r.n, 6);
        assert!((r.accuracy - 4.0 / 6.0).abs() < 1e-12);
        let s = r.per_class["s"];
        assert_eq!((s.precision, s.recall, s.support), (0.5, 0.5, 2));
        let n = r.per_class["n"];
        assert!((n.precision - 2.0 / 3.0).abs() < 1e-12 && (n.recall - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(r.per_class["c"].f1, 1.0);
        let macro_f1 = (0.5 + 1.0 + 2.0 / 3.0) / 3.0;
        assert!((r.macro_f1 - macro_f1).abs() < 1e-12);
        let weighted = (0.5 * 2.0 + 1.0 + 2.0 / 3.0 * 3.0) / 6.0;
        assert!((r.weighted_f1 - weighted).abs() < 1e-12);
        assert!(r.to_text().contains("macro f1"));
    }

    #[test]
    fn unseen_labels_are_appended() {
        let r = ClassificationReport::new(&["x"], &["y"], &[]);
        assert_eq!(r.labels, ["x", "y"]);
        assert_eq!(r.accuracy, 0.0);
    }
}
