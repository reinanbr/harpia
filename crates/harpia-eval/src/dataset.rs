//! Dataset evaluation: contamination against benchmark sets (shared with
//! `harpia_data::decontam`), label noise and duplicate statistics.

use harpia_data::decontam::{Contamination, Decontaminator};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Contamination of a dataset against named benchmarks.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ContaminationReport {
    /// Rows checked.
    pub rows: usize,
    /// Rows sharing at least `min_shared` n-grams with some benchmark item.
    pub contaminated: usize,
    /// Contaminated rows per benchmark.
    pub per_benchmark: BTreeMap<String, usize>,
    /// Row indices and what they matched.
    pub matches: Vec<(usize, Contamination)>,
}

/// Check `texts` against benchmark sets (`name → items`) with word `n`-grams.
pub fn contamination<S: AsRef<str> + Sync>(
    texts: &[S],
    benchmarks: &[(&str, Vec<String>)],
    n: usize,
    min_shared: usize,
) -> ContaminationReport {
    let mut d = Decontaminator::new(n);
    for (name, items) in benchmarks {
        d.add_benchmark(name, items.iter());
    }
    let mut r = ContaminationReport { rows: texts.len(), ..Default::default() };
    for (i, c) in d.check_all(texts, min_shared).into_iter().enumerate() {
        if let Some(c) = c {
            r.contaminated += 1;
            *r.per_benchmark.entry(c.benchmark.clone()).or_insert(0) += 1;
            r.matches.push((i, c));
        }
    }
    r
}

/// Label-noise and duplicate statistics of a labeled dataset.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LabelReport {
    /// Rows.
    pub rows: usize,
    /// Label → count.
    pub label_counts: BTreeMap<String, usize>,
    /// Rows whose normalized text appeared earlier.
    pub duplicate_rows: usize,
    /// Distinct normalized texts that appear with more than one label.
    pub conflicting_texts: usize,
    /// Rows involved in a label conflict.
    pub conflicting_rows: usize,
    /// Up to 20 examples: row indices sharing a text but disagreeing on the label.
    pub conflict_examples: Vec<Vec<usize>>,
    /// Majority-class share (a trivial baseline accuracy).
    pub majority_share: f64,
}

fn norm(text: &str) -> String {
    harpia_text::normalize::collapse_whitespace(&harpia_text::normalize::fold_case(text)).trim().to_string()
}

/// Duplicates and label conflicts (same text after case folding and whitespace collapse, but
/// different labels: a strong sign of annotation noise).
pub fn label_report<S: AsRef<str>, L: AsRef<str>>(rows: &[(S, L)]) -> LabelReport {
    let mut r = LabelReport { rows: rows.len(), ..Default::default() };
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    let mut order = Vec::new();
    for (i, (t, l)) in rows.iter().enumerate() {
        *r.label_counts.entry(l.as_ref().to_string()).or_insert(0) += 1;
        let k = norm(t.as_ref());
        let g = groups.entry(k.clone()).or_default();
        if g.is_empty() {
            order.push(k);
        } else {
            r.duplicate_rows += 1;
        }
        g.push(i);
    }
    for k in &order {
        let g = &groups[k];
        let first = rows[g[0]].1.as_ref();
        if g.iter().any(|&i| rows[i].1.as_ref() != first) {
            r.conflicting_texts += 1;
            r.conflicting_rows += g.len();
            if r.conflict_examples.len() < 20 {
                r.conflict_examples.push(g.clone());
            }
        }
    }
    r.majority_share = r.label_counts.values().max().map_or(0.0, |&m| m as f64 / rows.len().max(1) as f64);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_noise() {
        let rows = [("Great movie", "pos"), ("great  MOVIE", "neg"), ("bad", "neg"), ("bad", "neg"), ("ok", "pos")];
        let r = label_report(&rows);
        assert_eq!((r.duplicate_rows, r.conflicting_texts, r.conflicting_rows), (2, 1, 2));
        assert_eq!(r.conflict_examples, vec![vec![0, 1]]);
        assert_eq!(r.majority_share, 0.6);
    }

    #[test]
    fn contamination_counts() {
        let bench = vec!["the quick brown fox jumps over the lazy dog".to_string()];
        let texts = ["intro: the quick brown fox jumps over the lazy dog today", "nothing in common here at all"];
        let r = contamination(&texts, &[("toy", bench)], 5, 1);
        assert_eq!((r.rows, r.contaminated), (2, 1));
        assert_eq!(r.per_benchmark["toy"], 1);
        assert_eq!(r.matches[0].0, 0);
    }
}
