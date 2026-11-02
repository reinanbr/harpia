//! Benchmark decontamination: find training texts that share long word n-grams with
//! evaluation sets.

use harpia_text::similarity::{shingles, tokens};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A contamination hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contamination {
    /// Benchmark name.
    pub benchmark: String,
    /// Index of the benchmark item.
    pub item: usize,
    /// Fraction of the training text's n-grams found in that item.
    pub overlap: f64,
    /// Number of shared n-grams.
    pub shared: usize,
}

/// Index of n-grams of one or more evaluation sets.
#[derive(Debug, Clone)]
pub struct Decontaminator {
    n: usize,
    benchmarks: Vec<String>,
    index: HashMap<u64, Vec<(u32, u32)>>,
}

impl Decontaminator {
    /// New index with word n-gram size `n` (13 is a common choice for long texts; use 8 for
    /// short QA items).
    pub fn new(n: usize) -> Self {
        Self { n: n.max(1), benchmarks: Vec::new(), index: HashMap::new() }
    }

    /// Add the items of a benchmark.
    pub fn add_benchmark<I, S>(&mut self, name: &str, items: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let b = self.benchmarks.len() as u32;
        self.benchmarks.push(name.to_string());
        for (i, text) in items.into_iter().enumerate() {
            let mut sh = shingles(&tokens(text.as_ref()), self.n);
            sh.sort_unstable();
            sh.dedup();
            for h in sh {
                self.index.entry(h).or_default().push((b, i as u32));
            }
        }
    }

    /// The most overlapping benchmark item for `text`, if at least `min_shared` n-grams are
    /// shared.
    pub fn check(&self, text: &str, min_shared: usize) -> Option<Contamination> {
        let mut sh = shingles(&tokens(text), self.n);
        sh.sort_unstable();
        sh.dedup();
        if sh.is_empty() {
            return None;
        }
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
        for h in &sh {
            if let Some(posts) = self.index.get(h) {
                for &p in posts {
                    *counts.entry(p).or_default() += 1;
                }
            }
        }
        let ((b, i), shared) = counts.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))?;
        (shared >= min_shared.max(1)).then(|| Contamination {
            benchmark: self.benchmarks[b as usize].clone(),
            item: i as usize,
            overlap: shared as f64 / sh.len() as f64,
            shared,
        })
    }

    /// [`check`](Self::check) many texts in parallel.
    pub fn check_all<S: AsRef<str> + Sync>(&self, texts: &[S], min_shared: usize) -> Vec<Option<Contamination>> {
        texts.par_iter().map(|t| self.check(t.as_ref(), min_shared)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_leaked_item() {
        let mut d = Decontaminator::new(5);
        d.add_benchmark(
            "toyqa",
            [
                "What is the capital of Brazil? The capital of Brazil is Brasília.",
                "Unrelated question about cats and dogs here.",
            ],
        );
        let train = "Fun fact: the capital of Brazil is Brasília, built in 1960.";
        let hit = d.check(train, 1).unwrap();
        assert_eq!((hit.benchmark.as_str(), hit.item), ("toyqa", 0));
        assert!(d.check("nothing in common with any benchmark at all", 1).is_none());
    }
}
