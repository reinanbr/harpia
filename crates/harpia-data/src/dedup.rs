//! Exact and near-duplicate detection (MinHash + LSH), parallel with `rayon`.

use harpia_text::normalize::{collapse_whitespace, match_key};
use harpia_text::similarity::{MinHasher, minhash_similarity, shingle_set};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use xxhash_rust::xxh3::xxh3_64;

/// Hash of a text after case/accent folding and whitespace collapsing.
pub fn exact_key(text: &str) -> u64 {
    xxh3_64(match_key(&collapse_whitespace(text)).as_bytes())
}

/// Streaming exact deduplicator: [`is_new`](Self::is_new) returns `false` for repeats.
#[derive(Debug, Default, Clone)]
pub struct ExactDedup {
    seen: HashSet<u64>,
}

impl ExactDedup {
    /// Empty deduplicator.
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` the first time a (normalized) text is seen.
    pub fn is_new(&mut self, text: &str) -> bool {
        self.seen.insert(exact_key(text))
    }
}

/// MinHash-LSH parameters: `bands × rows` permutations; pairs with Jaccard above roughly
/// `(1/bands)^(1/rows)` become candidates, then are confirmed against `threshold`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LshParams {
    /// Number of bands.
    pub bands: usize,
    /// Rows per band.
    pub rows: usize,
    /// Word shingle size.
    pub shingle: usize,
    /// Estimated Jaccard similarity at or above which two texts are duplicates.
    pub threshold: f64,
    /// Hash seed.
    pub seed: u64,
}

impl Default for LshParams {
    fn default() -> Self {
        Self { bands: 20, rows: 6, shingle: 5, threshold: 0.8, seed: 1 }
    }
}

impl LshParams {
    /// Parameters tuned for a target `threshold`: picks bands/rows with 120 permutations so the
    /// LSH S-curve midpoint `(1/b)^(1/r)` is close to `threshold`.
    pub fn for_threshold(threshold: f64) -> Self {
        let mut best = (20, 6);
        let mut err = f64::MAX;
        for rows in 1..=24usize {
            if 120 % rows != 0 {
                continue;
            }
            let bands = 120 / rows;
            let mid = (1.0 / bands as f64).powf(1.0 / rows as f64);
            if (mid - threshold).abs() < err {
                err = (mid - threshold).abs();
                best = (bands, rows);
            }
        }
        Self { bands: best.0, rows: best.1, threshold, ..Self::default() }
    }
}

/// Near-duplicate deduplication over a collection: returns `keep[i] == false` for every text
/// that duplicates an earlier one (first occurrence wins).
///
/// ```
/// use harpia_data::dedup::{near_duplicates, LshParams};
/// let docs = [
///     "the harpy eagle lives in the tropical forests of central and south america",
///     "the harpy eagle lives in the tropical forests of central and south america!",
///     "completely different text about rust programming and compilers in general",
/// ];
/// let keep = near_duplicates(&docs, LshParams { shingle: 3, ..LshParams::default() });
/// assert_eq!(keep, [true, false, true]);
/// ```
pub fn near_duplicates<S: AsRef<str> + Sync>(texts: &[S], p: LshParams) -> Vec<bool> {
    let mh = MinHasher::new(p.bands * p.rows, p.seed);
    let sigs: Vec<Vec<u64>> = texts.par_iter().map(|t| mh.signature(&shingle_set(t.as_ref(), p.shingle))).collect();
    let mut buckets: Vec<HashMap<u64, Vec<usize>>> = vec![HashMap::new(); p.bands];
    let mut keep = vec![true; texts.len()];
    for (i, sig) in sigs.iter().enumerate() {
        if sig.iter().all(|&h| h == u64::MAX) {
            // too short to shingle: fall back to exact comparison via the exact key
            continue;
        }
        let mut dup = false;
        for (b, bucket) in buckets.iter_mut().enumerate() {
            let band = &sig[b * p.rows..(b + 1) * p.rows];
            let key = xxh3_64(&band.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>());
            let entry = bucket.entry(key).or_default();
            if !dup && entry.iter().any(|&j| keep[j] && minhash_similarity(sig, &sigs[j]) >= p.threshold) {
                dup = true;
            }
            entry.push(i);
        }
        if dup {
            keep[i] = false;
        }
    }
    // exact duplicates of texts too short to shingle
    let mut exact = ExactDedup::new();
    for (i, sig) in sigs.iter().enumerate() {
        if sig.iter().all(|&h| h == u64::MAX) && !exact.is_new(texts[i].as_ref()) {
            keep[i] = false;
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_normalizes() {
        let mut d = ExactDedup::new();
        assert!(d.is_new("Olá   Mundo"));
        assert!(!d.is_new("ola mundo"));
    }

    #[test]
    fn threshold_tuning() {
        let p = LshParams::for_threshold(0.8);
        assert_eq!(p.bands * p.rows, 120);
        let mid = (1.0 / p.bands as f64).powf(1.0 / p.rows as f64);
        assert!((mid - 0.8).abs() < 0.08, "{mid}");
    }

    #[test]
    fn short_texts_exact() {
        let keep = near_duplicates(&["oi", "Oi", "tchau"], LshParams::default());
        assert_eq!(keep, [true, false, true]);
    }
}
