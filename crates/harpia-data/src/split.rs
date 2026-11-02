//! Deterministic train/validation/test splits and sampling.

use crate::rng::SplitMix64;
use harpia_core::{HarpiaError, Result};
use std::collections::BTreeMap;
use xxhash_rust::xxh3::xxh3_64_with_seed;

/// Split ratios; must sum to 1 (within 1e-9).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratios {
    /// Train fraction.
    pub train: f64,
    /// Validation fraction.
    pub val: f64,
    /// Test fraction.
    pub test: f64,
}

impl Ratios {
    /// New ratios, validated.
    pub fn new(train: f64, val: f64, test: f64) -> Result<Self> {
        let r = Self { train, val, test };
        if [train, val, test].iter().any(|x| *x < 0.0) || ((train + val + test) - 1.0).abs() > 1e-9 {
            return Err(HarpiaError::invalid(format!(
                "split ratios must be non-negative and sum to 1, got {train}/{val}/{test}"
            )));
        }
        Ok(r)
    }
}

/// Indices of each split.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Split {
    /// Train indices.
    pub train: Vec<usize>,
    /// Validation indices.
    pub val: Vec<usize>,
    /// Test indices.
    pub test: Vec<usize>,
}

/// Shuffle `0..n` with `seed` and cut by `ratios`.
///
/// ```
/// use harpia_data::split::{random_split, Ratios};
/// let s = random_split(10, Ratios::new(0.8, 0.1, 0.1).unwrap(), 7);
/// assert_eq!((s.train.len(), s.val.len(), s.test.len()), (8, 1, 1));
/// assert_eq!(s, random_split(10, Ratios::new(0.8, 0.1, 0.1).unwrap(), 7));
/// ```
pub fn random_split(n: usize, ratios: Ratios, seed: u64) -> Split {
    let mut idx: Vec<usize> = (0..n).collect();
    SplitMix64::new(seed).shuffle(&mut idx);
    let n_train = (n as f64 * ratios.train).round() as usize;
    let n_val = ((n as f64 * ratios.val).round() as usize).min(n - n_train.min(n));
    let n_train = n_train.min(n);
    Split {
        train: idx[..n_train].to_vec(),
        val: idx[n_train..n_train + n_val].to_vec(),
        test: idx[n_train + n_val..].to_vec(),
    }
}

/// Assign by hashing a group key, so all records of a group land in the same split (and the
/// assignment is stable when the dataset grows).
pub fn group_split<S: AsRef<str>>(keys: &[S], ratios: Ratios, seed: u64) -> Split {
    let mut s = Split::default();
    for (i, k) in keys.iter().enumerate() {
        let u = (xxh3_64_with_seed(k.as_ref().as_bytes(), seed) >> 11) as f64 / (1u64 << 53) as f64;
        if u < ratios.train {
            s.train.push(i);
        } else if u < ratios.train + ratios.val {
            s.val.push(i);
        } else {
            s.test.push(i);
        }
    }
    s
}

/// Stratified split: [`random_split`] within each label, so label proportions are preserved.
pub fn stratified_split<S: AsRef<str>>(labels: &[S], ratios: Ratios, seed: u64) -> Split {
    let mut by: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, l) in labels.iter().enumerate() {
        by.entry(l.as_ref()).or_default().push(i);
    }
    let mut out = Split::default();
    for (j, (_, members)) in by.into_iter().enumerate() {
        let s = random_split(members.len(), ratios, seed.wrapping_add(j as u64));
        out.train.extend(s.train.iter().map(|&k| members[k]));
        out.val.extend(s.val.iter().map(|&k| members[k]));
        out.test.extend(s.test.iter().map(|&k| members[k]));
    }
    out.train.sort_unstable();
    out.val.sort_unstable();
    out.test.sort_unstable();
    out
}

/// Inverse-frequency class weights `n / (k · count)` in label order of first appearance.
pub fn class_weights<S: AsRef<str>>(labels: &[S]) -> Vec<(String, f64)> {
    let mut order: Vec<String> = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for l in labels {
        let l = l.as_ref().to_string();
        if !counts.contains_key(&l) {
            order.push(l.clone());
        }
        *counts.entry(l).or_default() += 1;
    }
    let k = order.len() as f64;
    let n = labels.len() as f64;
    order
        .into_iter()
        .map(|l| {
            let c = counts[&l] as f64;
            (l, n / (k * c))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_split_keeps_groups_together() {
        let keys = ["a", "b", "a", "c", "b", "a"];
        let s = group_split(&keys, Ratios::new(0.5, 0.25, 0.25).unwrap(), 3);
        for part in [&s.train, &s.val, &s.test] {
            for &i in part.iter() {
                for (j, k) in keys.iter().enumerate() {
                    if *k == keys[i] {
                        assert!(part.contains(&j));
                    }
                }
            }
        }
    }

    #[test]
    fn stratified_preserves_proportions() {
        let labels: Vec<&str> = (0..100).map(|i| if i < 80 { "neg" } else { "pos" }).collect();
        let s = stratified_split(&labels, Ratios::new(0.5, 0.25, 0.25).unwrap(), 1);
        let pos_train = s.train.iter().filter(|&&i| labels[i] == "pos").count();
        assert_eq!(pos_train, 10);
        assert_eq!(s.train.len() + s.val.len() + s.test.len(), 100);
        let w = class_weights(&labels);
        assert!((w[0].1 - 100.0 / (2.0 * 80.0)).abs() < 1e-12);
        assert!(Ratios::new(0.5, 0.5, 0.5).is_err());
    }
}
