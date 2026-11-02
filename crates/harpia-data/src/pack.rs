//! Sequence packing for training: many short tokenized samples into fixed-length rows, with
//! document boundaries so attention never crosses samples.

use harpia_core::{HarpiaError, Result};
use serde::{Deserialize, Serialize};

/// A tokenized sample: ids and per-token loss mask (`true` = train on this token).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TokenizedSample {
    /// Token ids.
    pub ids: Vec<u32>,
    /// Loss mask, same length as `ids`.
    pub loss_mask: Vec<bool>,
}

/// One packed row.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PackedSequence {
    /// Token ids (padded with `pad_id` to `seq_len` unless padding-free).
    pub ids: Vec<u32>,
    /// Loss mask (padding is `false`).
    pub loss_mask: Vec<bool>,
    /// Position ids restarting at 0 for each sample.
    pub position_ids: Vec<u32>,
    /// Cumulative sequence boundaries (`[0, len1, len1+len2, ...]`), as used by
    /// variable-length attention kernels.
    pub cu_seqlens: Vec<u32>,
}

/// Packing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PackStrategy {
    /// Fill rows in input order.
    #[default]
    Greedy,
    /// Best-fit decreasing: fewer rows and less padding, order not preserved.
    BestFitDecreasing,
}

/// What to do with samples longer than the row length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overflow {
    /// Keep the first `seq_len` tokens.
    #[default]
    Truncate,
    /// Skip the sample.
    Drop,
    /// Return an error.
    Error,
}

/// Packing options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackOptions {
    /// Row length in tokens.
    pub seq_len: usize,
    /// Padding token id.
    pub pad_id: u32,
    /// Strategy.
    pub strategy: PackStrategy,
    /// Overflow policy.
    pub overflow: Overflow,
    /// Do not pad rows (variable-length output).
    pub padding_free: bool,
}

impl PackOptions {
    /// Options for rows of `seq_len` tokens padded with `pad_id`.
    pub fn new(seq_len: usize, pad_id: u32) -> Self {
        Self { seq_len, pad_id, strategy: PackStrategy::Greedy, overflow: Overflow::Truncate, padding_free: false }
    }
}

/// Pack `samples` into rows.
///
/// ```
/// use harpia_data::pack::{pack, PackOptions, TokenizedSample};
/// let s = |n: u32| TokenizedSample { ids: (0..n).collect(), loss_mask: vec![true; n as usize] };
/// let rows = pack(vec![s(3), s(2), s(4)], PackOptions::new(5, 0)).unwrap();
/// assert_eq!(rows.len(), 2);
/// assert_eq!(rows[0].cu_seqlens, [0, 3, 5]);
/// assert_eq!(rows[1].ids, [0, 1, 2, 3, 0]);
/// assert_eq!(rows[1].loss_mask, [true, true, true, true, false]);
/// ```
pub fn pack(samples: Vec<TokenizedSample>, opts: PackOptions) -> Result<Vec<PackedSequence>> {
    if opts.seq_len == 0 {
        return Err(HarpiaError::invalid("seq_len must be > 0"));
    }
    let mut items = Vec::with_capacity(samples.len());
    for (i, mut s) in samples.into_iter().enumerate() {
        if s.ids.len() != s.loss_mask.len() {
            return Err(HarpiaError::invalid(format!("sample {i}: ids and loss_mask lengths differ")));
        }
        if s.ids.is_empty() {
            continue;
        }
        if s.ids.len() > opts.seq_len {
            match opts.overflow {
                Overflow::Truncate => {
                    s.ids.truncate(opts.seq_len);
                    s.loss_mask.truncate(opts.seq_len);
                }
                Overflow::Drop => continue,
                Overflow::Error => {
                    return Err(HarpiaError::invalid(format!(
                        "sample {i} has {} tokens > seq_len {}",
                        s.ids.len(),
                        opts.seq_len
                    )));
                }
            }
        }
        items.push(s);
    }
    let bins: Vec<Vec<usize>> = match opts.strategy {
        PackStrategy::Greedy => {
            let mut bins: Vec<Vec<usize>> = Vec::new();
            let mut used = 0;
            for (i, s) in items.iter().enumerate() {
                if bins.is_empty() || used + s.ids.len() > opts.seq_len {
                    bins.push(Vec::new());
                    used = 0;
                }
                bins.last_mut().expect("bin").push(i);
                used += s.ids.len();
            }
            bins
        }
        PackStrategy::BestFitDecreasing => {
            let mut order: Vec<usize> = (0..items.len()).collect();
            order.sort_by(|&a, &b| items[b].ids.len().cmp(&items[a].ids.len()).then(a.cmp(&b)));
            let mut bins: Vec<(usize, Vec<usize>)> = Vec::new();
            for i in order {
                let len = items[i].ids.len();
                let best = bins
                    .iter_mut()
                    .filter(|(used, _)| used + len <= opts.seq_len)
                    .min_by_key(|(used, _)| opts.seq_len - *used - len);
                match best {
                    Some((used, members)) => {
                        *used += len;
                        members.push(i);
                    }
                    None => bins.push((len, vec![i])),
                }
            }
            bins.into_iter().map(|(_, m)| m).collect()
        }
    };
    Ok(bins
        .into_iter()
        .map(|members| {
            let mut row = PackedSequence { cu_seqlens: vec![0], ..Default::default() };
            for i in members {
                let s = &items[i];
                row.ids.extend_from_slice(&s.ids);
                row.loss_mask.extend_from_slice(&s.loss_mask);
                row.position_ids.extend(0..s.ids.len() as u32);
                row.cu_seqlens.push(row.ids.len() as u32);
            }
            if !opts.padding_free {
                let pad = opts.seq_len - row.ids.len();
                row.ids.extend(std::iter::repeat_n(opts.pad_id, pad));
                row.loss_mask.extend(std::iter::repeat_n(false, pad));
                row.position_ids.extend(std::iter::repeat_n(0, pad));
            }
            row
        })
        .collect())
}

/// Fraction of real (non-padding) tokens in packed rows.
pub fn packing_efficiency(rows: &[PackedSequence]) -> f64 {
    let total: usize = rows.iter().map(|r| r.ids.len()).sum();
    let real: usize = rows.iter().map(|r| *r.cu_seqlens.last().unwrap_or(&0) as usize).sum();
    if total == 0 { 0.0 } else { real as f64 / total as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(n: usize) -> TokenizedSample {
        TokenizedSample { ids: vec![1; n], loss_mask: vec![true; n] }
    }

    #[test]
    fn bfd_uses_fewer_rows() {
        let input = vec![s(6), s(5), s(4), s(3), s(2)];
        let greedy = pack(input.clone(), PackOptions::new(10, 0)).unwrap();
        let bfd =
            pack(input, PackOptions { strategy: PackStrategy::BestFitDecreasing, ..PackOptions::new(10, 0) }).unwrap();
        assert!(bfd.len() <= greedy.len());
        assert_eq!(bfd.len(), 2);
        assert!((packing_efficiency(&bfd) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn overflow_policies() {
        assert_eq!(pack(vec![s(12)], PackOptions::new(10, 0)).unwrap()[0].ids.len(), 10);
        assert!(
            pack(vec![s(12)], PackOptions { overflow: Overflow::Drop, ..PackOptions::new(10, 0) }).unwrap().is_empty()
        );
        assert!(pack(vec![s(12)], PackOptions { overflow: Overflow::Error, ..PackOptions::new(10, 0) }).is_err());
        let free = pack(vec![s(3)], PackOptions { padding_free: true, ..PackOptions::new(10, 0) }).unwrap();
        assert_eq!(free[0].ids.len(), 3);
    }
}
