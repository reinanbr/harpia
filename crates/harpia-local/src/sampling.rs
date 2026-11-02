//! Token selection: penalties, temperature, top-k, top-p, min-p, seeded sampling, and an
//! optional per-token allow-list (constrained decoding).

use harpia_core::traits::GenerationParams;
use harpia_data::rng::SplitMix64;
use std::collections::HashMap;

/// Log-softmax of `logits`.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + logits.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
    logits.iter().map(|x| x - lse).collect()
}

/// Apply penalties. The repetition penalty (multiplicative) follows Hugging Face and covers every
/// token in `context` (prompt and generated); frequency and presence penalties (additive) follow
/// the OpenAI convention and cover only `generated`.
pub fn apply_penalties(logits: &mut [f32], context: &[u32], generated: &[u32], p: &GenerationParams) {
    if p.repetition_penalty != 1.0 {
        let seen: std::collections::HashSet<u32> = context.iter().copied().collect();
        for t in seen {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 { *l / p.repetition_penalty } else { *l * p.repetition_penalty };
            }
        }
    }
    if p.frequency_penalty == 0.0 && p.presence_penalty == 0.0 {
        return;
    }
    let mut counts: HashMap<u32, f32> = HashMap::new();
    for &t in generated {
        *counts.entry(t).or_default() += 1.0;
    }
    for (&t, &c) in &counts {
        if let Some(l) = logits.get_mut(t as usize) {
            *l -= p.frequency_penalty * c + p.presence_penalty;
        }
    }
}

/// Picks the next token.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: GenerationParams,
    rng: SplitMix64,
}

impl Sampler {
    /// Sampler for `params`; without a seed, a time-based seed is used.
    pub fn new(params: &GenerationParams) -> Self {
        let seed = params.seed.unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
        });
        Self { params: params.clone(), rng: SplitMix64::new(seed) }
    }

    /// Choose a token from (already penalized) `logits`. With `allowed`, only tokens for which it
    /// returns `true` can be chosen; `None` means no token is allowed.
    pub fn choose(&mut self, logits: &[f32], allowed: Option<&dyn Fn(u32) -> bool>) -> Option<u32> {
        let ok = |i: usize| allowed.is_none_or(|f| f(i as u32));
        let mut order: Vec<usize> = (0..logits.len()).filter(|&i| logits[i].is_finite()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        if self.params.temperature <= 0.0 {
            return order.into_iter().find(|&i| ok(i)).map(|i| i as u32);
        }
        let t = self.params.temperature;
        let maxl = logits[*order.first()?];
        let mut cand: Vec<(usize, f32)> = order.iter().map(|&i| (i, ((logits[i] - maxl) / t).exp())).collect();
        let total: f32 = cand.iter().map(|c| c.1).sum();
        cand.iter_mut().for_each(|c| c.1 /= total);
        if self.params.top_k > 0 {
            cand.truncate(self.params.top_k as usize);
        }
        if self.params.top_p < 1.0 {
            let mut acc = 0.0;
            let mut keep = 0;
            for c in &cand {
                keep += 1;
                acc += c.1;
                if acc >= self.params.top_p {
                    break;
                }
            }
            cand.truncate(keep.max(1));
        }
        if self.params.min_p > 0.0 {
            let pmax = cand.first().map(|c| c.1).unwrap_or(0.0);
            cand.retain(|c| c.1 >= self.params.min_p * pmax);
        }
        let cand: Vec<(usize, f32)> = cand.into_iter().filter(|c| ok(c.0)).collect();
        if cand.is_empty() {
            // the filters removed every allowed token: fall back to the best allowed one
            return order.into_iter().find(|&i| ok(i)).map(|i| i as u32);
        }
        let s: f32 = cand.iter().map(|c| c.1).sum();
        let mut r = self.rng.next_f64() as f32 * s;
        for (i, p) in &cand {
            r -= p;
            if r <= 0.0 {
                return Some(*i as u32);
            }
        }
        cand.last().map(|c| c.0 as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_allowed() {
        let mut s = Sampler::new(&GenerationParams::default());
        let logits = [0.1, 3.0, 2.0];
        assert_eq!(s.choose(&logits, None), Some(1));
        assert_eq!(s.choose(&logits, Some(&|t| t != 1)), Some(2));
        assert_eq!(s.choose(&logits, Some(&|_| false)), None);
    }

    #[test]
    fn seeded_sampling_is_deterministic_and_respects_top_k() {
        let p = GenerationParams { temperature: 1.0, top_k: 2, seed: Some(7), ..Default::default() };
        let logits = [1.0, 1.0, 1.0, -5.0, 0.9];
        let a: Vec<u32> = {
            let mut s = Sampler::new(&p);
            (0..50).filter_map(|_| s.choose(&logits, None)).collect()
        };
        let b: Vec<u32> = {
            let mut s = Sampler::new(&p);
            (0..50).filter_map(|_| s.choose(&logits, None)).collect()
        };
        assert_eq!(a, b);
        assert!(a.iter().all(|&t| t == 0 || t == 1));
    }

    #[test]
    fn penalties_and_log_softmax() {
        let mut l = vec![2.0, -1.0, 0.5];
        apply_penalties(
            &mut l,
            &[2, 0, 1, 1],
            &[0, 1, 1],
            &GenerationParams { repetition_penalty: 2.0, frequency_penalty: 0.1, ..Default::default() },
        );
        assert!((l[0] - (1.0 - 0.1)).abs() < 1e-6);
        assert!((l[2] - 0.25).abs() < 1e-6, "prompt tokens get the repetition penalty");
        assert!((l[1] - (-2.0 - 0.2)).abs() < 1e-6);
        let ls = log_softmax(&[1.0, 1.0]);
        assert!((ls[0] - 0.5f32.ln()).abs() < 1e-6);
    }
}
