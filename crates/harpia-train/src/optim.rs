//! Optimizer plumbing: AdamW parameter groups, learning-rate schedules, gradient clipping, losses.

use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor, Var};

use harpia_core::Result;
use serde::{Deserialize, Serialize};

pub(crate) fn err(e: candle_core::Error) -> harpia_core::HarpiaError {
    harpia_core::HarpiaError::Other(format!("training: {e}"))
}

/// Learning-rate schedule after warmup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Schedule {
    /// Linear decay to zero (BERT/transformers default).
    #[default]
    Linear,
    /// Cosine decay to zero.
    Cosine,
    /// Constant after warmup.
    Constant,
}

/// Learning rate at `step` (0-based) of `total`, with linear warmup over `warmup` steps.
///
/// ```
/// use harpia_train::optim::{lr_at, Schedule};
/// assert_eq!(lr_at(1e-3, 0, 10, 100, Schedule::Linear), 1e-4);
/// assert!((lr_at(1e-3, 55, 10, 100, Schedule::Linear) - 1e-3 * 45.0 / 90.0).abs() < 1e-12);
/// ```
pub fn lr_at(base: f64, step: usize, warmup: usize, total: usize, schedule: Schedule) -> f64 {
    if warmup > 0 && step < warmup {
        return base * (step + 1) as f64 / warmup as f64;
    }
    let span = total.saturating_sub(warmup).max(1) as f64;
    let progress = ((step - warmup.min(step)) as f64 / span).min(1.0);
    match schedule {
        Schedule::Linear => base * (1.0 - progress),
        Schedule::Cosine => base * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()),
        Schedule::Constant => base,
    }
}

/// `true` for parameters that should not get weight decay (biases and norms).
pub fn no_decay(name: &str) -> bool {
    let n = name.to_lowercase();
    n.ends_with(".bias") || n.contains("layernorm") || n.contains("layer_norm") || n.contains("norm.")
}

/// AdamW (PyTorch semantics, decoupled weight decay) fused with global-norm gradient clipping.
///
/// One parallel pass per parameter, in place, with the moments kept as plain `f32` buffers: the
/// generic candle optimizer allocates a full-size tensor per elementary op, which dominates the
/// step time for large embedding matrices. Biases and norms get no weight decay.
pub struct GroupedAdamW {
    params: Vec<(Var, bool, Vec<f32>, Vec<f32>)>,
    lr: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: i32,
}

impl GroupedAdamW {
    /// Optimizer over named variables (`no_decay` names skip weight decay).
    pub fn new(vars: &[(String, Var)], lr: f64, weight_decay: f64) -> Result<Self> {
        let params = vars
            .iter()
            .map(|(n, v)| {
                let k = v.as_tensor().elem_count();
                (v.clone(), !no_decay(n), vec![0f32; k], vec![0f32; k])
            })
            .collect();
        Ok(Self { params, lr, weight_decay, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0 })
    }

    /// Set the learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Clip gradients to a global L2 norm of `max_norm` (0 = off) and apply one AdamW update.
    /// Parameters without a gradient are skipped, as in PyTorch. Returns the pre-clip norm.
    pub fn step(&mut self, grads: &GradStore, max_norm: f64) -> Result<f64> {
        use rayon::prelude::*;
        let gs: Vec<Option<Vec<f32>>> = self
            .params
            .iter()
            .map(|(v, ..)| {
                grads
                    .get(v.as_tensor())
                    .map(|g| g.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>())
                    .transpose()
                    .map_err(err)
            })
            .collect::<Result<_>>()?;
        let sq: f64 =
            gs.par_iter().flatten().map(|g| g.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>()).sum();
        let norm = sq.sqrt();
        let clip = if max_norm > 0.0 && norm > max_norm { max_norm / (norm + 1e-6) } else { 1.0 };
        self.t += 1;
        let (b1, b2, eps, lr) = (self.beta1, self.beta2, self.eps, self.lr);
        let bc1 = 1.0 - b1.powi(self.t);
        let bc2 = 1.0 - b2.powi(self.t);
        for ((var, decay, m, v), g) in self.params.iter_mut().zip(gs) {
            let Some(g) = g else { continue };
            let shape = var.as_tensor().shape().clone();
            let mut p =
                var.as_tensor().to_dtype(DType::F32).and_then(|t| t.flatten_all()?.to_vec1::<f32>()).map_err(err)?;
            let wd = if *decay { self.weight_decay } else { 0.0 };
            p.par_iter_mut().zip(m.par_iter_mut()).zip(v.par_iter_mut()).zip(g.par_iter()).for_each(
                |(((p, m), v), g)| {
                    let g = f64::from(*g) * clip;
                    let mm = b1 * f64::from(*m) + (1.0 - b1) * g;
                    let vv = b2 * f64::from(*v) + (1.0 - b2) * g * g;
                    *m = mm as f32;
                    *v = vv as f32;
                    let mut x = f64::from(*p) * (1.0 - lr * wd);
                    x -= lr * (mm / bc1) / ((vv / bc2).sqrt() + eps);
                    *p = x as f32;
                },
            );
            let t = Tensor::from_vec(p, shape, var.as_tensor().device())
                .and_then(|t| t.to_dtype(var.as_tensor().dtype()))
                .map_err(err)?;
            var.set(&t).map_err(err)?;
        }
        Ok(norm)
    }
}

/// Scale gradients so their global L2 norm is at most `max_norm`; returns the norm before clipping.
pub fn clip_grad_norm(grads: &mut GradStore, vars: &[(String, Var)], max_norm: f64) -> Result<f64> {
    let mut sq = 0f64;
    for (_, v) in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            sq += g.to_dtype(DType::F64).and_then(|g| g.sqr()?.sum_all()?.to_scalar::<f64>()).map_err(err)?;
        }
    }
    let norm = sq.sqrt();
    if max_norm > 0.0 && norm > max_norm {
        let f = max_norm / (norm + 1e-6);
        for (_, v) in vars {
            if let Some(g) = grads.remove(v.as_tensor()) {
                grads.insert(v.as_tensor(), (g * f).map_err(err)?);
            }
        }
    }
    Ok(norm)
}

/// Mean cross-entropy over a batch with optional per-class weights (PyTorch semantics:
/// `sum_i w[y_i]·nll_i / sum_i w[y_i]`).
pub fn weighted_cross_entropy(logits: &Tensor, targets: &[u32], class_weights: Option<&[f32]>) -> Result<Tensor> {
    let dev = logits.device();
    let lp = candle_nn::ops::log_softmax(&logits.to_dtype(DType::F32).map_err(err)?, 1).map_err(err)?;
    let idx = Tensor::new(targets, dev).and_then(|t| t.unsqueeze(1)).map_err(err)?;
    let picked = lp.gather(&idx, 1).and_then(|t| t.squeeze(1)).map_err(err)?;
    let w: Vec<f32> = targets.iter().map(|&t| class_weights.map_or(1.0, |cw| cw[t as usize])).collect();
    let wsum: f32 = w.iter().sum();
    let w = Tensor::new(w.as_slice(), dev).map_err(err)?;
    (picked.mul(&w).and_then(|x| x.sum_all()).map_err(err)? * (-1.0 / f64::from(wsum))).map_err(err)
}

/// Sum of next-token log-probabilities over positions where `mask` is true (mask aligned with
/// the target token, i.e. `mask[i]` weights the prediction of `ids[i]` from position `i-1`).
/// Returns `(sum_logprob, count)`.
pub fn masked_token_logprob(logits: &Tensor, ids: &[u32], mask: &[bool]) -> Result<(Tensor, usize)> {
    let n = ids.len();
    let dev = logits.device();
    let lp =
        candle_nn::ops::log_softmax(&logits.narrow(0, 0, n - 1).and_then(|l| l.to_dtype(DType::F32)).map_err(err)?, 1)
            .map_err(err)?;
    let targets = Tensor::new(&ids[1..], dev).and_then(|t| t.unsqueeze(1)).map_err(err)?;
    let picked = lp.gather(&targets, 1).and_then(|t| t.squeeze(1)).map_err(err)?;
    let m: Vec<f32> = mask[1..].iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    let count = m.iter().filter(|x| **x > 0.0).count();
    let m = Tensor::new(m.as_slice(), dev).map_err(err)?;
    Ok((picked.mul(&m).and_then(|x| x.sum_all()).map_err(err)?, count))
}

/// Per-class F1 and macro F1 of predictions.
pub fn macro_f1(gold: &[u32], pred: &[u32], num_labels: usize) -> (f64, Vec<f64>) {
    let mut f1s = Vec::with_capacity(num_labels);
    for c in 0..num_labels as u32 {
        let tp = gold.iter().zip(pred).filter(|(g, p)| **g == c && **p == c).count() as f64;
        let fp = gold.iter().zip(pred).filter(|(g, p)| **g != c && **p == c).count() as f64;
        let fn_ = gold.iter().zip(pred).filter(|(g, p)| **g == c && **p != c).count() as f64;
        f1s.push(if tp == 0.0 { 0.0 } else { 2.0 * tp / (2.0 * tp + fp + fn_) });
    }
    (f1s.iter().sum::<f64>() / num_labels.max(1) as f64, f1s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules() {
        assert_eq!(lr_at(1.0, 99, 10, 100, Schedule::Linear), 1.0 - 89.0 / 90.0);
        assert!(lr_at(1.0, 100, 0, 100, Schedule::Cosine).abs() < 1e-12);
        assert_eq!(lr_at(0.5, 50, 0, 100, Schedule::Constant), 0.5);
        assert!(
            no_decay("roberta.encoder.layer.0.output.LayerNorm.weight") && no_decay("x.bias") && !no_decay("x.weight")
        );
    }

    #[test]
    fn weighted_ce_matches_hand_computation() {
        let logits = Tensor::new(&[[2.0f32, 0.0], [0.0, 1.0]], &candle_core::Device::Cpu).unwrap();
        let l = weighted_cross_entropy(&logits, &[0, 1], Some(&[1.0, 3.0])).unwrap().to_scalar::<f32>().unwrap();
        let nll0 = -(2f32.exp() / (2f32.exp() + 1.0)).ln();
        let nll1 = -(1f32.exp() / (1.0 + 1f32.exp())).ln();
        assert!((l - (nll0 + 3.0 * nll1) / 4.0).abs() < 1e-6);
        let (f, per) = macro_f1(&[0, 0, 1, 1], &[0, 1, 1, 1], 2);
        assert!((per[0] - 2.0 / 3.0).abs() < 1e-12 && (f - (2.0 / 3.0 + 0.8) / 2.0).abs() < 1e-12);
    }
}
