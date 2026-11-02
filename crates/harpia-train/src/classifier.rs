//! Full fine-tuning of encoder sequence classifiers and NLI cross-encoders.

use crate::backend::{Mode, TrainStore};
use crate::nn::EncoderClassifier;
use crate::optim::{GroupedAdamW, Schedule, err, lr_at, macro_f1, weighted_cross_entropy};
use candle_core::{Device, Tensor};
use harpia_core::{HarpiaError, Result};
use harpia_data::rng::SplitMix64;
use harpia_local::ModelDir;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One training example: a single text (`hypothesis: None`) or a `(premise, hypothesis)` pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Example {
    /// Text, or the premise (evidence) of a pair.
    pub premise: String,
    /// Hypothesis (claim) of a pair.
    #[serde(default)]
    pub hypothesis: Option<String>,
    /// Gold label name.
    pub label: String,
}

/// Classifier fine-tuning settings (defaults follow common BERT/NLI recipes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClassifierConfig {
    /// Label names of the fine-tuned model, in output order.
    pub labels: Vec<String>,
    /// Reuse the base head: new label → base label (e.g. `sustenta → entailment`). When every new
    /// label maps to a base label, the head rows are reordered and kept; otherwise the head is
    /// re-initialized.
    pub label_map: BTreeMap<String, String>,
    /// Epochs.
    pub epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Peak learning rate.
    pub lr: f64,
    /// Decoupled weight decay (not applied to biases and norms).
    pub weight_decay: f64,
    /// Fraction of steps used for linear warmup.
    pub warmup_ratio: f64,
    /// Schedule after warmup.
    pub schedule: Schedule,
    /// Maximum tokens per input.
    pub max_len: usize,
    /// Global gradient-norm clip (0 = off).
    pub grad_clip: f64,
    /// Weight the loss by inverse class frequency (`n / (k · count)`).
    pub class_weights: bool,
    /// Shuffling seed.
    pub seed: u64,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            labels: vec![],
            label_map: BTreeMap::new(),
            epochs: 3,
            batch_size: 16,
            lr: 2e-5,
            weight_decay: 0.01,
            warmup_ratio: 0.1,
            schedule: Schedule::Linear,
            max_len: 192,
            grad_clip: 1.0,
            class_weights: true,
            seed: 42,
        }
    }
}

/// Progress callback payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    /// 1-based epoch.
    pub epoch: usize,
    /// Optimizer steps so far.
    pub step: usize,
    /// Total steps.
    pub total_steps: usize,
    /// Loss of the last batch.
    pub loss: f64,
    /// Learning rate used.
    pub lr: f64,
    /// Seconds since start.
    pub elapsed: f64,
}

/// One epoch of the history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochReport {
    /// 1-based epoch.
    pub epoch: usize,
    /// Mean training loss.
    pub train_loss: f64,
    /// Macro F1 on the validation set.
    pub val_macro_f1: f64,
    /// Per-label F1 on the validation set.
    pub val_f1: BTreeMap<String, f64>,
}

/// Result of [`train_classifier`] (also written to `<out>/train.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainReport {
    /// Base model directory.
    pub base: String,
    /// Settings used.
    pub config: ClassifierConfig,
    /// Training / validation sizes.
    pub train_examples: usize,
    /// Validation size.
    pub val_examples: usize,
    /// Training label counts.
    pub class_counts: BTreeMap<String, usize>,
    /// Whether the base head was reused.
    pub head_reused: bool,
    /// Per-epoch history.
    pub history: Vec<EpochReport>,
    /// Best validation macro F1 (the saved checkpoint).
    pub best_val_macro_f1: f64,
    /// Wall-clock seconds.
    pub seconds: f64,
    /// Harpia version.
    pub harpia_version: String,
}

fn head_names(model_type: &str) -> (&'static str, &'static str) {
    if model_type == "bert" {
        ("classifier.weight", "classifier.bias")
    } else {
        ("classifier.out_proj.weight", "classifier.out_proj.bias")
    }
}

/// Fine-tune the classifier at `base` and save the best epoch (validation macro F1) to `out` in
/// Hugging Face format (`model.safetensors`, `config.json` with the new `id2label`, tokenizer
/// files, `train.json`).
pub fn train_classifier(
    base: &Path,
    out: &Path,
    cfg: &ClassifierConfig,
    train: &[Example],
    val: &[Example],
    mut on_progress: impl FnMut(&Progress),
) -> Result<TrainReport> {
    let t0 = Instant::now();
    if cfg.labels.is_empty() {
        return Err(HarpiaError::invalid("ClassifierConfig.labels is empty"));
    }
    let label_id: HashMap<&str, u32> = cfg.labels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let target = |e: &Example| {
        label_id
            .get(e.label.as_str())
            .copied()
            .ok_or_else(|| HarpiaError::invalid(format!("unknown label `{}`", e.label)))
    };
    let device = Device::Cpu;
    let md = ModelDir::open(base)?;
    let mut tensors = md.tensors(&device)?;

    // head reuse / remap
    let base_labels = md.num_labels().map(|n| md.labels(n)).unwrap_or_default();
    let (w_name, b_name) = head_names(md.model_type());
    let mapped: Option<Vec<usize>> = cfg
        .labels
        .iter()
        .map(|l| {
            let base_l = cfg.label_map.get(l).unwrap_or(l);
            base_labels.iter().position(|b| b.eq_ignore_ascii_case(base_l))
        })
        .collect();
    let head_reused = match (&mapped, tensors.get(w_name), tensors.get(b_name)) {
        (Some(rows), Some(w), Some(b)) => {
            let idx =
                Tensor::new(rows.iter().map(|&r| r as u32).collect::<Vec<_>>().as_slice(), &device).map_err(err)?;
            let w = w.index_select(&idx, 0).map_err(err)?;
            let b = b.index_select(&idx, 0).map_err(err)?;
            tensors.insert(w_name.into(), w);
            tensors.insert(b_name.into(), b);
            true
        }
        _ => {
            tensors.remove(w_name);
            tensors.remove(b_name);
            false
        }
    };

    let store = TrainStore::new(tensors, Mode::Full, &device, cfg.seed).map_err(err)?;
    let net = EncoderClassifier::new(&md.config, &store, cfg.labels.len()).map_err(err)?;
    let vars: Vec<(String, candle_core::Var)> = store.trainable();
    let tok = truncating_tokenizer(&md, cfg.max_len)?;
    let encode = |e: &Example| -> Result<Vec<u32>> {
        let enc = match &e.hypothesis {
            Some(h) => tok.encode((e.premise.as_str(), h.as_str()), true),
            None => tok.encode(e.premise.as_str(), true),
        }
        .map_err(|e| HarpiaError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    };
    let train_ids: Vec<Vec<u32>> = train.iter().map(encode).collect::<Result<_>>()?;
    let train_y: Vec<u32> = train.iter().map(target).collect::<Result<_>>()?;
    let val_ids: Vec<Vec<u32>> = val.iter().map(encode).collect::<Result<_>>()?;
    let val_y: Vec<u32> = val.iter().map(target).collect::<Result<_>>()?;

    let k = cfg.labels.len();
    let mut counts = vec![0usize; k];
    for &y in &train_y {
        counts[y as usize] += 1;
    }
    let weights: Option<Vec<f32>> = cfg.class_weights.then(|| {
        counts.iter().map(|&c| if c == 0 { 0.0 } else { train_y.len() as f32 / (k as f32 * c as f32) }).collect()
    });

    let steps_per_epoch = train_ids.len().div_ceil(cfg.batch_size.max(1));
    let total = steps_per_epoch * cfg.epochs;
    let warmup = (cfg.warmup_ratio * total as f64) as usize;
    let mut opt = GroupedAdamW::new(&vars, cfg.lr, cfg.weight_decay)?;
    let mut rng = SplitMix64::new(cfg.seed);
    let mut step = 0;
    let mut best = -1.0f64;
    let mut history = Vec::new();

    let evaluate = |net: &EncoderClassifier| -> Result<Vec<u32>> {
        let mut pred = Vec::with_capacity(val_ids.len());
        for chunk in val_ids.chunks(64) {
            let l = net
                .logits(chunk, &device)
                .map_err(err)?
                .detach()
                .argmax(1)
                .and_then(|t| t.to_vec1::<u32>())
                .map_err(err)?;
            pred.extend(l);
        }
        Ok(pred)
    };

    for epoch in 1..=cfg.epochs {
        let mut order: Vec<usize> = (0..train_ids.len()).collect();
        rng.shuffle(&mut order);
        let mut loss_sum = 0.0;
        for chunk in order.chunks(cfg.batch_size.max(1)) {
            let batch: Vec<Vec<u32>> = chunk.iter().map(|&i| train_ids[i].clone()).collect();
            let ys: Vec<u32> = chunk.iter().map(|&i| train_y[i]).collect();
            let t_fwd = Instant::now();
            let logits = net.logits(&batch, &device).map_err(err)?;
            let loss = weighted_cross_entropy(&logits, &ys, weights.as_deref())?;
            let lv = f64::from(loss.to_scalar::<f32>().map_err(err)?);
            let t_bwd = Instant::now();
            let grads = loss.backward().map_err(err)?;
            let t_clip = Instant::now();
            let lr = lr_at(cfg.lr, step, warmup, total, cfg.schedule);
            opt.set_lr(lr);
            let t_opt = Instant::now();
            opt.step(&grads, cfg.grad_clip)?;
            tracing::debug!(
                forward = t_bwd.duration_since(t_fwd).as_secs_f64(),
                backward = t_clip.duration_since(t_bwd).as_secs_f64(),
                clip_and_lr = t_opt.duration_since(t_clip).as_secs_f64(),
                optimizer = t_opt.elapsed().as_secs_f64(),
                "step timing"
            );
            step += 1;
            loss_sum += lv;
            on_progress(&Progress {
                epoch,
                step,
                total_steps: total,
                loss: lv,
                lr,
                elapsed: t0.elapsed().as_secs_f64(),
            });
        }
        let pred = evaluate(&net)?;
        let (f1, per) = macro_f1(&val_y, &pred, k);
        history.push(EpochReport {
            epoch,
            train_loss: loss_sum / steps_per_epoch.max(1) as f64,
            val_macro_f1: f1,
            val_f1: cfg.labels.iter().cloned().zip(per).collect(),
        });
        tracing::info!(epoch, val_macro_f1 = f1, "epoch done");
        if f1 > best {
            best = f1;
            save_checkpoint(&md, &store, out, &cfg.labels)?;
        }
    }
    let report = TrainReport {
        base: base.display().to_string(),
        config: cfg.clone(),
        train_examples: train.len(),
        val_examples: val.len(),
        class_counts: cfg.labels.iter().cloned().zip(counts).collect(),
        head_reused,
        history,
        best_val_macro_f1: best,
        seconds: t0.elapsed().as_secs_f64(),
        harpia_version: env!("CARGO_PKG_VERSION").into(),
    };
    let p = out.join("train.json");
    std::fs::write(&p, serde_json::to_string_pretty(&report)?).map_err(|e| HarpiaError::io(&p, e))?;
    Ok(report)
}

/// The model's tokenizer with longest-first truncation at `max_len` and no padding.
pub(crate) fn truncating_tokenizer(md: &ModelDir, max_len: usize) -> Result<tokenizers::Tokenizer> {
    let mut t = md.tokenizer()?.inner().clone();
    t.with_padding(None);
    t.with_truncation(Some(tokenizers::TruncationParams {
        max_length: max_len,
        strategy: tokenizers::TruncationStrategy::LongestFirst,
        stride: 0,
        direction: tokenizers::TruncationDirection::Right,
    }))
    .map_err(|e| HarpiaError::Tokenizer(e.to_string()))?;
    Ok(t)
}

/// Copy every non-weight file of `md` (tokenizer, special tokens...) into `out`.
pub(crate) fn copy_aux_files(md: &ModelDir, out: &Path) -> Result<()> {
    std::fs::create_dir_all(out).map_err(|e| HarpiaError::io(out, e))?;
    for entry in std::fs::read_dir(&md.root).map_err(|e| HarpiaError::io(&md.root, e))? {
        let path: PathBuf = entry.map_err(HarpiaError::from)?.path();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let is_weights =
            name.ends_with(".safetensors") || name == "model.safetensors.index.json" || name.ends_with(".bin");
        if path.is_file() && !is_weights && name != "config.json" && name != "train.json" {
            std::fs::copy(&path, out.join(&name)).map_err(|e| HarpiaError::io(&path, e))?;
        }
    }
    Ok(())
}

fn save_checkpoint(md: &ModelDir, store: &TrainStore, out: &Path, labels: &[String]) -> Result<()> {
    copy_aux_files(md, out)?;
    let weights = store.merged().map_err(err)?;
    candle_core::safetensors::save(&weights, out.join("model.safetensors")).map_err(err)?;
    let mut config = md.config.clone();
    let id2label: serde_json::Map<String, Value> =
        labels.iter().enumerate().map(|(i, l)| (i.to_string(), json!(l))).collect();
    let label2id: serde_json::Map<String, Value> =
        labels.iter().enumerate().map(|(i, l)| (l.clone(), json!(i))).collect();
    if let Some(o) = config.as_object_mut() {
        o.insert("id2label".into(), Value::Object(id2label));
        o.insert("label2id".into(), Value::Object(label2id));
        o.insert("num_labels".into(), json!(labels.len()));
    }
    let p = out.join("config.json");
    std::fs::write(&p, serde_json::to_string_pretty(&config)?).map_err(|e| HarpiaError::io(&p, e))
}
