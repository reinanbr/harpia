//! LoRA fine-tuning of decoder LLMs: supervised fine-tuning (SFT) on assistant-masked samples
//! and (experimental) DPO, exported as PEFT adapters.
//!
//! Each target weight becomes `W + (alpha / r) · B·A` with `W` frozen; the model is rebuilt from
//! the adapter state at every optimizer step, which keeps candle's model implementations
//! unchanged.

use crate::backend::{LoraPair, Mode, TrainStore};
use crate::classifier::copy_aux_files;
use crate::nn::DecoderLm;
use crate::optim::{GroupedAdamW, Schedule, err, lr_at, masked_token_logprob};
use candle_core::{DType, Device, Tensor, Var};
use harpia_core::{HarpiaError, Result};
use harpia_data::pack::TokenizedSample;
use harpia_data::rng::SplitMix64;
use harpia_local::ModelDir;
use harpia_local::adapter::{LoraAdapterConfig, peft_key};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

/// LoRA settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoraConfig {
    /// Rank.
    pub r: usize,
    /// Scaling numerator (`alpha / r`).
    pub alpha: f64,
    /// Linear layers to adapt (last path component of the weight name).
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 16,
            alpha: 32.0,
            target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

/// Settings shared by LoRA SFT and DPO.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoraTrainConfig {
    /// Adapter shape.
    pub lora: LoraConfig,
    /// Epochs over the samples.
    pub epochs: usize,
    /// Samples per optimizer step (gradient accumulation).
    pub grad_accum: usize,
    /// Peak learning rate.
    pub lr: f64,
    /// Weight decay on the adapter matrices.
    pub weight_decay: f64,
    /// Warmup fraction.
    pub warmup_ratio: f64,
    /// Schedule after warmup.
    pub schedule: Schedule,
    /// Global gradient-norm clip (0 = off).
    pub grad_clip: f64,
    /// Truncate samples to this many tokens.
    pub max_len: usize,
    /// Seed for initialization and shuffling.
    pub seed: u64,
    /// Also write merged full weights next to the adapter.
    pub save_merged: bool,
    /// DPO temperature (DPO only).
    pub beta: f64,
}

impl Default for LoraTrainConfig {
    fn default() -> Self {
        Self {
            lora: LoraConfig::default(),
            epochs: 1,
            grad_accum: 1,
            lr: 2e-4,
            weight_decay: 0.0,
            warmup_ratio: 0.03,
            schedule: Schedule::Cosine,
            grad_clip: 1.0,
            max_len: 2048,
            seed: 42,
            save_merged: false,
            beta: 0.1,
        }
    }
}

/// Report of a LoRA run (also written to `<out>/train.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraReport {
    /// `"sft"` or `"dpo"`.
    pub objective: String,
    /// Base model directory.
    pub base: String,
    /// Settings.
    pub config: LoraTrainConfig,
    /// Number of adapted weight matrices.
    pub adapted_weights: usize,
    /// Trainable parameters.
    pub trainable_params: usize,
    /// Optimizer steps.
    pub steps: usize,
    /// Loss per step.
    pub losses: Vec<f64>,
    /// Wall-clock seconds.
    pub seconds: f64,
    /// Harpia version.
    pub harpia_version: String,
}

fn init_lora(
    base: &HashMap<String, Tensor>,
    cfg: &LoraConfig,
    seed: u64,
    device: &Device,
) -> Result<BTreeMap<String, LoraPair>> {
    let mut rng = SplitMix64::new(seed);
    let mut out = BTreeMap::new();
    let mut names: Vec<&String> = base.keys().collect();
    names.sort();
    for name in names {
        let t = &base[name];
        let Some(stem) = name.strip_suffix(".weight") else { continue };
        let module = stem.rsplit('.').next().unwrap_or("");
        if !cfg.target_modules.iter().any(|m| m == module) || t.rank() != 2 {
            continue;
        }
        let (rows, cols) = t.dims2().map_err(err)?;
        // PEFT init: A ~ kaiming-uniform (bound 1/sqrt(fan_in)), B = 0, so training starts at the base model
        let bound = 1.0 / (cols as f64).sqrt();
        let a: Vec<f32> = (0..cfg.r * cols).map(|_| ((rng.next_f64() * 2.0 - 1.0) * bound) as f32).collect();
        let a = Var::from_tensor(&Tensor::from_vec(a, (cfg.r, cols), device).map_err(err)?).map_err(err)?;
        let b = Var::zeros((rows, cfg.r), DType::F32, device).map_err(err)?;
        out.insert(name.clone(), LoraPair { a, b });
    }
    if out.is_empty() {
        return Err(HarpiaError::invalid(format!("no weights match LoRA target modules {:?}", cfg.target_modules)));
    }
    Ok(out)
}

struct Run {
    md: ModelDir,
    store: TrainStore,
    net: DecoderLm,
    vars: Vec<(String, Var)>,
    opt: GroupedAdamW,
    device: Device,
}

impl Run {
    fn new(base: &Path, cfg: &LoraTrainConfig) -> Result<Self> {
        let device = Device::Cpu;
        let md = ModelDir::open(base)?;
        let tensors = md.tensors(&device)?;
        let lora = init_lora(&tensors, &cfg.lora, cfg.seed, &device)?;
        let store = TrainStore::new(tensors, Mode::Frozen, &device, cfg.seed)
            .map_err(err)?
            .with_lora(lora, cfg.lora.alpha / cfg.lora.r.max(1) as f64);
        let net = DecoderLm::new(&md.config, &store).map_err(err)?;
        let vars = store.trainable();
        let opt = GroupedAdamW::new(&vars, cfg.lr, cfg.weight_decay)?;
        Ok(Self { md, store, net, vars, opt, device })
    }

    fn update(&mut self, loss: &Tensor, cfg: &LoraTrainConfig, step: usize, total: usize) -> Result<f64> {
        let grads = loss.backward().map_err(err)?;
        self.opt.set_lr(lr_at(cfg.lr, step, (cfg.warmup_ratio * total as f64) as usize, total, cfg.schedule));
        self.opt.step(&grads, cfg.grad_clip)?;
        Ok(f64::from(loss.to_scalar::<f32>().map_err(err)?))
    }

    fn save(&self, out: &Path, cfg: &LoraTrainConfig, report: &LoraReport) -> Result<()> {
        std::fs::create_dir_all(out).map_err(|e| HarpiaError::io(out, e))?;
        let mut tensors = HashMap::new();
        for (name, p) in self.store.lora() {
            let module = name.strip_suffix(".weight").unwrap_or(name);
            tensors.insert(peft_key(module, "lora_A"), p.a.as_detached_tensor());
            tensors.insert(peft_key(module, "lora_B"), p.b.as_detached_tensor());
        }
        candle_core::safetensors::save(&tensors, out.join("adapter_model.safetensors")).map_err(err)?;
        let ac = LoraAdapterConfig {
            peft_type: "LORA".into(),
            r: cfg.lora.r,
            lora_alpha: cfg.lora.alpha,
            target_modules: cfg.lora.target_modules.clone(),
            lora_dropout: 0.0,
            bias: "none".into(),
            task_type: Some("CAUSAL_LM".into()),
            base_model_name_or_path: Some(self.md.root.display().to_string()),
        };
        write_json(&out.join("adapter_config.json"), &serde_json::to_value(ac)?)?;
        write_json(&out.join("train.json"), &serde_json::to_value(report)?)?;
        if cfg.save_merged {
            let merged_dir = out.join("merged");
            copy_aux_files(&self.md, &merged_dir)?;
            write_json(&merged_dir.join("config.json"), &self.md.config)?;
            candle_core::safetensors::save(&self.store.merged().map_err(err)?, merged_dir.join("model.safetensors"))
                .map_err(err)?;
        }
        Ok(())
    }

    fn report(&self, objective: &str, base: &Path, cfg: &LoraTrainConfig, losses: Vec<f64>, t0: Instant) -> LoraReport {
        LoraReport {
            objective: objective.into(),
            base: base.display().to_string(),
            config: cfg.clone(),
            adapted_weights: self.store.lora().len(),
            trainable_params: self.vars.iter().map(|(_, v)| v.as_tensor().elem_count()).sum(),
            steps: losses.len(),
            losses,
            seconds: t0.elapsed().as_secs_f64(),
            harpia_version: env!("CARGO_PKG_VERSION").into(),
        }
    }
}

fn write_json(p: &Path, v: &serde_json::Value) -> Result<()> {
    std::fs::write(p, serde_json::to_string_pretty(v)?).map_err(|e| HarpiaError::io(p, e))
}

fn truncate(s: &TokenizedSample, max: usize) -> (Vec<u32>, Vec<bool>) {
    let n = s.ids.len().min(max);
    (s.ids[..n].to_vec(), s.loss_mask[..n].to_vec())
}

/// LoRA supervised fine-tuning. Samples are token ids with an assistant-only loss mask (see
/// `harpia_data::template` and `harpia_data::pack`); the loss is the mean next-token NLL over
/// masked tokens. Writes a PEFT adapter (and `merged/` with `save_merged`) to `out`.
pub fn train_lora_sft(
    base: &Path,
    out: &Path,
    cfg: &LoraTrainConfig,
    samples: &[TokenizedSample],
    mut on_step: impl FnMut(usize, usize, f64),
) -> Result<LoraReport> {
    let t0 = Instant::now();
    let usable: Vec<(Vec<u32>, Vec<bool>)> = samples
        .iter()
        .map(|s| truncate(s, cfg.max_len))
        .filter(|(ids, m)| ids.len() >= 2 && m[1..].iter().any(|b| *b))
        .collect();
    if usable.is_empty() {
        return Err(HarpiaError::invalid("no sample has trainable (masked) tokens"));
    }
    let mut run = Run::new(base, cfg)?;
    let accum = cfg.grad_accum.max(1);
    let total = usable.len().div_ceil(accum) * cfg.epochs;
    let mut rng = SplitMix64::new(cfg.seed ^ 0x5F7);
    let mut losses = Vec::with_capacity(total);
    for _ in 0..cfg.epochs {
        let mut order: Vec<usize> = (0..usable.len()).collect();
        rng.shuffle(&mut order);
        for group in order.chunks(accum) {
            let mut sum: Option<Tensor> = None;
            let mut count = 0usize;
            for &i in group {
                let (ids, mask) = &usable[i];
                let logits = run.net.logits(ids, &run.device).map_err(err)?;
                let (lp, c) = masked_token_logprob(&logits, ids, mask)?;
                count += c;
                sum = Some(match sum {
                    None => lp,
                    Some(s) => (s + lp).map_err(err)?,
                });
            }
            let loss = (sum.expect("non-empty group") * (-1.0 / count.max(1) as f64)).map_err(err)?;
            let l = run.update(&loss, cfg, losses.len(), total)?;
            losses.push(l);
            on_step(losses.len(), total, l);
        }
    }
    let report = run.report("sft", base, cfg, losses, t0);
    run.save(out, cfg, &report)?;
    Ok(report)
}

/// A tokenized preference example: prompt, preferred and dispreferred continuations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreferenceIds {
    /// Prompt tokens (chat-formatted, ending with the assistant header).
    pub prompt: Vec<u32>,
    /// Preferred answer tokens (including the end-of-turn token).
    pub chosen: Vec<u32>,
    /// Dispreferred answer tokens.
    pub rejected: Vec<u32>,
}

fn answer_logprob(net: &DecoderLm, device: &Device, prompt: &[u32], answer: &[u32], max_len: usize) -> Result<Tensor> {
    let mut ids: Vec<u32> = prompt.iter().chain(answer).copied().collect();
    ids.truncate(max_len.max(2));
    let mask: Vec<bool> = (0..ids.len()).map(|i| i >= prompt.len()).collect();
    let logits = net.logits(&ids, device).map_err(err)?;
    Ok(masked_token_logprob(&logits, &ids, &mask)?.0)
}

/// DPO with LoRA (experimental): the frozen base model is the reference policy.
/// Loss: `-log σ(β · ((log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r))))`.
pub fn train_lora_dpo(
    base: &Path,
    out: &Path,
    cfg: &LoraTrainConfig,
    pairs: &[PreferenceIds],
    mut on_step: impl FnMut(usize, usize, f64),
) -> Result<LoraReport> {
    let t0 = Instant::now();
    if pairs.is_empty() {
        return Err(HarpiaError::invalid("no preference pairs"));
    }
    let mut run = Run::new(base, cfg)?;
    // reference log-probs: B = 0 at initialization, so the current model is the base model
    let mut reference = Vec::with_capacity(pairs.len());
    {
        for p in pairs {
            let c = answer_logprob(&run.net, &run.device, &p.prompt, &p.chosen, cfg.max_len)?
                .to_scalar::<f32>()
                .map_err(err)?;
            let r = answer_logprob(&run.net, &run.device, &p.prompt, &p.rejected, cfg.max_len)?
                .to_scalar::<f32>()
                .map_err(err)?;
            reference.push((c, r));
        }
    }
    let accum = cfg.grad_accum.max(1);
    let total = pairs.len().div_ceil(accum) * cfg.epochs;
    let mut rng = SplitMix64::new(cfg.seed ^ 0xD90);
    let mut losses = Vec::with_capacity(total);
    for _ in 0..cfg.epochs {
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        rng.shuffle(&mut order);
        for group in order.chunks(accum) {
            let mut sum: Option<Tensor> = None;
            for &i in group {
                let p = &pairs[i];
                let (rc, rr) = reference[i];
                let pc = answer_logprob(&run.net, &run.device, &p.prompt, &p.chosen, cfg.max_len)?;
                let pr = answer_logprob(&run.net, &run.device, &p.prompt, &p.rejected, cfg.max_len)?;
                let margin = ((pc - pr).map_err(err)? - f64::from(rc - rr)).map_err(err)?;
                let z = (margin * cfg.beta).map_err(err)?;
                // -log σ(z) = log(1 + e^{-z})
                let l = (z.neg().and_then(|x| x.exp()).map_err(err)? + 1.0).and_then(|x| x.log()).map_err(err)?;
                sum = Some(match sum {
                    None => l,
                    Some(s) => (s + l).map_err(err)?,
                });
            }
            let loss = (sum.expect("non-empty group") / group.len() as f64).map_err(err)?;
            let l = run.update(&loss, cfg, losses.len(), total)?;
            losses.push(l);
            on_step(losses.len(), total, l);
        }
    }
    let report = run.report("dpo", base, cfg, losses, t0);
    run.save(out, cfg, &report)?;
    Ok(report)
}
