//! LoRA adapters in the PEFT on-disk format (`adapter_config.json` + `adapter_model.safetensors`),
//! so adapters trained with `harpia-train` load in Python PEFT and vice versa.

use crate::loader::model_err;
use candle_core::{DType, Device, Tensor};
use harpia_core::{HarpiaError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// The subset of PEFT's `LoraConfig` that Harpia reads and writes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraAdapterConfig {
    /// Always `"LORA"`.
    pub peft_type: String,
    /// Rank.
    pub r: usize,
    /// Scaling numerator: the update is `(lora_alpha / r) · B·A`.
    pub lora_alpha: f64,
    /// Module names (last path component) the adapter targets, e.g. `q_proj`.
    pub target_modules: Vec<String>,
    /// Dropout used during training.
    #[serde(default)]
    pub lora_dropout: f64,
    /// Bias handling; Harpia supports `"none"`.
    #[serde(default = "none")]
    pub bias: String,
    /// `CAUSAL_LM` for decoders.
    #[serde(default)]
    pub task_type: Option<String>,
    /// Base model the adapter was trained on.
    #[serde(default)]
    pub base_model_name_or_path: Option<String>,
}

fn none() -> String {
    "none".into()
}

impl LoraAdapterConfig {
    /// `lora_alpha / r`.
    pub fn scale(&self) -> f64 {
        self.lora_alpha / self.r.max(1) as f64
    }
}

/// PEFT tensor name of the `lora_A` / `lora_B` matrix of `module` (e.g.
/// `model.layers.0.self_attn.q_proj`).
pub fn peft_key(module: &str, which: &str) -> String {
    format!("base_model.model.{module}.{which}.weight")
}

/// Merge the adapter in `adapter_dir` into `tensors` (`W += scale · B·A`, in f32, cast back).
pub fn merge_lora(tensors: &mut HashMap<String, Tensor>, adapter_dir: &Path, device: &Device) -> Result<()> {
    let cfg_path = adapter_dir.join("adapter_config.json");
    let cfg: LoraAdapterConfig =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).map_err(|e| HarpiaError::io(&cfg_path, e))?)?;
    if cfg.peft_type.to_uppercase() != "LORA" || cfg.bias != "none" {
        return Err(model_err(adapter_dir, "only LoRA adapters with bias = \"none\" are supported"));
    }
    let weights = adapter_dir.join("adapter_model.safetensors");
    let lora = candle_core::safetensors::load(&weights, device).map_err(|e| model_err(&weights, e))?;
    let scale = cfg.scale();
    let mut merged = 0;
    for (name, a) in &lora {
        let Some(module) = name.strip_prefix("base_model.model.").and_then(|n| n.strip_suffix(".lora_A.weight")) else {
            continue;
        };
        let b = lora
            .get(&peft_key(module, "lora_B"))
            .ok_or_else(|| model_err(adapter_dir, format!("missing lora_B for {module}")))?;
        let key = format!("{module}.weight");
        let w = tensors
            .get(&key)
            .ok_or_else(|| model_err(adapter_dir, format!("adapter targets unknown weight `{key}`")))?;
        let ce = |e: candle_core::Error| model_err(adapter_dir, e);
        let delta = b
            .to_dtype(DType::F32)
            .and_then(|b| b.matmul(&a.to_dtype(DType::F32)?))
            .and_then(|d| d * scale)
            .map_err(ce)?;
        let new = w.to_dtype(DType::F32).and_then(|w| w + delta).map_err(ce)?;
        let dtype = w.dtype();
        tensors.insert(key, new.to_dtype(dtype).map_err(ce)?);
        merged += 1;
    }
    if merged == 0 {
        return Err(model_err(adapter_dir, "adapter contains no lora_A/lora_B pairs"));
    }
    Ok(())
}
