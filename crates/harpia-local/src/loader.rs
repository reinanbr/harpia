//! Loading Hugging Face model directories (`config.json`, safetensors, `tokenizer.json`).

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use harpia_core::{HarpiaError, Result};
use harpia_data::tokenize::HfTokenizer;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub(crate) fn model_err(model: &Path, e: impl std::fmt::Display) -> HarpiaError {
    HarpiaError::Model { model: model.display().to_string(), message: e.to_string() }
}

/// A local model directory in Hugging Face layout.
#[derive(Debug, Clone)]
pub struct ModelDir {
    /// Directory path.
    pub root: PathBuf,
    /// Parsed `config.json`.
    pub config: Value,
}

impl ModelDir {
    /// Open `root` and parse its `config.json`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let p = root.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| HarpiaError::io(&p, e))?;
        Ok(Self { config: serde_json::from_str(&text)?, root })
    }

    /// Deserialize `config.json` into a model-specific config type.
    pub fn config_as<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_value(self.config.clone()).map_err(|e| model_err(&self.root, format!("config.json: {e}")))
    }

    /// Safetensors files: `model.safetensors`, or the shards listed in
    /// `model.safetensors.index.json`, or every `*.safetensors` file.
    pub fn weight_files(&self) -> Result<Vec<PathBuf>> {
        let single = self.root.join("model.safetensors");
        if single.is_file() {
            return Ok(vec![single]);
        }
        let index = self.root.join("model.safetensors.index.json");
        if index.is_file() {
            let v: Value =
                serde_json::from_str(&std::fs::read_to_string(&index).map_err(|e| HarpiaError::io(&index, e))?)?;
            let mut files: Vec<String> = v["weight_map"]
                .as_object()
                .map(|m| m.values().filter_map(Value::as_str).map(String::from).collect())
                .unwrap_or_default();
            files.sort();
            files.dedup();
            return Ok(files.into_iter().map(|f| self.root.join(f)).collect());
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.root)
            .map_err(|e| HarpiaError::io(&self.root, e))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(model_err(&self.root, "no .safetensors weights found"));
        }
        Ok(files)
    }

    /// All weight tensors by name, read into memory (no `unsafe` memory mapping).
    pub fn tensors(&self, device: &Device) -> Result<std::collections::HashMap<String, candle_core::Tensor>> {
        let mut tensors = std::collections::HashMap::new();
        for f in self.weight_files()? {
            let loaded = candle_core::safetensors::load(&f, device).map_err(|e| model_err(&f, e))?;
            tensors.extend(loaded);
        }
        Ok(tensors)
    }

    /// A `VarBuilder` over all weights.
    pub fn var_builder(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
        Ok(VarBuilder::from_tensors(self.tensors(device)?, dtype, device))
    }

    /// `tokenizer.json` of the model.
    pub fn tokenizer(&self) -> Result<HfTokenizer> {
        HfTokenizer::from_file(self.root.join("tokenizer.json"))
    }

    /// Parsed `tokenizer_config.json`, if present.
    pub fn tokenizer_config(&self) -> Option<Value> {
        std::fs::read_to_string(self.root.join("tokenizer_config.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
    }

    /// Labels in id order from `id2label` (falls back to `LABEL_i`).
    pub fn labels(&self, n: usize) -> Vec<String> {
        let map = self.config["id2label"].as_object();
        (0..n)
            .map(|i| {
                map.and_then(|m| m.get(&i.to_string()))
                    .and_then(Value::as_str)
                    .map(String::from)
                    .unwrap_or_else(|| format!("LABEL_{i}"))
            })
            .collect()
    }

    /// Number of labels from `id2label` / `num_labels`.
    pub fn num_labels(&self) -> Option<usize> {
        self.config["id2label"]
            .as_object()
            .map(|m| m.len())
            .or_else(|| self.config["num_labels"].as_u64().map(|n| n as usize))
    }

    /// `model_type` field.
    pub fn model_type(&self) -> &str {
        self.config["model_type"].as_str().unwrap_or("")
    }

    /// Token ids listed under `key` (number or array), e.g. `eos_token_id`.
    pub fn token_ids(&self, key: &str) -> Vec<u32> {
        match &self.config[key] {
            Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            Value::Array(a) => a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect(),
            _ => vec![],
        }
    }
}
