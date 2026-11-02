//! Run metadata shared by every trainer: dataset hash, hardware, and the model card written
//! next to `train.json`.

use harpia_core::{HarpiaError, Result};
use harpia_eval::report::{CardDataset, ModelCard};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Where a run executed.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Hardware {
    /// Operating system (`linux`, `macos`, `windows`).
    pub os: String,
    /// CPU architecture.
    pub arch: String,
    /// Logical CPUs available to the process.
    pub cpus: usize,
    /// Compute device (`cpu`, `cuda`, `metal`).
    pub device: String,
}

impl Hardware {
    /// Describe the current machine for `device`.
    pub fn current(device: &candle_core::Device) -> Self {
        Self {
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            cpus: std::thread::available_parallelism().map_or(1, usize::from),
            device: match device {
                candle_core::Device::Cpu => "cpu",
                candle_core::Device::Cuda(_) => "cuda",
                candle_core::Device::Metal(_) => "metal",
            }
            .into(),
        }
    }
}

/// Stable content hash (xxh3-128, hex) of a training set, so `train.json` and the model card
/// identify exactly which data was used.
pub fn data_hash<T: Serialize>(items: &[T]) -> Result<String> {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    for it in items {
        h.update(&serde_json::to_vec(it)?);
        h.update(b"\n");
    }
    Ok(format!("{:032x}", h.digest128()))
}

/// Write `<out>/README.md` from `card`.
pub fn write_card(out: &Path, card: &ModelCard) -> Result<()> {
    let p = out.join("README.md");
    std::fs::write(&p, card.to_markdown()).map_err(|e| HarpiaError::io(&p, e))
}

/// Model card skeleton shared by the trainers.
pub(crate) fn base_card(name: &str, description: &str, base: &Path, data_hash: &str, rows: usize) -> ModelCard {
    let base_model = std::fs::read_to_string(base.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("_name_or_path").and_then(|n| n.as_str()).map(String::from))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| base.display().to_string());
    ModelCard {
        name: name.into(),
        description: description.into(),
        base_model: Some(base_model),
        license: None,
        datasets: vec![CardDataset {
            name: "training data".into(),
            split: "train".into(),
            rows: Some(rows),
            license: None,
            hash: Some(data_hash.into()),
        }],
        limitations: vec![
            "The license of these weights must respect the base model's and the training data's licenses; set it before publishing.".into(),
            "Metrics come from the run's own validation split; evaluate on held-out data from your domain before relying on the model.".into(),
        ],
        ..ModelCard::default()
    }
}
