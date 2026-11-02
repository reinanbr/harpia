//! Reports: a JSON run summary, a Markdown leaderboard across runs/models, and a model card
//! generator (base model, data, hyperparameters, metrics vs. baseline, limitations, license).

use crate::harness::TaskResult;
use harpia_core::{HarpiaError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Summary of one evaluation run (one backend over several tasks).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    /// Output schema version.
    pub schema: u32,
    /// Model / backend name shown in leaderboards.
    pub model: String,
    /// Task → metric → value.
    pub results: BTreeMap<String, BTreeMap<String, f64>>,
    /// Task → rows evaluated.
    pub n: BTreeMap<String, usize>,
    /// Free-form metadata (commit, hardware, date...).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl RunSummary {
    /// Summarize task results (per-row samples are dropped).
    pub fn new(model: impl Into<String>, results: &[TaskResult]) -> Self {
        Self {
            schema: 1,
            model: model.into(),
            results: results.iter().map(|r| (r.task.clone(), r.metrics.clone())).collect(),
            n: results.iter().map(|r| (r.task.clone(), r.n)).collect(),
            metadata: BTreeMap::new(),
        }
    }

    /// Save as pretty JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, serde_json::to_vec_pretty(self)?).map_err(|e| HarpiaError::io(path, e))
    }

    /// Load from JSON.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Ok(serde_json::from_slice(&std::fs::read(path).map_err(|e| HarpiaError::io(path, e))?)?)
    }
}

/// Markdown leaderboard: one row per run, one column per `task/metric`, sorted by the mean of
/// all shown metrics (descending). Missing values show as `—`.
///
/// ```
/// use harpia_eval::report::{RunSummary, leaderboard};
/// use std::collections::BTreeMap;
/// let run = |m: &str, v: f64| RunSummary {
///     schema: 1, model: m.into(),
///     results: BTreeMap::from([("qa".into(), BTreeMap::from([("exact_match".into(), v)]))]),
///     n: BTreeMap::new(), metadata: BTreeMap::new() };
/// let md = leaderboard(&[run("small", 0.4), run("large", 0.7)], None);
/// assert!(md.find("large").unwrap() < md.find("small").unwrap());
/// ```
pub fn leaderboard(runs: &[RunSummary], metrics: Option<&[&str]>) -> String {
    let mut cols: BTreeSet<(String, String)> = BTreeSet::new();
    for r in runs {
        for (t, ms) in &r.results {
            for m in ms.keys() {
                if metrics.is_none_or(|f| f.contains(&m.as_str())) {
                    cols.insert((t.clone(), m.clone()));
                }
            }
        }
    }
    let cols: Vec<(String, String)> = cols.into_iter().collect();
    let value = |r: &RunSummary, (t, m): &(String, String)| r.results.get(t).and_then(|x| x.get(m)).copied();
    let mut order: Vec<(f64, &RunSummary)> = runs
        .iter()
        .map(|r| {
            let vals: Vec<f64> = cols.iter().filter_map(|c| value(r, c)).collect();
            (if vals.is_empty() { f64::NEG_INFINITY } else { vals.iter().sum::<f64>() / vals.len() as f64 }, r)
        })
        .collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.model.cmp(&b.1.model)));
    let mut s = String::from("| rank | model |");
    for (t, m) in &cols {
        s += &format!(" {t}/{m} |");
    }
    s += " mean |\n|---:|---|";
    s += &"---:|".repeat(cols.len() + 1);
    s.push('\n');
    for (i, (mean, r)) in order.iter().enumerate() {
        s += &format!("| {} | {} |", i + 1, r.model);
        for c in &cols {
            match value(r, c) {
                Some(v) => s += &format!(" {v:.4} |"),
                None => s += " — |",
            }
        }
        if mean.is_finite() {
            s += &format!(" {mean:.4} |\n");
        } else {
            s += " — |\n";
        }
    }
    s
}

/// Inputs of a model card.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelCard {
    /// Model name.
    pub name: String,
    /// What the model does.
    pub description: String,
    /// Base model id and its license.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
    /// License of the resulting weights (must respect the base model's and data licenses).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Training/eval datasets with licenses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<CardDataset>,
    /// Hyperparameters.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub hyperparameters: BTreeMap<String, Value>,
    /// Metrics of this model, by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
    /// Baseline metrics (same names) for comparison.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub baseline: BTreeMap<String, f64>,
    /// Name of the baseline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_name: Option<String>,
    /// Decision thresholds, if calibrated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thresholds: Option<Value>,
    /// Intended use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intended_use: Option<String>,
    /// Known limitations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub limitations: Vec<String>,
}

/// A dataset entry on a model card.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CardDataset {
    /// Name or path.
    pub name: String,
    /// Role (`train`, `validation`, `test`).
    pub split: String,
    /// Rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<usize>,
    /// License.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Content hash (e.g. of the dataset card).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

impl ModelCard {
    /// Render as Markdown (Hugging Face README style, with YAML front matter).
    pub fn to_markdown(&self) -> String {
        let mut s = String::from("---\n");
        if let Some(l) = &self.license {
            s += &format!("license: {}\n", yaml_scalar(l));
        }
        if let Some(b) = &self.base_model {
            s += &format!("base_model: {}\n", yaml_scalar(b));
        }
        s += "library_name: harpia\n---\n\n";
        s += &format!("# {}\n\n{}\n\n", self.name, self.description);
        if let Some(u) = &self.intended_use {
            s += &format!("## Intended use\n\n{u}\n\n");
        }
        s += "## Model details\n\n";
        s += &format!("- Base model: {}\n", self.base_model.as_deref().unwrap_or("none (trained from scratch)"));
        s += &format!("- License: {}\n\n", self.license.as_deref().unwrap_or("not specified"));
        if !self.datasets.is_empty() {
            s += "## Data\n\n| dataset | split | rows | license | hash |\n|---|---|---:|---|---|\n";
            for d in &self.datasets {
                s += &format!(
                    "| {} | {} | {} | {} | {} |\n",
                    d.name,
                    d.split,
                    d.rows.map_or("—".into(), |r| r.to_string()),
                    d.license.as_deref().unwrap_or("unknown"),
                    d.hash.as_deref().unwrap_or("—")
                );
            }
            s.push('\n');
        }
        if !self.hyperparameters.is_empty() {
            s += "## Hyperparameters\n\n| name | value |\n|---|---|\n";
            for (k, v) in &self.hyperparameters {
                s += &format!("| {k} | {} |\n", v.as_str().map_or_else(|| v.to_string(), String::from));
            }
            s.push('\n');
        }
        if !self.metrics.is_empty() {
            let base = self.baseline_name.as_deref().unwrap_or("baseline");
            if self.baseline.is_empty() {
                s += "## Evaluation\n\n| metric | value |\n|---|---:|\n";
                for (k, v) in &self.metrics {
                    s += &format!("| {k} | {v:.4} |\n");
                }
            } else {
                s += &format!("## Evaluation\n\n| metric | this model | {base} | Δ |\n|---|---:|---:|---:|\n");
                for (k, v) in &self.metrics {
                    match self.baseline.get(k) {
                        Some(b) => s += &format!("| {k} | {v:.4} | {b:.4} | {:+.4} |\n", v - b),
                        None => s += &format!("| {k} | {v:.4} | — | — |\n"),
                    }
                }
            }
            s.push('\n');
        }
        if let Some(t) = &self.thresholds {
            s += &format!(
                "## Decision thresholds\n\nTuned on validation data and applied frozen on test:\n\n```json\n{}\n```\n\n",
                serde_json::to_string_pretty(t).unwrap_or_default()
            );
        }
        s += "## Limitations\n\n";
        if self.limitations.is_empty() {
            s += "- Not documented. Evaluate on your own data before relying on this model.\n";
        }
        for l in &self.limitations {
            s += &format!("- {l}\n");
        }
        s
    }
}

fn yaml_scalar(s: &str) -> String {
    if s.chars().all(|c| c.is_alphanumeric() || "-._/".contains(c)) { s.to_string() } else { format!("{s:?}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_renders_delta_and_licenses() {
        let card = ModelCard {
            name: "nli-pt".into(),
            description: "Cross-encoder.".into(),
            base_model: Some("xlm-roberta-base".into()),
            license: Some("CC BY-NC 2.0".into()),
            datasets: vec![CardDataset {
                name: "pairs".into(),
                split: "train".into(),
                rows: Some(10),
                license: Some("MIT".into()),
                hash: None,
            }],
            metrics: BTreeMap::from([("macro_f1".into(), 0.8)]),
            baseline: BTreeMap::from([("macro_f1".into(), 0.6)]),
            limitations: vec!["Out-of-domain text degrades accuracy.".into()],
            ..Default::default()
        };
        let md = card.to_markdown();
        assert!(md.starts_with("---\nlicense: \"CC BY-NC 2.0\"\nbase_model: xlm-roberta-base\n"));
        assert!(md.contains("| macro_f1 | 0.8000 | 0.6000 | +0.2000 |"));
        assert!(md.contains("| pairs | train | 10 | MIT | — |"));
        assert!(md.contains("Out-of-domain"));
    }

    #[test]
    fn leaderboard_handles_missing() {
        let a = RunSummary {
            schema: 1,
            model: "a".into(),
            results: BTreeMap::from([("t".into(), BTreeMap::from([("m".into(), 0.5)]))]),
            n: BTreeMap::new(),
            metadata: BTreeMap::new(),
        };
        let b = RunSummary { model: "b".into(), results: BTreeMap::new(), ..a.clone() };
        let md = leaderboard(&[b, a], Some(&["m"]));
        assert!(md.contains("| 1 | a | 0.5000 | 0.5000 |"));
        assert!(md.contains("| 2 | b | — | — |"));
    }
}
