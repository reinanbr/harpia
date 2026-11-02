//! Composable curation pipeline over JSON records with a per-stage report.

use crate::dedup::ExactDedup;
use crate::io::{get_path, set_path};
use crate::pii::PiiDetector;
use harpia_core::Result;
use harpia_core::traits::{DataFilter, FilterDecision};
use harpia_text::normalize::{NormalizeOptions, normalize};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// One pipeline stage.
pub enum Stage {
    /// Normalize the text field.
    Normalize(NormalizeOptions),
    /// Redact PII in the text field.
    RedactPii(PiiDetector),
    /// Drop records rejected by a filter.
    Filter(Box<dyn DataFilter>),
    /// Drop exact duplicates (normalized) of the text field. Runs sequentially.
    ExactDedup,
}

impl Stage {
    fn name(&self) -> String {
        match self {
            Stage::Normalize(_) => "normalize".into(),
            Stage::RedactPii(_) => "pii".into(),
            Stage::Filter(f) => f.name().to_string(),
            Stage::ExactDedup => "exact_dedup".into(),
        }
    }
}

/// Counts per stage.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct StageReport {
    /// Stage name.
    pub stage: String,
    /// Records entering the stage.
    pub input: usize,
    /// Records dropped, by reason.
    pub dropped: BTreeMap<String, usize>,
    /// Records modified (normalization, redaction).
    pub modified: usize,
    /// Up to `samples_per_reason` dropped texts per reason, truncated to 200 chars.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub samples: BTreeMap<String, Vec<String>>,
}

/// Report of a whole run.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CurationReport {
    /// Records read.
    pub input: usize,
    /// Records kept.
    pub output: usize,
    /// Records skipped because the text field was missing or not a string.
    pub missing_field: usize,
    /// Per-stage details.
    pub stages: Vec<StageReport>,
}

/// A curation pipeline applied to a string field of JSON records.
///
/// ```
/// use harpia_data::curate::{Pipeline, Stage};
/// use harpia_data::filter::{QualityFilter, QualityThresholds};
/// use serde_json::json;
///
/// let pipe = Pipeline::new("text")
///     .stage(Stage::RedactPii(Default::default()))
///     .stage(Stage::Filter(Box::new(QualityFilter::new(QualityThresholds::minimal(), harpia_text::Lang::En.data()))))
///     .stage(Stage::ExactDedup);
/// let rows = vec![
///     json!({"text": "write to ana@example.com today please"}),
///     json!({"text": "hi"}),
///     json!({"text": "write to ana@example.com today please"}),
/// ];
/// let (kept, report) = pipe.run(rows).unwrap();
/// assert_eq!(kept.len(), 1);
/// assert_eq!(kept[0]["text"], "write to <EMAIL> today please");
/// assert_eq!(report.output, 1);
/// ```
pub struct Pipeline {
    field: String,
    stages: Vec<Stage>,
    samples_per_reason: usize,
}

impl Pipeline {
    /// Pipeline over the text at dotted path `field`.
    pub fn new(field: impl Into<String>) -> Self {
        Self { field: field.into(), stages: Vec::new(), samples_per_reason: 3 }
    }

    /// Append a stage.
    pub fn stage(mut self, s: Stage) -> Self {
        self.stages.push(s);
        self
    }

    /// How many dropped examples to keep per reason in the report (default 3).
    pub fn samples_per_reason(mut self, n: usize) -> Self {
        self.samples_per_reason = n;
        self
    }

    /// Run over in-memory records (parallel per stage, order preserved).
    pub fn run(&self, records: Vec<Value>) -> Result<(Vec<Value>, CurationReport)> {
        let mut report = CurationReport { input: records.len(), ..Default::default() };
        let field = self.field.as_str();
        let (mut rows, missing): (Vec<Value>, Vec<Value>) =
            records.into_iter().partition(|r| get_path(r, field).and_then(Value::as_str).is_some());
        report.missing_field = missing.len();
        for stage in &self.stages {
            let mut sr = StageReport { stage: stage.name(), input: rows.len(), ..Default::default() };
            let text_of = |r: &Value| get_path(r, field).and_then(Value::as_str).unwrap_or("").to_string();
            match stage {
                Stage::Normalize(opts) => {
                    let results: Vec<Option<String>> = rows
                        .par_iter()
                        .map(|r| {
                            let t = text_of(r);
                            match normalize(&t, opts) {
                                std::borrow::Cow::Owned(s) => Some(s),
                                std::borrow::Cow::Borrowed(_) => None,
                            }
                        })
                        .collect();
                    for (r, new) in rows.iter_mut().zip(results) {
                        if let Some(s) = new {
                            set_path(r, field, Value::String(s))?;
                            sr.modified += 1;
                        }
                    }
                }
                Stage::RedactPii(det) => {
                    let results: Vec<Option<String>> = rows
                        .par_iter()
                        .map(|r| {
                            let t = text_of(r);
                            let (clean, found) = det.redact(&t);
                            (!found.is_empty()).then(|| clean.into_owned())
                        })
                        .collect();
                    for (r, new) in rows.iter_mut().zip(results) {
                        if let Some(s) = new {
                            set_path(r, field, Value::String(s))?;
                            sr.modified += 1;
                        }
                    }
                }
                Stage::Filter(f) => {
                    let decisions: Vec<FilterDecision> = rows.par_iter().map(|r| f.decide(&text_of(r))).collect();
                    let mut kept = Vec::with_capacity(rows.len());
                    for (r, d) in rows.into_iter().zip(decisions) {
                        match d {
                            FilterDecision::Keep => kept.push(r),
                            FilterDecision::Drop(reason) => self.record_drop(&mut sr, reason, &text_of(&r)),
                        }
                    }
                    rows = kept;
                }
                Stage::ExactDedup => {
                    let mut d = ExactDedup::new();
                    let mut kept = Vec::with_capacity(rows.len());
                    for r in rows {
                        let t = text_of(&r);
                        if d.is_new(&t) {
                            kept.push(r);
                        } else {
                            self.record_drop(&mut sr, "duplicate".into(), &t);
                        }
                    }
                    rows = kept;
                }
            }
            report.stages.push(sr);
        }
        report.output = rows.len();
        Ok((rows, report))
    }

    fn record_drop(&self, sr: &mut StageReport, reason: String, text: &str) {
        let samples = sr.samples.entry(reason.clone()).or_default();
        if samples.len() < self.samples_per_reason {
            samples.push(text.chars().take(200).collect());
        }
        *sr.dropped.entry(reason).or_default() += 1;
    }
}
