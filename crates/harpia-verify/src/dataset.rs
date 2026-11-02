//! Dataset curation: verify the factual claims in a field of each JSONL row and flag or drop
//! rows whose claims are contradicted by trusted sources.

use crate::verify::{AnswerReport, ClaimVerifier, Status, status_name};
use harpia_core::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Summary of a dataset run.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DatasetReport {
    /// Rows read.
    pub rows: usize,
    /// Rows kept in the output.
    pub kept: usize,
    /// Rows dropped because a claim was contradicted (with `drop_contradicted`).
    pub dropped: usize,
    /// Rows without the field (kept unchanged).
    pub missing_field: usize,
    /// Claim counts per status.
    pub claims: BTreeMap<String, usize>,
}

/// Row-level verdict added under `"verification"`: the worst claim status and claim counts.
pub fn row_summary(report: &AnswerReport) -> Value {
    let worst = if report.claims.iter().any(|c| c.verification.status == Status::Contradicted) {
        "contradicted"
    } else if report.claims.iter().any(|c| c.verification.status == Status::NotEnoughInfo) {
        "not_enough_info"
    } else if !report.claims.is_empty() && report.claims.iter().all(|c| c.verification.status == Status::Supported) {
        "supported"
    } else if report.claims.iter().any(|c| matches!(c.verification.status, Status::Skipped { .. })) {
        "skipped"
    } else if report.claims.is_empty() {
        "no_claims"
    } else {
        "unavailable"
    };
    serde_json::json!({ "status": worst, "claims": report.counts, "details": report.claims })
}

/// Verify `field` of every row. Each kept row gets a `"verification"` object; with
/// `drop_contradicted`, rows with a contradicted claim are left out.
pub async fn verify_rows(
    verifier: &ClaimVerifier<'_>,
    rows: Vec<Value>,
    field: &str,
    drop_contradicted: bool,
) -> Result<(Vec<Value>, DatasetReport)> {
    let mut report = DatasetReport { rows: rows.len(), ..Default::default() };
    let mut out = Vec::with_capacity(rows.len());
    for mut row in rows {
        let Some(text) = row.get(field).and_then(Value::as_str).map(String::from) else {
            report.missing_field += 1;
            out.push(row);
            continue;
        };
        let r = verifier.verify_answer(&text).await;
        for c in &r.claims {
            *report.claims.entry(status_name(&c.verification.status).to_string()).or_insert(0) += 1;
        }
        let contradicted = r.claims.iter().any(|c| c.verification.status == Status::Contradicted);
        if drop_contradicted && contradicted {
            report.dropped += 1;
            continue;
        }
        if let Value::Object(m) = &mut row {
            m.insert("verification".into(), row_summary(&r));
        }
        out.push(row);
    }
    report.kept = out.len();
    Ok((out, report))
}
