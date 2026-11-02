//! Dataset cards: a JSON summary and a Markdown rendering of what a dataset contains and how
//! it was built.

use crate::curate::CurationReport;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// One source of a dataset.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Name or URL.
    pub name: String,
    /// License (SPDX id when known).
    pub license: Option<String>,
    /// Rows taken from this source.
    pub rows: u64,
}

/// Statistics of one split.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SplitStats {
    /// Rows.
    pub rows: u64,
    /// Tokens, when a tokenizer was used.
    pub tokens: Option<u64>,
}

/// A dataset card.
///
/// ```
/// use harpia_data::card::{DatasetCard, SplitStats};
/// let mut card = DatasetCard::new("toy-chat", "A toy chat dataset.");
/// card.license = Some("CC-BY-4.0".into());
/// card.splits.insert("train".into(), SplitStats { rows: 900, tokens: Some(120_000) });
/// let md = card.to_markdown();
/// assert!(md.contains("# toy-chat") && md.contains("| train | 900 | 120000 |"));
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DatasetCard {
    /// Dataset name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Overall license.
    pub license: Option<String>,
    /// Sources.
    pub sources: Vec<SourceInfo>,
    /// Split statistics by split name.
    pub splits: BTreeMap<String, SplitStats>,
    /// Language → rows.
    pub languages: BTreeMap<String, u64>,
    /// Curation report, if the dataset was curated with [`crate::curate::Pipeline`].
    pub curation: Option<CurationReport>,
    /// Other facts (dedup/decontamination statistics, tool versions, seeds...).
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl DatasetCard {
    /// New card.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self { name: name.into(), description: description.into(), ..Self::default() }
    }

    /// Markdown rendering.
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# {}\n\n{}\n", self.name, self.description);
        let _ = writeln!(s, "- **License:** {}", self.license.as_deref().unwrap_or("unspecified"));
        let _ = writeln!(s, "- **Built with:** harpia-data {}\n", env!("CARGO_PKG_VERSION"));
        if !self.splits.is_empty() {
            s.push_str("## Splits\n\n| split | rows | tokens |\n|---|---:|---:|\n");
            for (k, v) in &self.splits {
                let tok = v.tokens.map(|t| t.to_string()).unwrap_or_else(|| "–".into());
                let _ = writeln!(s, "| {k} | {} | {tok} |", v.rows);
            }
            s.push('\n');
        }
        if !self.sources.is_empty() {
            s.push_str("## Sources\n\n| source | license | rows |\n|---|---|---:|\n");
            for src in &self.sources {
                let _ = writeln!(
                    s,
                    "| {} | {} | {} |",
                    src.name,
                    src.license.as_deref().unwrap_or("unspecified"),
                    src.rows
                );
            }
            s.push('\n');
        }
        if !self.languages.is_empty() {
            s.push_str("## Languages\n\n");
            for (l, n) in &self.languages {
                let _ = writeln!(s, "- {l}: {n}");
            }
            s.push('\n');
        }
        if let Some(c) = &self.curation {
            let _ = writeln!(
                s,
                "## Curation\n\nInput {} rows → output {} rows ({} without the text field).\n",
                c.input, c.output, c.missing_field
            );
            s.push_str("| stage | input | dropped | modified |\n|---|---:|---|---:|\n");
            for st in &c.stages {
                let dropped = if st.dropped.is_empty() {
                    "0".to_string()
                } else {
                    st.dropped.iter().map(|(r, n)| format!("{r}: {n}")).collect::<Vec<_>>().join(", ")
                };
                let _ = writeln!(s, "| {} | {} | {} | {} |", st.stage, st.input, dropped, st.modified);
            }
            s.push('\n');
        }
        if !self.extra.is_empty() {
            s.push_str("## Other facts\n\n");
            for (k, v) in &self.extra {
                let _ = writeln!(s, "- **{k}:** {v}");
            }
        }
        s
    }
}
