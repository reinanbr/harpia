//! Wikidata entity search (`www.wikidata.org/w/api.php?action=wbsearchentities`): labels and
//! short descriptions (CC0). Useful for entity facts such as "X is a Y".

use super::{BoxFuture, SourceAdapter, base, stamp};
use crate::fetch::{Expect, Fetcher};
use crate::registry::Source;
use crate::work::Work;
use harpia_core::Result;
use harpia_http::Request;
use serde_json::Value;

/// The Wikidata adapter.
#[derive(Debug, Clone, Copy)]
pub struct Wikidata;

const API: &str = "https://www.wikidata.org";

/// Map a `wbsearchentities` response.
pub fn parse_search(v: &Value) -> Vec<Work> {
    v["search"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|e| {
            let label = e["label"].as_str()?;
            let desc = e["description"].as_str().unwrap_or("");
            Some(Work {
                title: label.to_string(),
                url: e["concepturi"].as_str().unwrap_or("").to_string(),
                text: if desc.is_empty() { label.to_string() } else { format!("{label} is {desc}.") },
                venue: "Wikidata".into(),
                ..Work::default()
            })
        })
        .collect()
}

impl SourceAdapter for Wikidata {
    fn name(&self) -> &str {
        "wikidata"
    }

    fn search<'a>(
        &'a self,
        f: &'a Fetcher,
        s: &'a Source,
        query: &'a str,
        k: usize,
    ) -> BoxFuture<'a, Result<Vec<Work>>> {
        Box::pin(async move {
            let req = Request::get(format!("{}/w/api.php", base(s, API)))
                .param("action", "wbsearchentities")
                .param("format", "json")
                .param("search", query.chars().take(200).collect::<String>())
                .param("language", s.lang.as_deref().unwrap_or("en"))
                .param("limit", k.min(50).to_string());
            let r = f.get(s, req, Expect::Json, "search: entity").await?;
            Ok(parse_search(&r.json()?).into_iter().map(|w| stamp(w, s, &r)).collect())
        })
    }
}
