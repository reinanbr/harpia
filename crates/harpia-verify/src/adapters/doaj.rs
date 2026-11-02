//! Directory of Open Access Journals (`doaj.org/api`): article search over open-access journals.

use super::{BoxFuture, SourceAdapter, base, encode, stamp, strip_tags};
use crate::fetch::{Expect, Fetcher};
use crate::registry::Source;
use crate::work::{Author, Work, normalize_doi};
use harpia_core::Result;
use harpia_http::Request;
use serde_json::Value;

/// The DOAJ adapter.
#[derive(Debug, Clone, Copy)]
pub struct Doaj;

const API: &str = "https://doaj.org/api";

/// Map one article.
pub fn parse_item(it: &Value) -> Work {
    let b = &it["bibjson"];
    let doi = b["identifier"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|i| i["type"].as_str().is_some_and(|t| t.eq_ignore_ascii_case("doi")))
        .and_then(|i| i["id"].as_str())
        .map(normalize_doi)
        .unwrap_or_default();
    let fulltext = b["link"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|l| l["type"].as_str() == Some("fulltext"))
        .and_then(|l| l["url"].as_str());
    Work {
        title: strip_tags(b["title"].as_str().unwrap_or("")),
        authors: b["author"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|a| a["name"].as_str())
            .map(Author::from_display)
            .collect(),
        year: b["year"].as_str().and_then(|y| y.parse().ok()),
        venue: b["journal"]["title"].as_str().unwrap_or("").to_string(),
        url: if doi.is_empty() { fulltext.unwrap_or("").to_string() } else { format!("https://doi.org/{doi}") },
        doi,
        text: strip_tags(b["abstract"].as_str().unwrap_or("")),
        license: b["journal"]["license"]
            .as_array()
            .and_then(|l| l.first())
            .and_then(|l| l["type"].as_str())
            .map(String::from),
        ..Work::default()
    }
}

impl SourceAdapter for Doaj {
    fn name(&self) -> &str {
        "doaj"
    }

    fn search<'a>(&'a self, f: &'a Fetcher, s: &'a Source, q: &'a str, k: usize) -> BoxFuture<'a, Result<Vec<Work>>> {
        Box::pin(async move {
            let q: String = q.chars().take(200).collect();
            let req = Request::get(format!("{}/search/articles/{}", base(s, API), encode(&q)))
                .param("pageSize", k.to_string());
            let r = f.get(s, req, Expect::Json, "search").await?;
            let v = r.json()?;
            Ok(v["results"].as_array().into_iter().flatten().map(|it| stamp(parse_item(it), s, &r)).collect())
        })
    }
}
