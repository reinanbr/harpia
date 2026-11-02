//! Semantic Scholar Graph API (`api.semanticscholar.org`): search and DOI / arXiv lookups. An
//! API key from `SEMANTIC_SCHOLAR_API_KEY` is sent when set (higher rate limits).

use super::{BoxFuture, SourceAdapter, base, encode, encode_doi, stamp, unavailable};
use crate::fetch::{Expect, Fetcher};
use crate::registry::Source;
use crate::work::{Author, Lookup, Work, WorkId, clean_doi, normalize_doi};
use harpia_core::Result;
use harpia_http::Request;
use serde_json::Value;

/// The Semantic Scholar adapter.
#[derive(Debug, Clone, Copy)]
pub struct SemanticScholar;

const API: &str = "https://api.semanticscholar.org/graph/v1";
const FIELDS: &str = "title,year,authors,venue,externalIds,url,abstract,openAccessPdf,isOpenAccess";

/// Map one paper.
pub fn parse_item(it: &Value) -> Work {
    Work {
        title: it["title"].as_str().unwrap_or("").to_string(),
        authors: it["authors"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|a| a["name"].as_str())
            .map(Author::from_display)
            .collect(),
        year: it["year"].as_i64().map(|y| y as i32),
        venue: it["venue"].as_str().unwrap_or("").to_string(),
        doi: normalize_doi(it["externalIds"]["DOI"].as_str().unwrap_or("")),
        url: it["url"].as_str().unwrap_or("").to_string(),
        text: it["abstract"].as_str().unwrap_or("").to_string(),
        pdf_url: it["openAccessPdf"]["url"].as_str().filter(|u| !u.is_empty()).map(String::from),
        ..Work::default()
    }
}

impl SourceAdapter for SemanticScholar {
    fn name(&self) -> &str {
        "semantic-scholar"
    }

    fn search<'a>(
        &'a self,
        f: &'a Fetcher,
        s: &'a Source,
        query: &'a str,
        k: usize,
    ) -> BoxFuture<'a, Result<Vec<Work>>> {
        Box::pin(async move {
            let q: String = query.chars().take(300).collect();
            let req = Request::get(format!("{}/paper/search", base(s, API)))
                .param("query", q)
                .param("limit", k.to_string())
                .param("fields", FIELDS);
            let r = f.get(s, req, Expect::Json, "search").await?;
            let v = r.json()?;
            Ok(v["data"].as_array().into_iter().flatten().map(|it| stamp(parse_item(it), s, &r)).collect())
        })
    }

    fn lookup<'a>(&'a self, f: &'a Fetcher, s: &'a Source, id: &'a WorkId) -> BoxFuture<'a, Lookup> {
        Box::pin(async move {
            let path = match id {
                WorkId::Doi(d) => format!("DOI:{}", encode_doi(&clean_doi(d))),
                WorkId::Arxiv(a) => format!("ARXIV:{}", encode(a.trim_start_matches("arXiv:"))),
                WorkId::Pmid(p) => format!("PMID:{}", encode(p)),
                WorkId::Isbn(_) => return Lookup::Unavailable("semantic scholar has no ISBN lookup".into()),
            };
            let req = Request::get(format!("{}/paper/{path}", base(s, API))).param("fields", FIELDS);
            match f.get(s, req, Expect::Json, "lookup").await {
                Ok(r) if r.status == 404 => Lookup::NotFound,
                Ok(r) if r.is_success() => match r.json() {
                    Ok(v) => Lookup::Found(Box::new(stamp(parse_item(&v), s, &r))),
                    Err(e) => unavailable(e),
                },
                Ok(r) => Lookup::Unavailable(format!("HTTP {}", r.status)),
                Err(e) => unavailable(e),
            }
        })
    }
}
