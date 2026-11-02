//! OpenAlex (`api.openalex.org`): DOI lookups and search, abstracts from the inverted index,
//! retraction flags and open-access PDF locations.

use super::{BoxFuture, SourceAdapter, base, encode_doi, stamp, unavailable};
use crate::fetch::{Expect, Fetcher};
use crate::registry::Source;
use crate::work::{Author, Lookup, Work, WorkId, clean_doi, normalize_doi};
use harpia_core::Result;
use harpia_http::Request;
use serde_json::Value;

/// The OpenAlex adapter.
#[derive(Debug, Clone, Copy)]
pub struct OpenAlex;

const API: &str = "https://api.openalex.org";

/// Rebuild an abstract from OpenAlex's `abstract_inverted_index`.
pub fn abstract_from_index(inv: &Value) -> String {
    let Some(map) = inv.as_object() else { return String::new() };
    let mut pos: Vec<(u64, &str)> = Vec::new();
    for (word, idxs) in map {
        for i in idxs.as_array().into_iter().flatten().filter_map(Value::as_u64) {
            pos.push((i, word));
        }
    }
    pos.sort_by_key(|p| p.0);
    pos.into_iter().map(|p| p.1).collect::<Vec<_>>().join(" ")
}

/// Map one OpenAlex work.
pub fn parse_item(it: &Value) -> Work {
    let authors = it["authorships"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|a| a["author"]["display_name"].as_str())
        .map(Author::from_display)
        .collect();
    let doi = normalize_doi(it["doi"].as_str().unwrap_or(""));
    let pdf =
        it["best_oa_location"]["pdf_url"].as_str().or_else(|| it["open_access"]["oa_url"].as_str()).map(String::from);
    let license = it["best_oa_location"]["license"]
        .as_str()
        .or_else(|| it["primary_location"]["license"].as_str())
        .map(String::from);
    Work {
        title: it["title"].as_str().or_else(|| it["display_name"].as_str()).unwrap_or("").to_string(),
        authors,
        year: it["publication_year"].as_i64().map(|y| y as i32),
        venue: it["primary_location"]["source"]["display_name"].as_str().unwrap_or("").to_string(),
        url: if doi.is_empty() {
            it["id"].as_str().unwrap_or("").to_string()
        } else {
            format!("https://doi.org/{doi}")
        },
        doi,
        text: abstract_from_index(&it["abstract_inverted_index"]),
        retracted: it.get("is_retracted").and_then(Value::as_bool),
        pdf_url: pdf,
        license,
        ..Work::default()
    }
}

impl SourceAdapter for OpenAlex {
    fn name(&self) -> &str {
        "openalex"
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
            let mut req =
                Request::get(format!("{}/works", base(s, API))).param("per-page", k.to_string()).param("search", q);
            if let Some(m) = s.contact() {
                req = req.param("mailto", m);
            }
            let r = f.get(s, req, Expect::Json, "search").await?;
            let v = r.json()?;
            Ok(v["results"].as_array().into_iter().flatten().map(|it| stamp(parse_item(it), s, &r)).collect())
        })
    }

    fn lookup<'a>(&'a self, f: &'a Fetcher, s: &'a Source, id: &'a WorkId) -> BoxFuture<'a, Lookup> {
        Box::pin(async move {
            let WorkId::Doi(doi) = id else { return Lookup::Unavailable("openalex looks up DOIs only".into()) };
            let mut req = Request::get(format!("{}/works/doi:{}", base(s, API), encode_doi(&clean_doi(doi))));
            if let Some(m) = s.contact() {
                req = req.param("mailto", m);
            }
            match f.get(s, req, Expect::Json, "lookup: DOI").await {
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
