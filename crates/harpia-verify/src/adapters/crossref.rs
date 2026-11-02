//! Crossref REST API (`api.crossref.org`): DOI lookups and bibliographic search, with
//! retraction/correction notices from `update-to` / `updated-by`.

use super::{BoxFuture, SourceAdapter, base, encode_doi, stamp, strip_tags, unavailable};
use crate::fetch::{Expect, Fetcher};
use crate::registry::Source;
use crate::work::{Author, Lookup, Work, WorkId, clean_doi, normalize_doi};
use harpia_core::Result;
use harpia_http::Request;
use serde_json::Value;

/// The Crossref adapter.
#[derive(Debug, Clone, Copy)]
pub struct Crossref;

const API: &str = "https://api.crossref.org";

fn first(v: &Value) -> String {
    match v {
        Value::Array(a) => a.first().and_then(Value::as_str).unwrap_or("").to_string(),
        Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

/// Map one Crossref `message` item.
pub fn parse_item(it: &Value) -> Work {
    let year = ["issued", "published-print", "published-online", "created"]
        .iter()
        .find_map(|k| it.get(*k)?.get("date-parts")?.get(0)?.get(0)?.as_i64().map(|y| y as i32));
    let authors = it
        .get("author")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    let family = x.get("family").or_else(|| x.get("name")).and_then(Value::as_str)?.to_string();
                    let given = x.get("given").and_then(Value::as_str).unwrap_or("");
                    Some(Author { full: format!("{given} {family}").trim().to_string(), family })
                })
                .collect()
        })
        .unwrap_or_default();
    let mut retracted = None;
    let mut corrections = Vec::new();
    for key in ["updated-by", "update-to"] {
        for u in it.get(key).and_then(Value::as_array).into_iter().flatten() {
            let t = u.get("type").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase();
            if matches!(t.as_str(), "retraction" | "removal" | "withdrawal") {
                retracted = Some(true);
            } else if !t.is_empty() {
                let doi = u.get("DOI").and_then(Value::as_str).unwrap_or("");
                corrections.push(format!("{t} {doi}").trim().to_string());
            }
        }
    }
    if it.get("type").and_then(Value::as_str) == Some("retraction") {
        retracted = Some(true);
    }
    let doi = normalize_doi(it.get("DOI").and_then(Value::as_str).unwrap_or(""));
    let license = it
        .get("license")
        .and_then(Value::as_array)
        .and_then(|l| l.first())
        .and_then(|l| l.get("URL"))
        .and_then(Value::as_str)
        .map(String::from);
    Work {
        title: strip_tags(&first(it.get("title").unwrap_or(&Value::Null))),
        authors,
        year,
        venue: first(it.get("container-title").unwrap_or(&Value::Null)),
        url: if doi.is_empty() {
            it.get("URL").and_then(Value::as_str).unwrap_or("").to_string()
        } else {
            format!("https://doi.org/{doi}")
        },
        doi,
        text: strip_tags(it.get("abstract").and_then(Value::as_str).unwrap_or("")),
        retracted,
        corrections,
        license,
        ..Work::default()
    }
}

impl SourceAdapter for Crossref {
    fn name(&self) -> &str {
        "crossref"
    }

    fn provides_text(&self) -> bool {
        false
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
            let mut req = Request::get(format!("{}/works", base(s, API)))
                .param("query.bibliographic", q)
                .param("rows", k.to_string())
                .param(
                    "select",
                    "DOI,title,author,issued,container-title,type,update-to,updated-by,URL,abstract,license",
                );
            if let Some(m) = s.contact() {
                req = req.param("mailto", m);
            }
            let r = f.get(s, req, Expect::Json, "search: bibliographic").await?;
            let v = r.json()?;
            Ok(v["message"]["items"].as_array().into_iter().flatten().map(|it| stamp(parse_item(it), s, &r)).collect())
        })
    }

    fn lookup<'a>(&'a self, f: &'a Fetcher, s: &'a Source, id: &'a WorkId) -> BoxFuture<'a, Lookup> {
        Box::pin(async move {
            let WorkId::Doi(doi) = id else { return Lookup::Unavailable("crossref looks up DOIs only".into()) };
            let mut req = Request::get(format!("{}/works/{}", base(s, API), encode_doi(&clean_doi(doi))));
            if let Some(m) = s.contact() {
                req = req.param("mailto", m);
            }
            match f.get(s, req, Expect::Json, "lookup: DOI").await {
                Ok(r) if r.status == 404 => Lookup::NotFound,
                Ok(r) if r.is_success() => match r.json() {
                    Ok(v) if v.get("message").is_some() => {
                        Lookup::Found(Box::new(stamp(parse_item(&v["message"]), s, &r)))
                    }
                    _ => Lookup::Unavailable("malformed Crossref response".into()),
                },
                Ok(r) => Lookup::Unavailable(format!("HTTP {}", r.status)),
                Err(e) => unavailable(e),
            }
        })
    }
}
