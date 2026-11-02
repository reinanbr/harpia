//! Europe PMC REST API (`www.ebi.ac.uk/europepmc`): life-science literature search with
//! abstracts, DOI / PMID lookups and open-access full-text links.

use super::{BoxFuture, SourceAdapter, base, stamp, strip_tags, unavailable};
use crate::fetch::{Expect, Fetcher};
use crate::registry::Source;
use crate::work::{Author, Lookup, Work, WorkId, normalize_doi};
use harpia_core::Result;
use harpia_http::Request;
use serde_json::Value;

/// The Europe PMC adapter.
#[derive(Debug, Clone, Copy)]
pub struct EuropePmc;

const API: &str = "https://www.ebi.ac.uk/europepmc/webservices/rest";

/// Map one result (`resultType=core`).
pub fn parse_item(it: &Value) -> Work {
    let authors = match it["authorList"]["author"].as_array() {
        Some(a) => a
            .iter()
            .filter_map(|x| {
                let family = x["lastName"].as_str()?.to_string();
                Some(Author { full: x["fullName"].as_str().unwrap_or(&family).to_string(), family })
            })
            .collect(),
        None => it["authorString"]
            .as_str()
            .unwrap_or("")
            .split(',')
            .map(|a| a.trim().trim_end_matches('.'))
            .filter(|a| !a.is_empty())
            .map(|a| Author { family: a.split_whitespace().next().unwrap_or("").to_string(), full: a.to_string() })
            .collect(),
    };
    let pdf = it["fullTextUrlList"]["fullTextUrl"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|u| u["documentStyle"].as_str() == Some("pdf") && u["availabilityCode"].as_str() == Some("OA"))
        .and_then(|u| u["url"].as_str())
        .map(String::from);
    let doi = normalize_doi(it["doi"].as_str().unwrap_or(""));
    let id = it["id"].as_str().unwrap_or("");
    let src = it["source"].as_str().unwrap_or("MED");
    Work {
        title: strip_tags(it["title"].as_str().unwrap_or("")),
        authors,
        year: it["pubYear"].as_str().and_then(|y| y.parse().ok()),
        venue: it["journalTitle"]
            .as_str()
            .or_else(|| it["journalInfo"]["journal"]["title"].as_str())
            .unwrap_or("")
            .to_string(),
        url: if doi.is_empty() {
            format!("https://europepmc.org/article/{src}/{id}")
        } else {
            format!("https://doi.org/{doi}")
        },
        doi,
        text: strip_tags(it["abstractText"].as_str().unwrap_or("")),
        retracted: it["pubTypeList"]["pubType"]
            .as_array()
            .map(|t| t.iter().any(|x| x.as_str().is_some_and(|s| s.to_ascii_lowercase().contains("retract")))),
        pdf_url: pdf,
        license: it["license"].as_str().map(String::from),
        ..Work::default()
    }
}

async fn query(
    f: &Fetcher,
    s: &Source,
    q: &str,
    k: usize,
    purpose: &str,
) -> Result<(crate::fetch::Fetched, Vec<Work>)> {
    let req = Request::get(format!("{}/search", base(s, API)))
        .param("query", q)
        .param("format", "json")
        .param("resultType", "core")
        .param("pageSize", k.to_string());
    let r = f.get(s, req, Expect::Json, purpose).await?;
    let v = r.json()?;
    let works =
        v["resultList"]["result"].as_array().into_iter().flatten().map(|it| stamp(parse_item(it), s, &r)).collect();
    Ok((r, works))
}

impl SourceAdapter for EuropePmc {
    fn name(&self) -> &str {
        "europepmc"
    }

    fn search<'a>(&'a self, f: &'a Fetcher, s: &'a Source, q: &'a str, k: usize) -> BoxFuture<'a, Result<Vec<Work>>> {
        Box::pin(async move { Ok(query(f, s, &q.chars().take(300).collect::<String>(), k, "search").await?.1) })
    }

    fn lookup<'a>(&'a self, f: &'a Fetcher, s: &'a Source, id: &'a WorkId) -> BoxFuture<'a, Lookup> {
        Box::pin(async move {
            let q = match id {
                WorkId::Doi(d) => format!("DOI:\"{}\"", crate::work::clean_doi(d)),
                WorkId::Pmid(p) => format!("EXT_ID:{p} AND SRC:MED"),
                _ => return Lookup::Unavailable("europepmc looks up DOIs and PMIDs only".into()),
            };
            match query(f, s, &q, 1, "lookup").await {
                Ok((_, w)) => w.into_iter().next().map_or(Lookup::NotFound, |w| Lookup::Found(Box::new(w))),
                Err(e) => unavailable(e),
            }
        })
    }
}
