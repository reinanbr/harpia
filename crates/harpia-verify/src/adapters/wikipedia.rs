//! Wikipedia through the MediaWiki Action API (`<lang>.wikipedia.org/w/api.php`): search plus
//! plain-text lead sections as evidence. Content is CC BY-SA 4.0 (recorded on every record).

use super::{BoxFuture, SourceAdapter, stamp};
use crate::fetch::{Expect, Fetcher};
use crate::registry::Source;
use crate::work::Work;
use harpia_core::Result;
use harpia_http::Request;
use serde_json::Value;

/// The Wikipedia adapter.
#[derive(Debug, Clone, Copy)]
pub struct Wikipedia;

/// Map the `query.pages` of a generator=search response, in search order.
pub fn parse_pages(v: &Value) -> Vec<Work> {
    let mut pages: Vec<&Value> = v["query"]["pages"].as_array().into_iter().flatten().collect();
    pages.sort_by_key(|p| p["index"].as_i64().unwrap_or(i64::MAX));
    pages
        .into_iter()
        .filter(|p| p.get("missing").is_none())
        .map(|p| Work {
            title: p["title"].as_str().unwrap_or("").to_string(),
            url: p["fullurl"].as_str().unwrap_or("").to_string(),
            text: p["extract"].as_str().unwrap_or("").trim().to_string(),
            venue: "Wikipedia".into(),
            ..Work::default()
        })
        .filter(|w| !w.text.is_empty())
        .collect()
}

impl SourceAdapter for Wikipedia {
    fn name(&self) -> &str {
        "wikipedia"
    }

    fn search<'a>(
        &'a self,
        f: &'a Fetcher,
        s: &'a Source,
        query: &'a str,
        k: usize,
    ) -> BoxFuture<'a, Result<Vec<Work>>> {
        Box::pin(async move {
            let lang = s.lang.as_deref().unwrap_or("en");
            let default = format!("https://{lang}.wikipedia.org");
            let base = s.base_url.as_deref().unwrap_or(&default).trim_end_matches('/');
            let req = Request::get(format!("{base}/w/api.php"))
                .param("action", "query")
                .param("format", "json")
                .param("formatversion", "2")
                .param("generator", "search")
                .param("gsrsearch", query.chars().take(300).collect::<String>())
                .param("gsrlimit", k.min(20).to_string())
                .param("prop", "extracts|info")
                .param("exintro", "1")
                .param("explaintext", "1")
                .param("exlimit", "max")
                .param("inprop", "url");
            let r = f.get(s, req, Expect::Json, "search").await?;
            Ok(parse_pages(&r.json()?).into_iter().map(|w| stamp(w, s, &r)).collect())
        })
    }
}
