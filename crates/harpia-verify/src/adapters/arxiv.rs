//! arXiv export API (`export.arxiv.org`, Atom). DTDs and entity declarations are refused
//! (untrusted input); only the handful of Atom elements needed are read.

use super::{BoxFuture, SourceAdapter, base, stamp, unavailable};
use crate::fetch::{Expect, Fetcher};
use crate::registry::Source;
use crate::work::{Author, Lookup, Work, WorkId};
use harpia_core::Result;
use harpia_data::html::decode_entities;
use harpia_http::Request;

/// The arXiv adapter.
#[derive(Debug, Clone, Copy)]
pub struct Arxiv;

const API: &str = "https://export.arxiv.org/api/query";

fn elements<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(p) = rest.find(&open) {
        let after = &rest[p + open.len()..];
        // Must be the whole tag name (e.g. `<title` but not `<titles`).
        if !after.starts_with(['>', ' ', '\t', '\n', '\r']) {
            rest = after;
            continue;
        }
        let Some(gt) = after.find('>') else { break };
        let body = &after[gt + 1..];
        let Some(end) = body.find(&close) else { break };
        out.push(&body[..end]);
        rest = &body[end + close.len()..];
    }
    out
}

fn text(xml: &str, tag: &str) -> String {
    elements(xml, tag)
        .first()
        .map(|t| decode_entities(t).split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

/// Parse an Atom feed from the arXiv API.
pub fn parse_feed(body: &str) -> Vec<Work> {
    if body.contains("<!DOCTYPE") || body.contains("<!ENTITY") {
        return Vec::new();
    }
    elements(body, "entry")
        .into_iter()
        .filter_map(|e| {
            let title = text(e, "title");
            if title.is_empty() || title == "Error" {
                return None;
            }
            let published = text(e, "published");
            let id = text(e, "id");
            let pdf = id.replace("/abs/", "/pdf/");
            Some(Work {
                title,
                authors: elements(e, "author")
                    .iter()
                    .map(|a| Author::from_display(&text(a, "name")))
                    .filter(|a| !a.family.is_empty())
                    .collect(),
                year: published.get(..4).and_then(|y| y.parse().ok()),
                venue: "arXiv".into(),
                doi: text(e, "arxiv:doi").to_ascii_lowercase(),
                text: text(e, "summary"),
                pdf_url: (!id.is_empty()).then_some(pdf),
                url: id,
                ..Work::default()
            })
        })
        .collect()
}

impl SourceAdapter for Arxiv {
    fn name(&self) -> &str {
        "arxiv"
    }

    fn search<'a>(
        &'a self,
        f: &'a Fetcher,
        s: &'a Source,
        query: &'a str,
        k: usize,
    ) -> BoxFuture<'a, Result<Vec<Work>>> {
        Box::pin(async move {
            let terms: Vec<String> = query.split_whitespace().take(12).map(|w| format!("all:{w}")).collect();
            let req = Request::get(base(s, API))
                .param("search_query", terms.join(" AND "))
                .param("max_results", k.to_string());
            let r = f.get(s, req, Expect::Xml, "search").await?;
            Ok(parse_feed(&r.text()).into_iter().map(|w| stamp(w, s, &r)).collect())
        })
    }

    fn lookup<'a>(&'a self, f: &'a Fetcher, s: &'a Source, id: &'a WorkId) -> BoxFuture<'a, Lookup> {
        Box::pin(async move {
            let WorkId::Arxiv(a) = id else { return Lookup::Unavailable("arxiv looks up arXiv ids only".into()) };
            let req = Request::get(base(s, API)).param("id_list", a.trim().trim_start_matches("arXiv:"));
            match f.get(s, req, Expect::Xml, "lookup: arXiv id").await {
                Ok(r) if r.is_success() => match parse_feed(&r.text()).into_iter().next() {
                    Some(w) => Lookup::Found(Box::new(stamp(w, s, &r))),
                    None => Lookup::NotFound,
                },
                Ok(r) => Lookup::Unavailable(format!("HTTP {}", r.status)),
                Err(e) => unavailable(e),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_atom_and_refuses_dtd() {
        let feed = r#"<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1706.03762v7</id>
<published>2017-06-12T17:57:34Z</published><title>Attention Is All
  You Need</title><summary>The dominant &amp; sequence models...</summary>
<author><name>Ashish Vaswani</name></author><author><name>Noam Shazeer</name></author></entry></feed>"#;
        let w = &parse_feed(feed)[0];
        assert_eq!(w.title, "Attention Is All You Need");
        assert_eq!(w.year, Some(2017));
        assert_eq!(w.authors[1].family, "Shazeer");
        assert_eq!(w.text, "The dominant & sequence models...");
        assert_eq!(w.pdf_url.as_deref(), Some("http://arxiv.org/pdf/1706.03762v7"));
        assert!(parse_feed(&format!("<!DOCTYPE x [<!ENTITY a 'b'>]>{feed}")).is_empty());
    }
}
