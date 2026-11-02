//! `GenericAllowlistedWeb`: user-added domains. Pages are fetched over HTTPS from allowed hosts
//! only, subject to `robots.txt`, and reduced to their main text. With a `search_url` template
//! the source's own search page is used to find pages (links must stay on the source's hosts).

use super::{BoxFuture, SourceAdapter, encode, stamp};
use crate::fetch::{Expect, Fetcher, url_parts};
use crate::registry::Source;
use crate::work::Work;
use harpia_core::{HarpiaError, Result};
use harpia_http::Request;

/// The generic web adapter.
#[derive(Debug, Clone, Copy)]
pub struct GenericAllowlistedWeb;

/// Fetch one page of `source` and extract its main text.
pub async fn fetch_page(f: &Fetcher, source: &Source, url: &str) -> Result<Work> {
    let r = f.get(source, Request::get(url), Expect::Html, "fetch page").await?;
    if !r.is_success() {
        return Err(HarpiaError::Remote { host: url_parts(url)?.host, message: format!("HTTP {}", r.status) });
    }
    let page = harpia_data::html::main_text(&r.text());
    Ok(stamp(
        Work { title: page.title.unwrap_or_default(), url: r.url.clone(), text: page.text, ..Work::default() },
        source,
        &r,
    ))
}

impl SourceAdapter for GenericAllowlistedWeb {
    fn name(&self) -> &str {
        "web"
    }

    fn search<'a>(
        &'a self,
        f: &'a Fetcher,
        s: &'a Source,
        query: &'a str,
        k: usize,
    ) -> BoxFuture<'a, Result<Vec<Work>>> {
        Box::pin(async move {
            let Some(template) = &s.search_url else {
                return Err(HarpiaError::invalid(format!(
                    "web source `{}` has no search_url; fetch pages by URL instead",
                    s.id
                )));
            };
            let url = template.replace("{query}", &encode(query));
            let r = f.get(s, Request::get(&url), Expect::Html, "search page").await?;
            let mut out = Vec::new();
            for link in harpia_data::html::links(&r.text(), &r.url) {
                if out.len() >= k {
                    break;
                }
                let same_page = link.split('?').next() == url.split('?').next();
                let allowed = url_parts(&link).is_ok_and(|p| s.allows_host(&p.host));
                if same_page || !allowed {
                    continue;
                }
                if let Ok(w) = fetch_page(f, s, &link).await
                    && !w.text.is_empty()
                {
                    out.push(w);
                }
            }
            Ok(out)
        })
    }
}
