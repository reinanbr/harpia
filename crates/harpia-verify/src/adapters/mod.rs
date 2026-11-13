//! Source adapters. Each built-in adapter sits behind its own cargo feature and talks to its
//! source's documented API through the [`Fetcher`] only. Implement [`SourceAdapter`] to add a
//! source (including private document stores that never touch the network).

use crate::fetch::Fetcher;
use crate::registry::Source;
use crate::work::{Lookup, Work, WorkId};
use harpia_core::Result;
use std::future::Future;
use std::pin::Pin;

/// A boxed, sendable future.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A trusted source behind a uniform interface.
pub trait SourceAdapter: Send + Sync {
    /// Adapter name, matched against `Source::adapter` in the registry.
    fn name(&self) -> &str;

    /// Search for works relevant to `query` (keywords or a bibliographic string).
    fn search<'a>(
        &'a self,
        fetcher: &'a Fetcher,
        source: &'a Source,
        query: &'a str,
        k: usize,
    ) -> BoxFuture<'a, Result<Vec<Work>>>;

    /// Direct lookup by identifier; unsupported identifiers yield `Lookup::Unavailable`.
    fn lookup<'a>(&'a self, fetcher: &'a Fetcher, source: &'a Source, id: &'a WorkId) -> BoxFuture<'a, Lookup> {
        let _ = (fetcher, source, id);
        Box::pin(async { Lookup::Unavailable(format!("{} does not support identifier lookups", self.name())) })
    }

    /// Whether `search` returns texts usable as claim evidence (abstracts, extracts, pages)
    /// rather than bibliographic metadata only.
    fn provides_text(&self) -> bool {
        true
    }
}

/// Percent-encode a path segment or query value (RFC 3986 unreserved characters kept).
pub fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Percent-encode a DOI for use in a URL path (keeps `/`).
pub fn encode_doi(doi: &str) -> String {
    doi.split('/').map(encode).collect::<Vec<_>>().join("/")
}

#[cfg_attr(
    not(any(
        feature = "crossref",
        feature = "openalex",
        feature = "semantic-scholar",
        feature = "arxiv",
        feature = "europepmc",
        feature = "doaj",
        feature = "wikipedia"
    )),
    allow(dead_code)
)]
pub(crate) fn base<'a>(source: &'a Source, default: &'a str) -> &'a str {
    source.base_url.as_deref().unwrap_or(default).trim_end_matches('/')
}

pub(crate) fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    harpia_data::html::decode_entities(&out).split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg_attr(
    not(any(
        feature = "crossref",
        feature = "openalex",
        feature = "semantic-scholar",
        feature = "arxiv",
        feature = "europepmc",
        feature = "doaj",
        feature = "wikipedia",
        feature = "web"
    )),
    allow(dead_code)
)]
pub(crate) fn stamp(mut w: Work, source: &Source, f: &crate::fetch::Fetched) -> Work {
    w.source = source.id.clone();
    w.tier = source.tier;
    if w.license.is_none() {
        w.license = source.license.clone();
    }
    w.retrieved_at = f.retrieved_at.clone();
    w.content_hash = f.sha256.clone();
    w
}

#[cfg_attr(
    not(any(
        feature = "crossref",
        feature = "openalex",
        feature = "semantic-scholar",
        feature = "arxiv",
        feature = "europepmc"
    )),
    allow(dead_code)
)]
/// Map a fetch error to `Lookup::Unavailable`.
pub(crate) fn unavailable(e: harpia_core::HarpiaError) -> Lookup {
    Lookup::Unavailable(e.to_string())
}

#[cfg(feature = "arxiv")]
pub mod arxiv;
#[cfg(feature = "crossref")]
pub mod crossref;
#[cfg(feature = "doaj")]
pub mod doaj;
#[cfg(feature = "europepmc")]
pub mod europepmc;
#[cfg(feature = "openalex")]
pub mod openalex;
#[cfg(feature = "semantic-scholar")]
pub mod semantic_scholar;
#[cfg(feature = "web")]
pub mod web;
#[cfg(feature = "wikipedia")]
pub mod wikidata;
#[cfg(feature = "wikipedia")]
pub mod wikipedia;

/// All adapters compiled into this build.
pub fn builtin() -> Vec<Box<dyn SourceAdapter>> {
    let adapters: Vec<Box<dyn SourceAdapter>> = vec![
        #[cfg(feature = "crossref")]
        Box::new(crossref::Crossref),
        #[cfg(feature = "openalex")]
        Box::new(openalex::OpenAlex),
        #[cfg(feature = "semantic-scholar")]
        Box::new(semantic_scholar::SemanticScholar),
        #[cfg(feature = "arxiv")]
        Box::new(arxiv::Arxiv),
        #[cfg(feature = "europepmc")]
        Box::new(europepmc::EuropePmc),
        #[cfg(feature = "doaj")]
        Box::new(doaj::Doaj),
        #[cfg(feature = "wikipedia")]
        Box::new(wikipedia::Wikipedia),
        #[cfg(feature = "wikipedia")]
        Box::new(wikidata::Wikidata),
        #[cfg(feature = "web")]
        Box::new(web::GenericAllowlistedWeb),
    ];
    adapters
}
