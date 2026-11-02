//! Trusted sources as a retrieval backend for `harpia_data::rag::RagPipeline`: search the
//! registry's text sources and turn the results into [`Document`]s that keep the source URL,
//! license, tier, retrieval time and content hash, so assembled contexts cite them.

use crate::adapters::SourceAdapter;
use crate::claims::{QueryMode, build_query};
use crate::fetch::{Disclosure, Fetcher};
use harpia_core::{Document, Result};
use harpia_text::lang::LangData;
use serde_json::json;

/// Search trusted sources for `query` and return documents (deduplicated by URL), plus the
/// disclosure log. Offline (and not frozen), nothing is fetched and the list is empty.
pub async fn trusted_documents(
    fetcher: &Fetcher,
    adapters: &[Box<dyn SourceAdapter>],
    query: &str,
    mode: QueryMode,
    lang: &LangData,
    per_source: usize,
) -> Result<(Vec<Document>, Disclosure)> {
    let mut docs: Vec<Document> = Vec::new();
    if fetcher.skip_reason().is_some() {
        return Ok((docs, fetcher.disclosure()));
    }
    let q = build_query(query, mode, lang, 10);
    for s in fetcher.registry().sources.iter().filter(|s| s.enabled && s.unavailable_reason().is_none()) {
        let Some(a) = adapters.iter().find(|a| a.name() == s.adapter && a.provides_text()) else { continue };
        let Ok(works) = a.search(fetcher, s, &q, per_source).await else { continue };
        for w in works.into_iter().filter(|w| !w.text.trim().is_empty()) {
            if docs.iter().any(|d| d.source.as_deref() == Some(w.url.as_str())) {
                continue;
            }
            let mut d =
                Document::new(format!("{}:{}", w.source, if w.url.is_empty() { &w.title } else { &w.url }), w.text);
            d.source = Some(w.url.clone());
            d.license = w.license.clone();
            d.metadata.insert("source_id".into(), json!(w.source));
            d.metadata.insert("tier".into(), json!(w.tier));
            d.metadata.insert("title".into(), json!(w.title));
            d.metadata.insert("retrieved_at".into(), json!(w.retrieved_at));
            d.metadata.insert("content_hash".into(), json!(w.content_hash));
            docs.push(d);
        }
    }
    Ok((docs, fetcher.disclosure()))
}
