//! Retrieval-augmented generation: a chunk store with dense (flat or HNSW) and BM25 retrieval,
//! hybrid fusion, optional cross-encoder reranking, metadata filters, persistence, and context
//! assembly within a token budget with citations back to document spans.
//!
//! ```
//! use harpia_core::{Document, traits::Embedder, Result};
//! use harpia_data::chunk::SentenceChunker;
//! use harpia_text::lang::Lang;
//! use harpia_data::rag::{RagConfig, RagPipeline};
//!
//! // A toy embedder: letter histogram.
//! struct Letters;
//! impl Embedder for Letters {
//!     fn dim(&self) -> usize { 26 }
//!     fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
//!         Ok(texts.iter().map(|t| {
//!             let mut v = vec![0.0; 26];
//!             for c in t.to_ascii_lowercase().bytes().filter(u8::is_ascii_lowercase) { v[(c - b'a') as usize] += 1.0; }
//!             v
//!         }).collect())
//!     }
//! }
//!
//! let chunker = SentenceChunker { max_chars: 200, overlap: 0, lang: Lang::En.data() };
//! let mut rag = RagPipeline::new(Box::new(chunker), &Letters, RagConfig::default());
//! rag.add_documents(&[Document::new("birds", "The harpy eagle hunts sloths. Owls hunt mice at night.")]).unwrap();
//! let ctx = rag.context("What does the harpy eagle hunt?", 50, &|t: &str| t.split_whitespace().count(), None).unwrap();
//! assert!(ctx.text.contains("[1]"));
//! assert_eq!(ctx.citations[0].doc_id, "birds");
//! ```

use crate::vector::{FlatIndex, HnswIndex, HnswParams, Metric, VectorIndex};
use harpia_core::traits::{Chunker, Embedder, Hit, Reranker, Retriever};
use harpia_core::{Document, HarpiaError, Result, Span};
use harpia_text::bm25::{Bm25, Bm25Params};
use harpia_text::lang::{Lang, LangData};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, OnceLock};

/// A stored chunk with provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkRecord {
    /// Chunk id (`<doc_id>#<n>`).
    pub id: String,
    /// Source document id.
    pub doc_id: String,
    /// Chunk text.
    pub text: String,
    /// Byte span in the source document.
    pub span: Span,
    /// Heading path (Markdown chunking).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headings: Vec<String>,
    /// Document source (path, URL).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Document license.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Document metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Metadata filter applied before ranking.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MetadataFilter {
    /// Field must equal the value.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub equals: BTreeMap<String, Value>,
    /// Field must equal one of the values.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub any_of: BTreeMap<String, Vec<Value>>,
    /// Restrict to these documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_ids: Option<Vec<String>>,
}

impl MetadataFilter {
    /// Whether `c` passes. `source` and `license` are matched as metadata fields too.
    pub fn matches(&self, c: &ChunkRecord) -> bool {
        let field = |k: &str| -> Option<Value> {
            match k {
                "source" => c.source.clone().map(Value::String),
                "license" => c.license.clone().map(Value::String),
                "doc_id" => Some(Value::String(c.doc_id.clone())),
                _ => c.metadata.get(k).cloned(),
            }
        };
        self.doc_ids.as_ref().is_none_or(|ids| ids.contains(&c.doc_id))
            && self.equals.iter().all(|(k, v)| field(k).as_ref() == Some(v))
            && self.any_of.iter().all(|(k, vs)| field(k).is_some_and(|f| vs.contains(&f)))
    }
}

/// Reciprocal rank fusion: `score = Σ 1 / (k + rank)` over the input lists (rank from 1).
///
/// ```
/// use harpia_core::traits::Hit;
/// let h = |id: &str| Hit { id: id.into(), score: 0.0, span: None };
/// let fused = harpia_data::rag::fuse_rrf(&[vec![h("a"), h("b")], vec![h("b"), h("c")]], 60.0);
/// assert_eq!(fused[0].id, "b");
/// ```
pub fn fuse_rrf(lists: &[Vec<Hit>], k: f32) -> Vec<Hit> {
    let mut acc: Vec<(String, f32, usize)> = Vec::new();
    let mut pos: HashMap<String, usize> = HashMap::new();
    for list in lists {
        for (rank, h) in list.iter().enumerate() {
            let s = 1.0 / (k + rank as f32 + 1.0);
            match pos.get(&h.id) {
                Some(&i) => acc[i].1 += s,
                None => {
                    pos.insert(h.id.clone(), acc.len());
                    acc.push((h.id.clone(), s, acc.len()));
                }
            }
        }
    }
    acc.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.2.cmp(&b.2)));
    acc.into_iter().map(|(id, score, _)| Hit { id, score, span: None }).collect()
}

/// Weighted fusion of min-max normalized scores (`weights[i]` for `lists[i]`).
pub fn fuse_weighted(lists: &[Vec<Hit>], weights: &[f32]) -> Vec<Hit> {
    let mut acc: Vec<(String, f32, usize)> = Vec::new();
    let mut pos: HashMap<String, usize> = HashMap::new();
    for (list, &w) in lists.iter().zip(weights) {
        let (lo, hi) =
            list.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), h| (lo.min(h.score), hi.max(h.score)));
        for h in list {
            let n = if hi > lo { (h.score - lo) / (hi - lo) } else { 1.0 };
            match pos.get(&h.id) {
                Some(&i) => acc[i].1 += w * n,
                None => {
                    pos.insert(h.id.clone(), acc.len());
                    acc.push((h.id.clone(), w * n, acc.len()));
                }
            }
        }
    }
    acc.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.2.cmp(&b.2)));
    acc.into_iter().map(|(id, score, _)| Hit { id, score, span: None }).collect()
}

/// Which retrievers to use and how to combine them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Fusion {
    /// Dense vectors only.
    Dense,
    /// BM25 only.
    Lexical,
    /// Reciprocal rank fusion of both.
    Rrf {
        /// RRF constant (60 is the usual choice).
        k: f32,
    },
    /// Weighted min-max fusion.
    Weighted {
        /// Weight of the dense scores (BM25 gets `1 - dense`).
        dense: f32,
    },
}

/// Vector index choice.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum IndexKind {
    /// Exact search.
    Flat,
    /// HNSW.
    Hnsw(HnswParams),
}

/// Pipeline configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RagConfig {
    /// Vector index.
    pub index: IndexKind,
    /// Similarity.
    pub metric: Metric,
    /// Fusion strategy.
    pub fusion: Fusion,
    /// Candidates fetched from each retriever before fusion/reranking.
    pub candidates: usize,
    /// Hits returned by `retrieve`.
    pub top_k: usize,
    /// Texts per embedding call.
    pub embed_batch: usize,
    /// Language for BM25 tokenization and stopwords (`"en"` or `"pt-br"`).
    pub lang: String,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            index: IndexKind::Flat,
            metric: Metric::Cosine,
            fusion: Fusion::Rrf { k: 60.0 },
            candidates: 50,
            top_k: 5,
            embed_batch: 32,
            lang: "en".into(),
        }
    }
}

/// A retrieved chunk with its scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Retrieved {
    /// The chunk.
    pub chunk: ChunkRecord,
    /// Final score (fusion or reranker).
    pub score: f32,
    /// Dense score, if retrieved densely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense: Option<f32>,
    /// BM25 score, if retrieved lexically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical: Option<f32>,
    /// Reranker score, if reranked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<f32>,
}

/// One citation in an assembled context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    /// Marker number (`[n]` in the context text).
    pub n: usize,
    /// Chunk id.
    pub chunk_id: String,
    /// Document id.
    pub doc_id: String,
    /// Byte span in the document.
    pub span: Span,
    /// Document source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Document license.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

/// Context text for a prompt plus its citations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Context {
    /// `[1] chunk text\n\n[2] ...`
    pub text: String,
    /// Citations in marker order.
    pub citations: Vec<Citation>,
    /// Tokens used (per the supplied counter).
    pub tokens: usize,
    /// Retrieved chunks that did not fit the budget.
    pub dropped: usize,
}

/// Chunk store with dense and lexical indexes.
#[derive(Debug)]
struct Store {
    chunks: Vec<ChunkRecord>,
    live: Vec<bool>,
    by_id: HashMap<String, usize>,
    vectors: VectorIndex,
    bm25: OnceLock<(Bm25, Vec<usize>)>,
}

/// The RAG pipeline: chunk → embed → index → retrieve → rerank → assemble.
pub struct RagPipeline<'a> {
    chunker: Box<dyn Chunker + 'a>,
    embedder: &'a dyn Embedder,
    reranker: Option<&'a dyn Reranker>,
    config: RagConfig,
    lang: Arc<LangData>,
    store: Store,
}

impl std::fmt::Debug for RagPipeline<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagPipeline").field("config", &self.config).field("chunks", &self.len()).finish_non_exhaustive()
    }
}

fn new_index(cfg: &RagConfig, dim: usize) -> VectorIndex {
    match cfg.index {
        IndexKind::Flat => VectorIndex::Flat(FlatIndex::new(dim, cfg.metric)),
        IndexKind::Hnsw(p) => VectorIndex::Hnsw(Box::new(HnswIndex::new(dim, cfg.metric, p))),
    }
}

fn lang_data(code: &str) -> Result<Arc<LangData>> {
    Lang::parse(code).map(Lang::data).ok_or_else(|| HarpiaError::invalid(format!("unsupported language {code:?}")))
}

impl<'a> RagPipeline<'a> {
    /// New empty pipeline. Panics if `config.lang` is not a supported language code.
    pub fn new(chunker: Box<dyn Chunker + 'a>, embedder: &'a dyn Embedder, config: RagConfig) -> Self {
        let lang = lang_data(&config.lang).expect("supported RAG language");
        let vectors = new_index(&config, embedder.dim());
        Self {
            chunker,
            embedder,
            reranker: None,
            lang,
            store: Store {
                chunks: Vec::new(),
                live: Vec::new(),
                by_id: HashMap::new(),
                vectors,
                bm25: OnceLock::new(),
            },
            config,
        }
    }

    /// Rerank fused candidates with a cross-encoder.
    pub fn with_reranker(mut self, reranker: &'a dyn Reranker) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Configuration.
    pub fn config(&self) -> &RagConfig {
        &self.config
    }

    /// Live chunks.
    pub fn len(&self) -> usize {
        self.store.by_id.len()
    }

    /// `true` when nothing is indexed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Live chunk by id.
    pub fn chunk(&self, id: &str) -> Option<&ChunkRecord> {
        self.store.by_id.get(id).map(|&i| &self.store.chunks[i])
    }

    /// Chunk, embed and index documents (re-adding a document id replaces its chunks).
    /// Returns the number of chunks added.
    pub fn add_documents(&mut self, docs: &[Document]) -> Result<usize> {
        let mut records = Vec::new();
        for d in docs {
            self.remove_document(&d.id);
            for (n, c) in self.chunker.chunk(&d.text)?.into_iter().enumerate() {
                let text = c.span.slice(&d.text).to_string();
                if text.trim().is_empty() {
                    continue;
                }
                records.push(ChunkRecord {
                    id: format!("{}#{n}", d.id),
                    doc_id: d.id.clone(),
                    text,
                    span: c.span,
                    headings: c.headings,
                    source: d.source.clone(),
                    license: d.license.clone(),
                    metadata: d.metadata.clone(),
                });
            }
        }
        for batch in records.chunks(self.config.embed_batch.max(1)) {
            let texts: Vec<&str> = batch.iter().map(|r| r.text.as_str()).collect();
            let vecs = self.embedder.embed(&texts)?;
            if vecs.len() != batch.len() {
                return Err(HarpiaError::invalid("embedder returned a different number of vectors"));
            }
            for (r, v) in batch.iter().zip(vecs) {
                self.store.vectors.add(&r.id, &v)?;
            }
        }
        let added = records.len();
        for r in records {
            self.store.by_id.insert(r.id.clone(), self.store.chunks.len());
            self.store.chunks.push(r);
            self.store.live.push(true);
        }
        self.store.bm25 = OnceLock::new();
        Ok(added)
    }

    /// Remove every chunk of a document; returns how many were removed.
    pub fn remove_document(&mut self, doc_id: &str) -> usize {
        let rows: Vec<usize> = (0..self.store.chunks.len())
            .filter(|&i| self.store.live[i] && self.store.chunks[i].doc_id == doc_id)
            .collect();
        for &i in &rows {
            self.store.live[i] = false;
            self.store.by_id.remove(&self.store.chunks[i].id);
            self.store.vectors.remove(&self.store.chunks[i].id);
        }
        if !rows.is_empty() {
            self.store.bm25 = OnceLock::new();
        }
        rows.len()
    }

    fn bm25(&self) -> &(Bm25, Vec<usize>) {
        self.store.bm25.get_or_init(|| {
            let rows: Vec<usize> = (0..self.store.chunks.len()).filter(|&i| self.store.live[i]).collect();
            let bm =
                Bm25::new(rows.iter().map(|&i| self.store.chunks[i].text.as_str()), &self.lang, Bm25Params::default());
            (bm, rows)
        })
    }

    /// Ranked chunks for `query` (the configured `top_k`).
    pub fn retrieve(&self, query: &str, filter: Option<&MetadataFilter>) -> Result<Vec<Retrieved>> {
        self.retrieve_top(query, self.config.top_k, filter)
    }

    /// The `k` best chunks for `query`, overriding the configured `top_k` (candidates are raised
    /// to at least `k`).
    pub fn retrieve_top(&self, query: &str, k: usize, filter: Option<&MetadataFilter>) -> Result<Vec<Retrieved>> {
        let n = self.config.candidates.max(k);
        let chunks = &self.store.chunks;
        let by_id = &self.store.by_id;
        let pass = |id: &str| filter.is_none_or(|f| by_id.get(id).is_some_and(|&i| f.matches(&chunks[i])));
        let dense = if matches!(self.config.fusion, Fusion::Lexical) {
            Vec::new()
        } else {
            let q = self
                .embedder
                .embed(&[query])?
                .pop()
                .ok_or_else(|| HarpiaError::invalid("embedder returned no vector"))?;
            self.store.vectors.search(&q, n, Some(&pass))?
        };
        let mut lexical: Vec<Hit> = if matches!(self.config.fusion, Fusion::Dense) {
            Vec::new()
        } else {
            let (bm, rows) = self.bm25();
            bm.scores(query)
                .into_iter()
                .enumerate()
                .filter(|&(i, s)| s > 0.0 && pass(&chunks[rows[i]].id))
                .map(|(i, s)| Hit { id: chunks[rows[i]].id.clone(), score: s as f32, span: Some(chunks[rows[i]].span) })
                .collect()
        };
        lexical.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| by_id[&a.id].cmp(&by_id[&b.id])));
        lexical.truncate(n);
        let fused = match self.config.fusion {
            Fusion::Dense => dense.clone(),
            Fusion::Lexical => lexical.clone(),
            Fusion::Rrf { k } => fuse_rrf(&[dense.clone(), lexical.clone()], k),
            Fusion::Weighted { dense: w } => fuse_weighted(&[dense.clone(), lexical.clone()], &[w, 1.0 - w]),
        };
        let score_of = |list: &[Hit], id: &str| list.iter().find(|h| h.id == id).map(|h| h.score);
        let mut out: Vec<Retrieved> = fused
            .into_iter()
            .take(n)
            .map(|h| Retrieved {
                chunk: chunks[by_id[&h.id]].clone(),
                score: h.score,
                dense: score_of(&dense, &h.id),
                lexical: score_of(&lexical, &h.id),
                rerank: None,
            })
            .collect();
        if let Some(rr) = self.reranker {
            let docs: Vec<&str> = out.iter().map(|r| r.chunk.text.as_str()).collect();
            let scores = rr.rerank(query, &docs)?;
            for (r, s) in out.iter_mut().zip(scores) {
                r.rerank = Some(s);
                r.score = s;
            }
            // Stable sort keeps fusion order among equal reranker scores.
            out.sort_by(|a, b| b.score.total_cmp(&a.score));
        }
        out.truncate(k);
        Ok(out)
    }

    /// Retrieve and assemble a context of at most `budget` tokens (per `count_tokens`), each
    /// chunk prefixed with a citation marker. Chunks that do not fit are skipped (smaller later
    /// ones may still fit).
    pub fn context(
        &self,
        query: &str,
        budget: usize,
        count_tokens: &dyn Fn(&str) -> usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Context> {
        Ok(assemble(&self.retrieve(query, filter)?, budget, count_tokens))
    }

    /// Save to a directory: `config.json`, `chunks.jsonl` (live chunks), `vectors.hvec`.
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| HarpiaError::io(dir, e))?;
        let cfg = dir.join("config.json");
        std::fs::write(&cfg, serde_json::to_vec_pretty(&self.config)?).map_err(|e| HarpiaError::io(&cfg, e))?;
        let mut w = crate::io::JsonlWriter::create(dir.join("chunks.jsonl"))?;
        for (i, c) in self.store.chunks.iter().enumerate() {
            if self.store.live[i] {
                w.write(c)?;
            }
        }
        w.finish()?;
        self.store.vectors.save(dir.join("vectors.hvec"))
    }

    /// Load a pipeline saved with [`RagPipeline::save`]. The embedder must be the one used to
    /// build it (its dimension is checked).
    pub fn load(dir: impl AsRef<Path>, chunker: Box<dyn Chunker + 'a>, embedder: &'a dyn Embedder) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg_path = dir.join("config.json");
        let config: RagConfig =
            serde_json::from_slice(&std::fs::read(&cfg_path).map_err(|e| HarpiaError::io(&cfg_path, e))?)?;
        let chunks: Vec<ChunkRecord> = crate::io::read_jsonl(dir.join("chunks.jsonl"))?;
        let vectors = VectorIndex::load(dir.join("vectors.hvec"))?;
        let mut p = Self::new(chunker, embedder, config);
        p.store.vectors = vectors;
        if p.store.vectors.len() != chunks.len() {
            return Err(HarpiaError::invalid(format!(
                "{}: {} chunks but {} vectors",
                dir.display(),
                chunks.len(),
                p.store.vectors.len()
            )));
        }
        for c in chunks {
            p.store.by_id.insert(c.id.clone(), p.store.chunks.len());
            p.store.chunks.push(c);
            p.store.live.push(true);
        }
        let probe = embedder.embed(&["dimension probe"])?;
        if probe.first().map(Vec::len) != Some(embedder.dim()) {
            return Err(HarpiaError::invalid("embedder dimension does not match its vectors"));
        }
        Ok(p)
    }
}

impl Retriever for RagPipeline<'_> {
    fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Hit>> {
        Ok(RagPipeline::retrieve(self, query, None)?
            .into_iter()
            .take(k)
            .map(|r| Hit { id: r.chunk.id, score: r.score, span: Some(r.chunk.span) })
            .collect())
    }
}

/// Assemble ranked chunks into a context of at most `budget` tokens.
pub fn assemble(ranked: &[Retrieved], budget: usize, count_tokens: &dyn Fn(&str) -> usize) -> Context {
    let mut text = String::new();
    let mut citations = Vec::new();
    let mut tokens = 0;
    let mut dropped = 0;
    for r in ranked {
        let block = format!("[{}] {}", citations.len() + 1, r.chunk.text.trim());
        let sep = if text.is_empty() { 0 } else { count_tokens("\n\n") };
        let t = count_tokens(&block) + sep;
        if tokens + t > budget {
            dropped += 1;
            continue;
        }
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(&block);
        tokens += t;
        citations.push(Citation {
            n: citations.len() + 1,
            chunk_id: r.chunk.id.clone(),
            doc_id: r.chunk.doc_id.clone(),
            span: r.chunk.span,
            source: r.chunk.source.clone(),
            license: r.chunk.license.clone(),
        });
    }
    Context { text, citations, tokens, dropped }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::SentenceChunker;

    /// Bag of hashed words: similar wording → similar vectors.
    struct Words;
    impl Embedder for Words {
        fn dim(&self) -> usize {
            64
        }
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0; 64];
                    for w in t.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
                        v[(xxhash_rust::xxh3::xxh3_64(w.as_bytes()) % 64) as usize] += 1.0;
                    }
                    v
                })
                .collect())
        }
    }

    struct PreferOwls;
    impl Reranker for PreferOwls {
        fn rerank(&self, _q: &str, docs: &[&str]) -> Result<Vec<f32>> {
            Ok(docs.iter().map(|d| if d.contains("Owls") { 1.0 } else { 0.0 }).collect())
        }
    }

    fn one_sentence() -> SentenceChunker {
        SentenceChunker { max_chars: 1, overlap: 0, lang: Lang::En.data() }
    }

    fn docs() -> Vec<Document> {
        let mut a = Document::new("birds", "The harpy eagle hunts sloths and monkeys. Owls hunt mice at night.");
        a.source = Some("birds.txt".into());
        a.license = Some("CC-BY-4.0".into());
        a.metadata.insert("lang".into(), "en".into());
        let mut b = Document::new("fish", "Piranhas live in rivers. Some eagles hunt fish too.");
        b.metadata.insert("lang".into(), "pt".into());
        vec![a, b]
    }

    #[test]
    fn hybrid_retrieval_filters_rerank_and_context() {
        for index in [IndexKind::Flat, IndexKind::Hnsw(HnswParams::default())] {
            for fusion in [Fusion::Dense, Fusion::Lexical, Fusion::Rrf { k: 60.0 }, Fusion::Weighted { dense: 0.5 }] {
                let cfg = RagConfig { index, fusion, top_k: 3, ..Default::default() };
                let mut rag = RagPipeline::new(Box::new(one_sentence()), &Words, cfg);
                assert_eq!(rag.add_documents(&docs()).unwrap(), 4);
                let hits = rag.retrieve("what does the harpy eagle hunt", None).unwrap();
                assert_eq!(hits[0].chunk.id, "birds#0", "{fusion:?} {index:?}");
                assert_eq!(hits[0].chunk.span.slice(&docs()[0].text), "The harpy eagle hunts sloths and monkeys.");
                let pt =
                    MetadataFilter { equals: BTreeMap::from([("lang".into(), "pt".into())]), ..Default::default() };
                let hits = rag.retrieve("eagle hunt", Some(&pt)).unwrap();
                assert!(!hits.is_empty() && hits.iter().all(|h| h.chunk.doc_id == "fish"), "{fusion:?}");
            }
        }
        let mut rag = RagPipeline::new(Box::new(one_sentence()), &Words, RagConfig::default());
        rag.add_documents(&docs()).unwrap();
        let words = |t: &str| t.split_whitespace().count();
        let ctx = rag.context("harpy eagle hunt", 12, &words, None).unwrap();
        assert!(ctx.tokens <= 12);
        assert_eq!(ctx.citations[0].chunk_id, "birds#0");
        assert_eq!(ctx.citations[0].license.as_deref(), Some("CC-BY-4.0"));
        assert!(ctx.text.starts_with("[1] The harpy eagle"));
        assert!(ctx.dropped > 0);
        // Reranking reorders.
        let mut rag2 =
            RagPipeline::new(Box::new(one_sentence()), &Words, RagConfig::default()).with_reranker(&PreferOwls);
        rag2.add_documents(&docs()).unwrap();
        let hits = rag2.retrieve("harpy eagle hunt", None).unwrap();
        assert!(hits[0].chunk.text.starts_with("Owls") && hits[0].rerank == Some(1.0));
        // Delete, persist, reload.
        assert_eq!(rag.remove_document("fish"), 2);
        assert_eq!(rag.len(), 2);
        assert!(rag.retrieve("piranhas rivers", None).unwrap().iter().all(|h| h.chunk.doc_id != "fish"));
        let dir = tempfile::tempdir().unwrap();
        rag.save(dir.path()).unwrap();
        let back = RagPipeline::load(dir.path(), Box::new(one_sentence()), &Words).unwrap();
        assert_eq!(back.retrieve("harpy", None).unwrap(), rag.retrieve("harpy", None).unwrap());
        assert_eq!(Retriever::retrieve(&back, "harpy", 1).unwrap()[0].id, "birds#0");
    }
}
