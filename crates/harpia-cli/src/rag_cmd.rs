//! `harpia rag index|query`: build and query `RagPipeline` indexes on disk.

use crate::text_cmd::{Strategy, build_chunker};
use crate::util::{OutputFormat, emit, read_records, read_text, write_records};
use clap::{Args, Subcommand, ValueEnum};
use harpia::core::traits::{Embedder, Tokenizer};
use harpia::core::{Document, HarpiaError, Result};
use harpia::data::chunk::RecursiveChunker;
use harpia::data::io::get_path;
use harpia::data::rag::{Fusion, IndexKind, MetadataFilter, RagConfig, RagPipeline, Retrieved, assemble};
use harpia::data::tokenize::HfTokenizer;
use harpia::data::vector::HnswParams;
use harpia::local::{EncoderEmbedder, SequenceClassifier};
use serde_json::{Value, json};
use std::path::Path;
use std::process::ExitCode;

#[derive(Subcommand)]
pub enum RagCommand {
    /// Chunk, embed and index JSONL documents into a directory.
    Index(IndexArgs),
    /// Retrieve chunks for a query (or a JSONL file of queries), optionally reranked and
    /// assembled into a cited context.
    Query(QueryArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum IndexType {
    /// Exact search.
    Flat,
    /// Approximate search (HNSW).
    Hnsw,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum FusionArg {
    /// Reciprocal rank fusion of dense and BM25.
    Rrf,
    /// Weighted min-max fusion (`--dense-weight`).
    Weighted,
    /// Dense vectors only.
    Dense,
    /// BM25 only.
    Lexical,
}

#[derive(Args)]
#[command(after_help = "Example:
  harpia rag index docs.jsonl --embedder models/e5-small -o idx --strategy markdown --size 1200
  harpia rag query idx 'Where does the harpy eagle live?' --embedder models/e5-small --budget 512

Every row needs a text field; `id`, `source` and `license` are recorded with each chunk and the
remaining fields become chunk metadata (usable in --filter). Re-indexing a document id replaces
its chunks.")]
pub struct IndexArgs {
    /// JSONL documents (`-` for stdin).
    pub input: String,
    /// Embedding model directory (the same one must be used for queries).
    #[arg(long)]
    pub embedder: String,
    /// Index directory.
    #[arg(short, long)]
    pub output: String,
    /// Add to an existing index (its saved settings are kept).
    #[arg(long)]
    pub append: bool,
    /// Text field.
    #[arg(long, default_value = "text")]
    pub field: String,
    /// Document id field (rows without it get `row-<n>`).
    #[arg(long, default_value = "id")]
    pub id_field: String,
    /// Source field (path, URL).
    #[arg(long, default_value = "source")]
    pub source_field: String,
    /// License field.
    #[arg(long, default_value = "license")]
    pub license_field: String,
    /// Chunking strategy.
    #[arg(long, value_enum, default_value_t = Strategy::Recursive)]
    pub strategy: Strategy,
    /// Chunk size (characters; tokens for `token`).
    #[arg(long, default_value_t = 1000)]
    pub size: usize,
    /// Chunk overlap.
    #[arg(long, default_value_t = 0)]
    pub overlap: usize,
    /// `tokenizer.json` for `--strategy token`.
    #[arg(long)]
    pub tokenizer: Option<String>,
    /// Language for sentence chunking, BM25 tokenization and stopwords (en, pt-br).
    #[arg(long, default_value = "en")]
    pub lang: String,
    /// Vector index.
    #[arg(long, value_enum, default_value_t = IndexType::Flat)]
    pub index: IndexType,
    /// How dense and BM25 results are combined.
    #[arg(long, value_enum, default_value_t = FusionArg::Rrf)]
    pub fusion: FusionArg,
    /// Dense weight for `--fusion weighted` (BM25 gets the rest).
    #[arg(long, default_value_t = 0.5)]
    pub dense_weight: f32,
    /// Default number of hits per query.
    #[arg(long, default_value_t = 5)]
    pub top_k: usize,
    /// Candidates fetched from each retriever before fusion and reranking.
    #[arg(long, default_value_t = 50)]
    pub candidates: usize,
}

fn as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn documents(rows: Vec<Value>, a: &IndexArgs) -> Result<Vec<Document>> {
    rows.into_iter()
        .enumerate()
        .map(|(i, row)| {
            let text = get_path(&row, &a.field)
                .and_then(Value::as_str)
                .ok_or_else(|| HarpiaError::invalid(format!("row {}: no string field `{}`", i + 1, a.field)))?
                .to_string();
            let id = get_path(&row, &a.id_field).and_then(as_string).unwrap_or_else(|| format!("row-{}", i + 1));
            let mut doc = Document::new(id, text);
            doc.source = get_path(&row, &a.source_field).and_then(as_string);
            doc.license = get_path(&row, &a.license_field).and_then(as_string);
            if let Value::Object(m) = row {
                let skip = [&a.field, &a.id_field, &a.source_field, &a.license_field];
                doc.metadata = m.into_iter().filter(|(k, _)| !skip.contains(&k)).collect();
            }
            Ok(doc)
        })
        .collect()
}

pub fn index(a: IndexArgs, format: OutputFormat) -> Result<ExitCode> {
    let emb = EncoderEmbedder::load(Path::new(&a.embedder))?;
    let chunker = build_chunker(a.strategy, a.size, a.overlap, &a.lang, a.tokenizer.as_deref())?;
    let out = Path::new(&a.output);
    let exists = out.join("config.json").is_file();
    let mut pipe = if a.append && exists {
        RagPipeline::load(out, chunker, &emb)?
    } else {
        if exists {
            return Err(HarpiaError::invalid(format!(
                "{} already holds an index; pass --append to add to it or choose another directory",
                out.display()
            )));
        }
        crate::util::lang(&a.lang)?;
        let config = RagConfig {
            index: match a.index {
                IndexType::Flat => IndexKind::Flat,
                IndexType::Hnsw => IndexKind::Hnsw(HnswParams::default()),
            },
            fusion: match a.fusion {
                FusionArg::Rrf => Fusion::Rrf { k: 60.0 },
                FusionArg::Weighted => Fusion::Weighted { dense: a.dense_weight },
                FusionArg::Dense => Fusion::Dense,
                FusionArg::Lexical => Fusion::Lexical,
            },
            top_k: a.top_k,
            candidates: a.candidates,
            lang: a.lang.clone(),
            ..RagConfig::default()
        };
        RagPipeline::new(chunker, &emb, config)
    };
    let docs = documents(read_records(&a.input)?, &a)?;
    let added = pipe.add_documents(&docs)?;
    pipe.save(out)?;
    emit(
        format,
        json!({"documents": docs.len(), "chunks_added": added, "chunks_total": pipe.len(), "dim": emb.dim(), "path": a.output}),
        || format!("indexed {} documents ({added} chunks; {} in the index) into {}", docs.len(), pipe.len(), a.output),
    );
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
#[command(after_help = "Examples:
  harpia rag query idx 'harpy eagle diet' --embedder models/e5-small --k 3
  harpia rag query idx --queries questions.jsonl --field question --embedder models/e5-small -o retrieved.jsonl
  harpia eval rag retrieved.jsonl --k 5

With --queries, each row gets `retrieved` (document ids, best first, deduplicated), `contexts`
(chunk texts) and `hits`, the input `harpia eval rag` expects alongside `relevant`.")]
pub struct QueryArgs {
    /// Index directory (from `harpia rag index`).
    pub index: String,
    /// Query text (`-` for stdin); omit with --queries.
    #[arg(required_unless_present = "queries", conflicts_with = "queries")]
    pub query: Option<String>,
    /// JSONL file of queries.
    #[arg(long)]
    pub queries: Option<String>,
    /// Query field for --queries.
    #[arg(long, default_value = "question")]
    pub field: String,
    /// Embedding model directory used to build the index.
    #[arg(long)]
    pub embedder: String,
    /// Cross-encoder directory to rerank fused candidates.
    #[arg(long)]
    pub reranker: Option<String>,
    /// Hits per query (default: the index's `top_k`).
    #[arg(long)]
    pub k: Option<usize>,
    /// Metadata filter as JSON: `{"equals": {...}, "any_of": {...}, "doc_ids": [...]}`.
    #[arg(long)]
    pub filter: Option<String>,
    /// Assemble a context of at most this many tokens with `[n]` citation markers.
    #[arg(long)]
    pub budget: Option<usize>,
    /// `tokenizer.json` that counts --budget tokens (whitespace-separated words otherwise).
    #[arg(long)]
    pub tokenizer: Option<String>,
    /// Output file for --queries (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

pub fn query(a: QueryArgs, format: OutputFormat) -> Result<ExitCode> {
    let emb = EncoderEmbedder::load(Path::new(&a.embedder))?;
    let rr = a.reranker.as_deref().map(|p| SequenceClassifier::load(Path::new(p))).transpose()?;
    let mut pipe = RagPipeline::load(&a.index, Box::new(RecursiveChunker::new(1000)), &emb as &dyn Embedder)?;
    if let Some(r) = &rr {
        pipe = pipe.with_reranker(r);
    }
    let k = a.k.unwrap_or(pipe.config().top_k);
    let filter: Option<MetadataFilter> = a.filter.as_deref().map(serde_json::from_str).transpose()?;
    let tok = a.tokenizer.as_deref().map(HfTokenizer::from_file).transpose()?;
    let count = |t: &str| match &tok {
        Some(tk) => tk.count(t).unwrap_or_else(|_| t.split_whitespace().count()),
        None => t.split_whitespace().count(),
    };
    let run = |q: &str| -> Result<(Vec<Retrieved>, Option<Value>)> {
        let hits = pipe.retrieve_top(q, k, filter.as_ref())?;
        let ctx = a.budget.map(|b| json!(assemble(&hits, b, &count)));
        Ok((hits, ctx))
    };

    if let Some(path) = &a.queries {
        let mut rows = read_records(path)?;
        for (i, row) in rows.iter_mut().enumerate() {
            let q = get_path(row, &a.field)
                .and_then(Value::as_str)
                .ok_or_else(|| HarpiaError::invalid(format!("row {}: no string field `{}`", i + 1, a.field)))?
                .to_string();
            let (hits, ctx) = run(&q)?;
            let mut retrieved: Vec<&str> = Vec::new();
            for h in &hits {
                if !retrieved.contains(&h.chunk.doc_id.as_str()) {
                    retrieved.push(&h.chunk.doc_id);
                }
            }
            if let Value::Object(m) = row {
                m.insert("retrieved".into(), json!(retrieved));
                m.insert("contexts".into(), json!(hits.iter().map(|h| &h.chunk.text).collect::<Vec<_>>()));
                m.insert("hits".into(), json!(hits));
                if let Some(c) = ctx {
                    m.insert("context".into(), c);
                }
            }
        }
        write_records(a.output.as_deref(), &rows)?;
        if a.output.is_some() {
            emit(format, json!({"queries": rows.len()}), || format!("answered {} queries", rows.len()));
        }
        return Ok(ExitCode::SUCCESS);
    }

    let q = match a.query.as_deref() {
        Some("-") => read_text("-")?,
        Some(q) => q.to_string(),
        None => unreachable!("clap requires a query or --queries"),
    };
    let (hits, ctx) = run(q.trim())?;
    let mut payload = json!({"query": q.trim(), "hits": hits});
    if let Some(c) = &ctx {
        payload["context"] = c.clone();
    }
    emit(format, payload, || {
        let mut s: Vec<String> = hits
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let src = h.chunk.source.as_deref().map(|s| format!(" ({s})")).unwrap_or_default();
                format!("{}. [{:.4}] {}{src}\n   {}", i + 1, h.score, h.chunk.id, h.chunk.text.replace('\n', " "))
            })
            .collect();
        if let Some(c) = &ctx {
            s.push(format!("\n--- context ({} tokens)\n{}", c["tokens"], c["text"].as_str().unwrap_or("")));
        }
        s.join("\n")
    });
    Ok(ExitCode::SUCCESS)
}

pub fn run(c: RagCommand, format: OutputFormat) -> Result<ExitCode> {
    match c {
        RagCommand::Index(a) => index(a, format),
        RagCommand::Query(a) => query(a, format),
    }
}
