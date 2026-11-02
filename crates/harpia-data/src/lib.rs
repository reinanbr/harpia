//! # harpia-data
//!
//! Dataset tooling for LLM training, evaluation and RAG.
//!
//! | module | what it gives you |
//! |---|---|
//! | [`io`] | streaming JSONL/CSV with gzip/zstd, dotted field paths |
//! | [`cache`] | [`JsonlCache`]: resumable keyed results for incremental jobs |
//! | [`shard`] | sharded datasets with a manifest |
//! | [`filter`] | heuristic quality filters, presets, TOML config, language filter |
//! | [`pii`] | PII and secret detection/redaction (incl. CPF/CNPJ) |
//! | [`dedup`] | exact and MinHash-LSH near-duplicate removal |
//! | [`decontam`] | benchmark decontamination by n-gram overlap |
//! | [`split`] | seeded, grouped and stratified splits; class weights |
//! | [`curate`] | a pipeline of stages with a curation report |
//! | [`format`](mod@format) | OpenAI messages / ShareGPT / Alpaca / prompt-completion / preferences |
//! | [`template`] | Jinja chat templates with assistant spans and loss masks |
//! | `tokenize` | Hugging Face tokenizers (feature `tokenizers`) |
//! | [`pack`] | sequence packing with document boundaries |
//! | [`chunk`] | chunking for RAG with byte spans |
//! | [`card`] | dataset cards |
//! | [`rng`] | the deterministic PRNG behind every shuffle |
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cache;
pub mod card;
pub mod chunk;
pub mod curate;
pub mod decontam;
pub mod dedup;
pub mod filter;
pub mod format;
pub mod io;
pub mod pack;
pub mod pii;
pub mod rng;
pub mod shard;
pub mod split;
#[cfg(feature = "tokenizers")]
pub mod spm;
pub mod template;
#[cfg(feature = "tokenizers")]
pub mod tokenize;

pub use cache::JsonlCache;
pub use io::{JsonlReader, JsonlWriter, read_jsonl};
