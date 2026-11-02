//! # harpia-local
//!
//! Local inference in pure Rust on [candle](https://github.com/huggingface/candle), CPU by
//! default. Models load from Hugging Face directories (`config.json`, safetensors,
//! `tokenizer.json`) without memory mapping, so the crate contains no `unsafe`.
//!
//! | type | models | implements |
//! |---|---|---|
//! | [`SequenceClassifier`] | BERT, RoBERTa, XLM-RoBERTa `*ForSequenceClassification` | `PairClassifier`, `Reranker` |
//! | [`EncoderEmbedder`] | BERT / XLM-R encoders (sentence-transformers) | `Embedder` |
//! | [`MarianTranslator`] | MarianMT (`opus-mt-*`) | `Translator` |
//! | [`TextGenerator`] | Llama and Qwen2 families | `Model`, `Provider` |
//!
//! [`ModelCache`] loads each model once per process and shares it behind `Arc`.
//! [`testing`] writes tiny random-weight models for tests.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod adapter;
mod cache;
pub mod decoder;
mod encoder;
#[cfg(feature = "hub")]
pub mod hub;
mod loader;
mod marian;
pub mod sampling;
pub mod testing;

pub use cache::ModelCache;
pub use candle_core::{DType, Device};
pub use decoder::{CausalLm, TextGenerator};
pub use encoder::{EncoderEmbedder, Pooling, SequenceClassifier, softmax};
pub use loader::ModelDir;
pub use marian::{MarianTranslator, Search};
