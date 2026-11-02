//! # harpia-eval
//!
//! Evaluation for LLMs and datasets.
//!
//! - [`harness`]: tasks in TOML/JSON over JSONL (generation, multiple choice by
//!   log-likelihood, perplexity / bits-per-byte, classification), few-shot with a seed, against
//!   any `Provider` or local `Model`, cached and resumable.
//! - [`judge`]: LLM-as-judge with rubric scores, position-swapped pairwise comparison and
//!   agreement statistics.
//! - [`rag`]: recall@k, precision@k, MRR, nDCG, context recall, faithfulness with a
//!   `PairClassifier`, answer relevance with an `Embedder`.
//! - [`metrics`]: exact match, token F1, BLEU, chrF, ROUGE-N/L, edit distance.
//! - [`classification`]: accuracy, per-class P/R/F1, macro/weighted F1, confusion matrix.
//! - [`calibration`]: decision rules as data, threshold grid search on validation, frozen
//!   application on test, best-of-N windows with evidence spans.
//! - [`dataset`]: contamination, label noise and duplicate statistics.
//! - [`report`]: JSON summaries, Markdown leaderboards and model cards.
//! - [`cost`]: prices, cost tracking and `tracing` spans for LLM calls.
//!
//! ```
//! use harpia_eval::metrics::{token_f1, corpus_bleu};
//! assert_eq!(token_f1("Brasília", "brasília"), 1.0);
//! assert!(corpus_bleu(&["the harpy eagle hunts"], &["the harpy eagle hunts"]) > 0.99);
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod calibration;
pub mod classification;
pub mod cost;
pub mod dataset;
pub mod harness;
pub mod judge;
pub mod metrics;
pub mod rag;
pub mod report;
pub mod testing;
