//! # harpia-learn
//!
//! Learning material for [Harpia](https://github.com/reinanbr/harpia): every lesson is an
//! integration test in `tests/lessons/`, so it is compiled, run and kept correct by CI. Read
//! them in order; each one prints what it does (`cargo test -p harpia-learn -- --nocapture`).
//!
//! | lesson | topic |
//! |---|---|
//! | `01_text` | normalization, words and sentences with byte spans, readability, fuzzy matching, similarity |
//! | `02_tokenization` | tokenizers, offsets, pairs, chat templates with assistant-only loss masks, packing |
//! | `03_models` | reading model directories; decoders, classifiers, embeddings and translation |
//! | `04_model_review` | grading models (metrics, eval harness, calibration) and correcting one with LoRA |
//! | `05_article_review` | reviewing a LaTeX article with the [`review`] module (an errei-style auditor) |
//!
//! Every model in the lessons is a *tiny random-weight model* written to a temporary
//! directory by `harpia::local::testing`, so nothing is downloaded and the lessons run
//! offline in seconds. Replace the directory with a real Hugging Face model directory (for
//! example `Qwen2.5-0.5B-Instruct` or an XLM-R NLI cross-encoder) and the same code does
//! real work.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod review;

/// Directory of the bundled fixtures (a small LaTeX article, its bibliography and a corpus).
pub fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}
