//! # harpia-text
//!
//! Text primitives for LLM data curation, RAG and evaluation. Everything works on `&str` and
//! reports **byte spans** into the original text (see [`harpia_core::Span`] and
//! [`harpia_core::LineIndex`]).
//!
//! | module | what it gives you |
//! |---|---|
//! | [`normalize`] | NFC/NFKC, accent stripping, case folding, whitespace and control chars |
//! | [`segment`] | words and sentences with byte spans, abbreviation-aware |
//! | [`stats`] | Flesch, MATTR, burstiness, repetition and character-ratio signals |
//! | [`phrases`] | case/accent-insensitive multi-phrase matching |
//! | [`langid`] | language identification (feature `langid`) |
//! | [`similarity`] | shingles, Jaccard/containment, shared spans, fingerprint index, MinHash |
//! | [`fuzzy`] | rapidfuzz-style ratios |
//! | [`bm25`] | lexical retrieval |
//!
//! ```
//! use harpia_text::{segment, Lang};
//!
//! let text = "O Dr. Silva chegou. Ele trouxe 3,5 kg de café.";
//! let sents: Vec<&str> = segment::sentences(text, &Lang::PtBr.data())
//!     .map(|s| s.slice(text))
//!     .collect();
//! assert_eq!(sents, ["O Dr. Silva chegou.", "Ele trouxe 3,5 kg de café."]);
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bm25;
pub mod fuzzy;
pub mod lang;
#[cfg(feature = "langid")]
pub mod langid;
pub mod normalize;
pub mod phrases;
pub mod segment;
pub mod similarity;
pub mod stats;

pub use lang::{Lang, LangData};
