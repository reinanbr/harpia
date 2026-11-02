//! # harpia
//!
//! *Sharp-eyed LLM tooling in pure Rust.* This umbrella crate re-exports the Harpia crates
//! behind feature flags, so one dependency gives you the whole toolkit:
//!
//! | module | crate | feature | status |
//! |---|---|---|---|
//! | [`core`] | `harpia-core` | always | stable |
//! | `text` | `harpia-text` | `text` (default) | stable |
//! | `data` | `harpia-data` | `data` (default) | stable |
//! | `structured` | `harpia-structured` | `structured` (default) | stable |
//! | — | `harpia-http`, `harpia-client`, `harpia-local`, `harpia-train`, `harpia-eval`, `harpia-verify` | — | planned |
//!
//! ```
//! # #[cfg(all(feature = "data", feature = "structured"))] {
//! use harpia::data::pii::PiiDetector;
//! use harpia::structured::parse_llm_json;
//!
//! #[derive(serde::Deserialize)]
//! struct Row { text: String }
//! let row: Row = parse_llm_json("Here: {text: 'mail me at ana@example.com'}").unwrap();
//! let (clean, _) = PiiDetector::default().redact(&row.text);
//! assert_eq!(clean, "mail me at <EMAIL>");
//! # }
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use harpia_core as core;
#[cfg(feature = "data")]
pub use harpia_data as data;
#[cfg(feature = "structured")]
pub use harpia_structured as structured;
#[cfg(feature = "text")]
pub use harpia_text as text;

pub use harpia_core::{HarpiaError, NetworkPolicy, Result};
