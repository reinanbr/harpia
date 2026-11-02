//! # harpia-structured
//!
//! Getting reliable structured data out of LLMs:
//!
//! - [`extract_json`] / [`parse_llm_json`]: find and deserialize JSON in chatty output;
//! - [`repair`](repair::repair): fix almost-JSON (quotes, commas, brackets, comments, Python
//!   literals, truncation);
//! - [`IncrementalJson`]: best-effort values while a response streams;
//! - [`constraint`]: choice, regex and JSON-Schema constraints producing token masks for
//!   constrained decoding;
//! - [`toolcall`]: parse tool calls in OpenAI, Anthropic and Hermes/Qwen formats;
//! - `schema` (feature `schema`): full JSON Schema validation and schemas derived from Rust
//!   types.
//!
//! ```
//! #[derive(serde::Deserialize)]
//! struct Out { answer: String }
//! let o: Out = harpia_structured::parse_llm_json("```json\n{'answer': 'forty-two',}\n```").unwrap();
//! assert_eq!(o.answer, "forty-two");
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod constraint;
pub mod extract;
pub mod incremental;
pub mod repair;
#[cfg(feature = "schema")]
pub mod schema;
pub mod toolcall;

pub use extract::{extract_all, extract_json, parse_llm_json};
pub use incremental::IncrementalJson;
