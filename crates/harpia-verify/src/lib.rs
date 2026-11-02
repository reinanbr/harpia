//! # harpia-verify
//!
//! Grounding and fact-checking for LLM pipelines, using **only sources you trust**:
//!
//! - **grounding answers**: split an answer into claims and check each one ([`ClaimVerifier`]);
//! - **web-grounded RAG**: trusted sources as documents for `RagPipeline` ([`rag`]);
//! - **dataset curation**: flag or drop rows whose claims are contradicted ([`dataset`]);
//! - **reference checking**: do cited works exist and match their metadata
//!   ([`ReferenceVerifier`], [`bib`], [`reference::check_quote`]).
//!
//! Output is always an indicator with confidence and evidence, never a verdict. Every network
//! access goes through [`Fetcher`]: offline by default (results are `Skipped` without opening a
//! socket), HTTPS only, hard host allowlist on every request and redirect hop, size and content
//! type limits, `robots.txt` for HTML, no cookies or JavaScript, a [`Disclosure`] log of all
//! outgoing queries, and a content store whose `Frozen` mode replays runs offline.
//!
//! **Limitations.** "Not found" does not mean false. `NotEnoughInfo` is the normal outcome when
//! sources lack the point. NLI models degrade out of domain. Source trust is the user's
//! configuration, not Harpia's judgment.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod adapters;
pub mod bib;
pub mod claims;
pub mod dataset;
pub mod fetch;
#[cfg(feature = "llm")]
pub mod llm;
pub mod rag;
pub mod reference;
pub mod registry;
pub mod robots;
pub mod store;
pub mod time;
pub mod verify;
pub mod work;

pub use adapters::SourceAdapter;
pub use claims::{Claim, QueryMode};
pub use fetch::{Disclosure, FetchConfig, Fetcher, Mode};
pub use reference::{ReferenceConfig, ReferenceQuery, ReferenceResult, ReferenceStatus, ReferenceVerifier};
pub use registry::{Source, SourceKind, SourceRegistry};
pub use verify::{AnswerReport, ClaimVerifier, Evidence, NliLabels, Status, Verification, VerifyConfig};
pub use work::{Author, Lookup, Work, WorkId};
