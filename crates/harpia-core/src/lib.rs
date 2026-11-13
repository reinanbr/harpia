//! # harpia-core
//!
//! Shared vocabulary of the [Harpia](https://github.com/reinanbr/harpia) toolkit:
//!
//! - [`HarpiaError`] / [`Result`]: one error type for every crate, carrying context but never
//!   secrets;
//! - [`types`]: messages, conversations, training samples, documents, scores, usage and cost;
//! - [`span`]: byte spans and [`LineIndex`] for precise source locations;
//! - [`network`]: the explicit [`NetworkPolicy`] (offline by default);
//! - [`traits`]: the abstractions that let data pipelines, local models and API providers be
//!   swapped freely.
//!
//! ```
//! use harpia_core::types::{Conversation, Message};
//!
//! let conv = Conversation::new()
//!     .with(Message::system("You are terse."))
//!     .with(Message::user("2+2?"))
//!     .with(Message::assistant("4"));
//! assert_eq!(conv.messages.len(), 3);
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod network;
pub mod span;
pub mod traits;
pub mod types;

pub use error::{HarpiaError, Result};
pub use network::NetworkPolicy;
pub use span::{LineIndex, Span};
pub use types::{
    ChatSample, Conversation, Cost, Document, Label, Message, PreferencePair, Role, Scores, TokenUsage, ToolCall,
};
