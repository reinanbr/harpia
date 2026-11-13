//! # harpia-client
//!
//! Typed async LLM clients behind one trait, [`Provider`](harpia_core::traits::Provider):
//!
//! | provider | feature | API |
//! |---|---|---|
//! | [`OpenAiCompatible`] | `openai-compatible` | Chat Completions + Embeddings (OpenAI, vLLM, llama.cpp, LM Studio, OpenRouter...) |
//! | [`Anthropic`] | `anthropic` | Messages API, `anthropic-version: 2026-10-21` |
//! | [`Ollama`] | `ollama` | `/api/chat`, `/api/embed` |
//!
//! All requests go through [`harpia_http::HttpClient`], so they inherit its offline default,
//! rate limits, retries and circuit breaker. On top of the providers:
//!
//! - [`chat_json`] / [`chat_typed`]: structured output with validation and corrective retries;
//! - [`WithMiddleware`], [`UsageTracker`], [`DefaultSystemPrompt`]: request/response hooks;
//! - [`BatchRunner`]: bounded concurrency, RPM/TPM budgets, resumable via `JsonlCache`;
//! - [`collect_stream`]: fold a streamed reply into one chunk.
//!
//! Sampling parameters map as follows: OpenAI-compatible sends `temperature`, `top_p`, `stop`,
//! `seed` and penalties (plus `top_k`/`min_p`/`repetition_penalty` with `extended_sampling`);
//! Ollama sends everything as `options`; Anthropic sends only `max_tokens` and
//! `stop_sequences`, because API version `2026-10-21` rejects sampling parameters.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "anthropic")]
pub mod anthropic;
mod batch;
// The HTTP helpers are only used by the provider modules; without any provider feature they
// are dead code, which is expected.
#[cfg_attr(not(any(feature = "openai-compatible", feature = "anthropic", feature = "ollama")), allow(dead_code))]
mod common;
mod middleware;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai-compatible")]
pub mod openai;
mod structured;
mod synth;

#[cfg(feature = "anthropic")]
pub use anthropic::Anthropic;
pub use batch::{BatchReport, BatchRunner};
pub use common::{Secret, SseEvent, SseParser, collect_stream};
pub use middleware::{DefaultSystemPrompt, Middleware, UsageTracker, WithMiddleware};
#[cfg(feature = "ollama")]
pub use ollama::Ollama;
#[cfg(feature = "openai-compatible")]
pub use openai::OpenAiCompatible;
pub use structured::{chat_json, chat_typed, strict_schema};
pub use synth::{SynthTemplate, request_key, synthesize};
