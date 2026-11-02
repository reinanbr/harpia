//! # harpia-http
//!
//! Polite HTTP for LLM and data pipelines.
//!
//! - **Offline by default**: with [`NetworkPolicy::Offline`](harpia_core::NetworkPolicy) every
//!   request fails with [`HarpiaError::NetworkDisabled`](harpia_core::HarpiaError) before any
//!   socket is opened ([`testing::ConnectionCounter`] proves it in tests).
//! - Per-host minimum intervals ([`ratelimit::HostLimiter`]) and token buckets
//!   ([`ratelimit::TokenBucket`]) for RPM/TPM budgets.
//! - Retries on transport errors, 429 and 5xx with exponential backoff and jitter, honoring
//!   `Retry-After` in both forms ([`parse_retry_after`]).
//! - A per-host circuit breaker with [`HttpClient::status`].
//! - An optional TTL cache ([`cache::DiskCache`], pure-Rust `redb`), keyed on method, URL,
//!   sorted parameters (minus excluded ones such as `mailto`) and body hash.
//! - Size caps for buffered bodies and capped downloads with a content-addressed cache.
//! - An async API and a [`blocking`] facade.
//!
//! ```no_run
//! use harpia_core::NetworkPolicy;
//! use harpia_http::{CacheConfig, HttpConfig, Request, blocking::HttpClient};
//! use std::time::Duration;
//!
//! let cfg = HttpConfig::new(NetworkPolicy::Online)
//!     .contact("lab@example.org")
//!     .host_interval("api.openalex.org", Duration::from_millis(200))
//!     .with_cache(CacheConfig::new("/tmp/harpia-http.redb"));
//! let client = HttpClient::new(cfg)?;
//! let work: serde_json::Value = client.get_json(Request::get("https://api.openalex.org/works/W2741809807"))?;
//! # Ok::<(), harpia_core::HarpiaError>(())
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod blocking;
pub mod cache;
mod client;
mod config;
pub mod ratelimit;
pub mod testing;

pub use client::{ByteStream, HostStatus, HttpClient, Method, Request, Response, StreamingResponse, parse_retry_after};
pub use config::{CacheConfig, HttpConfig};
