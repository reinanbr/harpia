//! Blocking facade over [`HttpClient`](crate::HttpClient) for synchronous programs.
//!
//! Each client owns a small current-thread Tokio runtime. Do not call these methods from inside
//! an async runtime (Tokio panics on nested `block_on`); use the async client there.

use crate::client::{HttpClient as AsyncClient, Request, Response};
use crate::config::HttpConfig;
use harpia_core::{HarpiaError, Result};

/// Synchronous polite HTTP client.
///
/// ```
/// let c = harpia_http::blocking::HttpClient::new(Default::default()).unwrap();
/// assert!(c.send(harpia_http::Request::get("https://example.org")).is_err()); // offline
/// ```
pub struct HttpClient {
    rt: tokio::runtime::Runtime,
    inner: AsyncClient,
}

impl std::fmt::Debug for HttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl HttpClient {
    /// Build a client and its runtime.
    pub fn new(cfg: HttpConfig) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| HarpiaError::Other(format!("cannot start runtime: {e}")))?;
        let inner = { rt.block_on(async { AsyncClient::new(cfg) })? };
        Ok(Self { rt, inner })
    }

    /// The async client (e.g. for [`status`](AsyncClient::status)).
    pub fn inner(&self) -> &AsyncClient {
        &self.inner
    }

    /// See [`AsyncClient::send`].
    pub fn send(&self, req: Request) -> Result<Response> {
        self.rt.block_on(self.inner.send(req))
    }

    /// See [`AsyncClient::get_json`].
    pub fn get_json<T: serde::de::DeserializeOwned>(&self, req: Request) -> Result<T> {
        self.rt.block_on(self.inner.get_json(req))
    }

    /// See [`AsyncClient::download_to_file`].
    pub fn download_to_file(&self, req: Request, path: &std::path::Path, max_bytes: u64) -> Result<u64> {
        self.rt.block_on(self.inner.download_to_file(req, path, max_bytes))
    }

    /// See [`AsyncClient::download`].
    pub fn download(&self, url: &str, max_bytes: usize) -> Result<Vec<u8>> {
        self.rt.block_on(self.inner.download(url, max_bytes))
    }
}
