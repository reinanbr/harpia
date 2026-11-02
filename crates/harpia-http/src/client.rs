//! The async client.

use crate::cache::DiskCache;
use crate::config::HttpConfig;
use crate::ratelimit::HostLimiter;
use bytes::Bytes;
use futures_util::StreamExt;
use harpia_core::traits::Cache;
use harpia_core::{HarpiaError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// GET.
    Get,
    /// POST.
    Post,
}

/// A request description.
#[derive(Clone)]
pub struct Request {
    /// Method.
    pub method: Method,
    /// Absolute URL.
    pub url: String,
    /// Query parameters (appended to the URL).
    pub params: Vec<(String, String)>,
    /// Extra headers.
    pub headers: Vec<(String, String)>,
    /// Body (POST).
    pub body: Option<Vec<u8>>,
    /// Use the response cache (when configured). Default: `true` for GET, `false` for POST.
    pub cache: bool,
    /// Per-request timeout override.
    pub timeout: Option<Duration>,
}

impl std::fmt::Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Request")
            .field("method", &self.method)
            .field("url", &self.url)
            .field("headers", &self.headers.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>())
            .field("body_bytes", &self.body.as_ref().map(Vec::len))
            .finish()
    }
}

impl Request {
    /// GET `url`.
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: Method::Get,
            url: url.into(),
            params: vec![],
            headers: vec![],
            body: None,
            cache: true,
            timeout: None,
        }
    }

    /// POST a JSON body to `url`.
    pub fn post_json(url: impl Into<String>, body: &serde_json::Value) -> Self {
        Self {
            method: Method::Post,
            url: url.into(),
            params: vec![],
            headers: vec![("content-type".into(), "application/json".into())],
            body: Some(serde_json::to_vec(body).expect("JSON value serializes")),
            cache: false,
            timeout: None,
        }
    }

    /// Add a query parameter.
    pub fn param(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.params.push((k.into(), v.into()));
        self
    }

    /// Add a header.
    pub fn header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.headers.push((k.into(), v.into()));
        self
    }

    /// Enable or disable the cache for this request.
    pub fn cached(mut self, yes: bool) -> Self {
        self.cache = yes;
        self
    }

    /// Override the timeout.
    pub fn timeout(mut self, t: Duration) -> Self {
        self.timeout = Some(t);
        self
    }
}

/// A buffered response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Response {
    /// Status code.
    pub status: u16,
    /// Body bytes.
    pub body: Vec<u8>,
    /// Served from the cache.
    pub from_cache: bool,
}

impl Response {
    /// `true` for 2xx.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Body as UTF-8 (lossy).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// Body as JSON.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        Ok(serde_json::from_slice(&self.body)?)
    }
}

/// A byte stream of a streaming response body.
pub type ByteStream = Pin<Box<dyn futures_core::Stream<Item = Result<Bytes>> + Send>>;

/// A streaming response: status known, body not yet read.
pub struct StreamingResponse {
    /// Status code.
    pub status: u16,
    /// The body stream.
    pub body: ByteStream,
}

/// Availability of a host in this session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostStatus {
    /// Host name.
    pub host: String,
    /// Failed requests (after retries) so far.
    pub failures: u32,
    /// Consecutive failures.
    pub consecutive: u32,
    /// Last error message.
    pub last_error: Option<String>,
    /// Disabled for the rest of the session.
    pub disabled: bool,
}

#[derive(Debug, Default)]
struct Breaker {
    hosts: HashMap<String, HostStatus>,
}

/// The polite HTTP client. Cheap to share behind `Arc`; all methods take `&self`.
///
/// ```
/// use harpia_http::{HttpClient, HttpConfig, Request};
/// # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
/// let client = HttpClient::new(HttpConfig::default()).unwrap(); // offline by default
/// let err = client.send(Request::get("https://api.example.org/x")).await.unwrap_err();
/// assert!(err.to_string().contains("offline"));
/// assert_eq!(client.requests_sent(), 0);
/// # });
/// ```
pub struct HttpClient {
    cfg: HttpConfig,
    inner: reqwest::Client,
    limiter: HostLimiter,
    cache: Option<DiskCache>,
    download_dir: Option<PathBuf>,
    breaker: Mutex<Breaker>,
    sent: AtomicU64,
    refused: AtomicU64,
    jitter_state: AtomicU64,
}

impl std::fmt::Debug for HttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpClient")
            .field("config", &self.cfg)
            .field("sent", &self.requests_sent())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "tls")]
fn install_crypto_provider() {
    // process-wide; ignore "already installed"
    let _ = rustls::crypto::ring::default_provider().install_default();
}

#[cfg(not(feature = "tls"))]
fn install_crypto_provider() {}

/// Parse a `Retry-After` value (delta-seconds or HTTP date) into a delay.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let v = value.trim();
    if let Ok(secs) = v.parse::<f64>() {
        return (secs >= 0.0 && secs.is_finite()).then(|| Duration::from_secs_f64(secs));
    }
    let when = httpdate::parse_http_date(v).ok()?;
    Some(when.duration_since(SystemTime::now()).unwrap_or(Duration::ZERO))
}

impl HttpClient {
    /// Build a client.
    pub fn new(cfg: HttpConfig) -> Result<Self> {
        install_crypto_provider();
        let inner = reqwest::Client::builder()
            .user_agent(cfg.user_agent())
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| HarpiaError::Other(format!("cannot build HTTP client: {e}")))?;
        let cache = match &cfg.cache {
            Some(c) => match DiskCache::open(&c.path, c.ttl) {
                Ok(db) => Some(db),
                Err(e) => {
                    tracing::warn!("http cache disabled: {e}");
                    None
                }
            },
            None => None,
        };
        let download_dir =
            cfg.cache.as_ref().map(|c| c.download_dir.clone().unwrap_or_else(|| c.path.with_extension("downloads")));
        Ok(Self {
            limiter: HostLimiter::new(cfg.host_intervals.clone(), cfg.default_interval),
            inner,
            cache,
            download_dir,
            breaker: Mutex::new(Breaker::default()),
            sent: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            jitter_state: AtomicU64::new(0x9E37_79B9_7F4A_7C15),
            cfg,
        })
    }

    /// Configuration in use.
    pub fn config(&self) -> &HttpConfig {
        &self.cfg
    }

    /// Requests actually sent over the network (cache hits and refusals excluded).
    pub fn requests_sent(&self) -> u64 {
        self.sent.load(Ordering::Relaxed)
    }

    /// Requests refused by the offline policy.
    pub fn requests_refused(&self) -> u64 {
        self.refused.load(Ordering::Relaxed)
    }

    /// Per-host status (only hosts that failed at least once).
    pub fn status(&self) -> Vec<HostStatus> {
        let b = self.breaker.lock().expect("breaker lock");
        let mut v: Vec<HostStatus> = b.hosts.values().cloned().collect();
        v.sort_by(|a, b| a.host.cmp(&b.host));
        v
    }

    /// `true` if `host` was disabled by the circuit breaker.
    pub fn is_disabled(&self, host: &str) -> bool {
        self.breaker.lock().expect("breaker lock").hosts.get(host).is_some_and(|h| h.disabled)
    }

    fn full_url(&self, req: &Request) -> Result<reqwest::Url> {
        let mut url = reqwest::Url::parse(&req.url).map_err(|e| HarpiaError::invalid(format!("bad URL: {e}")))?;
        if !req.params.is_empty() {
            let mut q = url.query_pairs_mut();
            for (k, v) in &req.params {
                q.append_pair(k, v);
            }
        }
        Ok(url)
    }

    fn cache_key(&self, req: &Request) -> String {
        let excluded: Vec<String> = self
            .cfg
            .cache
            .as_ref()
            .map(|c| c.excluded_params.iter().map(|p| p.to_lowercase()).collect())
            .unwrap_or_default();
        let mut params: Vec<&(String, String)> =
            req.params.iter().filter(|(k, _)| !excluded.contains(&k.to_lowercase())).collect();
        params.sort();
        let mut key = format!("{:?} {}", req.method, req.url);
        for (k, v) in params {
            key.push_str(&format!("&{k}={v}"));
        }
        if let Some(b) = &req.body {
            key.push_str(&format!(" body:{:016x}", xxhash_rust::xxh3::xxh3_64(b)));
        }
        key
    }

    fn check(&self, url: &reqwest::Url) -> Result<String> {
        let host = url.host_str().unwrap_or("").to_string();
        if let Err(e) = self.cfg.policy.check(&host) {
            self.refused.fetch_add(1, Ordering::Relaxed);
            return Err(e);
        }
        if url.scheme() == "https" && !cfg!(feature = "tls") {
            return Err(HarpiaError::FeatureDisabled { what: "HTTPS", feature: "tls", krate: "harpia-http" });
        }
        if !matches!(url.scheme(), "http" | "https") {
            return Err(HarpiaError::invalid(format!("unsupported URL scheme `{}`", url.scheme())));
        }
        if self.is_disabled(&host) {
            let last = self.breaker.lock().expect("breaker lock").hosts.get(&host).and_then(|h| h.last_error.clone());
            return Err(HarpiaError::Remote {
                host: host.clone(),
                message: format!("disabled for this session after repeated failures ({})", last.unwrap_or_default()),
            });
        }
        Ok(host)
    }

    fn jitter(&self, max: Duration) -> Duration {
        // xorshift on a shared state: cheap, good enough for jitter
        let mut x = self.jitter_state.fetch_add(0x9E37_79B9, Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        max.mul_f64((x >> 11) as f64 / (1u64 << 53) as f64)
    }

    fn backoff(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        let exp = self.cfg.backoff.saturating_mul(1u32 << attempt.min(16));
        let d = exp + self.jitter(self.cfg.backoff / 2);
        retry_after.map_or(d, |r| r.max(d)).min(self.cfg.max_delay)
    }

    fn record_success(&self, host: &str) {
        if let Some(h) = self.breaker.lock().expect("breaker lock").hosts.get_mut(host) {
            h.consecutive = 0;
        }
    }

    fn record_failure(&self, host: &str, err: &str) {
        let mut b = self.breaker.lock().expect("breaker lock");
        let h = b.hosts.entry(host.to_string()).or_insert_with(|| HostStatus {
            host: host.to_string(),
            failures: 0,
            consecutive: 0,
            last_error: None,
            disabled: false,
        });
        h.failures += 1;
        h.consecutive += 1;
        h.last_error = Some(err.to_string());
        if h.consecutive >= self.cfg.failures_to_disable.max(1) {
            h.disabled = true;
        }
    }

    fn build(&self, req: &Request, url: reqwest::Url) -> reqwest::RequestBuilder {
        let mut b = match req.method {
            Method::Get => self.inner.get(url),
            Method::Post => self.inner.post(url),
        };
        for (k, v) in &req.headers {
            b = b.header(k.as_str(), v.as_str());
        }
        if let Some(body) = &req.body {
            b = b.body(body.clone());
        }
        if let Some(t) = req.timeout {
            b = b.timeout(t);
        }
        b
    }

    /// Send with rate limiting and retries, returning the raw response once a non-retryable
    /// status arrives.
    async fn send_with_retries(&self, req: &Request, url: &reqwest::Url, host: &str) -> Result<reqwest::Response> {
        let mut last_err = String::new();
        for attempt in 0..=self.cfg.retries {
            self.limiter.wait(host).await;
            self.sent.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(method = ?req.method, host, attempt, "http request");
            match self.build(req, url.clone()).send().await {
                Err(e) => {
                    last_err =
                        if e.is_timeout() { "timeout".into() } else { format!("transport error: {}", e.without_url()) };
                }
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if status == 429 || status >= 500 {
                        last_err = format!("HTTP {status}");
                        let ra =
                            resp.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(parse_retry_after);
                        if attempt < self.cfg.retries {
                            tokio::time::sleep(self.backoff(attempt, ra)).await;
                        }
                        continue;
                    }
                    self.record_success(host);
                    return Ok(resp);
                }
            }
            if attempt < self.cfg.retries {
                tokio::time::sleep(self.backoff(attempt, None)).await;
            }
        }
        self.record_failure(host, &last_err);
        Err(HarpiaError::Remote { host: host.to_string(), message: last_err })
    }

    /// Send a request and buffer the response (up to `max_body_bytes`). Non-2xx statuses are
    /// returned as responses; transport failures, exhausted retries, the offline policy and
    /// the circuit breaker produce errors.
    pub async fn send(&self, req: Request) -> Result<Response> {
        let url = self.full_url(&req)?;
        let host = self.check(&url)?;
        let key = (req.cache && self.cache.is_some()).then(|| self.cache_key(&req));
        if let (Some(k), Some(c)) = (&key, &self.cache)
            && let Ok(Some(bytes)) = c.get(k)
            && let Ok(mut r) = serde_json::from_slice::<Response>(&bytes)
        {
            r.from_cache = true;
            return Ok(r);
        }
        let resp = self.send_with_retries(&req, &url, &host).await?;
        let status = resp.status().as_u16();
        let body = read_capped(resp, self.cfg.max_body_bytes, &host).await?;
        let out = Response { status, body, from_cache: false };
        if let (Some(k), Some(c), Some(cc)) = (&key, &self.cache, &self.cfg.cache)
            && cc.statuses.contains(&status)
            && let Ok(bytes) = serde_json::to_vec(&out)
            && let Err(e) = c.put(k, &bytes, None)
        {
            tracing::warn!("http cache write failed: {e}");
        }
        Ok(out)
    }

    /// Send and stream the body (for server-sent events and large downloads). Never cached.
    pub async fn send_streaming(&self, req: Request) -> Result<StreamingResponse> {
        let url = self.full_url(&req)?;
        let host = self.check(&url)?;
        let resp = self.send_with_retries(&req, &url, &host).await?;
        let status = resp.status().as_u16();
        let body = resp.bytes_stream().map(move |r| {
            r.map_err(|e| HarpiaError::Remote { host: host.clone(), message: e.without_url().to_string() })
        });
        Ok(StreamingResponse { status, body: Box::pin(body) })
    }

    /// GET JSON (2xx only; other statuses become [`HarpiaError::Remote`]).
    pub async fn get_json<T: serde::de::DeserializeOwned>(&self, req: Request) -> Result<T> {
        let host = reqwest::Url::parse(&req.url).ok().and_then(|u| u.host_str().map(String::from)).unwrap_or_default();
        let r = self.send(req).await?;
        if !r.is_success() {
            return Err(HarpiaError::Remote { host, message: format!("HTTP {}", r.status) });
        }
        r.json()
    }

    /// Download binary content (e.g. an open-access PDF) with a hard size limit, cached on
    /// disk under the SHA-256 of the URL when a cache is configured.
    pub async fn download(&self, url: &str, max_bytes: usize) -> Result<Vec<u8>> {
        let path = self
            .download_dir
            .as_ref()
            .map(|d| d.join(Sha256::digest(url.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>()));
        if let Some(p) = &path
            && let Ok(data) = std::fs::read(p)
        {
            return Ok(data);
        }
        let req = Request::get(url).cached(false);
        let parsed = self.full_url(&req)?;
        let host = self.check(&parsed)?;
        let resp = self.send_with_retries(&req, &parsed, &host).await?;
        if !resp.status().is_success() {
            return Err(HarpiaError::Remote { host, message: format!("HTTP {}", resp.status().as_u16()) });
        }
        let data = read_capped(resp, max_bytes, &host).await?;
        if let Some(p) = &path {
            let _ = p.parent().map(std::fs::create_dir_all);
            if let Err(e) = std::fs::write(p, &data) {
                tracing::warn!("download cache write failed: {e}");
            }
        }
        Ok(data)
    }
}

async fn read_capped(resp: reqwest::Response, max: usize, host: &str) -> Result<Vec<u8>> {
    if resp.content_length().is_some_and(|n| n as usize > max) {
        return Err(HarpiaError::Remote { host: host.into(), message: format!("response larger than {max} bytes") });
    }
    let mut out = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|e| HarpiaError::Remote { host: host.into(), message: e.without_url().to_string() })?;
        if out.len() + chunk.len() > max {
            return Err(HarpiaError::Remote {
                host: host.into(),
                message: format!("response larger than {max} bytes"),
            });
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}
