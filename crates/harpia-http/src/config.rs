//! Client configuration.

use harpia_core::NetworkPolicy;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Response cache settings.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheConfig {
    /// Database file (created with its parent directories).
    pub path: PathBuf,
    /// Entries older than this are ignored.
    pub ttl: Duration,
    /// Statuses worth caching (default 200 and 404).
    pub statuses: Vec<u16>,
    /// Query parameters left out of the cache key (contact e-mails, API keys...).
    pub excluded_params: Vec<String>,
    /// Directory for content-addressed binary downloads (default: next to `path`).
    pub download_dir: Option<PathBuf>,
}

impl CacheConfig {
    /// Cache at `path` with a 30-day TTL.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            ttl: Duration::from_secs(30 * 24 * 3600),
            statuses: vec![200, 404],
            excluded_params: vec!["mailto".into(), "api_key".into(), "apikey".into(), "key".into()],
            download_dir: None,
        }
    }
}

/// Configuration of an [`HttpClient`](crate::HttpClient).
///
/// ```
/// use harpia_http::HttpConfig;
/// use harpia_core::NetworkPolicy;
/// use std::time::Duration;
/// let cfg = HttpConfig::new(NetworkPolicy::Online)
///     .contact("lab@example.org")
///     .host_interval("api.crossref.org", Duration::from_millis(200));
/// assert!(cfg.user_agent().contains("mailto:lab@example.org"));
/// ```
#[derive(Clone)]
pub struct HttpConfig {
    /// Network policy; `Offline` (default) refuses every request.
    pub policy: NetworkPolicy,
    /// Product token for the User-Agent (default `harpia/<version>`).
    pub product: String,
    /// Contact e-mail added to the User-Agent (polite-pool etiquette of public APIs).
    pub contact_email: Option<String>,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Retries after the first attempt.
    pub retries: u32,
    /// Base delay of exponential backoff.
    pub backoff: Duration,
    /// Cap on any single wait (backoff or `Retry-After`).
    pub max_delay: Duration,
    /// Minimum interval between requests to a host.
    pub host_intervals: HashMap<String, Duration>,
    /// Interval for hosts not in `host_intervals`.
    pub default_interval: Duration,
    /// Consecutive failed requests after which a host is disabled for the session.
    pub failures_to_disable: u32,
    /// Maximum size of a buffered response body.
    pub max_body_bytes: usize,
    /// Optional response cache.
    pub cache: Option<CacheConfig>,
    /// Header names whose values are never logged (case-insensitive).
    pub sensitive_headers: Vec<String>,
}

impl std::fmt::Debug for HttpConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpConfig")
            .field("policy", &self.policy)
            .field("user_agent", &self.user_agent())
            .field("timeout", &self.timeout)
            .field("retries", &self.retries)
            .field("cache", &self.cache.as_ref().map(|c| &c.path))
            .finish_non_exhaustive()
    }
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self::new(NetworkPolicy::Offline)
    }
}

impl HttpConfig {
    /// Defaults with the given policy.
    pub fn new(policy: NetworkPolicy) -> Self {
        Self {
            policy,
            product: format!("harpia/{}", env!("CARGO_PKG_VERSION")),
            contact_email: None,
            timeout: Duration::from_secs(60),
            retries: 3,
            backoff: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            host_intervals: HashMap::new(),
            default_interval: Duration::ZERO,
            failures_to_disable: 3,
            max_body_bytes: 32 * 1024 * 1024,
            cache: None,
            sensitive_headers: vec!["authorization".into(), "x-api-key".into(), "api-key".into(), "cookie".into()],
        }
    }

    /// Set the contact e-mail.
    pub fn contact(mut self, email: impl Into<String>) -> Self {
        self.contact_email = Some(email.into());
        self
    }

    /// Minimum interval between requests to `host`.
    pub fn host_interval(mut self, host: impl Into<String>, every: Duration) -> Self {
        self.host_intervals.insert(host.into(), every);
        self
    }

    /// Enable the response cache.
    pub fn with_cache(mut self, cache: CacheConfig) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Retries and backoff base.
    pub fn retries(mut self, retries: u32, backoff: Duration) -> Self {
        self.retries = retries;
        self.backoff = backoff;
        self
    }

    /// The User-Agent header value.
    pub fn user_agent(&self) -> String {
        match &self.contact_email {
            Some(e) => format!("{} (+https://github.com/harpia-rs/harpia; mailto:{e})", self.product),
            None => format!("{} (+https://github.com/harpia-rs/harpia)", self.product),
        }
    }
}
