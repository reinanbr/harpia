//! The only way `harpia-verify` talks to the network.
//!
//! Every request is checked before a socket is opened:
//! - HTTPS only (plain HTTP is refused, except loopback hosts when explicitly enabled for tests);
//! - the host must be one of the requesting source's domains, and every redirect hop must stay on
//!   a host allowed by the registry (enforced inside the HTTP client);
//! - response bodies are capped and their content type must match what the caller expects;
//! - HTML pages are fetched only when the site's `robots.txt` allows it;
//! - no cookies, no JavaScript, no credentials beyond a source's own documented API key.
//!
//! Each outgoing request is appended to a [`Disclosure`] log (what was sent, to whom, why).
//! Fetched content is written to an optional [`ContentStore`] with its URL, time and hash;
//! [`Mode::Frozen`] answers only from that store, which works offline and makes runs
//! reproducible.

use crate::registry::{Source, SourceRegistry};
use crate::robots::Robots;
use crate::store::{ContentStore, StoredMeta, sha256_hex};
use crate::time::now_rfc3339;
use harpia_core::{HarpiaError, NetworkPolicy, Result};
use harpia_http::{HttpClient, HttpConfig, RedirectPolicy, Request};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Where answers come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Always fetch (and record in the store when one is configured).
    #[default]
    Live,
    /// Use the store when it has the response, otherwise fetch.
    CachedFirst,
    /// Only the store; never the network (allowed offline).
    Frozen,
}

/// What content a request expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    /// `application/json` (or `+json`).
    Json,
    /// XML / Atom.
    Xml,
    /// An HTML page (subject to `robots.txt`).
    Html,
    /// A PDF.
    Pdf,
    /// Any `text/*`.
    Text,
}

impl Expect {
    fn accepts(self, content_type: Option<&str>, body: &[u8]) -> bool {
        let ct = content_type.unwrap_or("").to_ascii_lowercase();
        match self {
            Self::Json => {
                ct.contains("json") || (ct.is_empty() && body.first().is_some_and(|b| *b == b'{' || *b == b'['))
            }
            Self::Xml => ct.contains("xml") || ct.contains("atom"),
            Self::Html => ct.contains("html"),
            Self::Pdf => ct.contains("pdf") || body.starts_with(b"%PDF"),
            Self::Text => ct.starts_with("text/") || ct.is_empty(),
        }
    }
}

/// One entry of the disclosure log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisclosureEntry {
    /// Source id.
    pub source: String,
    /// Host contacted.
    pub host: String,
    /// URL including the query string (the contact e-mail is redacted).
    pub url: String,
    /// Why (e.g. `"search: claim keywords"`).
    pub purpose: String,
    /// When (RFC 3339).
    pub at: String,
    /// `true` when answered from the content store without contacting the host.
    pub from_store: bool,
}

/// Everything sent (or replayed) during a run, returned with results.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Disclosure {
    /// Entries in order.
    pub entries: Vec<DisclosureEntry>,
}

impl Disclosure {
    /// Requests that actually left the machine.
    pub fn network_requests(&self) -> usize {
        self.entries.iter().filter(|e| !e.from_store).count()
    }
}

/// A fetched (or replayed) response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    /// HTTP status.
    pub status: u16,
    /// Body.
    pub body: Vec<u8>,
    /// Content type.
    pub content_type: Option<String>,
    /// URL fetched (final URL after redirects when known).
    pub url: String,
    /// Retrieval time (RFC 3339).
    pub retrieved_at: String,
    /// SHA-256 of the body.
    pub sha256: String,
    /// Served from the content store.
    pub from_store: bool,
}

impl Fetched {
    /// Body as UTF-8 (lossy).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// `true` for 2xx.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Parse the body as JSON.
    pub fn json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::from_slice(&self.body)?)
    }
}

/// Fetcher settings.
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Network policy (`Offline` by default: nothing leaves the machine).
    pub policy: NetworkPolicy,
    /// Store usage.
    pub mode: Mode,
    /// Content store directory (required for `Frozen` and `CachedFirst`).
    pub store: Option<PathBuf>,
    /// Maximum body size in bytes (default 8 MiB; PDFs use `max_pdf_bytes`).
    pub max_bytes: usize,
    /// Maximum PDF size in bytes (default 30 MiB).
    pub max_pdf_bytes: usize,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Retries for transient failures.
    pub retries: u32,
    /// Base delay of exponential backoff between retries.
    pub backoff: Duration,
    /// Consecutive failures after which a host is disabled for the session (circuit breaker).
    pub failures_to_disable: u32,
    /// Product token used for `robots.txt` matching and the User-Agent.
    pub agent: String,
    /// Accept `http://` to loopback hosts (local mock servers in tests). Never enable this for
    /// real sources.
    pub allow_loopback_http: bool,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            policy: NetworkPolicy::Offline,
            mode: Mode::Live,
            store: None,
            max_bytes: 8 * 1024 * 1024,
            max_pdf_bytes: 30 * 1024 * 1024,
            timeout: Duration::from_secs(30),
            retries: 2,
            backoff: Duration::from_millis(500),
            failures_to_disable: 3,
            agent: "harpia".into(),
            allow_loopback_http: false,
        }
    }
}

/// Parsed pieces of an absolute URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlParts {
    /// Scheme, lowercase.
    pub scheme: String,
    /// Host, lowercase.
    pub host: String,
    /// `host[:port]` as written (lowercase).
    pub authority: String,
    /// Path and query (`/` when empty).
    pub path: String,
}

/// Split an absolute `http(s)` URL. URLs with credentials (`user@host`) are rejected.
pub fn url_parts(url: &str) -> Result<UrlParts> {
    let (scheme, rest) =
        url.split_once("://").ok_or_else(|| HarpiaError::invalid(format!("not an absolute URL: {url}")))?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..end];
    if authority.contains('@') {
        return Err(HarpiaError::invalid("URLs with credentials are not allowed"));
    }
    let host = match authority.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or("").to_string(),
        None => authority.split(':').next().unwrap_or("").to_string(),
    };
    if host.is_empty() {
        return Err(HarpiaError::invalid(format!("URL without host: {url}")));
    }
    let path = rest[end..].split('#').next().unwrap_or("");
    Ok(UrlParts {
        scheme: scheme.to_ascii_lowercase(),
        host: host.to_ascii_lowercase(),
        authority: authority.to_ascii_lowercase(),
        path: if path.is_empty() || path.starts_with('?') { format!("/{path}") } else { path.to_string() },
    })
}

fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

/// Network gatekeeper shared by all adapters.
pub struct Fetcher {
    registry: Arc<SourceRegistry>,
    cfg: FetchConfig,
    http: HttpClient,
    store: Option<ContentStore>,
    log: Mutex<Vec<DisclosureEntry>>,
    robots: Mutex<HashMap<String, Robots>>,
}

impl std::fmt::Debug for Fetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fetcher")
            .field("policy", &self.cfg.policy)
            .field("mode", &self.cfg.mode)
            .finish_non_exhaustive()
    }
}

impl Fetcher {
    /// Build a fetcher for `registry`.
    pub fn new(registry: SourceRegistry, cfg: FetchConfig) -> Result<Self> {
        registry.validate()?;
        let registry = Arc::new(registry);
        let mut http_cfg = HttpConfig::new(cfg.policy).retries(cfg.retries, cfg.backoff);
        http_cfg.failures_to_disable = cfg.failures_to_disable;
        http_cfg.timeout = cfg.timeout;
        http_cfg.max_body_bytes = cfg.max_bytes.max(cfg.max_pdf_bytes);
        http_cfg.product = format!("{}/{}", cfg.agent, env!("CARGO_PKG_VERSION"));
        let mut wildcard_interval = Duration::ZERO;
        for s in &registry.sources {
            let every = Duration::from_millis(s.min_interval_ms);
            for d in &s.domains {
                if d.starts_with("*.") {
                    wildcard_interval = wildcard_interval.max(every);
                } else {
                    http_cfg = http_cfg.host_interval(d.to_ascii_lowercase(), every);
                }
            }
            if http_cfg.contact_email.is_none()
                && let Some(c) = s.contact()
            {
                http_cfg = http_cfg.contact(c);
            }
        }
        http_cfg.default_interval = wildcard_interval;
        let reg = registry.clone();
        let loopback = cfg.allow_loopback_http;
        http_cfg.redirects = RedirectPolicy::Checked {
            max: 5,
            check: Arc::new(move |scheme, host| {
                (scheme == "https" || (loopback && scheme == "http" && is_loopback(host))) && reg.allows_host(host)
            }),
        };
        let store = match &cfg.store {
            Some(d) => Some(ContentStore::open(d)?),
            None => None,
        };
        if matches!(cfg.mode, Mode::Frozen | Mode::CachedFirst) && store.is_none() {
            return Err(HarpiaError::invalid("frozen and cached-first modes need a content store directory"));
        }
        Ok(Self {
            http: HttpClient::new(http_cfg)?,
            registry,
            cfg,
            store,
            log: Mutex::new(Vec::new()),
            robots: Mutex::new(HashMap::new()),
        })
    }

    /// The registry.
    pub fn registry(&self) -> &SourceRegistry {
        &self.registry
    }

    /// Settings.
    pub fn config(&self) -> &FetchConfig {
        &self.cfg
    }

    /// Why nothing can be fetched at all (offline and not frozen), if so.
    pub fn skip_reason(&self) -> Option<String> {
        match (self.cfg.policy, self.cfg.mode) {
            (NetworkPolicy::Offline, Mode::Live) => {
                Some("network access is disabled (offline); pass --online or use a frozen content store".into())
            }
            _ => None,
        }
    }

    /// Copy of the disclosure log so far.
    pub fn disclosure(&self) -> Disclosure {
        Disclosure { entries: self.log.lock().expect("disclosure lock").clone() }
    }

    /// Requests sent over the network by the underlying HTTP client.
    pub fn requests_sent(&self) -> u64 {
        self.http.requests_sent()
    }

    /// Per-host circuit-breaker status.
    pub fn host_status(&self) -> Vec<harpia_http::HostStatus> {
        self.http.status()
    }

    fn check_url(&self, source: &Source, url: &str) -> Result<UrlParts> {
        let p = url_parts(url)?;
        let plain_ok = self.cfg.allow_loopback_http && p.scheme == "http" && is_loopback(&p.host);
        if p.scheme != "https" && !plain_ok {
            return Err(HarpiaError::invalid(format!("only HTTPS is allowed: {url}")));
        }
        if !source.enabled || !source.allows_host(&p.host) {
            return Err(HarpiaError::invalid(format!("host `{}` is not allowed for source `{}`", p.host, source.id)));
        }
        if let Some(r) = source.unavailable_reason() {
            return Err(HarpiaError::invalid(r));
        }
        Ok(p)
    }

    fn key(req: &Request) -> String {
        let mut params: Vec<&(String, String)> = req.params.iter().filter(|(k, _)| k != "mailto").collect();
        params.sort();
        let mut k = format!("GET {}", req.url);
        for (a, b) in params {
            k.push_str(&format!("&{a}={b}"));
        }
        k
    }

    fn display_url(req: &Request) -> String {
        if req.params.is_empty() {
            return req.url.clone();
        }
        let q: Vec<String> = req
            .params
            .iter()
            .map(|(k, v)| if k == "mailto" { format!("{k}=<redacted>") } else { format!("{k}={v}") })
            .collect();
        format!("{}?{}", req.url, q.join("&"))
    }

    /// GET `req` on behalf of `source`, expecting `expect`. Non-2xx statuses are returned (with
    /// unchecked content type); policy violations, transport failures and frozen-store misses are
    /// errors.
    pub async fn get(&self, source: &Source, req: Request, expect: Expect, purpose: &str) -> Result<Fetched> {
        let parts = self.check_url(source, &req.url)?;
        if expect == Expect::Html && !self.robots_allow(source, &parts).await? {
            return Err(HarpiaError::invalid(format!("robots.txt of {} disallows {}", parts.host, parts.path)));
        }
        self.get_unchecked(source, req, expect, purpose, &parts).await
    }

    async fn get_unchecked(
        &self,
        source: &Source,
        req: Request,
        expect: Expect,
        purpose: &str,
        parts: &UrlParts,
    ) -> Result<Fetched> {
        let key = Self::key(&req);
        let entry = |from_store: bool| DisclosureEntry {
            source: source.id.clone(),
            host: parts.host.clone(),
            url: Self::display_url(&req),
            purpose: purpose.into(),
            at: now_rfc3339(),
            from_store,
        };
        if matches!(self.cfg.mode, Mode::Frozen | Mode::CachedFirst)
            && let Some(store) = &self.store
            && let Some(s) = store.get(&key)?
        {
            self.log.lock().expect("disclosure lock").push(entry(true));
            return Ok(Fetched {
                status: s.meta.status,
                content_type: s.meta.content_type,
                url: s.meta.url,
                retrieved_at: s.meta.retrieved_at,
                sha256: s.meta.sha256,
                body: s.body,
                from_store: true,
            });
        }
        if self.cfg.mode == Mode::Frozen {
            return Err(HarpiaError::Remote {
                host: parts.host.clone(),
                message: "not in the frozen content store".into(),
            });
        }
        self.cfg.policy.check(&parts.host)?;
        self.log.lock().expect("disclosure lock").push(entry(false));
        let mut req = req.cached(false);
        if let Some(key) = source.api_key()
            && source.adapter == "semantic-scholar"
        {
            req = req.header("x-api-key", key);
        }
        let r = self.http.send(req).await?;
        let cap = if expect == Expect::Pdf { self.cfg.max_pdf_bytes } else { self.cfg.max_bytes };
        if r.body.len() > cap {
            return Err(HarpiaError::Remote {
                host: parts.host.clone(),
                message: format!("response larger than {cap} bytes"),
            });
        }
        let ok = (200..300).contains(&r.status);
        if ok && !expect.accepts(r.content_type.as_deref(), &r.body) {
            return Err(HarpiaError::Remote {
                host: parts.host.clone(),
                message: format!(
                    "unexpected content type {:?} (wanted {expect:?})",
                    r.content_type.unwrap_or_default()
                ),
            });
        }
        let fetched = Fetched {
            status: r.status,
            sha256: sha256_hex(&r.body),
            content_type: r.content_type,
            url: r.url.unwrap_or_else(|| Self::display_url_plain(&key)),
            retrieved_at: now_rfc3339(),
            body: r.body,
            from_store: false,
        };
        if let Some(store) = &self.store
            && (ok || fetched.status == 404)
        {
            let meta = StoredMeta {
                key,
                url: fetched.url.clone(),
                status: fetched.status,
                content_type: fetched.content_type.clone(),
                retrieved_at: fetched.retrieved_at.clone(),
                sha256: fetched.sha256.clone(),
            };
            if let Err(e) = store.put(&meta, &fetched.body) {
                tracing::warn!("content store write failed: {e}");
            }
        }
        Ok(fetched)
    }

    fn display_url_plain(key: &str) -> String {
        key.trim_start_matches("GET ").split('&').next().unwrap_or("").to_string()
    }

    async fn robots_allow(&self, source: &Source, parts: &UrlParts) -> Result<bool> {
        let origin = format!("{}://{}", parts.scheme, parts.authority);
        let cached = self.robots.lock().expect("robots lock").get(&origin).cloned();
        let robots = match cached {
            Some(r) => r,
            None => {
                let url = format!("{origin}/robots.txt");
                let rp = UrlParts { path: "/robots.txt".into(), ..parts.clone() };
                let r = match self.get_unchecked(source, Request::get(&url), Expect::Text, "robots.txt", &rp).await {
                    Ok(f) if f.is_success() => Robots::parse(&f.text(), &self.cfg.agent),
                    Ok(f) if (400..500).contains(&f.status) => Robots::allow_all(),
                    Ok(_) => Robots::disallow_all(),
                    Err(e @ HarpiaError::NetworkDisabled { .. }) => return Err(e),
                    Err(_) => Robots::disallow_all(),
                };
                self.robots.lock().expect("robots lock").insert(origin, r.clone());
                r
            }
        };
        Ok(robots.allowed(&parts.path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_parts_rules() {
        let p = url_parts("https://API.crossref.org/works?q=1#frag").unwrap();
        assert_eq!((p.scheme.as_str(), p.host.as_str(), p.path.as_str()), ("https", "api.crossref.org", "/works?q=1"));
        assert_eq!(url_parts("https://x.org").unwrap().path, "/");
        assert_eq!(url_parts("https://x.org?a=1").unwrap().path, "/?a=1");
        assert_eq!(url_parts("http://[::1]:8080/a").unwrap().host, "::1");
        assert_eq!(url_parts("https://x.org:8443/a").unwrap().authority, "x.org:8443");
        assert!(url_parts("https://user:pw@x.org/").is_err());
        assert!(url_parts("x.org/a").is_err());
    }

    #[test]
    fn content_type_expectations() {
        assert!(Expect::Json.accepts(Some("application/vnd.api+json; charset=utf-8"), b"{}"));
        assert!(Expect::Json.accepts(None, b"[1]"));
        assert!(!Expect::Json.accepts(Some("text/html"), b"<html>"));
        assert!(Expect::Pdf.accepts(Some("application/octet-stream"), b"%PDF-1.7"));
        assert!(Expect::Xml.accepts(Some("application/atom+xml"), b""));
        assert!(!Expect::Html.accepts(Some("application/pdf"), b""));
    }

    #[test]
    fn frozen_needs_store_and_offline_skips() {
        let reg = SourceRegistry::builtin();
        assert!(Fetcher::new(reg.clone(), FetchConfig { mode: Mode::Frozen, ..Default::default() }).is_err());
        let f = Fetcher::new(reg, FetchConfig::default()).unwrap();
        assert!(f.skip_reason().unwrap().contains("offline"));
    }
}
