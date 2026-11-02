//! Hugging Face Hub downloads (feature `hub`), through the polite `harpia-http` client.
//!
//! Nothing is downloaded unless the caller passes [`NetworkPolicy::Online`]; files go to an
//! explicit cache directory (`<cache>/<owner>--<name>/<revision>/`), are written atomically, and
//! files already present are not fetched again. Set `HF_TOKEN` for gated or private repositories.

use harpia_core::{HarpiaError, NetworkPolicy, Result};
use harpia_http::blocking::HttpClient;
use harpia_http::{HttpConfig, Request};
use std::path::{Path, PathBuf};

/// Files Harpia needs from a model repository (weights in safetensors, configs, tokenizers).
pub const DEFAULT_PATTERNS: &[&str] = &["*.json", "*.safetensors", "*.spm", "*.model", "*.txt"];

/// Shell-style match supporting `*` (any run) and `?` (one char).
pub fn glob_match(pattern: &str, name: &str) -> bool {
    fn go(p: &[char], n: &[char]) -> bool {
        match (p.first(), n.first()) {
            (None, None) => true,
            (Some('*'), _) => go(&p[1..], n) || (!n.is_empty() && go(p, &n[1..])),
            (Some('?'), Some(_)) => go(&p[1..], &n[1..]),
            (Some(a), Some(b)) if a == b => go(&p[1..], &n[1..]),
            _ => false,
        }
    }
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    go(&p, &n)
}

/// A Hub client with a cache directory.
pub struct Hub {
    http: HttpClient,
    endpoint: String,
    cache: PathBuf,
    token: Option<String>,
    /// Maximum size of one file (default 50 GB).
    pub max_file_bytes: u64,
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hub").field("endpoint", &self.endpoint).field("cache", &self.cache).finish_non_exhaustive()
    }
}

impl Hub {
    /// Client storing files under `cache_dir`. Offline policies refuse every download.
    pub fn new(policy: NetworkPolicy, cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let mut cfg = HttpConfig::new(policy);
        cfg.timeout = std::time::Duration::from_secs(3600);
        Ok(Self {
            http: HttpClient::new(cfg)?,
            endpoint: std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into()),
            cache: cache_dir.into(),
            token: std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
            max_file_bytes: 50 * 1024 * 1024 * 1024,
        })
    }

    /// Use another endpoint (mirrors, tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into().trim_end_matches('/').to_string();
        self
    }

    fn auth(&self, r: Request) -> Request {
        match &self.token {
            Some(t) => r.header("authorization", format!("Bearer {t}")),
            None => r,
        }
    }

    /// Local directory of `repo` at `revision`.
    pub fn snapshot_dir(&self, repo: &str, revision: &str) -> PathBuf {
        self.cache.join(repo.replace('/', "--")).join(revision)
    }

    /// File names in `repo` at `revision`.
    pub fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<String>> {
        let url = format!("{}/api/models/{repo}/revision/{revision}", self.endpoint);
        let v: serde_json::Value = self.http.get_json(self.auth(Request::get(url).cached(false)))?;
        Ok(v["siblings"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s["rfilename"].as_str().map(String::from))
            .collect())
    }

    /// Download every file of `repo` at `revision` matching one of `patterns` (default
    /// [`DEFAULT_PATTERNS`]) and return the local directory, ready for `ModelDir::open`.
    pub fn snapshot(&self, repo: &str, revision: &str, patterns: Option<&[&str]>) -> Result<PathBuf> {
        let patterns = patterns.unwrap_or(DEFAULT_PATTERNS);
        let dir = self.snapshot_dir(repo, revision);
        let files: Vec<String> = self
            .list_files(repo, revision)?
            .into_iter()
            .filter(|f| patterns.iter().any(|p| glob_match(p, f)))
            .collect();
        if files.is_empty() {
            return Err(HarpiaError::Remote {
                host: self.endpoint.clone(),
                message: format!("{repo}@{revision}: no files match {patterns:?}"),
            });
        }
        for f in &files {
            let url = format!("{}/{repo}/resolve/{revision}/{f}", self.endpoint);
            let dest = dir.join(f);
            self.http.download_to_file(self.auth(Request::get(url).cached(false)), &dest, self.max_file_bytes)?;
        }
        Ok(dir)
    }

    /// `true` when `dir` looks like a usable snapshot (has `config.json`).
    pub fn is_complete(dir: &Path) -> bool {
        dir.join("config.json").is_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globbing() {
        assert!(glob_match("*.safetensors", "model-00001-of-00002.safetensors"));
        assert!(glob_match("*.json", "tokenizer.json") && !glob_match("*.json", "model.bin"));
        assert!(glob_match("model-?.bin", "model-1.bin"));
    }

    #[test]
    fn offline_refuses() {
        let hub = Hub::new(NetworkPolicy::Offline, std::env::temp_dir().join("harpia-hub-test")).unwrap();
        assert!(matches!(hub.list_files("x/y", "main"), Err(HarpiaError::NetworkDisabled { .. })));
    }
}
