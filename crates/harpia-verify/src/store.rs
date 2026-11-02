//! Content store: every fetched response saved with its URL, retrieval time and SHA-256, so
//! verification runs can be replayed exactly (`Mode::Frozen`).
//!
//! Layout: `<dir>/<sha256(key)>.json` (metadata) and `<dir>/<sha256(key)>.body`.

use harpia_core::{HarpiaError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Metadata of a stored response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredMeta {
    /// Request key (method, URL with sorted parameters).
    pub key: String,
    /// URL that was fetched.
    pub url: String,
    /// HTTP status.
    pub status: u16,
    /// Content type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Retrieval time (RFC 3339, UTC).
    pub retrieved_at: String,
    /// SHA-256 of the body, hex.
    pub sha256: String,
}

/// A stored response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stored {
    /// Metadata.
    pub meta: StoredMeta,
    /// Body bytes.
    pub body: Vec<u8>,
}

/// Directory-backed content store.
#[derive(Debug, Clone)]
pub struct ContentStore {
    dir: PathBuf,
}

/// Hex SHA-256.
pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

impl ContentStore {
    /// Open (creating) a store directory.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| HarpiaError::io(&dir, e))?;
        Ok(Self { dir })
    }

    fn paths(&self, key: &str) -> (PathBuf, PathBuf) {
        let h = sha256_hex(key.as_bytes());
        (self.dir.join(format!("{h}.json")), self.dir.join(format!("{h}.body")))
    }

    /// Fetch a stored response.
    pub fn get(&self, key: &str) -> Result<Option<Stored>> {
        let (m, b) = self.paths(key);
        let Ok(meta) = std::fs::read(&m) else { return Ok(None) };
        let meta: StoredMeta = serde_json::from_slice(&meta)?;
        let body = std::fs::read(&b).map_err(|e| HarpiaError::io(&b, e))?;
        if sha256_hex(&body) != meta.sha256 {
            return Err(HarpiaError::Validation(format!("stored body for {} does not match its hash", meta.url)));
        }
        Ok(Some(Stored { meta, body }))
    }

    /// Store a response (body first, then metadata, each atomically).
    pub fn put(&self, meta: &StoredMeta, body: &[u8]) -> Result<()> {
        let (m, b) = self.paths(&meta.key);
        write_atomic(&b, body)?;
        write_atomic(&m, &serde_json::to_vec_pretty(meta)?)
    }

    /// Number of stored responses.
    pub fn len(&self) -> usize {
        std::fs::read_dir(&self.dir)
            .map(|d| d.filter_map(|e| e.ok()).filter(|e| e.path().extension().is_some_and(|x| x == "json")).count())
            .unwrap_or(0)
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data).map_err(|e| HarpiaError::io(&tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| HarpiaError::io(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_tamper_detection() {
        let dir = tempfile::tempdir().unwrap();
        let s = ContentStore::open(dir.path()).unwrap();
        let meta = StoredMeta {
            key: "GET https://x.org/a".into(),
            url: "https://x.org/a".into(),
            status: 200,
            content_type: Some("text/plain".into()),
            retrieved_at: "2026-11-02T00:00:00Z".into(),
            sha256: sha256_hex(b"hello"),
        };
        s.put(&meta, b"hello").unwrap();
        assert_eq!(s.get(&meta.key).unwrap().unwrap().body, b"hello");
        assert_eq!(s.len(), 1);
        assert!(s.get("other").unwrap().is_none());
        let (_, body) = s.paths(&meta.key);
        std::fs::write(body, b"tampered").unwrap();
        assert!(s.get(&meta.key).is_err());
    }
}
