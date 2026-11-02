//! Response cache in a pure-Rust embedded database (`redb`).

use harpia_core::traits::Cache;
use harpia_core::{HarpiaError, Result};
use redb::{Database, ReadableDatabase, TableDefinition};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("responses");

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn db_err(e: impl std::fmt::Display) -> HarpiaError {
    HarpiaError::Other(format!("http cache: {e}"))
}

/// A TTL key/value store. Values are stored as `expires_at (u64 LE) || payload`.
pub struct DiskCache {
    db: Database,
    default_ttl: Duration,
}

impl std::fmt::Debug for DiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskCache").field("default_ttl", &self.default_ttl).finish_non_exhaustive()
    }
}

impl DiskCache {
    /// Open or create the database at `path`.
    pub fn open(path: &Path, default_ttl: Duration) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| HarpiaError::io(parent, e))?;
        }
        let db = Database::create(path).map_err(db_err)?;
        let w = db.begin_write().map_err(db_err)?;
        w.open_table(TABLE).map_err(db_err)?;
        w.commit().map_err(db_err)?;
        Ok(Self { db, default_ttl })
    }
}

impl Cache for DiskCache {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let r = self.db.begin_read().map_err(db_err)?;
        let t = r.open_table(TABLE).map_err(db_err)?;
        let Some(v) = t.get(key).map_err(db_err)? else { return Ok(None) };
        let bytes = v.value();
        if bytes.len() < 8 {
            return Ok(None);
        }
        let expires = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"));
        Ok((expires > now_secs()).then(|| bytes[8..].to_vec()))
    }

    fn put(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
        let expires = now_secs() + ttl.unwrap_or(self.default_ttl).as_secs();
        let mut buf = Vec::with_capacity(8 + value.len());
        buf.extend_from_slice(&expires.to_le_bytes());
        buf.extend_from_slice(value);
        let w = self.db.begin_write().map_err(db_err)?;
        {
            let mut t = w.open_table(TABLE).map_err(db_err)?;
            t.insert(key, buf.as_slice()).map_err(db_err)?;
        }
        w.commit().map_err(db_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_and_expire() {
        let dir = tempfile::tempdir().unwrap();
        let c = DiskCache::open(&dir.path().join("c.redb"), Duration::from_secs(60)).unwrap();
        c.put("k", b"v", None).unwrap();
        assert_eq!(c.get("k").unwrap().as_deref(), Some(&b"v"[..]));
        c.put("old", b"x", Some(Duration::ZERO)).unwrap();
        assert_eq!(c.get("old").unwrap(), None);
        assert_eq!(c.get("missing").unwrap(), None);
    }
}
