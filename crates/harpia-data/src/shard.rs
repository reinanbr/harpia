//! Sharded JSONL datasets with a manifest (`manifest.json`) recording row counts, byte sizes
//! and content hashes, so large datasets can be processed in parallel and resumed.

use crate::io::{JsonlReader, JsonlWriter};
use harpia_core::{HarpiaError, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use xxhash_rust::xxh3::Xxh3;

/// One shard in a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardInfo {
    /// File name relative to the dataset directory.
    pub file: String,
    /// Number of records.
    pub rows: u64,
    /// File size in bytes.
    pub bytes: u64,
    /// XXH3-64 of the file contents, hex.
    pub xxh3: String,
}

/// Dataset manifest.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest format version.
    pub version: u32,
    /// Shards in order.
    pub shards: Vec<ShardInfo>,
}

impl Manifest {
    /// Total rows.
    pub fn rows(&self) -> u64 {
        self.shards.iter().map(|s| s.rows).sum()
    }

    /// Load `dir/manifest.json`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let p = dir.as_ref().join("manifest.json");
        let text = std::fs::read_to_string(&p).map_err(|e| HarpiaError::io(&p, e))?;
        Ok(serde_json::from_str(&text)?)
    }
}

/// Writes records into numbered shards of at most `rows_per_shard` records.
pub struct ShardWriter {
    dir: PathBuf,
    prefix: String,
    rows_per_shard: u64,
    current: Option<(JsonlWriter, String, u64)>,
    manifest: Manifest,
}

impl ShardWriter {
    /// New writer into `dir` (created), files named `{prefix}-00000.jsonl`, ...
    pub fn new(dir: impl AsRef<Path>, prefix: &str, rows_per_shard: u64) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| HarpiaError::io(&dir, e))?;
        Ok(Self {
            dir,
            prefix: prefix.to_string(),
            rows_per_shard: rows_per_shard.max(1),
            current: None,
            manifest: Manifest { version: 1, shards: vec![] },
        })
    }

    /// Write one record.
    pub fn write<T: Serialize>(&mut self, record: &T) -> Result<()> {
        if self.current.as_ref().is_some_and(|(_, _, n)| *n >= self.rows_per_shard) {
            self.close_current()?;
        }
        if self.current.is_none() {
            let name = format!("{}-{:05}.jsonl", self.prefix, self.manifest.shards.len());
            self.current = Some((JsonlWriter::create(self.dir.join(&name))?, name, 0));
        }
        let (w, _, n) = self.current.as_mut().expect("shard open");
        w.write(record)?;
        *n += 1;
        Ok(())
    }

    fn close_current(&mut self) -> Result<()> {
        if let Some((w, name, rows)) = self.current.take() {
            w.finish()?;
            let path = self.dir.join(&name);
            let data = std::fs::read(&path).map_err(|e| HarpiaError::io(&path, e))?;
            let mut h = Xxh3::new();
            h.update(&data);
            self.manifest.shards.push(ShardInfo {
                file: name,
                rows,
                bytes: data.len() as u64,
                xxh3: format!("{:016x}", h.digest()),
            });
        }
        Ok(())
    }

    /// Close the last shard and write `manifest.json`.
    pub fn finish(mut self) -> Result<Manifest> {
        self.close_current()?;
        let p = self.dir.join("manifest.json");
        std::fs::write(&p, serde_json::to_string_pretty(&self.manifest)?).map_err(|e| HarpiaError::io(&p, e))?;
        Ok(self.manifest)
    }
}

/// Iterate over every record of a sharded dataset, in manifest order.
pub fn read_shards<T: DeserializeOwned + 'static>(dir: impl AsRef<Path>) -> Result<impl Iterator<Item = Result<T>>> {
    let dir = dir.as_ref().to_path_buf();
    let manifest = Manifest::load(&dir)?;
    let mut readers = Vec::new();
    for s in &manifest.shards {
        readers.push(JsonlReader::<T>::open(dir.join(&s.file))?);
    }
    Ok(readers.into_iter().flatten().map(|r| r.map(|(_, v)| v)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ShardWriter::new(dir.path(), "train", 3).unwrap();
        for i in 0..7 {
            w.write(&serde_json::json!({"i": i})).unwrap();
        }
        let m = w.finish().unwrap();
        assert_eq!(m.shards.len(), 3);
        assert_eq!(m.rows(), 7);
        assert_eq!(Manifest::load(dir.path()).unwrap(), m);
        let back: Vec<serde_json::Value> = read_shards(dir.path()).unwrap().collect::<Result<_>>().unwrap();
        assert_eq!(back.len(), 7);
        assert_eq!(back[6]["i"], 6);
    }
}
