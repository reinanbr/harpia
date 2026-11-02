//! Resumable keyed JSONL cache for incremental jobs (translation, annotation, synthetic data,
//! LLM labeling): already-computed keys are skipped and new results are appended immediately,
//! so an interrupted job resumes where it stopped.

use crate::io::{JsonlReader, JsonlWriter};
use harpia_core::Result;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
struct Entry<K, V> {
    key: K,
    value: V,
}

/// A JSONL-backed map from `K` to `V`, persisted as `{"key": ..., "value": ...}` lines.
///
/// ```
/// use harpia_data::JsonlCache;
/// let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("translations.jsonl");
/// {
///     let mut c: JsonlCache<String, String> = JsonlCache::open(&path).unwrap();
///     c.insert("hello".into(), "olá".into()).unwrap();
/// }
/// let c: JsonlCache<String, String> = JsonlCache::open(&path).unwrap();
/// assert_eq!(c.get(&"hello".to_string()).map(String::as_str), Some("olá"));
/// let todo = c.missing(["hello".to_string(), "world".to_string()]);
/// assert_eq!(todo, ["world"]);
/// ```
pub struct JsonlCache<K, V> {
    path: PathBuf,
    map: HashMap<K, V>,
    writer: Option<JsonlWriter>,
}

impl<K, V> JsonlCache<K, V>
where
    K: Eq + Hash + Clone + Serialize + DeserializeOwned,
    V: Serialize + DeserializeOwned,
{
    /// Load `path` if it exists (later lines win for repeated keys).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut map = HashMap::new();
        if path.exists() {
            for row in JsonlReader::<Entry<K, V>>::open(&path)? {
                let (_, e) = row?;
                map.insert(e.key, e.value);
            }
        }
        Ok(Self { path, map, writer: None })
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Cached value.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// `true` if `key` is cached.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Store and persist (flushed immediately).
    pub fn insert(&mut self, key: K, value: V) -> Result<()> {
        if self.writer.is_none() {
            self.writer = Some(JsonlWriter::append(&self.path)?.flush_every(1));
        }
        self.writer.as_mut().expect("writer").write(&Entry { key: &key, value: &value })?;
        self.map.insert(key, value);
        Ok(())
    }

    /// Keys from `keys` that are not cached yet, in order, without duplicates.
    pub fn missing<I: IntoIterator<Item = K>>(&self, keys: I) -> Vec<K> {
        let mut seen = std::collections::HashSet::new();
        keys.into_iter().filter(|k| !self.map.contains_key(k) && seen.insert(k.clone())).collect()
    }

    /// Iterate over cached entries.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter()
    }
}
