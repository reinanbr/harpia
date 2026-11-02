//! A process-wide model cache: each model directory is loaded once per type and shared.
//!
//! This is the only global state in Harpia, and it is explicit: models are inserted only through
//! [`ModelCache::get_or_load`] and can be evicted with [`ModelCache::evict`].

use harpia_core::Result;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

type Entry = Arc<dyn Any + Send + Sync>;

static CACHE: LazyLock<Mutex<HashMap<(PathBuf, TypeId), Entry>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Shared, load-once model handles.
///
/// ```no_run
/// use harpia_local::{ModelCache, SequenceClassifier};
/// let clf = ModelCache::get_or_load("models/nli", SequenceClassifier::load)?;
/// let same = ModelCache::get_or_load("models/nli", SequenceClassifier::load)?; // no reload
/// assert!(std::sync::Arc::ptr_eq(&clf, &same));
/// # Ok::<(), harpia_core::HarpiaError>(())
/// ```
pub struct ModelCache;

impl ModelCache {
    /// The cached `T` for `dir`, loading it with `load` the first time. Concurrent first calls
    /// may load twice; only one copy is kept.
    pub fn get_or_load<T, F>(dir: impl AsRef<Path>, load: F) -> Result<Arc<T>>
    where
        T: Send + Sync + 'static,
        F: FnOnce(&Path) -> Result<T>,
    {
        let path = dir.as_ref().canonicalize().unwrap_or_else(|_| dir.as_ref().to_path_buf());
        let key = (path.clone(), TypeId::of::<T>());
        if let Some(e) = CACHE.lock().expect("model cache lock").get(&key).cloned() {
            return Ok(e.downcast::<T>().expect("type keyed"));
        }
        let model: Entry = Arc::new(load(&path)?);
        let mut c = CACHE.lock().expect("model cache lock");
        let e = c.entry(key).or_insert(model).clone();
        Ok(e.downcast::<T>().expect("type keyed"))
    }

    /// Drop every cached model loaded from `dir`.
    pub fn evict(dir: impl AsRef<Path>) {
        let path = dir.as_ref().canonicalize().unwrap_or_else(|_| dir.as_ref().to_path_buf());
        CACHE.lock().expect("model cache lock").retain(|(p, _), _| *p != path);
    }

    /// Number of cached models.
    pub fn len() -> usize {
        CACHE.lock().expect("model cache lock").len()
    }

    /// `true` when nothing is cached.
    pub fn is_empty() -> bool {
        Self::len() == 0
    }
}
