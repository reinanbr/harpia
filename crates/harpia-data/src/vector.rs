//! Vector indexes for retrieval: an exact flat index (batched, parallel with `rayon`) and an
//! approximate HNSW graph, both with incremental add/delete and on-disk persistence.
//!
//! Vectors live in one contiguous little-endian `f32` block. The file format puts that block at
//! a 64-byte aligned offset right after a fixed header, so it can be memory-mapped by readers
//! that want to; Harpia itself reads it with plain I/O (the crate has no `unsafe`).
//!
//! File layout (`.hvec`): magic `HRPV`, format version `u32`, dim `u32`, metric `u32`, rows
//! `u64`, JSON metadata length `u64`, padding to 64 bytes, `rows × dim` `f32` values, then the JSON
//! metadata (ids, tombstones and, for HNSW, the graph).
//!
//! Determinism: HNSW levels come from a seeded [`SplitMix64`], insertion is sequential, and ties
//! are broken by row index, so the same inputs and seed build the same graph on every platform.

use crate::rng::SplitMix64;
use harpia_core::traits::Hit;
use harpia_core::{HarpiaError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{Read, Write};
use std::path::Path;

/// Similarity metric. Higher scores are better for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Cosine similarity (vectors are L2-normalized on insert, queries on search).
    #[default]
    Cosine,
    /// Raw dot product.
    Dot,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalized(v: &[f32]) -> Vec<f32> {
    let n = dot(v, v).sqrt();
    if n == 0.0 { v.to_vec() } else { v.iter().map(|x| x / n).collect() }
}

/// Rows of vectors with string ids and tombstones (shared by both indexes).
#[derive(Debug, Clone, PartialEq, Default)]
struct Store {
    dim: usize,
    metric: Metric,
    ids: Vec<String>,
    data: Vec<f32>,
    deleted: Vec<bool>,
    by_id: HashMap<String, usize>,
}

impl Store {
    fn new(dim: usize, metric: Metric) -> Self {
        Self { dim, metric, ..Default::default() }
    }

    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    fn prepare(&self, v: &[f32]) -> Result<Vec<f32>> {
        if v.len() != self.dim {
            return Err(HarpiaError::invalid(format!("vector has dimension {}, index expects {}", v.len(), self.dim)));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(HarpiaError::invalid("vector contains NaN or infinity"));
        }
        Ok(match self.metric {
            Metric::Cosine => normalized(v),
            Metric::Dot => v.to_vec(),
        })
    }

    /// Append a row (tombstoning an existing row with the same id). Returns the new row index.
    fn push(&mut self, id: &str, v: &[f32]) -> Result<usize> {
        let v = self.prepare(v)?;
        if let Some(old) = self.by_id.get(id).copied() {
            self.deleted[old] = true;
        }
        let row = self.ids.len();
        self.ids.push(id.to_string());
        self.data.extend_from_slice(&v);
        self.deleted.push(false);
        self.by_id.insert(id.to_string(), row);
        Ok(row)
    }

    fn remove(&mut self, id: &str) -> bool {
        match self.by_id.remove(id) {
            Some(row) => {
                self.deleted[row] = true;
                true
            }
            None => false,
        }
    }

    fn live(&self) -> usize {
        self.by_id.len()
    }

    fn hit(&self, row: usize, score: f32) -> Hit {
        Hit { id: self.ids[row].clone(), score, span: None }
    }
}

#[derive(Serialize, Deserialize)]
struct FileMeta {
    kind: String,
    ids: Vec<String>,
    deleted: Vec<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hnsw: Option<HnswGraph>,
}

const MAGIC: &[u8; 4] = b"HRPV";
const VERSION: u32 = 1;
const HEADER: usize = 64;

fn write_file(path: &Path, store: &Store, meta: &FileMeta) -> Result<()> {
    let io = |e| HarpiaError::io(path, e);
    let json = serde_json::to_vec(meta)?;
    let tmp = path.with_extension("hvec.tmp");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(io)?);
        let mut header = Vec::with_capacity(HEADER);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&(store.dim as u32).to_le_bytes());
        header.extend_from_slice(
            &(match store.metric {
                Metric::Cosine => 0u32,
                Metric::Dot => 1u32,
            })
            .to_le_bytes(),
        );
        header.extend_from_slice(&(store.ids.len() as u64).to_le_bytes());
        header.extend_from_slice(&(json.len() as u64).to_le_bytes());
        header.resize(HEADER, 0);
        f.write_all(&header).map_err(io)?;
        for x in &store.data {
            f.write_all(&x.to_le_bytes()).map_err(io)?;
        }
        f.write_all(&json).map_err(io)?;
        f.flush().map_err(io)?;
    }
    std::fs::rename(&tmp, path).map_err(io)
}

fn read_file(path: &Path) -> Result<(Store, FileMeta)> {
    let io = |e| HarpiaError::io(path, e);
    let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(io)?);
    let mut header = [0u8; HEADER];
    f.read_exact(&mut header).map_err(io)?;
    let u32_at = |i: usize| u32::from_le_bytes(header[i..i + 4].try_into().expect("4 bytes"));
    let u64_at = |i: usize| u64::from_le_bytes(header[i..i + 8].try_into().expect("8 bytes"));
    if &header[..4] != MAGIC || u32_at(4) != VERSION {
        return Err(HarpiaError::invalid(format!(
            "{} is not a Harpia vector index (version {VERSION})",
            path.display()
        )));
    }
    let dim = u32_at(8) as usize;
    let metric = if u32_at(12) == 1 { Metric::Dot } else { Metric::Cosine };
    let rows = u64_at(16) as usize;
    let json_len = u64_at(24) as usize;
    let mut raw = vec![0u8; rows * dim * 4];
    f.read_exact(&mut raw).map_err(io)?;
    let data = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect();
    let mut json = vec![0u8; json_len];
    f.read_exact(&mut json).map_err(io)?;
    let meta: FileMeta = serde_json::from_slice(&json)?;
    if meta.ids.len() != rows || meta.deleted.len() != rows {
        return Err(HarpiaError::invalid(format!("{}: row count mismatch", path.display())));
    }
    let by_id = meta.ids.iter().enumerate().filter(|(i, _)| !meta.deleted[*i]).map(|(i, id)| (id.clone(), i)).collect();
    let store = Store { dim, metric, ids: meta.ids.clone(), data, deleted: meta.deleted.clone(), by_id };
    Ok((store, meta))
}

/// A row filter: return `false` to exclude an id from results.
pub type IdFilter<'a> = &'a (dyn Fn(&str) -> bool + Sync);

/// Exact nearest-neighbour index (brute force, parallel).
///
/// ```
/// use harpia_data::vector::{FlatIndex, Metric};
/// let mut ix = FlatIndex::new(2, Metric::Cosine);
/// ix.add("east", &[1.0, 0.0]).unwrap();
/// ix.add("north", &[0.0, 1.0]).unwrap();
/// let hits = ix.search(&[0.9, 0.1], 1, None).unwrap();
/// assert_eq!(hits[0].id, "east");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FlatIndex {
    store: Store,
}

impl FlatIndex {
    /// Empty index.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self { store: Store::new(dim, metric) }
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.store.dim
    }

    /// Live (non-deleted) vectors.
    pub fn len(&self) -> usize {
        self.store.live()
    }

    /// `true` when there are no live vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add or replace a vector.
    pub fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        self.store.push(id, vector).map(|_| ())
    }

    /// Delete by id; returns whether it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        self.store.remove(id)
    }

    /// Whether `id` is present.
    pub fn contains(&self, id: &str) -> bool {
        self.store.by_id.contains_key(id)
    }

    /// Top-`k` hits, best first (ties by insertion order).
    pub fn search(&self, query: &[f32], k: usize, filter: Option<IdFilter<'_>>) -> Result<Vec<Hit>> {
        let q = self.store.prepare(query)?;
        let s = &self.store;
        let mut scored: Vec<(usize, f32)> = (0..s.ids.len())
            .into_par_iter()
            .filter(|&i| !s.deleted[i] && filter.is_none_or(|f| f(&s.ids[i])))
            .map(|i| (i, dot(&q, s.row(i))))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(scored.into_iter().take(k).map(|(i, sc)| s.hit(i, sc)).collect())
    }

    /// Search several queries in parallel.
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<Hit>>> {
        queries.par_iter().map(|q| self.search(q, k, None)).collect()
    }

    /// Drop tombstoned rows.
    pub fn compact(&mut self) {
        let fresh = Store::new(self.store.dim, self.store.metric);
        let old = std::mem::replace(&mut self.store, fresh);
        for i in 0..old.ids.len() {
            if !old.deleted[i] {
                self.store.ids.push(old.ids[i].clone());
                self.store.data.extend_from_slice(old.row(i));
                self.store.deleted.push(false);
                self.store.by_id.insert(old.ids[i].clone(), self.store.ids.len() - 1);
            }
        }
    }

    /// Save to a `.hvec` file (atomic rename).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let meta = FileMeta {
            kind: "flat".into(),
            ids: self.store.ids.clone(),
            deleted: self.store.deleted.clone(),
            hnsw: None,
        };
        write_file(path.as_ref(), &self.store, &meta)
    }

    /// Load from a `.hvec` file written by [`FlatIndex::save`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let (store, meta) = read_file(path.as_ref())?;
        if meta.kind != "flat" {
            return Err(HarpiaError::invalid(format!("{} holds a {} index", path.as_ref().display(), meta.kind)));
        }
        Ok(Self { store })
    }
}

/// HNSW build/search parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswParams {
    /// Neighbours per node on upper layers (layer 0 keeps `2 * m`).
    pub m: usize,
    /// Candidate list size while building.
    pub ef_construction: usize,
    /// Candidate list size while searching (raised to `k` when smaller).
    pub ef_search: usize,
    /// Seed for level assignment.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self { m: 16, ef_construction: 200, ef_search: 64, seed: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
struct HnswGraph {
    params: Option<HnswParams>,
    /// `links[node][level]` = neighbour rows.
    links: Vec<Vec<Vec<u32>>>,
    entry: Option<u32>,
    max_level: usize,
    rng_state: u64,
}

#[derive(Clone, Copy, PartialEq)]
struct Cand(f32, usize);
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    // Max-heap by score; ties prefer the lower row.
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
    }
}

/// Approximate nearest-neighbour index (Malkov & Yashunin's HNSW).
///
/// Deleted vectors stay in the graph as routing nodes and are filtered from results; call
/// [`HnswIndex::rebuild`] after many deletions.
#[derive(Debug, Clone, PartialEq)]
pub struct HnswIndex {
    store: Store,
    params: HnswParams,
    g: HnswGraph,
    rng: SplitMix64,
}

impl HnswIndex {
    /// Empty index.
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        Self {
            store: Store::new(dim, metric),
            params,
            g: HnswGraph { params: Some(params), ..Default::default() },
            rng: SplitMix64::new(params.seed),
        }
    }

    /// Parameters.
    pub fn params(&self) -> HnswParams {
        self.params
    }

    /// Set the search candidate list size.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.params.ef_search = ef.max(1);
    }

    /// Live vectors.
    pub fn len(&self) -> usize {
        self.store.live()
    }

    /// `true` when there are no live vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `id` is present.
    pub fn contains(&self, id: &str) -> bool {
        self.store.by_id.contains_key(id)
    }

    fn sim(&self, q: &[f32], i: usize) -> f32 {
        dot(q, self.store.row(i))
    }

    fn random_level(&mut self) -> usize {
        let ml = 1.0 / (self.params.m.max(2) as f64).ln();
        let u = 1.0 - self.rng.next_f64(); // (0, 1]
        ((-u.ln()) * ml).floor() as usize
    }

    fn search_layer(&self, q: &[f32], entry: &[usize], ef: usize, level: usize) -> Vec<Cand> {
        let mut visited: HashSet<usize> = entry.iter().copied().collect();
        let mut candidates: BinaryHeap<Cand> = entry.iter().map(|&e| Cand(self.sim(q, e), e)).collect();
        // Min-heap of current results via Reverse ordering.
        let mut results: BinaryHeap<std::cmp::Reverse<Cand>> =
            candidates.iter().map(|&c| std::cmp::Reverse(c)).collect();
        while let Some(c) = candidates.pop() {
            let worst = results.peek().map_or(f32::NEG_INFINITY, |r| r.0.0);
            if c.0 < worst && results.len() >= ef {
                break;
            }
            for &n in self.g.links[c.1].get(level).map_or(&[][..], |v| v.as_slice()) {
                let n = n as usize;
                if !visited.insert(n) {
                    continue;
                }
                let s = self.sim(q, n);
                let worst = results.peek().map_or(f32::NEG_INFINITY, |r| r.0.0);
                if results.len() < ef || s > worst {
                    candidates.push(Cand(s, n));
                    results.push(std::cmp::Reverse(Cand(s, n)));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut out: Vec<Cand> = results.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        out
    }

    /// Heuristic neighbour selection (keeps diverse neighbours, as in the HNSW paper).
    fn select(&self, candidates: &[Cand], m: usize) -> Vec<usize> {
        let mut chosen: Vec<usize> = Vec::with_capacity(m);
        for c in candidates {
            if chosen.len() >= m {
                break;
            }
            let row = self.store.row(c.1);
            if chosen.iter().all(|&o| dot(row, self.store.row(o)) < c.0) {
                chosen.push(c.1);
            }
        }
        // Fill up with the best remaining ones so nodes keep enough links.
        for c in candidates {
            if chosen.len() >= m {
                break;
            }
            if !chosen.contains(&c.1) {
                chosen.push(c.1);
            }
        }
        chosen
    }

    fn max_links(&self, level: usize) -> usize {
        if level == 0 { self.params.m * 2 } else { self.params.m }
    }

    /// Add or replace a vector.
    pub fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        let row = self.store.push(id, vector)?;
        let level = self.random_level();
        self.g.links.push(vec![Vec::new(); level + 1]);
        let Some(entry) = self.g.entry else {
            self.g.entry = Some(row as u32);
            self.g.max_level = level;
            return Ok(());
        };
        let q = self.store.row(row).to_vec();
        let mut ep = vec![entry as usize];
        for l in (level + 1..=self.g.max_level).rev() {
            ep = vec![self.search_layer(&q, &ep, 1, l)[0].1];
        }
        for l in (0..=level.min(self.g.max_level)).rev() {
            let cands = self.search_layer(&q, &ep, self.params.ef_construction, l);
            let neighbours = self.select(&cands, self.params.m);
            self.g.links[row][l] = neighbours.iter().map(|&n| n as u32).collect();
            let cap = self.max_links(l);
            for &n in &neighbours {
                let mut list: Vec<usize> = self.g.links[n][l].iter().map(|&x| x as usize).collect();
                list.push(row);
                if list.len() > cap {
                    let base = self.store.row(n).to_vec();
                    let mut c: Vec<Cand> = list.iter().map(|&x| Cand(self.sim(&base, x), x)).collect();
                    c.sort_by(|a, b| b.cmp(a));
                    list = self.select(&c, cap);
                }
                self.g.links[n][l] = list.into_iter().map(|x| x as u32).collect();
            }
            ep = cands.iter().map(|c| c.1).collect();
        }
        if level > self.g.max_level {
            self.g.max_level = level;
            self.g.entry = Some(row as u32);
        }
        Ok(())
    }

    /// Delete by id (tombstone); returns whether it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        self.store.remove(id)
    }

    /// Approximate top-`k`, best first.
    pub fn search(&self, query: &[f32], k: usize, filter: Option<IdFilter<'_>>) -> Result<Vec<Hit>> {
        let q = self.store.prepare(query)?;
        let Some(entry) = self.g.entry else { return Ok(Vec::new()) };
        let mut ep = vec![entry as usize];
        for l in (1..=self.g.max_level).rev() {
            ep = vec![self.search_layer(&q, &ep, 1, l)[0].1];
        }
        // Over-fetch when rows are excluded so filtered searches still return k hits.
        let excluded = self.store.ids.len() - self.store.live();
        let ef = self.params.ef_search.max(k) + if filter.is_some() { k * 4 } else { 0 } + excluded.min(k * 4);
        let s = &self.store;
        Ok(self
            .search_layer(&q, &ep, ef, 0)
            .into_iter()
            .filter(|c| !s.deleted[c.1] && filter.is_none_or(|f| f(&s.ids[c.1])))
            .take(k)
            .map(|c| s.hit(c.1, c.0))
            .collect())
    }

    /// Rebuild the graph from live vectors only (drops tombstones; same seed, same result).
    pub fn rebuild(&mut self) -> Result<()> {
        let mut fresh = HnswIndex::new(self.store.dim, self.store.metric, self.params);
        for i in 0..self.store.ids.len() {
            if !self.store.deleted[i] {
                let v = self.store.row(i).to_vec();
                fresh.add(&self.store.ids[i].clone(), &v)?;
            }
        }
        *self = fresh;
        Ok(())
    }

    /// Save to a `.hvec` file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut g = self.g.clone();
        g.params = Some(self.params);
        g.rng_state = self.rng.state();
        let meta = FileMeta {
            kind: "hnsw".into(),
            ids: self.store.ids.clone(),
            deleted: self.store.deleted.clone(),
            hnsw: Some(g),
        };
        write_file(path.as_ref(), &self.store, &meta)
    }

    /// Load from a `.hvec` file written by [`HnswIndex::save`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let (store, meta) = read_file(path.as_ref())?;
        let g = meta
            .hnsw
            .filter(|_| meta.kind == "hnsw")
            .ok_or_else(|| HarpiaError::invalid(format!("{} does not hold an HNSW index", path.as_ref().display())))?;
        let params = g.params.unwrap_or_default();
        let rng = SplitMix64::from_state(g.rng_state);
        Ok(Self { store, params, g, rng })
    }
}

/// Either index kind behind one interface.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorIndex {
    /// Exact.
    Flat(FlatIndex),
    /// Approximate.
    Hnsw(Box<HnswIndex>),
}

impl VectorIndex {
    /// Add or replace.
    pub fn add(&mut self, id: &str, v: &[f32]) -> Result<()> {
        match self {
            Self::Flat(i) => i.add(id, v),
            Self::Hnsw(i) => i.add(id, v),
        }
    }
    /// Delete.
    pub fn remove(&mut self, id: &str) -> bool {
        match self {
            Self::Flat(i) => i.remove(id),
            Self::Hnsw(i) => i.remove(id),
        }
    }
    /// Search.
    pub fn search(&self, q: &[f32], k: usize, filter: Option<IdFilter<'_>>) -> Result<Vec<Hit>> {
        match self {
            Self::Flat(i) => i.search(q, k, filter),
            Self::Hnsw(i) => i.search(q, k, filter),
        }
    }
    /// Live vectors.
    pub fn len(&self) -> usize {
        match self {
            Self::Flat(i) => i.len(),
            Self::Hnsw(i) => i.len(),
        }
    }
    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Save.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        match self {
            Self::Flat(i) => i.save(path),
            Self::Hnsw(i) => i.save(path),
        }
    }
    /// Load either kind.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let (_, meta) = read_file(path)?;
        if meta.kind == "hnsw" {
            Ok(Self::Hnsw(Box::new(HnswIndex::load(path)?)))
        } else {
            Ok(Self::Flat(FlatIndex::load(path)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut r = SplitMix64::new(seed);
        (0..n).map(|_| (0..dim).map(|_| r.next_f64() as f32 * 2.0 - 1.0).collect()).collect()
    }

    #[test]
    fn flat_add_replace_remove_persist() {
        let mut ix = FlatIndex::new(3, Metric::Cosine);
        ix.add("a", &[1.0, 0.0, 0.0]).unwrap();
        ix.add("b", &[0.0, 1.0, 0.0]).unwrap();
        ix.add("a", &[0.0, 0.0, 1.0]).unwrap(); // replace
        assert_eq!(ix.len(), 2);
        assert_eq!(ix.search(&[0.0, 0.0, 2.0], 1, None).unwrap()[0].id, "a");
        assert!(ix.add("x", &[1.0]).is_err());
        assert!(ix.add("x", &[f32::NAN, 0.0, 0.0]).is_err());
        let only_b = |id: &str| id == "b";
        assert_eq!(ix.search(&[0.0, 0.0, 1.0], 5, Some(&only_b)).unwrap().len(), 1);
        assert!(ix.remove("b") && !ix.remove("b"));
        let dir = tempfile::tempdir().unwrap();
        ix.save(dir.path().join("f.hvec")).unwrap();
        let back = FlatIndex::load(dir.path().join("f.hvec")).unwrap();
        assert_eq!(back.search(&[0.0, 0.0, 1.0], 3, None).unwrap(), ix.search(&[0.0, 0.0, 1.0], 3, None).unwrap());
        let mut c = back.clone();
        c.compact();
        assert_eq!((c.len(), c.store.ids.len()), (1, 1));
        assert!(HnswIndex::load(dir.path().join("f.hvec")).is_err());
    }

    #[test]
    fn hnsw_recall_and_determinism() {
        let data = random_vectors(800, 16, 1);
        let queries = random_vectors(50, 16, 2);
        let mut flat = FlatIndex::new(16, Metric::Cosine);
        let mut h =
            HnswIndex::new(16, Metric::Cosine, HnswParams { m: 12, ef_construction: 100, ef_search: 64, seed: 3 });
        for (i, v) in data.iter().enumerate() {
            flat.add(&i.to_string(), v).unwrap();
            h.add(&i.to_string(), v).unwrap();
        }
        let mut found = 0;
        for q in &queries {
            let exact: HashSet<String> = flat.search(q, 10, None).unwrap().into_iter().map(|h| h.id).collect();
            found += h.search(q, 10, None).unwrap().iter().filter(|x| exact.contains(&x.id)).count();
        }
        let recall = found as f64 / (queries.len() * 10) as f64;
        assert!(recall > 0.95, "recall@10 = {recall}");
        // Same seed and inputs → identical graph and results.
        let mut h2 = HnswIndex::new(16, Metric::Cosine, h.params());
        for (i, v) in data.iter().enumerate() {
            h2.add(&i.to_string(), v).unwrap();
        }
        assert_eq!(h2.g, h.g);
        // Deletions disappear from results; persistence round-trips.
        let top = h.search(&queries[0], 1, None).unwrap()[0].id.clone();
        h.remove(&top);
        assert!(h.search(&queries[0], 10, None).unwrap().iter().all(|x| x.id != top));
        let dir = tempfile::tempdir().unwrap();
        h.save(dir.path().join("h.hvec")).unwrap();
        let back = VectorIndex::load(dir.path().join("h.hvec")).unwrap();
        assert_eq!(back.search(&queries[1], 5, None).unwrap(), h.search(&queries[1], 5, None).unwrap());
        h.rebuild().unwrap();
        assert_eq!(h.len(), 799);
    }
}
