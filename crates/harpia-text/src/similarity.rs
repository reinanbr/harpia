//! Word k-gram shingles, set similarity, longest shared span, a sampled fingerprint index and
//! MinHash signatures.
//!
//! The shingle hash is **XXH3-64** of the space-joined, lowercased, accent-stripped words. It is
//! part of the stable API: indexes and signatures built with one version remain valid in later
//! versions of the same major release.

use crate::normalize::match_key;
use crate::segment::words;
use harpia_core::Span;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

/// A normalized word with its span in the original text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// Lowercase, accent-stripped form.
    pub norm: String,
    /// Span in the original text.
    pub span: Span,
}

/// Normalized word tokens of `text`.
pub fn tokens(text: &str) -> Vec<Token> {
    words(text).map(|s| Token { norm: match_key(s.slice(text)), span: s }).collect()
}

/// Stable 64-bit hash of a string.
pub fn hash64(s: &str) -> u64 {
    xxh3_64(s.as_bytes())
}

/// Hash of each word `k`-gram, in order (position `i` covers tokens `i..i+k`).
///
/// ```
/// use harpia_text::similarity::{shingles, tokens};
/// let toks = tokens("a b c d");
/// assert_eq!(shingles(&toks, 2).len(), 3);
/// assert!(shingles(&toks, 5).is_empty());
/// ```
pub fn shingles(toks: &[Token], k: usize) -> Vec<u64> {
    if k == 0 || toks.len() < k {
        return Vec::new();
    }
    let mut buf = String::new();
    toks.windows(k)
        .map(|w| {
            buf.clear();
            for (i, t) in w.iter().enumerate() {
                if i > 0 {
                    buf.push(' ');
                }
                buf.push_str(&t.norm);
            }
            hash64(&buf)
        })
        .collect()
}

/// Shingle set of `text`.
pub fn shingle_set(text: &str, k: usize) -> HashSet<u64> {
    shingles(&tokens(text), k).into_iter().collect()
}

/// Jaccard similarity `|A∩B| / |A∪B|` (0 when either is empty).
pub fn jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    inter as f64 / (a.len() + b.len() - inter) as f64
}

/// Containment `|A∩B| / min(|A|, |B|)` (0 when either is empty).
pub fn containment(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.intersection(b).count() as f64 / a.len().min(b.len()) as f64
}

/// Longest contiguous run of `seq` covered by shingles in `other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SharedSpan {
    /// Span in the text `toks` came from.
    pub span: Span,
    /// Number of words in the run.
    pub words: usize,
}

/// Longest contiguous stretch of `seq` (shingles of `toks`, size `k`) whose shingles all
/// appear in `other`. `None` when nothing is shared.
///
/// ```
/// use harpia_text::similarity::*;
/// let a = "the cat sat on the mat today";
/// let b = shingle_set("yesterday the cat sat on the mat", 3);
/// let toks = tokens(a);
/// let s = longest_shared_span(&shingles(&toks, 3), &b, &toks, 3).unwrap();
/// assert_eq!(s.span.slice(a), "the cat sat on the mat");
/// assert_eq!(s.words, 6);
/// ```
pub fn longest_shared_span(seq: &[u64], other: &HashSet<u64>, toks: &[Token], k: usize) -> Option<SharedSpan> {
    let mut best: Option<SharedSpan> = None;
    let mut i = 0;
    while i < seq.len() {
        if other.contains(&seq[i]) {
            let mut j = i;
            while j + 1 < seq.len() && other.contains(&seq[j + 1]) {
                j += 1;
            }
            let n = j - i + k;
            if best.is_none_or(|b| n > b.words) {
                best = Some(SharedSpan { span: toks[i].span.cover(&toks[j + k - 1].span), words: n });
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    best
}

/// Inverted index over a sample of shingles (`hash % modulus == 0`), for finding candidate
/// documents that share text with a query before an exact comparison.
#[derive(Debug, Clone)]
pub struct FingerprintIndex {
    k: usize,
    modulus: u64,
    postings: HashMap<u64, Vec<u32>>,
    names: Vec<String>,
    shingles: Vec<Vec<u64>>,
}

impl FingerprintIndex {
    /// New index with shingle size `k`, keeping roughly `1/modulus` of the shingles.
    pub fn new(k: usize, modulus: u64) -> Self {
        Self { k, modulus: modulus.max(1), postings: HashMap::new(), names: Vec::new(), shingles: Vec::new() }
    }

    /// Shingle size.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of documents.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    fn sampled(&self, h: u64) -> bool {
        h.is_multiple_of(self.modulus)
    }

    /// Add a document; returns its id.
    pub fn add(&mut self, name: impl Into<String>, text: &str) -> u32 {
        let sh = shingles(&tokens(text), self.k);
        self.add_shingles(name.into(), sh)
    }

    fn add_shingles(&mut self, name: String, sh: Vec<u64>) -> u32 {
        let id = u32::try_from(self.names.len()).expect("more than u32::MAX documents");
        let mut seen = HashSet::new();
        for &h in &sh {
            if self.sampled(h) && seen.insert(h) {
                self.postings.entry(h).or_default().push(id);
            }
        }
        self.names.push(name);
        self.shingles.push(sh);
        id
    }

    /// Add many documents, shingling them in parallel.
    pub fn extend<'a, I>(&mut self, docs: I)
    where
        I: IntoParallelIterator<Item = (String, &'a str)>,
    {
        let k = self.k;
        let prepared: Vec<(String, Vec<u64>)> =
            docs.into_par_iter().map(|(n, t)| (n, shingles(&tokens(t), k))).collect();
        for (n, sh) in prepared {
            self.add_shingles(n, sh);
        }
    }

    /// Documents sharing at least `min_hits` sampled shingles with `query`, as `(id, hits)`
    /// sorted by hits descending.
    pub fn query(&self, query: &[u64], min_hits: usize) -> Vec<(u32, usize)> {
        let mut hits: HashMap<u32, usize> = HashMap::new();
        let uniq: HashSet<u64> = query.iter().copied().filter(|h| self.sampled(*h)).collect();
        for h in uniq {
            if let Some(ids) = self.postings.get(&h) {
                for &id in ids {
                    *hits.entry(id).or_default() += 1;
                }
            }
        }
        let mut out: Vec<(u32, usize)> = hits.into_iter().filter(|(_, n)| *n >= min_hits.max(1)).collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    /// Name of document `id`.
    pub fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }

    /// All shingles of document `id`, in order.
    pub fn doc_shingles(&self, id: u32) -> &[u64] {
        &self.shingles[id as usize]
    }
}

/// MinHash signature generator with `num_perm` seeded hash functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinHasher {
    num_perm: usize,
    seed: u64,
}

impl MinHasher {
    /// New hasher with `num_perm` permutations and a base `seed`.
    pub fn new(num_perm: usize, seed: u64) -> Self {
        Self { num_perm: num_perm.max(1), seed }
    }

    /// Number of permutations.
    pub fn num_perm(&self) -> usize {
        self.num_perm
    }

    /// Signature of a shingle set (all `u64::MAX` when empty).
    pub fn signature<'a, I: IntoIterator<Item = &'a u64>>(&self, shingles: I) -> Vec<u64> {
        let mut sig = vec![u64::MAX; self.num_perm];
        for &s in shingles {
            let bytes = s.to_le_bytes();
            for (i, slot) in sig.iter_mut().enumerate() {
                let h = xxh3_64_with_seed(&bytes, self.seed.wrapping_add(i as u64));
                if h < *slot {
                    *slot = h;
                }
            }
        }
        sig
    }

    /// Signature of `text` with word `k`-gram shingles.
    pub fn signature_of(&self, text: &str, k: usize) -> Vec<u64> {
        self.signature(&shingle_set(text, k))
    }
}

/// Estimated Jaccard similarity from two signatures of the same [`MinHasher`].
pub fn minhash_similarity(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).filter(|(x, y)| x == y && **x != u64::MAX).count() as f64 / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accents_and_case_do_not_matter() {
        assert_eq!(shingle_set("Ação Rápida já", 2), shingle_set("acao rapida JA", 2));
    }

    #[test]
    fn set_metrics() {
        let a = shingle_set("a b c d e", 2);
        let b = shingle_set("a b c x y", 2);
        assert!((jaccard(&a, &b) - 2.0 / 6.0).abs() < 1e-12);
        assert!((containment(&a, &b) - 0.5).abs() < 1e-12);
        assert_eq!(jaccard(&a, &HashSet::new()), 0.0);
    }

    #[test]
    fn index_finds_overlapping_doc() {
        let mut idx = FingerprintIndex::new(3, 1);
        idx.add("x", "completely unrelated words about something else entirely");
        let id = idx.add("y", "the quick brown fox jumps over the lazy dog");
        let q = shingles(&tokens("a quick brown fox jumps over a fence"), 3);
        let hits = idx.query(&q, 2);
        assert_eq!(hits[0].0, id);
        assert_eq!(idx.name(id), "y");
    }

    #[test]
    fn minhash_estimates_jaccard() {
        let mh = MinHasher::new(256, 7);
        let a = mh.signature_of(&(0..200).map(|i| format!("w{i} ")).collect::<String>(), 1);
        let b = mh.signature_of(&(100..300).map(|i| format!("w{i} ")).collect::<String>(), 1);
        let est = minhash_similarity(&a, &b);
        assert!((est - 1.0 / 3.0).abs() < 0.1, "{est}");
        assert_eq!(minhash_similarity(&a, &a), 1.0);
    }
}
