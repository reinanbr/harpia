//! Okapi BM25 over a fixed set of passages.

use crate::lang::LangData;
use crate::normalize::match_key;
use crate::segment::words;
use std::collections::HashMap;

/// BM25 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// Term-frequency saturation (typically 1.2–2.0).
    pub k1: f64,
    /// Length normalization (0 = none, 1 = full).
    pub b: f64,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

/// An in-memory BM25 index.
///
/// ```
/// use harpia_text::{bm25::Bm25, Lang};
/// let docs = ["Cats purr when happy.", "Dogs bark at strangers.", "Happy dogs wag tails."];
/// let idx = Bm25::new(docs, &Lang::En.data(), Default::default());
/// let top = idx.search("happy dogs", 2);
/// assert_eq!(top[0].0, 2);
/// ```
#[derive(Debug, Clone)]
pub struct Bm25 {
    params: Bm25Params,
    postings: HashMap<String, Vec<(u32, u32)>>,
    doc_len: Vec<u32>,
    avg_len: f64,
    stopwords: std::collections::HashSet<String>,
}

fn terms<'a>(text: &'a str, stop: &'a std::collections::HashSet<String>) -> impl Iterator<Item = String> + 'a {
    words(text).map(move |s| match_key(s.slice(text))).filter(move |t| !stop.contains(t))
}

impl Bm25 {
    /// Index `docs` (ids are their positions). Stopwords of `lang` are ignored.
    pub fn new<I, S>(docs: I, lang: &LangData, params: Bm25Params) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let stopwords: std::collections::HashSet<String> = lang.stopwords.iter().map(|s| match_key(s)).collect();
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut doc_len = Vec::new();
        for (id, d) in docs.into_iter().enumerate() {
            let id = u32::try_from(id).expect("too many documents");
            let mut tf: HashMap<String, u32> = HashMap::new();
            let mut n = 0u32;
            for t in terms(d.as_ref(), &stopwords) {
                *tf.entry(t).or_default() += 1;
                n += 1;
            }
            for (t, c) in tf {
                postings.entry(t).or_default().push((id, c));
            }
            doc_len.push(n);
        }
        let avg_len = if doc_len.is_empty() {
            0.0
        } else {
            doc_len.iter().map(|&l| l as f64).sum::<f64>() / doc_len.len() as f64
        };
        Self { params, postings, doc_len, avg_len, stopwords }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.doc_len.len()
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.doc_len.is_empty()
    }

    /// Score of every document for `query`.
    pub fn scores(&self, query: &str) -> Vec<f64> {
        let n = self.doc_len.len() as f64;
        let mut scores = vec![0.0; self.doc_len.len()];
        let mut qterms: Vec<String> = terms(query, &self.stopwords).collect();
        qterms.sort();
        qterms.dedup();
        for t in qterms {
            let Some(post) = self.postings.get(&t) else { continue };
            let df = post.len() as f64;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(id, tf) in post {
                let tf = tf as f64;
                let len = self.doc_len[id as usize] as f64;
                let norm = 1.0 - self.params.b + self.params.b * len / self.avg_len.max(1e-9);
                scores[id as usize] += idf * tf * (self.params.k1 + 1.0) / (tf + self.params.k1 * norm);
            }
        }
        scores
    }

    /// Top-`k` `(doc id, score)` with positive score, best first.
    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f64)> {
        let mut s: Vec<(usize, f64)> = self.scores(query).into_iter().enumerate().filter(|(_, v)| *v > 0.0).collect();
        s.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        s.truncate(k);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lang;

    #[test]
    fn rare_terms_weigh_more() {
        let docs = ["o gato dorme", "o gato come", "o cão dorme"];
        let idx = Bm25::new(docs, &Lang::PtBr.data(), Bm25Params::default());
        let s = idx.scores("cão");
        assert!(s[2] > 0.0 && s[0] == 0.0);
        assert!(idx.search("o", 3).is_empty(), "stopwords are ignored");
        assert_eq!(idx.search("Cão DORME", 1)[0].0, 2);
    }
}
