//! Generation metrics: exact match, token F1, BLEU, chrF, ROUGE-N/L and edit distance.
//!
//! Every function works on one prediction/reference pair; corpus-level variants accumulate
//! sufficient statistics first (as sacreBLEU does) instead of averaging sentence scores. Each
//! metric also exists as a [`Metric`] implementation so the harness can pick metrics by name.

use harpia_core::traits::Metric;
use std::collections::HashMap;

/// SQuAD-style answer normalization: lowercase, drop punctuation and the English articles
/// `a`/`an`/`the`, collapse whitespace.
///
/// ```
/// assert_eq!(harpia_eval::metrics::normalize_answer("The  Harpy, eagle!"), "harpy eagle");
/// ```
pub fn normalize_answer(s: &str) -> String {
    let lower = s.to_lowercase();
    let no_punct: String =
        lower.chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect();
    no_punct.split_whitespace().filter(|w| !matches!(*w, "a" | "an" | "the")).collect::<Vec<_>>().join(" ")
}

/// `1.0` when the normalized strings are equal.
pub fn exact_match(prediction: &str, reference: &str) -> f64 {
    f64::from(u8::from(normalize_answer(prediction) == normalize_answer(reference)))
}

/// Token-level F1 over normalized words (SQuAD definition).
///
/// ```
/// // "harpy eagle" vs "harpy eagle of brazil": precision 2/2, recall 2/4.
/// let f = harpia_eval::metrics::token_f1("the harpy eagle", "harpy eagle of brazil");
/// assert!((f - 2.0 / 3.0).abs() < 1e-12);
/// ```
pub fn token_f1(prediction: &str, reference: &str) -> f64 {
    let p = normalize_answer(prediction);
    let r = normalize_answer(reference);
    let pt: Vec<&str> = p.split_whitespace().collect();
    let rt: Vec<&str> = r.split_whitespace().collect();
    if pt.is_empty() || rt.is_empty() {
        return f64::from(u8::from(pt.is_empty() && rt.is_empty()));
    }
    let common = overlap(&counts(&pt), &counts(&rt));
    if common == 0 {
        return 0.0;
    }
    let precision = common as f64 / pt.len() as f64;
    let recall = common as f64 / rt.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

fn counts<T: std::hash::Hash + Eq + Clone>(items: &[T]) -> HashMap<T, usize> {
    let mut m = HashMap::new();
    for t in items {
        *m.entry(t.clone()).or_insert(0) += 1;
    }
    m
}

fn overlap<T: std::hash::Hash + Eq>(a: &HashMap<T, usize>, b: &HashMap<T, usize>) -> usize {
    a.iter().map(|(k, &n)| n.min(b.get(k).copied().unwrap_or(0))).sum()
}

fn ngrams<T: Clone + std::hash::Hash + Eq>(toks: &[T], n: usize) -> HashMap<Vec<T>, usize> {
    let mut m = HashMap::new();
    if n == 0 || toks.len() < n {
        return m;
    }
    for w in toks.windows(n) {
        *m.entry(w.to_vec()).or_insert(0) += 1;
    }
    m
}

/// Tokenization used by BLEU: whitespace split with punctuation separated (close to sacreBLEU's
/// `13a` tokenizer for plain text).
pub fn bleu_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in s.split_whitespace() {
        let mut cur = String::new();
        for c in word.chars() {
            if c.is_alphanumeric() {
                cur.push(c);
            } else {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(c.to_string());
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

/// BLEU sufficient statistics (accumulate over a corpus, then call [`BleuStats::score`]).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BleuStats {
    /// Clipped n-gram matches for n = 1..=4.
    pub matches: [usize; 4],
    /// Candidate n-gram totals for n = 1..=4.
    pub totals: [usize; 4],
    /// Candidate length (tokens).
    pub hyp_len: usize,
    /// Reference length (tokens; closest reference when several are given).
    pub ref_len: usize,
}

impl BleuStats {
    /// Statistics of one hypothesis against one or more references.
    pub fn of(hypothesis: &str, references: &[&str]) -> Self {
        let h = bleu_tokens(hypothesis);
        let refs: Vec<Vec<String>> = references.iter().map(|r| bleu_tokens(r)).collect();
        let mut s = Self { hyp_len: h.len(), ..Self::default() };
        s.ref_len = refs.iter().map(Vec::len).min_by_key(|&l| (l.abs_diff(h.len()), l)).unwrap_or(0);
        for n in 1..=4 {
            let hc = ngrams(&h, n);
            let mut max_ref: HashMap<Vec<String>, usize> = HashMap::new();
            for r in &refs {
                for (g, c) in ngrams(r, n) {
                    let e = max_ref.entry(g).or_insert(0);
                    *e = (*e).max(c);
                }
            }
            s.matches[n - 1] = overlap(&hc, &max_ref);
            s.totals[n - 1] = h.len().saturating_sub(n - 1);
        }
        s
    }

    /// BLEU in `0.0..=1.0`. With `smooth`, orders above 1 use add-one smoothing (Lin & Och,
    /// 2004), which keeps sentence-level scores informative; corpus BLEU uses `smooth = false`.
    pub fn score(&self, smooth: bool) -> f64 {
        if self.hyp_len == 0 {
            return 0.0;
        }
        let mut log_sum = 0.0;
        for n in 0..4 {
            let (m, t) = if smooth && n > 0 {
                (self.matches[n] as f64 + 1.0, self.totals[n] as f64 + 1.0)
            } else {
                (self.matches[n] as f64, self.totals[n] as f64)
            };
            if m == 0.0 || t == 0.0 {
                return 0.0;
            }
            log_sum += (m / t).ln() / 4.0;
        }
        let bp =
            if self.hyp_len >= self.ref_len { 1.0 } else { (1.0 - self.ref_len as f64 / self.hyp_len as f64).exp() };
        bp * log_sum.exp()
    }
}

impl std::ops::AddAssign for BleuStats {
    fn add_assign(&mut self, o: Self) {
        for n in 0..4 {
            self.matches[n] += o.matches[n];
            self.totals[n] += o.totals[n];
        }
        self.hyp_len += o.hyp_len;
        self.ref_len += o.ref_len;
    }
}

/// Smoothed sentence BLEU against one or more references.
pub fn sentence_bleu(hypothesis: &str, references: &[&str]) -> f64 {
    BleuStats::of(hypothesis, references).score(true)
}

/// Corpus BLEU (statistics summed over all pairs, no smoothing), as reported in papers.
pub fn corpus_bleu(hypotheses: &[&str], references: &[&str]) -> f64 {
    let mut total = BleuStats::default();
    for (h, r) in hypotheses.iter().zip(references) {
        total += BleuStats::of(h, &[r]);
    }
    total.score(false)
}

/// chrF statistics: per character order, (matches, hypothesis n-grams, reference n-grams).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChrfStats(pub Vec<[usize; 3]>);

impl ChrfStats {
    /// Character n-gram statistics for orders 1..=`order`, whitespace removed (sacreBLEU chrF).
    pub fn of(hypothesis: &str, reference: &str, order: usize) -> Self {
        let h: Vec<char> = hypothesis.chars().filter(|c| !c.is_whitespace()).collect();
        let r: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();
        Self(
            (1..=order)
                .map(|n| {
                    let hc = ngrams(&h, n);
                    let rc = ngrams(&r, n);
                    [overlap(&hc, &rc), h.len().saturating_sub(n - 1), r.len().saturating_sub(n - 1)]
                })
                .collect(),
        )
    }

    /// chrF-β in `0.0..=1.0`: F-score of the precision and recall averaged over orders.
    pub fn score(&self, beta: f64) -> f64 {
        let mut p_sum = 0.0;
        let mut r_sum = 0.0;
        let mut n = 0.0;
        for &[m, h, r] in &self.0 {
            if h == 0 && r == 0 {
                continue;
            }
            p_sum += if h > 0 { m as f64 / h as f64 } else { 0.0 };
            r_sum += if r > 0 { m as f64 / r as f64 } else { 0.0 };
            n += 1.0;
        }
        if n == 0.0 {
            return 0.0;
        }
        let (p, r) = (p_sum / n, r_sum / n);
        let b2 = beta * beta;
        if p + r == 0.0 { 0.0 } else { (1.0 + b2) * p * r / (b2 * p + r) }
    }
}

impl std::ops::AddAssign<&ChrfStats> for ChrfStats {
    fn add_assign(&mut self, o: &ChrfStats) {
        if self.0.is_empty() {
            self.0 = vec![[0; 3]; o.0.len()];
        }
        for (a, b) in self.0.iter_mut().zip(&o.0) {
            for i in 0..3 {
                a[i] += b[i];
            }
        }
    }
}

/// Sentence chrF (character order 6, β = 2).
pub fn chrf(hypothesis: &str, reference: &str) -> f64 {
    ChrfStats::of(hypothesis, reference, 6).score(2.0)
}

/// Corpus chrF (statistics summed over all pairs).
pub fn corpus_chrf(hypotheses: &[&str], references: &[&str]) -> f64 {
    let mut total = ChrfStats::default();
    for (h, r) in hypotheses.iter().zip(references) {
        total += &ChrfStats::of(h, r, 6);
    }
    total.score(2.0)
}

/// Precision, recall and F1.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Prf {
    /// Precision.
    pub precision: f64,
    /// Recall.
    pub recall: f64,
    /// F1.
    pub f1: f64,
}

impl Prf {
    fn from_counts(common: usize, pred: usize, gold: usize) -> Self {
        let precision = if pred == 0 { 0.0 } else { common as f64 / pred as f64 };
        let recall = if gold == 0 { 0.0 } else { common as f64 / gold as f64 };
        let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
        Self { precision, recall, f1 }
    }
}

fn rouge_tokens(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(String::from).collect()
}

/// ROUGE-N on lowercase alphanumeric tokens.
pub fn rouge_n(prediction: &str, reference: &str, n: usize) -> Prf {
    let p = ngrams(&rouge_tokens(prediction), n);
    let r = ngrams(&rouge_tokens(reference), n);
    Prf::from_counts(overlap(&p, &r), p.values().sum(), r.values().sum())
}

/// ROUGE-L (longest common subsequence) on lowercase alphanumeric tokens.
pub fn rouge_l(prediction: &str, reference: &str) -> Prf {
    let p = rouge_tokens(prediction);
    let r = rouge_tokens(reference);
    Prf::from_counts(lcs_len(&p, &r), p.len(), r.len())
}

fn lcs_len<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for x in a {
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { cur[j].max(prev[j + 1]) };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Levenshtein distance in characters.
///
/// ```
/// assert_eq!(harpia_eval::metrics::edit_distance("kitten", "sitting"), 3);
/// ```
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// `1 - distance / max_len`, in `0.0..=1.0`.
pub fn edit_similarity(a: &str, b: &str) -> f64 {
    let n = a.chars().count().max(b.chars().count());
    if n == 0 { 1.0 } else { 1.0 - edit_distance(a, b) as f64 / n as f64 }
}

/// A named generation metric usable through the [`Metric`] trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenMetric {
    /// [`exact_match`].
    ExactMatch,
    /// [`token_f1`].
    TokenF1,
    /// [`sentence_bleu`].
    Bleu,
    /// [`chrf`].
    Chrf,
    /// ROUGE-1 F1.
    Rouge1,
    /// ROUGE-2 F1.
    Rouge2,
    /// ROUGE-L F1.
    RougeL,
    /// [`edit_similarity`].
    EditSimilarity,
}

impl GenMetric {
    /// All metrics.
    pub const ALL: [GenMetric; 8] = [
        Self::ExactMatch,
        Self::TokenF1,
        Self::Bleu,
        Self::Chrf,
        Self::Rouge1,
        Self::Rouge2,
        Self::RougeL,
        Self::EditSimilarity,
    ];

    /// Parse a metric name (`exact_match`, `token_f1`, `bleu`, `chrf`, `rouge1`, `rouge2`,
    /// `rouge_l`, `edit_similarity`).
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.name() == name)
    }
}

impl Metric for GenMetric {
    fn name(&self) -> &str {
        match self {
            Self::ExactMatch => "exact_match",
            Self::TokenF1 => "token_f1",
            Self::Bleu => "bleu",
            Self::Chrf => "chrf",
            Self::Rouge1 => "rouge1",
            Self::Rouge2 => "rouge2",
            Self::RougeL => "rouge_l",
            Self::EditSimilarity => "edit_similarity",
        }
    }

    fn score(&self, prediction: &str, reference: &str) -> f64 {
        match self {
            Self::ExactMatch => exact_match(prediction, reference),
            Self::TokenF1 => token_f1(prediction, reference),
            Self::Bleu => sentence_bleu(prediction, &[reference]),
            Self::Chrf => chrf(prediction, reference),
            Self::Rouge1 => rouge_n(prediction, reference, 1).f1,
            Self::Rouge2 => rouge_n(prediction, reference, 2).f1,
            Self::RougeL => rouge_l(prediction, reference).f1,
            Self::EditSimilarity => edit_similarity(prediction, reference),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_and_f1() {
        assert_eq!(exact_match("The Eagle.", "eagle"), 1.0);
        assert_eq!(exact_match("eagle", "hawk"), 0.0);
        assert_eq!(token_f1("", ""), 1.0);
        assert_eq!(token_f1("a", "b c"), 0.0);
    }

    #[test]
    fn bleu_identity_and_brevity() {
        let s = "the harpy eagle is the largest raptor in the americas";
        assert!((sentence_bleu(s, &[s]) - 1.0).abs() < 1e-12);
        assert!((corpus_bleu(&[s], &[s]) - 1.0).abs() < 1e-12);
        // Hand computation: hyp "the cat sat" vs ref "the cat sat on the mat": precisions 1,
        // so BLEU = BP = exp(1 - 6/3) = e^-1.
        let b = BleuStats::of("the cat sat", &["the cat sat on the mat"]);
        assert_eq!(b.matches, [3, 2, 1, 0]);
        assert_eq!(b.score(false), 0.0);
        // Add-one smoothing: (3/3) · ((2+1)/(2+1)) · ((1+1)/(1+1)) · ((0+1)/(0+1)) = 1, so only BP remains.
        assert!((b.score(true) - (-1.0f64).exp()).abs() < 1e-12);
        assert_eq!(bleu_tokens("Hi, there!"), ["Hi", ",", "there", "!"]);
    }

    #[test]
    fn chrf_and_rouge() {
        assert!((chrf("harpy", "harpy") - 1.0).abs() < 1e-12);
        assert_eq!(chrf("abc", "xyz"), 0.0);
        assert!(chrf("harpy eagle", "harpy eagles") > 0.8);
        assert!((corpus_chrf(&["a b"], &["ab"]) - 1.0).abs() < 1e-12, "whitespace ignored");
        let r = rouge_n("the cat sat", "the cat sat on the mat", 1);
        assert!((r.precision - 1.0).abs() < 1e-12 && (r.recall - 0.5).abs() < 1e-12);
        let l = rouge_l("police killed the gunman", "the gunman killed police");
        assert_eq!(l.precision, 0.5); // LCS = "the gunman" or "killed police" → 2/4
    }

    #[test]
    fn edit_and_trait() {
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("ação", "acao"), 2);
        assert_eq!(edit_similarity("", ""), 1.0);
        for m in GenMetric::ALL {
            assert_eq!(GenMetric::parse(m.name()), Some(m));
            assert!((m.score("same text here", "same text here") - 1.0).abs() < 1e-12, "{}", m.name());
        }
    }
}
