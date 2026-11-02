//! Text statistics used as quality signals for data filtering and as readability metrics.

use crate::lang::{LangData, SyllableRule};
use crate::segment::{alpha_words, sentences, words};
use std::collections::HashMap;

/// Syllables in `word` according to `rule` (heuristic, at least 1).
///
/// ```
/// use harpia_text::{lang::SyllableRule, stats::syllables};
/// assert_eq!(syllables("país", SyllableRule::Portuguese), 2);
/// assert_eq!(syllables("poeta", SyllableRule::Portuguese), 3);
/// assert_eq!(syllables("coração", SyllableRule::Portuguese), 3);
/// assert_eq!(syllables("table", SyllableRule::English), 2);
/// ```
pub fn syllables(word: &str, rule: SyllableRule) -> usize {
    match rule {
        SyllableRule::Portuguese => syllables_pt(word),
        _ => syllables_en(word),
    }
}

fn syllables_pt(word: &str) -> usize {
    const VOWELS: &str = "aeiouáéíóúâêôãõàüy";
    const STRONG: &str = "aeoáéóâêôãõà";
    let mut count: usize = 0;
    let mut prev: Option<char> = None;
    for ch in word.to_lowercase().chars() {
        if VOWELS.contains(ch) {
            match prev {
                None => count += 1,
                Some(p) => {
                    if "íú".contains(ch) || (STRONG.contains(ch) && STRONG.contains(p) && !"ãõ".contains(p)) {
                        count += 1;
                    }
                }
            }
            prev = Some(ch);
        } else {
            prev = None;
        }
    }
    count.max(1)
}

fn syllables_en(word: &str) -> usize {
    let w = word.to_lowercase();
    let is_v = |c: char| "aeiouy".contains(c);
    let chars: Vec<char> = w.chars().collect();
    let mut count: usize = 0;
    let mut prev_v = false;
    for &c in &chars {
        let v = is_v(c);
        if v && !prev_v {
            count += 1;
        }
        prev_v = v;
    }
    let n = chars.len();
    if n > 2 && chars[n - 1] == 'e' && !is_v(chars[n - 2]) && !(chars[n - 2] == 'l' && !is_v(chars[n - 3])) {
        count = count.saturating_sub(1usize);
    }
    count.max(1)
}

/// Flesch formula coefficients: `a − b·ASL − c·ASW`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FleschFormula {
    /// Intercept.
    pub a: f64,
    /// Words-per-sentence weight.
    pub b: f64,
    /// Syllables-per-word weight.
    pub c: f64,
}

impl FleschFormula {
    /// Original English Flesch Reading Ease.
    pub const ENGLISH: FleschFormula = FleschFormula { a: 206.835, b: 1.015, c: 84.6 };
    /// Portuguese adaptation (Martins et al., 1996).
    pub const PORTUGUESE: FleschFormula = FleschFormula { a: 248.835, b: 1.015, c: 84.6 };
}

/// Result of [`flesch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flesch {
    /// Reading ease score (higher = easier).
    pub score: f64,
    /// Average sentence length (words per sentence).
    pub asl: f64,
    /// Average syllables per word.
    pub asw: f64,
}

/// Flesch reading ease, using the formula matching `lang.syllables`.
/// Returns `None` for texts with fewer than 2 sentences or 30 words.
pub fn flesch(text: &str, lang: &LangData) -> Option<Flesch> {
    let formula = match lang.syllables {
        SyllableRule::Portuguese => FleschFormula::PORTUGUESE,
        _ => FleschFormula::ENGLISH,
    };
    flesch_with(text, lang, formula)
}

/// Flesch with explicit coefficients.
pub fn flesch_with(text: &str, lang: &LangData, f: FleschFormula) -> Option<Flesch> {
    let n_sents = sentences(text, lang).filter(|s| alpha_words(s.slice(text)).next().is_some()).count();
    let ws: Vec<&str> = alpha_words(text).map(|s| s.slice(text)).collect();
    if n_sents < 2 || ws.len() < 30 {
        return None;
    }
    let asl = ws.len() as f64 / n_sents as f64;
    let asw = ws.iter().map(|w| syllables(w, lang.syllables)).sum::<usize>() as f64 / ws.len() as f64;
    Some(Flesch { score: f.a - f.b * asl - f.c * asw, asl, asw })
}

/// Moving-average type-token ratio (Covington & McFall, 2010) over lowercase alphabetic words.
/// Returns `None` below 20 words; for texts shorter than `window`, the plain TTR.
pub fn mattr(text: &str, window: usize) -> Option<f64> {
    let toks: Vec<String> = alpha_words(text).map(|s| s.slice(text).to_lowercase()).collect();
    if toks.len() < 20 || window == 0 {
        return None;
    }
    if toks.len() <= window {
        let uniq: std::collections::HashSet<&String> = toks.iter().collect();
        return Some(uniq.len() as f64 / toks.len() as f64);
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for t in &toks[..window] {
        *counts.entry(t).or_default() += 1;
    }
    let mut sum = counts.len() as f64 / window as f64;
    let mut n = 1usize;
    for i in window..toks.len() {
        let out = toks[i - window].as_str();
        if let Some(c) = counts.get_mut(out) {
            *c -= 1;
            if *c == 0 {
                counts.remove(out);
            }
        }
        *counts.entry(&toks[i]).or_default() += 1;
        sum += counts.len() as f64 / window as f64;
        n += 1;
    }
    Some(sum / n as f64)
}

/// Population coefficient of variation (`σ/μ`); 0 for fewer than 2 values or zero mean.
pub fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    var.sqrt() / mean
}

/// Burstiness of sentence lengths: `(σ − μ) / (σ + μ)` in `-1..=1` (higher = more varied).
pub fn burstiness(text: &str, lang: &LangData) -> Option<f64> {
    let lens: Vec<f64> = sentences(text, lang).map(|s| words(s.slice(text)).count() as f64).collect();
    if lens.len() < 2 {
        return None;
    }
    let mean = lens.iter().sum::<f64>() / lens.len() as f64;
    let sd = (lens.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / lens.len() as f64).sqrt();
    Some(if sd + mean == 0.0 { 0.0 } else { (sd - mean) / (sd + mean) })
}

/// Fraction of word characters covered by the most frequent word `n`-gram (Gopher-style
/// "top n-gram" repetition signal).
pub fn top_ngram_fraction(text: &str, n: usize) -> f64 {
    let ws: Vec<String> = words(text).map(|s| s.slice(text).to_lowercase()).collect();
    if n == 0 || ws.len() < n {
        return 0.0;
    }
    let total: usize = ws.iter().map(|w| w.chars().count()).sum();
    let mut counts: HashMap<&[String], usize> = HashMap::new();
    for g in ws.windows(n) {
        *counts.entry(g).or_default() += 1;
    }
    let Some((gram, c)) = counts.into_iter().max_by_key(|(g, c)| (*c, std::cmp::Reverse(g.to_vec()))) else {
        return 0.0;
    };
    if c < 2 || total == 0 {
        return 0.0;
    }
    let len: usize = gram.iter().map(|w| w.chars().count()).sum();
    (len * c) as f64 / total as f64
}

/// Fraction of word characters inside word `n`-grams that occur more than once (Gopher-style
/// "duplicate n-gram" signal).
pub fn duplicate_ngram_fraction(text: &str, n: usize) -> f64 {
    let ws: Vec<String> = words(text).map(|s| s.slice(text).to_lowercase()).collect();
    if n == 0 || ws.len() < n {
        return 0.0;
    }
    let total: usize = ws.iter().map(|w| w.chars().count()).sum();
    let mut counts: HashMap<&[String], usize> = HashMap::new();
    for g in ws.windows(n) {
        *counts.entry(g).or_default() += 1;
    }
    let mut covered = vec![false; ws.len()];
    for (i, g) in ws.windows(n).enumerate() {
        if counts[g] > 1 {
            covered[i..i + n].iter_mut().for_each(|c| *c = true);
        }
    }
    let dup: usize = ws.iter().zip(&covered).filter(|(_, c)| **c).map(|(w, _)| w.chars().count()).sum();
    if total == 0 { 0.0 } else { dup as f64 / total as f64 }
}

/// Character-level and line-level ratios used by heuristic quality filters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TextSignals {
    /// Number of characters.
    pub chars: usize,
    /// Number of words.
    pub words: usize,
    /// Mean word length in characters.
    pub mean_word_len: f64,
    /// Fraction of alphabetic characters among non-whitespace characters.
    pub alpha_ratio: f64,
    /// Fraction of digits among non-whitespace characters.
    pub digit_ratio: f64,
    /// Fraction of uppercase among alphabetic characters.
    pub upper_ratio: f64,
    /// Fraction of symbol/punctuation characters among non-whitespace characters.
    pub symbol_ratio: f64,
    /// Number of non-empty lines.
    pub lines: usize,
    /// Fraction of lines starting with a bullet (`-`, `*`, `•`, `·`).
    pub bullet_line_ratio: f64,
    /// Fraction of lines ending with an ellipsis.
    pub ellipsis_line_ratio: f64,
    /// Fraction of non-empty lines that are exact duplicates of an earlier line.
    pub duplicate_line_ratio: f64,
    /// Fraction of words that are stopwords.
    pub stopword_ratio: f64,
}

/// Compute [`TextSignals`] in one pass over characters plus one over lines.
pub fn signals(text: &str, lang: &LangData) -> TextSignals {
    let (mut non_ws, mut alpha, mut digit, mut upper, mut symbol) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut chars = 0;
    for c in text.chars() {
        chars += 1;
        if c.is_whitespace() {
            continue;
        }
        non_ws += 1;
        if c.is_alphabetic() {
            alpha += 1;
            if c.is_uppercase() {
                upper += 1;
            }
        } else if c.is_numeric() {
            digit += 1;
        } else {
            symbol += 1;
        }
    }
    let ws: Vec<&str> = words(text).map(|s| s.slice(text)).collect();
    let stop = ws.iter().filter(|w| lang.is_stopword(w)).count();
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let mut seen = std::collections::HashSet::new();
    let dup_lines = lines.iter().filter(|l| !seen.insert(**l)).count();
    let bullets = lines.iter().filter(|l| l.starts_with(['-', '*', '•', '·'])).count();
    let ellipsis = lines.iter().filter(|l| l.ends_with("...") || l.ends_with('…')).count();
    let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };
    TextSignals {
        chars,
        words: ws.len(),
        mean_word_len: ratio(ws.iter().map(|w| w.chars().count()).sum(), ws.len()),
        alpha_ratio: ratio(alpha, non_ws),
        digit_ratio: ratio(digit, non_ws),
        upper_ratio: ratio(upper, alpha),
        symbol_ratio: ratio(symbol, non_ws),
        lines: lines.len(),
        bullet_line_ratio: ratio(bullets, lines.len()),
        ellipsis_line_ratio: ratio(ellipsis, lines.len()),
        duplicate_line_ratio: ratio(dup_lines, lines.len()),
        stopword_ratio: ratio(stop, ws.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lang;

    #[test]
    fn pt_syllables() {
        for (w, n) in [("saúde", 3), ("pão", 1), ("leem", 2), ("a", 1), ("bola", 2), ("computador", 4)] {
            assert_eq!(syllables(w, SyllableRule::Portuguese), n, "{w}");
        }
    }

    #[test]
    fn flesch_pt_matches_formula() {
        let text = "O gato subiu no telhado. ".repeat(8);
        let f = flesch(&text, &Lang::PtBr.data()).unwrap();
        assert!((f.asl - 5.0).abs() < 1e-12);
        let expected = 248.835 - 1.015 * f.asl - 84.6 * f.asw;
        assert!((f.score - expected).abs() < 1e-9);
        assert!(flesch("Curto.", &Lang::PtBr.data()).is_none());
    }

    #[test]
    fn mattr_basic() {
        // identical words: every window has 1 type out of 10 tokens
        let m = mattr(&"palavra ".repeat(25), 10).unwrap();
        assert!((m - 0.1).abs() < 1e-12);
        assert!(mattr("poucas palavras", 10).is_none());
        let uniq: String =
            ["alfa", "beta", "gama", "delta", "épsilon"].iter().cycle().take(25).map(|w| format!("{w} ")).collect();
        assert!((mattr(&uniq, 10).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn repetition_signals() {
        let rep = "buy now buy now buy now buy now";
        assert!(top_ngram_fraction(rep, 2) > 0.9);
        assert!(duplicate_ngram_fraction(rep, 2) > 0.99);
        assert_eq!(duplicate_ngram_fraction("a b c d e f", 2), 0.0);
        assert!((coefficient_of_variation(&[1.0, 1.0, 1.0])).abs() < 1e-12);
    }

    #[test]
    fn signals_ratios() {
        let s = signals("- Item one\n- Item one\nThe end...", &Lang::En.data());
        assert_eq!(s.lines, 3);
        assert!((s.duplicate_line_ratio - 1.0 / 3.0).abs() < 1e-12);
        assert!((s.bullet_line_ratio - 2.0 / 3.0).abs() < 1e-12);
        assert!((s.ellipsis_line_ratio - 1.0 / 3.0).abs() < 1e-12);
        assert!(s.stopword_ratio > 0.0);
    }
}
