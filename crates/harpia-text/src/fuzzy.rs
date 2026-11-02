//! Fuzzy string matching with rapidfuzz-compatible definitions, all returning `0.0..=1.0`.
//!
//! The functions compare the strings as given (by Unicode scalar values); normalize first with
//! [`crate::phrases::fold`] or [`crate::normalize::match_key`] for case/accent-insensitive use.

/// Length of the longest common subsequence of two char slices.
fn lcs_len(a: &[char], b: &[char]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for &ca in a {
        for (j, &cb) in b.iter().enumerate() {
            cur[j + 1] = if ca == cb { prev[j] + 1 } else { cur[j].max(prev[j + 1]) };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn ratio_chars(a: &[char], b: &[char]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * lcs_len(a, b) as f64 / total as f64
}

/// Normalized Indel similarity: `2·LCS / (|a| + |b|)` (rapidfuzz `fuzz.ratio`).
///
/// ```
/// use harpia_text::fuzzy::ratio;
/// assert_eq!(ratio("abc", "abc"), 1.0);
/// assert!((ratio("this is a test", "this is a test!") - 0.9655).abs() < 1e-3);
/// ```
pub fn ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    ratio_chars(&a, &b)
}

/// Best [`ratio`] of the shorter string against any equally long substring of the longer one
/// (rapidfuzz `fuzz.partial_ratio`). Useful to find a quote inside a page.
///
/// ```
/// use harpia_text::fuzzy::partial_ratio;
/// assert_eq!(partial_ratio("brown fox", "the quick brown fox jumps"), 1.0);
/// ```
pub fn partial_ratio(a: &str, b: &str) -> f64 {
    partial_ratio_with_offset(a, b).0
}

/// [`partial_ratio`] plus the char offset in the longer string where the best window starts.
pub fn partial_ratio_with_offset(a: &str, b: &str) -> (f64, usize) {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.is_empty() {
        return (if long.is_empty() { 1.0 } else { 0.0 }, 0);
    }
    let m = short.len();
    let short_set: std::collections::HashSet<char> = short.iter().copied().collect();
    let mut best = (0.0f64, 0usize);
    let consider = |best: &mut (f64, usize), start: usize, end: usize| {
        let r = ratio_chars(&short, &long[start..end]);
        if r > best.0 {
            *best = (r, start);
        }
    };
    // windows partially overlapping the start of `long`
    for len in 1..m.min(long.len()) {
        if short_set.contains(&long[len - 1]) {
            consider(&mut best, 0, len);
        }
    }
    // full windows
    for start in 0..=long.len().saturating_sub(m) {
        if short_set.contains(&long[start]) || short_set.contains(&long[start + m - 1]) {
            consider(&mut best, start, start + m);
            if best.0 == 1.0 {
                return best;
            }
        }
    }
    // windows partially overlapping the end
    for start in long.len().saturating_sub(m) + 1..long.len() {
        if short_set.contains(&long[start]) {
            consider(&mut best, start, long.len());
        }
    }
    best
}

fn sorted_tokens(s: &str) -> Vec<&str> {
    let mut t: Vec<&str> = s.split_whitespace().collect();
    t.sort_unstable();
    t
}

/// [`ratio`] after sorting whitespace-separated tokens (rapidfuzz `fuzz.token_sort_ratio`).
pub fn token_sort_ratio(a: &str, b: &str) -> f64 {
    ratio(&sorted_tokens(a).join(" "), &sorted_tokens(b).join(" "))
}

/// rapidfuzz `fuzz.token_set_ratio`: compares the shared tokens with each side's leftovers;
/// returns 1.0 when one token set contains the other.
///
/// ```
/// use harpia_text::fuzzy::token_set_ratio;
/// assert_eq!(token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear"), 1.0);
/// ```
pub fn token_set_ratio(a: &str, b: &str) -> f64 {
    use std::collections::BTreeSet;
    let ta: BTreeSet<&str> = a.split_whitespace().collect();
    let tb: BTreeSet<&str> = b.split_whitespace().collect();
    if ta.is_empty() || tb.is_empty() {
        return if ta.is_empty() && tb.is_empty() { 1.0 } else { 0.0 };
    }
    let inter: Vec<&str> = ta.intersection(&tb).copied().collect();
    let da: Vec<&str> = ta.difference(&tb).copied().collect();
    let db: Vec<&str> = tb.difference(&ta).copied().collect();
    if !inter.is_empty() && (da.is_empty() || db.is_empty()) {
        return 1.0;
    }
    let sect = inter.join(" ");
    let join = |d: &[&str]| {
        if sect.is_empty() {
            d.join(" ")
        } else if d.is_empty() {
            sect.clone()
        } else {
            format!("{sect} {}", d.join(" "))
        }
    };
    let (sa, sb) = (join(&da), join(&db));
    let mut best = ratio(&sa, &sb);
    if !sect.is_empty() {
        best = best.max(ratio(&sect, &sa)).max(ratio(&sect, &sb));
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_basics() {
        assert_eq!(ratio("", ""), 1.0);
        assert_eq!(ratio("abc", ""), 0.0);
        assert!((ratio("kitten", "sitting") - 8.0 / 13.0).abs() < 1e-12);
        assert_eq!(ratio("ação", "ação"), 1.0);
    }

    #[test]
    fn partial() {
        assert_eq!(partial_ratio("fox", "the brown fox"), 1.0);
        let (r, off) = partial_ratio_with_offset("brwn fox", "the quick brown fox");
        assert!(r >= 0.875 && (9..=11).contains(&off), "{r} {off}");
        assert_eq!(partial_ratio("", "x"), 0.0);
    }

    #[test]
    fn token_ratios() {
        assert_eq!(token_sort_ratio("new york mets", "mets new york"), 1.0);
        let r = token_set_ratio("great book about cats", "a great book about dogs");
        assert!(r > 0.7 && r < 1.0, "{r}");
    }

    proptest::proptest! {
        #[test]
        fn ratio_symmetric_and_bounded(a in "\\PC{0,20}", b in "\\PC{0,20}") {
            let r = ratio(&a, &b);
            proptest::prop_assert!((0.0..=1.0).contains(&r));
            proptest::prop_assert!((r - ratio(&b, &a)).abs() < 1e-12);
            let p = partial_ratio(&a, &b);
            proptest::prop_assert!((0.0..=1.0).contains(&p));
        }
    }
}
