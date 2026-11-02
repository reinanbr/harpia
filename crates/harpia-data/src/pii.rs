//! Detection and redaction of personally identifiable information and secrets.
//!
//! Detected kinds: e-mail addresses, phone numbers, IPv4/IPv6 addresses, URLs with embedded
//! credentials, payment card numbers (Luhn-checked), API-key-like secrets, and Brazilian
//! CPF/CNPJ (check digits verified). Matched values are never logged.

use harpia_core::Span;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::LazyLock;

/// Kind of PII.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PiiKind {
    /// E-mail address.
    Email,
    /// Phone number.
    Phone,
    /// IP address.
    IpAddress,
    /// URL containing `user:password@`.
    UrlCredentials,
    /// Payment card number.
    CreditCard,
    /// API key or token.
    Secret,
    /// Brazilian individual taxpayer id.
    Cpf,
    /// Brazilian company id.
    Cnpj,
}

impl PiiKind {
    /// Default replacement token, e.g. `<EMAIL>`.
    pub fn token(self) -> &'static str {
        match self {
            PiiKind::Email => "<EMAIL>",
            PiiKind::Phone => "<PHONE>",
            PiiKind::IpAddress => "<IP>",
            PiiKind::UrlCredentials => "<URL_CREDENTIALS>",
            PiiKind::CreditCard => "<CARD>",
            PiiKind::Secret => "<SECRET>",
            PiiKind::Cpf => "<CPF>",
            PiiKind::Cnpj => "<CNPJ>",
        }
    }
}

/// One detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PiiMatch {
    /// Kind.
    pub kind: PiiKind,
    /// Span in the text.
    pub span: Span,
}

/// Checksum validator for a matched value.
type Validator = fn(&str) -> bool;

macro_rules! re {
    ($name:ident, $pat:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($pat).expect("valid PII regex"));
    };
}

re!(EMAIL, r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b");
re!(URL_CRED, r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s/:@]+:[^\s/@]+@[^\s/]+");
re!(IPV4, r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b");
re!(IPV6, r"(?i)\b(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b");
re!(CARD, r"\b(?:\d[ \-]?){12,18}\d\b");
re!(CPF, r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b");
re!(CNPJ, r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b");
re!(PHONE, r"(?:\+\d{1,3}[ .\-]?)?(?:\(\d{2,3}\)|\b\d{2,3})[ .\-]?\d{4,5}[ .\-]?\d{4}\b");
re!(
    SECRET,
    r"\b(?:sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}|xox[baprs]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|hf_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_\-]{20,})\b"
);

fn digits(s: &str) -> Vec<u32> {
    s.chars().filter_map(|c| c.to_digit(10)).collect()
}

/// Luhn checksum.
pub fn luhn_valid(number: &str) -> bool {
    let d = digits(number);
    if d.len() < 12 {
        return false;
    }
    let sum: u32 = d
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &x)| {
            if i % 2 == 1 {
                let y = x * 2;
                if y > 9 { y - 9 } else { y }
            } else {
                x
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

/// CPF check digits.
///
/// ```
/// assert!(harpia_data::pii::cpf_valid("529.982.247-25"));
/// assert!(!harpia_data::pii::cpf_valid("529.982.247-26"));
/// ```
pub fn cpf_valid(cpf: &str) -> bool {
    let d = digits(cpf);
    if d.len() != 11 || d.iter().all(|&x| x == d[0]) {
        return false;
    }
    let check = |n: usize| {
        let s: u32 = (0..n).map(|i| d[i] * (n as u32 + 1 - i as u32)).sum();
        let r = (s * 10) % 11;
        if r == 10 { 0 } else { r }
    };
    check(9) == d[9] && check(10) == d[10]
}

/// CNPJ check digits.
///
/// ```
/// assert!(harpia_data::pii::cnpj_valid("11.222.333/0001-81"));
/// ```
pub fn cnpj_valid(cnpj: &str) -> bool {
    let d = digits(cnpj);
    if d.len() != 14 || d.iter().all(|&x| x == d[0]) {
        return false;
    }
    let calc = |n: usize| {
        let weights: &[u32] =
            if n == 12 { &[5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2] } else { &[6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2] };
        let s: u32 = (0..n).map(|i| d[i] * weights[i]).sum();
        let r = s % 11;
        if r < 2 { 0 } else { 11 - r }
    };
    calc(12) == d[12] && calc(13) == d[13]
}

/// Which kinds to detect.
#[derive(Debug, Clone)]
pub struct PiiDetector {
    kinds: Vec<PiiKind>,
}

impl Default for PiiDetector {
    fn default() -> Self {
        Self {
            kinds: vec![
                PiiKind::UrlCredentials,
                PiiKind::Email,
                PiiKind::Secret,
                PiiKind::Cnpj,
                PiiKind::Cpf,
                PiiKind::CreditCard,
                PiiKind::IpAddress,
                PiiKind::Phone,
            ],
        }
    }
}

impl PiiDetector {
    /// Detect only `kinds` (earlier kinds win on overlaps).
    pub fn with_kinds(kinds: Vec<PiiKind>) -> Self {
        Self { kinds }
    }

    /// All non-overlapping detections, sorted by position.
    pub fn detect(&self, text: &str) -> Vec<PiiMatch> {
        let mut found: Vec<PiiMatch> = Vec::new();
        let overlaps =
            |found: &[PiiMatch], s: &Span| found.iter().any(|m| m.span.start < s.end && s.start < m.span.end);
        for &kind in &self.kinds {
            let (re, validate): (&Regex, Option<Validator>) = match kind {
                PiiKind::Email => (&EMAIL, None),
                PiiKind::UrlCredentials => (&URL_CRED, None),
                PiiKind::IpAddress => (&IPV4, None),
                PiiKind::CreditCard => (&CARD, Some(luhn_valid)),
                PiiKind::Secret => (&SECRET, None),
                PiiKind::Cpf => (&CPF, Some(cpf_valid)),
                PiiKind::Cnpj => (&CNPJ, Some(cnpj_valid)),
                PiiKind::Phone => (&PHONE, None),
            };
            let mut push = |m: regex::Match<'_>| {
                let span = Span::new(m.start(), m.end());
                if validate.is_none_or(|v| v(m.as_str())) && !overlaps(&found, &span) {
                    found.push(PiiMatch { kind, span });
                }
            };
            re.find_iter(text).for_each(&mut push);
            if kind == PiiKind::IpAddress {
                IPV6.find_iter(text).for_each(&mut push);
            }
        }
        found.sort_by_key(|m| m.span.start);
        found
    }

    /// Replace detections with their kind token (`<EMAIL>`, ...). Borrows when nothing is found.
    ///
    /// ```
    /// use harpia_data::pii::PiiDetector;
    /// let (clean, found) = PiiDetector::default().redact("Mail ana@exemplo.com.br, CPF 529.982.247-25.");
    /// assert_eq!(clean, "Mail <EMAIL>, CPF <CPF>.");
    /// assert_eq!(found.len(), 2);
    /// ```
    pub fn redact<'a>(&self, text: &'a str) -> (Cow<'a, str>, Vec<PiiMatch>) {
        let found = self.detect(text);
        if found.is_empty() {
            return (Cow::Borrowed(text), found);
        }
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for m in &found {
            out.push_str(&text[last..m.span.start]);
            out.push_str(m.kind.token());
            last = m.span.end;
        }
        out.push_str(&text[last..]);
        (Cow::Owned(out), found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_kinds() {
        let d = PiiDetector::default();
        let t = "key sk-abcdefghijklmnopqrstuvwxyz012345 card 4111 1111 1111 1111 ip 192.168.0.1 \
                 tel +55 (11) 98765-4321 cnpj 11.222.333/0001-81 db postgres://u:p4ss@host/db";
        let kinds: Vec<PiiKind> = d.detect(t).into_iter().map(|m| m.kind).collect();
        for k in [
            PiiKind::Secret,
            PiiKind::CreditCard,
            PiiKind::IpAddress,
            PiiKind::Phone,
            PiiKind::Cnpj,
            PiiKind::UrlCredentials,
        ] {
            assert!(kinds.contains(&k), "{k:?} not in {kinds:?}");
        }
    }

    #[test]
    fn invalid_numbers_are_kept() {
        let d = PiiDetector::default();
        assert!(d.detect("order 4111 1111 1111 1112").iter().all(|m| m.kind != PiiKind::CreditCard));
        assert!(d.detect("id 111.111.111-11").iter().all(|m| m.kind != PiiKind::Cpf));
        let (c, f) = d.redact("nothing here");
        assert!(matches!(c, Cow::Borrowed(_)) && f.is_empty());
    }
}
