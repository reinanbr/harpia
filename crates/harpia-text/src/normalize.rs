//! Unicode normalization and cleanup. Every function returns `Cow<str>` and borrows the input
//! when nothing needs to change.

use std::borrow::Cow;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick, is_nfkc_quick};

/// Canonical composition (NFC).
pub fn nfc(text: &str) -> Cow<'_, str> {
    if is_nfc_quick(text.chars()) == IsNormalized::Yes { Cow::Borrowed(text) } else { Cow::Owned(text.nfc().collect()) }
}

/// Compatibility composition (NFKC): folds ligatures, full-width forms, superscripts...
pub fn nfkc(text: &str) -> Cow<'_, str> {
    if is_nfkc_quick(text.chars()) == IsNormalized::Yes {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(text.nfkc().collect())
    }
}

/// Remove diacritics (NFKD, then drop combining marks).
///
/// ```
/// use harpia_text::normalize::strip_accents;
/// assert_eq!(strip_accents("ação científica"), "acao cientifica");
/// assert!(matches!(strip_accents("plain"), std::borrow::Cow::Borrowed(_)));
/// ```
pub fn strip_accents(text: &str) -> Cow<'_, str> {
    if text.is_ascii() {
        return Cow::Borrowed(text);
    }
    Cow::Owned(text.nfkd().filter(|c| !is_combining_mark(*c)).collect())
}

/// Lowercase (Unicode-aware); borrows when already lowercase.
pub fn fold_case(text: &str) -> Cow<'_, str> {
    if text.chars().any(char::is_uppercase) { Cow::Owned(text.to_lowercase()) } else { Cow::Borrowed(text) }
}

/// Collapse runs of whitespace into one space and trim the ends.
///
/// ```
/// use harpia_text::normalize::collapse_whitespace;
/// assert_eq!(collapse_whitespace("  a \n\t b  "), "a b");
/// ```
pub fn collapse_whitespace(text: &str) -> Cow<'_, str> {
    let mut prev_space = true;
    let needs = text.chars().any(|c| {
        let bad = c.is_whitespace() && (prev_space || c != ' ');
        prev_space = c.is_whitespace();
        bad
    }) || text.ends_with(char::is_whitespace);
    if !needs {
        return Cow::Borrowed(text);
    }
    Cow::Owned(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Remove control characters except `\n` and `\t`, and normalize `\r\n`/`\r` to `\n`.
pub fn remove_control_chars(text: &str) -> Cow<'_, str> {
    let bad = |c: char| (c.is_control() && c != '\n' && c != '\t') || is_invisible(c);
    if !text.chars().any(bad) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() != Some(&'\n') {
                out.push('\n');
            }
        } else if !bad(c) {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

/// Zero-width and BOM characters that are invisible but not `char::is_control`.
fn is_invisible(c: char) -> bool {
    matches!(c, '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' | '\u{00AD}')
}

/// Normalization steps for [`normalize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizeOptions {
    /// Apply NFKC (otherwise NFC).
    pub nfkc: bool,
    /// Remove control and invisible characters.
    pub remove_control: bool,
    /// Collapse whitespace.
    pub collapse_whitespace: bool,
    /// Lowercase.
    pub lowercase: bool,
    /// Strip diacritics.
    pub strip_accents: bool,
}

impl Default for NormalizeOptions {
    fn default() -> Self {
        Self { nfkc: false, remove_control: true, collapse_whitespace: false, lowercase: false, strip_accents: false }
    }
}

/// Apply several normalization steps, allocating only when something changes.
pub fn normalize<'a>(text: &'a str, opts: &NormalizeOptions) -> Cow<'a, str> {
    let mut cur: Cow<'a, str> = if opts.nfkc { nfkc(text) } else { nfc(text) };
    if opts.remove_control {
        cur = chain(cur, remove_control_chars);
    }
    if opts.collapse_whitespace {
        cur = chain(cur, collapse_whitespace);
    }
    if opts.lowercase {
        cur = chain(cur, fold_case);
    }
    if opts.strip_accents {
        cur = chain(cur, strip_accents);
    }
    cur
}

fn chain<'a>(cur: Cow<'a, str>, f: fn(&str) -> Cow<'_, str>) -> Cow<'a, str> {
    match cur {
        Cow::Borrowed(s) => f(s),
        Cow::Owned(s) => Cow::Owned(f(&s).into_owned()),
    }
}

/// Key for accent- and case-insensitive comparisons.
pub fn match_key(text: &str) -> String {
    strip_accents(&text.to_lowercase()).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrows_when_clean() {
        assert!(matches!(nfc("abc"), Cow::Borrowed(_)));
        assert!(matches!(collapse_whitespace("a b"), Cow::Borrowed(_)));
        assert!(matches!(remove_control_chars("a\nb"), Cow::Borrowed(_)));
        assert!(matches!(fold_case("abc"), Cow::Borrowed(_)));
    }

    #[test]
    fn fixes() {
        assert_eq!(nfkc("ﬁ"), "fi");
        assert_eq!(nfc("e\u{301}"), "é");
        assert_eq!(remove_control_chars("a\r\nb\rc\u{200B}d\u{7}"), "a\nb\ncd");
        assert_eq!(collapse_whitespace("a  b"), "a b");
        assert_eq!(collapse_whitespace("a\nb"), "a b");
        let opts =
            NormalizeOptions { lowercase: true, strip_accents: true, collapse_whitespace: true, ..Default::default() };
        assert_eq!(normalize("  Ação\u{200B}  RÁPIDA ", &opts), "acao rapida");
    }
}
