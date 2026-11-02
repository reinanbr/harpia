//! Zero-copy word and sentence segmentation with byte spans.
//!
//! - **Words**: a letter followed by letters/digits, allowing inner hyphens and apostrophes
//!   between letters (`guarda-chuva`, `d'água`, `don't`); or a number with `.`/`,` groups
//!   (`3,5`, `1.000.000`).
//! - **Sentences**: split after `.`, `!`, `?` or `…` (plus closing quotes/brackets) followed by
//!   whitespace or the end, except after known abbreviations, single-letter initials, or when
//!   the next word starts in lowercase.

use crate::lang::LangData;
use harpia_core::Span;
use unicode_normalization::char::is_combining_mark;

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || is_combining_mark(c)
}

fn is_joiner(c: char) -> bool {
    matches!(c, '-' | '\'' | '’' | '\u{2010}' | '\u{2011}')
}

/// Iterator over word spans. See [`words`].
#[derive(Debug, Clone)]
pub struct Words<'a> {
    text: &'a str,
    pos: usize,
    alpha_only: bool,
}

impl Iterator for Words<'_> {
    type Item = Span;

    fn next(&mut self) -> Option<Span> {
        let bytes_len = self.text.len();
        while self.pos < bytes_len {
            let rest = &self.text[self.pos..];
            let mut it = rest.char_indices().peekable();
            let (_, first) = *it.peek()?;
            let start = self.pos;
            if is_letter(first) {
                // letter-initial word
                let mut end = start + first.len_utf8();
                it.next();
                while let Some(&(i, c)) = it.peek() {
                    let ok = if self.alpha_only { is_letter(c) || is_combining_mark(c) } else { is_word_char(c) };
                    if ok {
                        end = start + i + c.len_utf8();
                        it.next();
                    } else if is_joiner(c) {
                        let mut look = it.clone();
                        look.next();
                        match look.peek() {
                            Some(&(j, n)) if is_letter(n) => {
                                end = start + j + n.len_utf8();
                                it.next();
                                it.next();
                            }
                            _ => break,
                        }
                    } else {
                        break;
                    }
                }
                self.pos = end;
                return Some(Span::new(start, end));
            } else if first.is_ascii_digit() && !self.alpha_only {
                let b = self.text.as_bytes();
                let mut end = start;
                while end < bytes_len && b[end].is_ascii_digit() {
                    end += 1;
                }
                while end + 1 < bytes_len && (b[end] == b'.' || b[end] == b',') && b[end + 1].is_ascii_digit() {
                    end += 1;
                    while end < bytes_len && b[end].is_ascii_digit() {
                        end += 1;
                    }
                }
                self.pos = end;
                return Some(Span::new(start, end));
            } else {
                self.pos += first.len_utf8();
            }
        }
        None
    }
}

/// Words (letters, inner joiners, digits) as byte spans.
///
/// ```
/// use harpia_text::segment::words;
/// let t = "O guarda-chuva custa 1.500,00 reais.";
/// let w: Vec<&str> = words(t).map(|s| s.slice(t)).collect();
/// assert_eq!(w, ["O", "guarda-chuva", "custa", "1.500,00", "reais"]);
/// ```
pub fn words(text: &str) -> Words<'_> {
    Words { text, pos: 0, alpha_only: false }
}

/// Alphabetic words only (no digits), as byte spans.
pub fn alpha_words(text: &str) -> Words<'_> {
    Words { text, pos: 0, alpha_only: true }
}

/// Number of [`words`].
pub fn word_count(text: &str) -> usize {
    words(text).count()
}

fn is_terminal(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

fn is_closer(c: char) -> bool {
    matches!(c, '"' | '”' | '’' | '\'' | ')' | ']' | '»' | '}')
}

/// Sentence spans (trimmed of surrounding whitespace). See the module docs for the rules.
///
/// ```
/// use harpia_text::{segment::sentences, Lang};
/// let t = "See Fig. 2 for details. It works! Does it? Yes.";
/// let s: Vec<&str> = sentences(t, &Lang::En.data()).map(|s| s.slice(t)).collect();
/// assert_eq!(s, ["See Fig. 2 for details.", "It works!", "Does it?", "Yes."]);
/// ```
pub fn sentences(text: &str, lang: &LangData) -> std::vec::IntoIter<Span> {
    sentence_spans(text, lang).into_iter()
}

fn sentence_spans(text: &str, lang: &LangData) -> Vec<Span> {
    let mut out = Vec::new();
    let skip_ws = |mut i: usize| {
        while let Some(c) = text[i..].chars().next() {
            if c.is_whitespace() {
                i += c.len_utf8();
            } else {
                break;
            }
        }
        i
    };
    let mut start = skip_ws(0);
    let mut i = start;
    while i < text.len() {
        let c = text[i..].chars().next().expect("char boundary");
        if !is_terminal(c) {
            i += c.len_utf8();
            continue;
        }
        let punct_start = i;
        let mut end = i;
        // run of terminals then closers
        while let Some(ch) = text[end..].chars().next() {
            if is_terminal(ch) {
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        while let Some(ch) = text[end..].chars().next() {
            if is_closer(ch) {
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        let next = text[end..].chars().next();
        if next.is_some_and(|n| !n.is_whitespace()) {
            i = end;
            continue;
        }
        if text[punct_start..].starts_with('.') && end < text.len() {
            let before = &text[start..punct_start];
            let tok: String = before
                .chars()
                .rev()
                .take_while(|c| c.is_alphanumeric() || *c == '.' || *c == '_')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<String>()
                .to_lowercase();
            let single_letter = tok.chars().count() == 1 && tok.chars().all(char::is_alphabetic);
            if lang.abbreviations.contains(&tok) || single_letter {
                i = end;
                continue;
            }
            let after = skip_ws(end);
            if text[after..].chars().next().is_some_and(char::is_lowercase) {
                i = end;
                continue;
            }
        }
        if start < punct_start || start < end {
            out.push(Span::new(start, end));
        }
        start = skip_ws(end);
        i = start;
    }
    let tail = text[start..].trim_end();
    if !tail.is_empty() {
        out.push(Span::new(start, start + tail.len()));
    }
    out
}

/// Paragraph spans: blocks separated by one or more blank lines.
pub fn paragraphs(text: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let mut last_end = 0;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let blank = line.trim().is_empty();
        if blank {
            if let Some(s) = start.take() {
                out.push(Span::new(s, last_end));
            }
        } else {
            if start.is_none() {
                start = Some(offset + (line.len() - line.trim_start().len()));
            }
            last_end = offset + line.trim_end().len();
        }
        offset += line.len();
    }
    if let Some(s) = start {
        out.push(Span::new(s, last_end));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lang;

    fn sents(t: &str, l: Lang) -> Vec<&str> {
        sentences(t, &l.data()).map(|s| s.slice(t)).collect()
    }

    #[test]
    fn words_with_joiners_and_numbers() {
        let t = "d'água, don't -x- 3.14 e 2,5% são-paulinos";
        let w: Vec<&str> = words(t).map(|s| s.slice(t)).collect();
        assert_eq!(w, ["d'água", "don't", "x", "3.14", "e", "2,5", "são-paulinos"]);
        let a: Vec<&str> = alpha_words("abc1 def").map(|s| s.slice("abc1 def")).collect();
        assert_eq!(a, ["abc", "def"]);
    }

    #[test]
    fn pt_sentences() {
        assert_eq!(
            sents("Segundo Silva et al. (2020), o efeito é real. Ver p. 12. Fim", Lang::PtBr),
            ["Segundo Silva et al. (2020), o efeito é real.", "Ver p. 12.", "Fim"]
        );
        assert_eq!(sents("J. R. R. Tolkien escreveu. Outro.", Lang::PtBr), ["J. R. R. Tolkien escreveu.", "Outro."]);
        assert_eq!(sents("Valor de 3.5 unidades. Ok", Lang::PtBr), ["Valor de 3.5 unidades.", "Ok"]);
        assert_eq!(sents("Ele disse \"pare.\" Depois saiu.", Lang::PtBr), ["Ele disse \"pare.\"", "Depois saiu."]);
        assert_eq!(sents("  ", Lang::PtBr), Vec::<&str>::new());
    }

    #[test]
    fn lowercase_continuation() {
        assert_eq!(
            sents("Usa-se o méd. geral. depois continua. Fim.", Lang::PtBr),
            ["Usa-se o méd. geral. depois continua.", "Fim."]
        );
    }

    #[test]
    fn paragraphs_split_on_blank_lines() {
        let t = "a b\nc\n\n  d\n\n\ne ";
        let p: Vec<&str> = paragraphs(t).into_iter().map(|s| s.slice(t)).collect();
        assert_eq!(p, ["a b\nc", "d", "e"]);
    }

    proptest::proptest! {
        #[test]
        fn spans_are_valid(s in "\\PC{0,80}") {
            for sp in words(&s).chain(sentences(&s, &Lang::PtBr.data())) {
                proptest::prop_assert!(s.is_char_boundary(sp.start) && s.is_char_boundary(sp.end));
                proptest::prop_assert!(sp.start <= sp.end && sp.end <= s.len());
            }
        }
    }
}
