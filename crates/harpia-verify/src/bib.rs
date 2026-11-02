//! Minimal BibTeX reader for reference checking: entries, fields in braces / quotes / bare,
//! nested braces, `@string` macros, `#` concatenation. LaTeX markup is reduced to plain text.

use crate::reference::ReferenceQuery;
use harpia_core::{HarpiaError, Result};
use std::collections::{BTreeMap, HashMap};

/// One BibTeX entry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BibEntry {
    /// Entry type, lowercase (`article`).
    pub kind: String,
    /// Citation key.
    pub key: String,
    /// Fields (lowercase names), values with braces kept.
    pub fields: BTreeMap<String, String>,
    /// 1-based line of the `@`.
    pub line: usize,
}

impl BibEntry {
    /// Field as plain text (braces and simple LaTeX commands removed).
    pub fn get(&self, name: &str) -> Option<String> {
        self.fields.get(name).map(|v| plain(v)).filter(|v| !v.is_empty())
    }
}

/// Reduce LaTeX markup to text: accent commands (`\'e` → `é`), `\&`, braces, `~`.
pub fn plain(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '{' | '}' => {}
            '~' => out.push(' '),
            '\\' if i + 1 < chars.len() => {
                let cmd = chars[i + 1];
                let accent = match cmd {
                    '\'' => Some('\u{301}'),
                    '`' => Some('\u{300}'),
                    '^' => Some('\u{302}'),
                    '~' => Some('\u{303}'),
                    '"' => Some('\u{308}'),
                    'c' if chars.get(i + 2).is_some_and(|n| *n == '{' || *n == ' ') => Some('\u{327}'),
                    _ => None,
                };
                if let Some(mark) = accent {
                    let mut j = i + 2;
                    while j < chars.len() && matches!(chars[j], '{' | ' ') {
                        j += 1;
                    }
                    if j < chars.len() {
                        let base = if chars[j] == '\\' && chars.get(j + 1) == Some(&'i') {
                            j += 1;
                            'i'
                        } else {
                            chars[j]
                        };
                        out.extend(unicode_normalization::UnicodeNormalization::nfc([base, mark].into_iter()));
                        i = j + 1;
                        continue;
                    }
                } else if "&%$#_".contains(cmd) {
                    out.push(cmd);
                    i += 2;
                    continue;
                } else if cmd.is_ascii_alphabetic() {
                    // Drop a command name such as \textit or \emph; its argument stays.
                    let mut j = i + 1;
                    while j < chars.len() && chars[j].is_ascii_alphabetic() {
                        j += 1;
                    }
                    i = j;
                    continue;
                }
                i += 1;
                continue;
            }
            _ => out.push(c),
        }
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Parse BibTeX source.
pub fn parse(src: &str) -> Result<Vec<BibEntry>> {
    let b: Vec<char> = src.chars().collect();
    let mut strings: HashMap<String, String> = HashMap::new();
    let mut out = Vec::new();
    let mut i = 0;
    let line_at = |pos: usize| b[..pos].iter().filter(|c| **c == '\n').count() + 1;
    while i < b.len() {
        if b[i] != '@' {
            i += 1;
            continue;
        }
        let at = i;
        i += 1;
        let start = i;
        while i < b.len() && b[i].is_alphanumeric() {
            i += 1;
        }
        let kind: String = b[start..i].iter().collect::<String>().to_lowercase();
        while i < b.len() && b[i].is_whitespace() {
            i += 1;
        }
        if i >= b.len() || !matches!(b[i], '{' | '(') {
            continue;
        }
        let close = if b[i] == '{' { '}' } else { ')' };
        i += 1;
        if kind == "comment" || kind == "preamble" {
            let mut depth = 1;
            while i < b.len() && depth > 0 {
                match b[i] {
                    '{' | '(' => depth += 1,
                    '}' | ')' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            continue;
        }
        let mut entry = BibEntry { kind: kind.clone(), line: line_at(at), ..Default::default() };
        if kind != "string" {
            let ks = i;
            while i < b.len() && b[i] != ',' && b[i] != close {
                i += 1;
            }
            entry.key = b[ks..i].iter().collect::<String>().trim().to_string();
            if i < b.len() && b[i] == ',' {
                i += 1;
            }
        }
        // Fields.
        loop {
            while i < b.len() && (b[i].is_whitespace() || b[i] == ',') {
                i += 1;
            }
            if i >= b.len() {
                return Err(HarpiaError::invalid(format!("unterminated BibTeX entry at line {}", entry.line)));
            }
            if b[i] == close {
                i += 1;
                break;
            }
            let ns = i;
            while i < b.len() && !matches!(b[i], '=' | ',') && b[i] != close {
                i += 1;
            }
            let name = b[ns..i].iter().collect::<String>().trim().to_lowercase();
            if i >= b.len() || b[i] != '=' {
                continue;
            }
            i += 1;
            let mut value = String::new();
            loop {
                while i < b.len() && b[i].is_whitespace() {
                    i += 1;
                }
                if i >= b.len() {
                    break;
                }
                match b[i] {
                    '{' => {
                        let mut depth = 1;
                        i += 1;
                        let vs = i;
                        while i < b.len() && depth > 0 {
                            match b[i] {
                                '{' => depth += 1,
                                '}' => depth -= 1,
                                _ => {}
                            }
                            i += 1;
                        }
                        value.extend(&b[vs..i.saturating_sub(1)]);
                    }
                    '"' => {
                        i += 1;
                        let vs = i;
                        let mut depth = 0;
                        while i < b.len() && !(b[i] == '"' && depth == 0) {
                            match b[i] {
                                '{' => depth += 1,
                                '}' => depth -= 1,
                                _ => {}
                            }
                            i += 1;
                        }
                        value.extend(&b[vs..i]);
                        i += 1;
                    }
                    _ => {
                        let vs = i;
                        while i < b.len() && !matches!(b[i], ',' | '#') && b[i] != close && !b[i].is_whitespace() {
                            i += 1;
                        }
                        let word: String = b[vs..i].iter().collect();
                        value.push_str(strings.get(&word.to_lowercase()).map_or(word.as_str(), String::as_str));
                    }
                }
                while i < b.len() && b[i].is_whitespace() {
                    i += 1;
                }
                if i < b.len() && b[i] == '#' {
                    i += 1;
                    continue;
                }
                break;
            }
            if kind == "string" {
                strings.insert(name, value);
            } else {
                entry.fields.insert(name, value);
            }
        }
        if kind != "string" {
            out.push(entry);
        }
    }
    Ok(out)
}

/// Author family names from a BibTeX `author` field (`Last, First and First Last and others`).
pub fn family_names(field: &str) -> Vec<String> {
    let text = plain(field);
    text.split(" and ")
        .map(str::trim)
        .filter(|a| !a.is_empty() && *a != "others")
        .map(|a| match a.split_once(',') {
            Some((last, _)) => last.trim().to_string(),
            None => a.split_whitespace().last().unwrap_or("").to_string(),
        })
        .collect()
}

impl BibEntry {
    /// As a reference query.
    pub fn to_query(&self) -> ReferenceQuery {
        let year = self
            .get("year")
            .or_else(|| self.get("date"))
            .and_then(|y| y.chars().filter(char::is_ascii_digit).take(4).collect::<String>().parse().ok());
        ReferenceQuery {
            key: self.key.clone(),
            kind: self.kind.clone(),
            title: self.get("title").unwrap_or_default(),
            authors: self
                .fields
                .get("author")
                .or_else(|| self.fields.get("editor"))
                .map(|a| family_names(a))
                .unwrap_or_default(),
            year,
            venue: self
                .get("journal")
                .or_else(|| self.get("journaltitle"))
                .or_else(|| self.get("booktitle"))
                .unwrap_or_default(),
            doi: self.get("doi").unwrap_or_default(),
            arxiv: self
                .get("eprint")
                .filter(|_| self.get("archiveprefix").is_none_or(|p| p.eq_ignore_ascii_case("arxiv")))
                .unwrap_or_default(),
            isbn: self.get("isbn").unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_entries_strings_and_latex() {
        let src = r#"
@string{nat = "Nature"}
@comment{ignored {nested} }
@article{lecun2015,
  title = {Deep {L}earning},
  author = {LeCun, Yann and Bengio, Yoshua and Geoffrey Hinton},
  journal = nat,
  year = 2015,
  doi = {10.1038/nature14539},
}
@techreport{martins1996, title = "Readability formulas applied to textbooks in Brazilian Portuguese",
  author = {Martins, Teresa B. F. and others}, year = {1996}, institution = {ICMSC-USP} }
@book{pt, title = {Detec{\c c}{\~a}o de pl\'agio \& c\'opia}, author = {Jos\'e da Silva}, year = {2021}}
"#;
        let e = parse(src).unwrap();
        assert_eq!(e.len(), 3);
        let q = e[0].to_query();
        assert_eq!(q.title, "Deep Learning");
        assert_eq!(q.authors, ["LeCun", "Bengio", "Hinton"]);
        assert_eq!((q.venue.as_str(), q.year, q.doi.as_str()), ("Nature", Some(2015), "10.1038/nature14539"));
        assert_eq!(e[0].line, 4);
        let m = e[1].to_query();
        assert_eq!((m.kind.as_str(), m.authors.len()), ("techreport", 1));
        assert_eq!(e[2].to_query().title, "Detecção de plágio & cópia");
        assert_eq!(e[2].to_query().authors, ["Silva"]);
        assert!(parse("@article{x, title = {open").is_err());
    }
}
