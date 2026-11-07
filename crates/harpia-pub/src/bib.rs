//! Tolerant BibTeX/BibLaTeX parsing with the position of every entry and field, plus name
//! parsing, required fields per entry type and identifier validators (DOI, ISBN, arXiv).
//!
//! The `.bib` is untrusted input: nothing is executed, entries are capped at
//! [`MAX_ENTRIES`] and `@string` expansion is bounded.

use crate::source::SourceFile;
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

/// Maximum number of entries read from one file.
pub const MAX_ENTRIES: usize = 20_000;

/// A field (`name = value`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BibField {
    /// Lowercase name.
    pub name: String,
    /// Value without outer delimiters, `@string`s expanded, trimmed.
    pub value: String,
    /// Raw value as written.
    pub raw: String,
    /// Byte offset of the field name in the file.
    pub offset: usize,
}

/// One entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BibEntry {
    /// Lowercase type (`article`, `book`…).
    pub kind: String,
    /// Citation key.
    pub key: String,
    /// Offset of the `@`.
    pub start: usize,
    /// Offset of the key.
    pub key_offset: usize,
    /// Offset after the closing delimiter.
    pub end: usize,
    /// Fields by lowercase name (first occurrence wins).
    pub fields: BTreeMap<String, BibField>,
    /// Repeated fields (second and later occurrences).
    pub duplicate_fields: Vec<BibField>,
}

impl BibEntry {
    /// Field value or `""`.
    pub fn get(&self, name: &str) -> &str {
        self.fields.get(name).map_or("", |f| f.value.as_str())
    }

    /// First non-empty of several fields.
    pub fn get_any(&self, names: &[&str]) -> &str {
        names.iter().map(|n| self.get(n)).find(|v| !v.trim().is_empty()).unwrap_or("")
    }

    /// Field present and non-blank.
    pub fn has(&self, name: &str) -> bool {
        self.fields.get(name).is_some_and(|f| !f.value.trim().is_empty())
    }

    /// Offset of a field (or of the entry when absent).
    pub fn field_offset(&self, name: &str) -> usize {
        self.fields.get(name).map_or(self.start, |f| f.offset)
    }

    /// Year from `year` or `date`.
    pub fn year(&self) -> Option<i32> {
        year_of(self.get_any(&["year", "date"]))
    }

    /// Title as plain text.
    pub fn title(&self) -> String {
        plain(self.get("title"))
    }
}

/// A syntax problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BibProblem {
    /// What is wrong (English).
    pub kind: BibProblemKind,
    /// Byte offset.
    pub offset: usize,
}

/// Kinds of [`BibProblem`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BibProblemKind {
    /// Too many entries; the rest was ignored.
    TooManyEntries,
    /// `@type{` without a closing delimiter.
    Unterminated(String),
    /// `@type{` without a key.
    MissingKey(String),
    /// Field without `name = value`.
    MalformedField,
    /// Missing comma after a field.
    MissingComma(String),
}

/// A parsed `.bib` file.
#[derive(Debug, Clone)]
pub struct BibFile {
    /// The source.
    pub source: SourceFile,
    /// Entries in file order.
    pub entries: Vec<BibEntry>,
    /// Syntax problems.
    pub problems: Vec<BibProblem>,
    /// `@string` macros (lowercase name → value).
    pub strings: BTreeMap<String, String>,
}

fn months(tok: &str) -> Option<&'static str> {
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].into_iter().find(|m| *m == tok)
}

fn find_close(b: &[u8], mut i: usize, open: u8) -> Option<usize> {
    let mut depth = 0i32;
    while i < b.len() {
        match b[i] {
            b'\\' => {
                i += 2;
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if open == b'{' && depth == 0 {
                    return Some(i);
                }
            }
            b')' if open == b'(' && depth == 0 => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

fn skip_space(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// Read a value with `#` concatenation: (value, raw, next index).
fn read_value(body: &str, mut i: usize, strings: &BTreeMap<String, String>) -> (String, String, usize) {
    let b = body.as_bytes();
    let start = i;
    let mut parts = String::new();
    loop {
        i = skip_space(b, i);
        if i >= b.len() {
            break;
        }
        match b[i] {
            b'{' => {
                let mut depth = 0;
                let mut j = i;
                while j < b.len() {
                    match b[j] {
                        b'\\' => {
                            j += 2;
                            continue;
                        }
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                let j = j.min(b.len());
                parts.push_str(&body[i + 1..j]);
                i = (j + 1).min(b.len());
            }
            b'"' => {
                let mut j = i + 1;
                let mut depth = 0;
                while j < b.len() {
                    match b[j] {
                        b'\\' => {
                            j += 2;
                            continue;
                        }
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        b'"' if depth == 0 => break,
                        _ => {}
                    }
                    j += 1;
                }
                let j = j.min(b.len());
                parts.push_str(&body[i + 1..j]);
                i = (j + 1).min(b.len());
            }
            _ => {
                let j = body[i..]
                    .find(|c: char| c.is_whitespace() || c == ',' || c == '#' || c == '}')
                    .map_or(b.len(), |k| i + k);
                if j == i {
                    break;
                }
                let tok = &body[i..j];
                let low = tok.to_lowercase();
                match strings.get(&low) {
                    Some(v) => parts.push_str(v),
                    None => parts.push_str(months(&low).unwrap_or(tok)),
                }
                i = j;
            }
        }
        i = skip_space(b, i);
        if i < b.len() && b[i] == b'#' {
            i += 1;
            continue;
        }
        break;
    }
    let value: String = parts.chars().take(10_000).collect();
    (value, body[start..i].to_string(), i)
}

/// `name =` at `i`: (lowercase name, name offset, index after `=` and spaces).
fn read_name(body: &str, i: usize) -> Option<(String, usize, usize)> {
    let b = body.as_bytes();
    let s = skip_space(b, i);
    if s >= b.len() || !b[s].is_ascii_alphabetic() {
        return None;
    }
    let mut j = s + 1;
    while j < b.len() && (b[j].is_ascii_alphanumeric() || matches!(b[j], b'_' | b':' | b'.' | b'+' | b'-')) {
        j += 1;
    }
    let k = skip_space(b, j);
    if b.get(k) != Some(&b'=') {
        return None;
    }
    Some((body[s..j].to_lowercase(), s, skip_space(b, k + 1)))
}

/// Parse a `.bib` file, never failing: problems are collected in [`BibFile::problems`].
pub fn parse(source: SourceFile) -> BibFile {
    let text = source.text.clone();
    let b = text.as_bytes();
    let mut bib = BibFile { source, entries: Vec::new(), problems: Vec::new(), strings: BTreeMap::new() };
    let mut pos = 0;
    let mut seen = 0;
    while let Some(k) = text[pos..].find('@') {
        let at = pos + k;
        // @ type {
        let mut j = skip_space(b, at + 1);
        let ts = j;
        while j < b.len() && b[j].is_ascii_alphabetic() {
            j += 1;
        }
        let etype = text[ts..j].to_lowercase();
        let o = skip_space(b, j);
        if etype.is_empty() || o >= b.len() || !matches!(b[o], b'{' | b'(') {
            pos = at + 1;
            continue;
        }
        seen += 1;
        if seen > MAX_ENTRIES {
            bib.problems.push(BibProblem { kind: BibProblemKind::TooManyEntries, offset: at });
            break;
        }
        let open = b[o];
        let Some(close) = find_close(b, o, open) else {
            bib.problems.push(BibProblem { kind: BibProblemKind::Unterminated(etype), offset: at });
            pos = o + 1;
            continue;
        };
        let body_start = o + 1;
        let body = &text[body_start..close];
        pos = close + 1;
        if etype == "comment" || etype == "preamble" {
            continue;
        }
        if etype == "string" {
            if let Some((name, _, vstart)) = read_name(body, 0) {
                let (v, _, _) = read_value(body, vstart, &bib.strings);
                bib.strings.insert(name, v);
            }
            continue;
        }
        // key
        let bb = body.as_bytes();
        let ks = skip_space(bb, 0);
        let mut ke = ks;
        while ke < bb.len()
            && !matches!(bb[ke], b',' | b'{' | b'}' | b'(' | b')' | b'"' | b'#' | b'%' | b'\'' | b'=')
            && !bb[ke].is_ascii_whitespace()
        {
            ke += 1;
        }
        let comma = skip_space(bb, ke);
        if ke == ks || bb.get(comma) != Some(&b',') {
            bib.problems.push(BibProblem { kind: BibProblemKind::MissingKey(etype), offset: at });
            continue;
        }
        let mut entry = BibEntry {
            kind: etype,
            key: body[ks..ke].to_string(),
            start: at,
            key_offset: body_start + ks,
            end: close + 1,
            fields: BTreeMap::new(),
            duplicate_fields: Vec::new(),
        };
        let mut i = comma + 1;
        while i < bb.len() {
            let Some((name, noff, vstart)) = read_name(body, i) else {
                match body[i..].find(',') {
                    Some(n) => {
                        if !body[i..i + n].trim().is_empty() {
                            bib.problems
                                .push(BibProblem { kind: BibProblemKind::MalformedField, offset: body_start + i });
                        }
                        i += n + 1;
                        continue;
                    }
                    None => {
                        if !body[i..].trim().is_empty() {
                            bib.problems
                                .push(BibProblem { kind: BibProblemKind::MalformedField, offset: body_start + i });
                        }
                        break;
                    }
                }
            };
            let (val, raw, j) = read_value(body, vstart, &bib.strings);
            let f = BibField { name: name.clone(), value: val.trim().to_string(), raw, offset: body_start + noff };
            if entry.fields.contains_key(&name) {
                entry.duplicate_fields.push(f);
            } else {
                entry.fields.insert(name.clone(), f);
            }
            i = skip_space(bb, j);
            if i < bb.len() && bb[i] == b',' {
                i += 1;
            } else if i < bb.len() {
                bib.problems.push(BibProblem { kind: BibProblemKind::MissingComma(name), offset: body_start + i });
                match body[i..].find(',') {
                    Some(n) => i += n + 1,
                    None => break,
                }
            }
        }
        bib.entries.push(entry);
    }
    bib
}

static ACC_SYM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\\([`'^~"=.])\s*(?:\{\s*)?\\?([A-Za-z])(?:\s*\})?"#).expect("regex"));
static ACC_LET: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\([cuvHkr])(?:\s+|\s*\{\s*)\\?([A-Za-z])(?:\s*\})?").expect("regex"));
static CMD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\[A-Za-z]+\s*").expect("regex"));

fn combining(acc: &str) -> char {
    match acc {
        "'" => '\u{301}',
        "`" => '\u{300}',
        "^" => '\u{302}',
        "~" => '\u{303}',
        "\"" => '\u{308}',
        "=" => '\u{304}',
        "." => '\u{307}',
        "c" => '\u{327}',
        "u" => '\u{306}',
        "v" => '\u{30C}',
        "H" => '\u{30B}',
        "k" => '\u{328}',
        _ => '\u{30A}',
    }
}

/// BibTeX value → plain text (accents composed, braces and commands removed).
pub fn plain(value: &str) -> String {
    let f = |c: &regex::Captures<'_>| -> String {
        let mut s = String::new();
        s.push_str(&c[2]);
        s.push(combining(&c[1]));
        s.nfc().collect()
    };
    let s = ACC_SYM.replace_all(value, f);
    let s = ACC_LET.replace_all(&s, f);
    let s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_").replace("\\$", "$");
    let s = CMD.replace_all(&s, "");
    let s = s.replace(['{', '}'], "").replace('~', " ").replace("---", "—").replace("--", "–");
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ------------------------------------------------------------------ names

const VON: &[&str] = &[
    "da", "de", "do", "das", "dos", "di", "du", "van", "von", "der", "den", "la", "le", "del", "della", "e", "y",
    "ten", "ter", "zu",
];
const SUFFIX: &[&str] = &["jr", "júnior", "junior", "filho", "neto", "sobrinho", "segundo", "ii", "iii", "iv"];

/// A personal or corporate name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Name {
    /// Given names.
    pub first: String,
    /// Particles (`da`, `van`…).
    pub von: String,
    /// Family name.
    pub last: String,
    /// Suffix (`Jr.`, `Filho`…).
    pub jr: String,
    /// As written.
    pub raw: String,
    /// Protected by braces (`{IBGE}`): a corporate author.
    pub literal: bool,
}

/// Split at `sep` (case-insensitive word or char) only at brace depth 0.
fn split_top(s: &str, sep: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut last = 0;
    let lower = s.to_lowercase();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ if depth == 0 => {
                let hit = if sep == " and " {
                    // whitespace + and + whitespace
                    b[i].is_ascii_whitespace() && {
                        let rest = &lower[i..];
                        let t = rest.trim_start();
                        let ws = rest.len() - t.len();
                        t.starts_with("and") && t[3..].starts_with(char::is_whitespace) && {
                            let after = &t[3..];
                            let n = ws + 3 + (after.len() - after.trim_start().len());
                            out.push(s[last..i].trim().to_string());
                            i += n;
                            last = i;
                            true
                        }
                    }
                } else if sep == "," {
                    b[i] == b',' && {
                        out.push(s[last..i].trim().to_string());
                        i += 1;
                        last = i;
                        true
                    }
                } else {
                    b[i].is_ascii_whitespace() && {
                        out.push(s[last..i].trim().to_string());
                        while i < b.len() && b[i].is_ascii_whitespace() {
                            i += 1;
                        }
                        last = i;
                        true
                    }
                };
                if hit {
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }
    out.push(s[last.min(s.len())..].trim().to_string());
    out.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Split an `author` field on top-level ` and `.
pub fn split_authors(value: &str) -> Vec<String> {
    split_top(value, " and ")
}

/// Parse one BibTeX name (`Last, First`, `First von Last`, `{Corporate}`).
pub fn parse_name(raw: &str) -> Name {
    let raw = raw.trim();
    if raw.starts_with('{') && raw.ends_with('}') && raw.matches('{').count() == 1 {
        return Name {
            first: String::new(),
            von: String::new(),
            last: plain(raw),
            jr: String::new(),
            raw: raw.into(),
            literal: true,
        };
    }
    let parts = split_top(raw, ",");
    if parts.len() >= 2 {
        let first = plain(parts.last().map_or("", String::as_str));
        let jr = if parts.len() == 3 { plain(&parts[1]) } else { String::new() };
        let mut words: Vec<&str> = parts[0].split_whitespace().collect();
        let mut von = Vec::new();
        while words.len() > 1 && VON.contains(&words[0].to_lowercase().as_str()) {
            von.push(words.remove(0));
        }
        return Name {
            first,
            von: plain(&von.join(" ")),
            last: plain(&words.join(" ")),
            jr,
            raw: raw.into(),
            literal: false,
        };
    }
    let mut words = split_top(raw, " ");
    if words.len() == 1 {
        return Name {
            first: String::new(),
            von: String::new(),
            last: plain(&words[0]),
            jr: String::new(),
            raw: raw.into(),
            literal: false,
        };
    }
    let mut jr = String::new();
    if words.len() > 2
        && SUFFIX.contains(&words.last().map(|w| w.to_lowercase()).unwrap_or_default().trim_end_matches('.'))
    {
        jr = words.pop().unwrap_or_default();
    }
    let last = words.pop().unwrap_or_default();
    let mut von = Vec::new();
    while words.len() > 1 && words.last().is_some_and(|w| VON.contains(&w.to_lowercase().as_str())) {
        von.insert(0, words.pop().unwrap_or_default());
    }
    Name {
        first: plain(&words.join(" ")),
        von: plain(&von.join(" ")),
        last: plain(&last),
        jr: plain(&jr),
        raw: raw.into(),
        literal: false,
    }
}

/// Parse an `author`/`editor` field (`others` dropped).
pub fn parse_authors(value: &str) -> Vec<Name> {
    split_authors(value).iter().filter(|a| !a.eq_ignore_ascii_case("others")).map(|a| parse_name(a)).collect()
}

// ------------------------------------------------------------------ fields

/// Required fields per entry type; alternatives separated by `|`.
pub fn required_fields(kind: &str) -> Option<&'static [&'static str]> {
    Some(match kind {
        "article" => &["author", "title", "journal|journaltitle", "year|date"],
        "book" => &["author|editor", "title", "publisher", "year|date"],
        "inbook" => &["author|editor", "title", "chapter|pages", "publisher", "year|date"],
        "incollection" => &["author", "title", "booktitle", "publisher", "year|date"],
        "inproceedings" | "conference" => &["author", "title", "booktitle", "year|date"],
        "phdthesis" | "mastersthesis" => &["author", "title", "school|institution", "year|date"],
        "thesis" => &["author", "title", "type", "institution|school", "year|date"],
        "techreport" => &["author", "title", "institution", "year|date"],
        "manual" | "booklet" => &["title"],
        "misc" => &[],
        "online" => &["title", "url", "urldate|urlaccessdate"],
        "proceedings" => &["title", "year|date"],
        "unpublished" => &["author", "title", "note"],
        _ => return None,
    })
}

static DOI_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)").expect("regex"));
static DOI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^10\.\d{4,9}/\S+$").expect("regex"));
static ARXIV_NEW: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?$").expect("regex"));
static ARXIV_OLD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(?:arxiv:)?[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?$").expect("regex"));
static YEAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(1[5-9]\d\d|20\d\d)\b").expect("regex"));

/// Strip `https://doi.org/` / `doi:` and a trailing period.
pub fn normalize_doi(v: &str) -> String {
    DOI_PREFIX.replace(v.trim(), "").trim().trim_end_matches('.').to_string()
}

/// `10.<prefix>/<suffix>` after normalization.
pub fn valid_doi(v: &str) -> bool {
    DOI.is_match(&normalize_doi(v))
}

/// New (`2101.00001v2`) or old (`hep-th/9901001`) arXiv identifiers.
pub fn valid_arxiv(v: &str) -> bool {
    let v = v.trim();
    ARXIV_NEW.is_match(v) || ARXIV_OLD.is_match(v)
}

/// ISBN-10 or ISBN-13 check digit.
pub fn valid_isbn(v: &str) -> bool {
    let d: Vec<char> = v.chars().filter(|c| !c.is_whitespace() && *c != '-').map(|c| c.to_ascii_uppercase()).collect();
    if d.len() == 10 && d[..9].iter().all(char::is_ascii_digit) && (d[9].is_ascii_digit() || d[9] == 'X') {
        let total: u32 = d
            .iter()
            .enumerate()
            .map(|(i, c)| (10 - i as u32) * if *c == 'X' { 10 } else { c.to_digit(10).unwrap_or(0) })
            .sum();
        return total.is_multiple_of(11);
    }
    if d.len() == 13 && d.iter().all(char::is_ascii_digit) {
        let total: u32 =
            d.iter().enumerate().map(|(i, c)| if i % 2 == 0 { 1 } else { 3 } * c.to_digit(10).unwrap_or(0)).sum();
        return total.is_multiple_of(10);
    }
    false
}

/// First plausible year (1500–2099).
pub fn year_of(v: &str) -> Option<i32> {
    YEAR.captures(v).and_then(|c| c[1].parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Encoding;

    fn bib(t: &str) -> BibFile {
        parse(SourceFile::new("r.bib", "/r.bib", t.to_string(), Encoding::Utf8))
    }

    #[test]
    fn parses_entries_strings_and_problems() {
        let b = bib(r#"@string{jn = "Journal of X"}
@article{silva2020,
  author = {Silva, Jo{\~a}o and Souza, Maria},
  title = "A {GPU} study",
  journal = jn # " B",
  year = 2020,
  month = jan,
  title = {dup},
}
@book{nokey}
@misc{x, note = {a} b = {c}}
"#);
        assert_eq!(b.entries.len(), 2);
        let e = &b.entries[0];
        assert_eq!(e.key, "silva2020");
        assert_eq!(e.get("journal"), "Journal of X B");
        assert_eq!(e.get("month"), "jan");
        assert_eq!(e.year(), Some(2020));
        assert_eq!(e.title(), "A GPU study");
        assert_eq!(e.duplicate_fields.len(), 1);
        assert_eq!(&b.source.text[e.key_offset..e.key_offset + 9], "silva2020");
        assert!(b.problems.iter().any(|p| matches!(p.kind, BibProblemKind::MissingKey(_))));
        assert!(b.problems.iter().any(|p| matches!(p.kind, BibProblemKind::MissingComma(_))));
        let names = parse_authors(e.get("author"));
        assert_eq!(names[0].last, "Silva");
        assert_eq!(names[0].first, "João");
    }

    #[test]
    fn names() {
        assert_eq!(parse_name("Ludwig van Beethoven").last, "Beethoven");
        assert_eq!(parse_name("Ludwig van Beethoven").von, "van");
        assert_eq!(parse_name("José da Silva Filho").jr, "Filho");
        assert!(parse_name("{Instituto Brasileiro}").literal);
        assert_eq!(split_authors("A and B AND {C and D}").len(), 3);
    }

    #[test]
    fn identifiers() {
        assert!(valid_doi("https://doi.org/10.1000/xyz.1"));
        assert_eq!(normalize_doi("doi: 10.1000/x."), "10.1000/x");
        assert!(!valid_doi("10.10/x"));
        assert!(valid_isbn("978-0-306-40615-7"));
        assert!(valid_isbn("0-306-40615-2"));
        assert!(!valid_isbn("978-0-306-40615-8"));
        assert!(valid_arxiv("2101.00001v2") && valid_arxiv("hep-th/9901001") && !valid_arxiv("21.1"));
        assert_eq!(plain(r"Jos\'{e} \c{C}a\~ao --- {X}"), "José Çaão — X");
    }
}
