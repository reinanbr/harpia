//! Source files with a line index and a *masked* view (comments and verbatim blanked).
//!
//! All offsets are **byte** offsets into [`SourceFile::text`]. Masking replaces every byte of
//! a comment or verbatim body with an ASCII space (multi-byte characters become several
//! spaces), so offsets in the masked view and in the original text are interchangeable.

use serde::Serialize;
use std::path::{Path, PathBuf};

/// Environments whose body is never parsed as LaTeX.
pub const VERBATIM_ENVS: &[&str] = &["verbatim", "verbatim*", "lstlisting", "minted", "Verbatim", "comment"];

/// Files larger than this are truncated when read (5 MiB).
pub const MAX_FILE_BYTES: usize = 5 * 1024 * 1024;

/// A `%` comment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comment {
    /// Byte offset of the `%`.
    pub offset: usize,
    /// Text after the `%`, up to (not including) the newline.
    pub text: String,
}

/// A position (or range) in a project file, 1-based, columns in characters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Location {
    /// Path relative to the project root.
    pub file: PathBuf,
    /// 1-based line.
    pub line: usize,
    /// 1-based column (characters).
    pub column: Option<usize>,
    /// 1-based end line.
    pub end_line: Option<usize>,
    /// 1-based end column (exclusive, characters).
    pub end_column: Option<usize>,
}

impl Location {
    /// `file:line` (no column).
    pub fn short(&self) -> String {
        format!("{}:{}", slash(&self.file), self.line)
    }
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.column {
            Some(c) => write!(f, "{}:{}:{}", slash(&self.file), self.line, c),
            None => write!(f, "{}:{}", slash(&self.file), self.line),
        }
    }
}

/// A path with `/` separators, for stable reports across platforms.
pub fn slash(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

/// Text encoding a file was decoded with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Encoding {
    /// UTF-8 (a BOM is stripped).
    Utf8,
    /// ISO-8859-1 fallback (every byte maps to one code point).
    Latin1,
}

/// One project file (`.tex` or `.bib`) held in memory.
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// Path relative to the project root.
    pub path: PathBuf,
    /// Absolute path.
    pub abspath: PathBuf,
    /// Decoded text.
    pub text: String,
    /// Encoding used to decode.
    pub encoding: Encoding,
    line_starts: Vec<usize>,
    masked: String,
    comments: Vec<Comment>,
}

impl SourceFile {
    /// Build from text already in memory.
    pub fn new(path: impl Into<PathBuf>, abspath: impl Into<PathBuf>, text: String, encoding: Encoding) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(text.match_indices('\n').map(|(i, _)| i + 1));
        let (masked, comments) = build_mask(&text);
        Self { path: path.into(), abspath: abspath.into(), text, encoding, line_starts, masked, comments }
    }

    /// Read a file from disk (UTF-8, falling back to Latin-1), truncated at [`MAX_FILE_BYTES`].
    pub fn read(abspath: &Path, relpath: &Path) -> std::io::Result<Self> {
        let mut raw = std::fs::read(abspath)?;
        raw.truncate(MAX_FILE_BYTES);
        let (text, enc) = decode(raw);
        Ok(Self::new(relpath, abspath, text, enc))
    }

    /// Text with comments and verbatim bodies replaced by spaces (newlines kept).
    pub fn masked(&self) -> &str {
        &self.masked
    }

    /// `%` comments, in order.
    pub fn comments(&self) -> &[Comment] {
        &self.comments
    }

    /// 1-based (line, column) of a byte offset; the column counts characters.
    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        let offset = floor_char(&self.text, offset.min(self.text.len()));
        let idx = self.line_starts.partition_point(|&s| s <= offset) - 1;
        let col = self.text[self.line_starts[idx]..offset].chars().count() + 1;
        (idx + 1, col)
    }

    /// Location of `start` (and optionally the range up to `end`, exclusive).
    pub fn location(&self, start: usize, end: Option<usize>) -> Location {
        let (line, col) = self.line_col(start);
        match end {
            None => Location { file: self.path.clone(), line, column: Some(col), end_line: None, end_column: None },
            Some(e) => {
                let last = prev_char(&self.text, e.max(start + 1)).max(start);
                let (el, ec) = self.line_col(last);
                Location {
                    file: self.path.clone(),
                    line,
                    column: Some(col),
                    end_line: Some(el),
                    end_column: Some(ec + 1),
                }
            }
        }
    }

    /// Text of a 1-based line (without the newline); empty when out of range.
    pub fn line_text(&self, line: usize) -> &str {
        if line == 0 || line > self.line_starts.len() {
            return "";
        }
        let start = self.line_starts[line - 1];
        let end = if line < self.line_starts.len() { self.line_starts[line] - 1 } else { self.text.len() };
        &self.text[start..end.max(start)]
    }

    /// Number of lines.
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// One-line excerpt of `text[start..end]`, clipped to `limit` characters.
    pub fn excerpt(&self, start: usize, end: usize, limit: usize) -> String {
        let s = floor_char(&self.text, start.min(self.text.len()));
        let e = floor_char(&self.text, end.min(self.text.len())).max(s);
        clip(&self.text[s..e], limit)
    }
}

/// Collapse whitespace and clip to `limit` characters (adding `…`).
pub fn clip(text: &str, limit: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= limit {
        flat
    } else {
        let mut s: String = flat.chars().take(limit.saturating_sub(1)).collect();
        s.push('…');
        s
    }
}

/// Largest char boundary `<= i`.
pub fn floor_char(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Start of the character that ends right before byte `i`.
fn prev_char(s: &str, i: usize) -> usize {
    let i = floor_char(s, i);
    s[..i].char_indices().next_back().map_or(0, |(j, _)| j)
}

fn decode(mut raw: Vec<u8>) -> (String, Encoding) {
    if raw.starts_with(&[0xEF, 0xBB, 0xBF]) {
        raw.drain(..3);
    }
    match String::from_utf8(raw) {
        Ok(s) => (s, Encoding::Utf8),
        Err(e) => (e.into_bytes().iter().map(|&b| b as char).collect(), Encoding::Latin1),
    }
}

fn blank(buf: &mut [u8], start: usize, end: usize) {
    for b in &mut buf[start..end] {
        if *b != b'\n' {
            *b = b' ';
        }
    }
}

/// Number of consecutive backslashes ending right before `i`.
pub(crate) fn backslashes_before(b: &[u8], i: usize) -> usize {
    b[..i].iter().rev().take_while(|&&c| c == b'\\').count()
}

fn build_mask(text: &str) -> (String, Vec<Comment>) {
    let mut buf = text.as_bytes().to_vec();
    // 1) verbatim environments
    for env in VERBATIM_ENVS {
        let open = format!("\\begin{{{env}}}");
        let close = format!("\\end{{{env}}}");
        let mut pos = 0;
        while let Some(i) = text[pos..].find(&open) {
            let body = pos + i + open.len();
            match text[body..].find(&close) {
                Some(j) => {
                    blank(&mut buf, body, body + j);
                    pos = body + j + close.len();
                }
                None => break,
            }
        }
    }
    // 2) \verb|...| and \lstinline|...|
    for cmd in ["\\verb", "\\lstinline"] {
        let mut pos = 0;
        while let Some(i) = text[pos..].find(cmd) {
            let mut j = pos + i + cmd.len();
            pos = j;
            if text[j..].starts_with('*') {
                j += 1;
            }
            let Some(delim) = text[j..].chars().next() else { break };
            if delim.is_alphanumeric() || delim.is_whitespace() || delim == '{' || delim == '_' {
                continue;
            }
            let body = j + delim.len_utf8();
            let line_end = text[body..].find('\n').map_or(text.len(), |k| body + k);
            if let Some(k) = text[body..line_end].find(delim) {
                blank(&mut buf, body, body + k);
                pos = body + k + delim.len_utf8();
            }
        }
    }
    // 3) comments: '%' preceded by an even number of backslashes, on the partially masked text
    let mut comments = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        if buf[i] == b'%' && backslashes_before(&buf, i).is_multiple_of(2) {
            let end = buf[i..].iter().position(|&c| c == b'\n').map_or(buf.len(), |k| i + k);
            comments.push(Comment { offset: i, text: text[i + 1..end].to_string() });
            blank(&mut buf, i, end);
            i = end;
            continue;
        }
        i += 1;
    }
    // Every replaced byte is ASCII space and untouched bytes are the original UTF-8, but a
    // multi-byte char may have been partially blanked only if a range boundary split it, which
    // cannot happen: all boundaries above sit on ASCII delimiters.
    let masked = String::from_utf8(buf).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
    (masked, comments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(t: &str) -> SourceFile {
        SourceFile::new("a.tex", "/x/a.tex", t.to_string(), Encoding::Utf8)
    }

    #[test]
    fn masks_comments_and_keeps_offsets() {
        let s = src("ação % comentário\n50\\% real\n");
        assert_eq!(s.masked().len(), s.text.len());
        assert!(!s.masked().contains("comentário"));
        assert!(s.masked().contains("50\\% real"));
        assert_eq!(s.comments().len(), 1);
        assert_eq!(s.comments()[0].text, " comentário");
    }

    #[test]
    fn masks_verbatim() {
        let s = src("a\n\\begin{verbatim}\n% not a comment \\cite{x}\n\\end{verbatim}\n\\verb|\\ref{y}| b");
        assert!(!s.masked().contains("cite"));
        assert!(!s.masked().contains("ref{y}"));
        assert!(s.comments().is_empty());
    }

    #[test]
    fn line_col_counts_chars() {
        let s = src("é\nação x");
        let off = s.text.find('x').unwrap();
        assert_eq!(s.line_col(off), (2, 6));
        assert_eq!(s.line_text(2), "ação x");
        let loc = s.location(off, Some(off + 1));
        assert_eq!((loc.line, loc.column, loc.end_column), (2, Some(6), Some(7)));
    }
}
