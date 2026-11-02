//! Byte spans into source text and conversions to human-readable positions.
//!
//! All Harpia spans are byte ranges into the original `&str`, always on `char` boundaries.
//! [`LineIndex`] converts them to char offsets, UTF-16 offsets (for editors/LSP) and 1-based
//! line:column.

use serde::{Deserialize, Serialize};
use std::ops::Range;

/// A byte range into some source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct Span {
    /// Start byte (inclusive).
    pub start: usize,
    /// End byte (exclusive).
    pub end: usize,
}

impl Span {
    /// New span. Panics in debug builds if `start > end`.
    pub fn new(start: usize, end: usize) -> Self {
        debug_assert!(start <= end, "span start {start} > end {end}");
        Self { start, end }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// `true` for an empty span.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// The slice of `text` covered by this span.
    ///
    /// ```
    /// use harpia_core::Span;
    /// assert_eq!(Span::new(6, 11).slice("hello world"), "world");
    /// ```
    pub fn slice<'a>(&self, text: &'a str) -> &'a str {
        &text[self.start..self.end]
    }

    /// As a `Range<usize>`.
    pub fn range(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Smallest span covering both.
    pub fn cover(&self, other: &Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }

    /// Shift by `offset` bytes (e.g. when the text was a slice of a larger document).
    pub fn shift(&self, offset: usize) -> Span {
        Span::new(self.start + offset, self.end + offset)
    }
}

impl From<Range<usize>> for Span {
    fn from(r: Range<usize>) -> Self {
        Span::new(r.start, r.end)
    }
}

/// 1-based line and column (column counted in `char`s).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineCol {
    /// 1-based line.
    pub line: usize,
    /// 1-based column, in Unicode scalar values.
    pub column: usize,
}

/// Precomputed line starts for fast offset → line:column conversion.
///
/// ```
/// use harpia_core::LineIndex;
/// let text = "olá\nmundo";
/// let idx = LineIndex::new(text);
/// let pos = idx.line_col(text, text.find("mundo").unwrap());
/// assert_eq!((pos.line, pos.column), (2, 1));
/// assert_eq!(idx.char_offset(text, 4), 3); // "olá" is 4 bytes, 3 chars
/// ```
#[derive(Debug, Clone)]
pub struct LineIndex {
    starts: Vec<usize>,
    len: usize,
}

impl LineIndex {
    /// Index `text`.
    pub fn new(text: &str) -> Self {
        let mut starts = vec![0];
        starts.extend(text.match_indices('\n').map(|(i, _)| i + 1));
        Self { starts, len: text.len() }
    }

    /// Number of lines.
    pub fn line_count(&self) -> usize {
        self.starts.len()
    }

    /// 0-based line containing byte `offset`.
    pub fn line_of(&self, offset: usize) -> usize {
        let offset = offset.min(self.len);
        match self.starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        }
    }

    /// Byte offset where 0-based `line` starts.
    pub fn line_start(&self, line: usize) -> Option<usize> {
        self.starts.get(line).copied()
    }

    /// 1-based line:column of byte `offset` (must be a char boundary of `text`).
    pub fn line_col(&self, text: &str, offset: usize) -> LineCol {
        let line = self.line_of(offset);
        let start = self.starts[line];
        let column = text[start..offset.min(text.len())].chars().count() + 1;
        LineCol { line: line + 1, column }
    }

    /// Char offset of byte `offset`.
    pub fn char_offset(&self, text: &str, offset: usize) -> usize {
        text[..offset.min(text.len())].chars().count()
    }

    /// UTF-16 code unit offset of byte `offset`.
    pub fn utf16_offset(&self, text: &str, offset: usize) -> usize {
        text[..offset.min(text.len())].chars().map(char::len_utf16).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_col_and_offsets() {
        let text = "a\nbç d\n\nend";
        let idx = LineIndex::new(text);
        assert_eq!(idx.line_count(), 4);
        let d = text.find('d').unwrap();
        assert_eq!(idx.line_col(text, d), LineCol { line: 2, column: 4 });
        assert_eq!(idx.line_col(text, text.find("end").unwrap()), LineCol { line: 4, column: 1 });
        assert_eq!(idx.char_offset(text, d), 5);
        assert_eq!(idx.utf16_offset(text, text.len()), text.chars().count());
        assert_eq!(idx.line_of(text.len()), 3);
    }

    #[test]
    fn span_ops() {
        let s = Span::new(2, 5);
        assert_eq!(s.len(), 3);
        assert_eq!(s.shift(10), Span::new(12, 15));
        assert_eq!(s.cover(&Span::new(4, 9)), Span::new(2, 9));
    }
}
