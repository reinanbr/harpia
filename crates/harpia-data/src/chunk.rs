//! Chunking strategies for RAG. Every chunk is a byte [`Span`] into the original text, so
//! citations can point back to the source.

use harpia_core::traits::{Chunk, Chunker, Tokenizer};
use harpia_core::{HarpiaError, Result, Span};
use harpia_text::LangData;
use harpia_text::segment::sentences;
use std::sync::Arc;

fn trimmed(text: &str, s: Span) -> Option<Span> {
    let piece = s.slice(text);
    let start = s.start + (piece.len() - piece.trim_start().len());
    let end = s.start + piece.trim_end().len();
    (start < end).then(|| Span::new(start, end))
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Fixed windows of `size` characters overlapping by `overlap` characters.
#[derive(Debug, Clone, Copy)]
pub struct FixedChunker {
    /// Window size in characters.
    pub size: usize,
    /// Overlap in characters (must be < `size`).
    pub overlap: usize,
}

impl Chunker for FixedChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>> {
        if self.size == 0 || self.overlap >= self.size {
            return Err(HarpiaError::invalid("FixedChunker needs size > overlap"));
        }
        let bounds: Vec<usize> = text.char_indices().map(|(i, _)| i).chain(std::iter::once(text.len())).collect();
        let n = bounds.len() - 1;
        let mut out = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + self.size).min(n);
            if let Some(s) = trimmed(text, Span::new(bounds[start], bounds[end])) {
                out.push(Chunk { span: s, headings: vec![] });
            }
            if end == n {
                break;
            }
            start = end - self.overlap;
        }
        Ok(out)
    }
}

/// Groups whole sentences up to `max_chars`, repeating the last `overlap` sentences.
#[derive(Debug, Clone)]
pub struct SentenceChunker {
    /// Maximum chunk size in characters (a single longer sentence becomes its own chunk).
    pub max_chars: usize,
    /// Sentences repeated between consecutive chunks.
    pub overlap: usize,
    /// Abbreviation lists for sentence splitting.
    pub lang: Arc<LangData>,
}

impl Chunker for SentenceChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>> {
        let sents: Vec<Span> = sentences(text, &self.lang).collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < sents.len() {
            let mut j = i;
            while j + 1 < sents.len() && char_len(&text[sents[i].start..sents[j + 1].end]) <= self.max_chars {
                j += 1;
            }
            out.push(Chunk { span: Span::new(sents[i].start, sents[j].end), headings: vec![] });
            if j + 1 >= sents.len() {
                break;
            }
            // overlap, but make sure the next chunk includes at least one new sentence
            let mut k = (j + 1).saturating_sub(self.overlap).max(i + 1);
            while k <= j && char_len(&text[sents[k].start..sents[j + 1].end]) > self.max_chars {
                k += 1;
            }
            i = k;
        }
        Ok(out)
    }
}

/// Recursive splitting on a hierarchy of separators (paragraphs, lines, sentences, words,
/// characters), merging neighbours up to `max_chars`.
#[derive(Debug, Clone)]
pub struct RecursiveChunker {
    /// Maximum chunk size in characters.
    pub max_chars: usize,
    /// Separators from coarsest to finest; `""` splits into characters.
    pub separators: Vec<String>,
}

impl RecursiveChunker {
    /// Default separators: `"\n\n"`, `"\n"`, `". "`, `" "`, `""`.
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars, separators: ["\n\n", "\n", ". ", " ", ""].iter().map(|s| s.to_string()).collect() }
    }

    fn split(&self, text: &str, span: Span, level: usize, out: &mut Vec<Span>) {
        let piece = span.slice(text);
        if char_len(piece) <= self.max_chars || level >= self.separators.len() {
            out.push(span);
            return;
        }
        let sep = &self.separators[level];
        if !sep.is_empty() && !piece.contains(sep.as_str()) {
            return self.split(text, span, level + 1, out);
        }
        // pieces keep their trailing separator
        let mut parts: Vec<Span> = Vec::new();
        if sep.is_empty() {
            let mut cur = span.start;
            for (i, c) in piece.char_indices() {
                let end = span.start + i + c.len_utf8();
                if char_len(&text[cur..end]) >= self.max_chars {
                    parts.push(Span::new(cur, end));
                    cur = end;
                }
            }
            if cur < span.end {
                parts.push(Span::new(cur, span.end));
            }
        } else {
            let mut cur = span.start;
            for (i, _) in piece.match_indices(sep.as_str()) {
                let end = span.start + i + sep.len();
                parts.push(Span::new(cur, end));
                cur = end;
            }
            if cur < span.end {
                parts.push(Span::new(cur, span.end));
            }
        }
        // merge greedily
        let mut acc: Option<Span> = None;
        for p in parts {
            let merged = acc.map(|a| a.cover(&p)).unwrap_or(p);
            if char_len(merged.slice(text)) <= self.max_chars {
                acc = Some(merged);
                continue;
            }
            if let Some(a) = acc.take() {
                out.push(a);
            }
            if char_len(p.slice(text)) > self.max_chars {
                self.split(text, p, level + 1, out);
            } else {
                acc = Some(p);
            }
        }
        if let Some(a) = acc {
            out.push(a);
        }
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>> {
        if self.max_chars == 0 {
            return Err(HarpiaError::invalid("max_chars must be > 0"));
        }
        let mut spans = Vec::new();
        self.split(text, Span::new(0, text.len()), 0, &mut spans);
        Ok(spans.into_iter().filter_map(|s| trimmed(text, s)).map(|span| Chunk { span, headings: vec![] }).collect())
    }
}

/// Windows of `max_tokens` tokens overlapping by `overlap` tokens.
pub struct TokenChunker {
    /// Tokenizer used to measure chunks.
    pub tokenizer: Arc<dyn Tokenizer>,
    /// Tokens per chunk.
    pub max_tokens: usize,
    /// Overlap in tokens.
    pub overlap: usize,
}

impl Chunker for TokenChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>> {
        if self.max_tokens == 0 || self.overlap >= self.max_tokens {
            return Err(HarpiaError::invalid("TokenChunker needs max_tokens > overlap"));
        }
        let enc = self.tokenizer.encode(text, false)?;
        let offs: Vec<Span> = enc.offsets.into_iter().filter(|s| !s.is_empty()).collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < offs.len() {
            let j = (i + self.max_tokens).min(offs.len());
            out.push(Chunk { span: Span::new(offs[i].start, offs[j - 1].end), headings: vec![] });
            if j == offs.len() {
                break;
            }
            i = j - self.overlap;
        }
        Ok(out)
    }
}

/// Splits Markdown by headings, recording the heading path of each chunk; long sections are
/// split further with a [`RecursiveChunker`].
#[derive(Debug, Clone)]
pub struct MarkdownChunker {
    /// Maximum chunk size in characters.
    pub max_chars: usize,
}

impl Chunker for MarkdownChunker {
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>> {
        let rec = RecursiveChunker::new(self.max_chars);
        let mut out = Vec::new();
        let mut path: Vec<(usize, String)> = Vec::new();
        let mut section_start = 0;
        let mut headings_for_section: Vec<String> = Vec::new();
        let mut in_code = false;
        let mut offset = 0;
        let flush = |start: usize, end: usize, heads: &[String], out: &mut Vec<Chunk>| -> Result<()> {
            if start >= end {
                return Ok(());
            }
            for c in rec.chunk(&text[start..end])? {
                out.push(Chunk { span: c.span.shift(start), headings: heads.to_vec() });
            }
            Ok(())
        };
        for line in text.split_inclusive('\n') {
            let t = line.trim_start();
            if t.starts_with("```") || t.starts_with("~~~") {
                in_code = !in_code;
            }
            let level = t.chars().take_while(|c| *c == '#').count();
            if !in_code && (1..=6).contains(&level) && t[level..].starts_with([' ', '\t']) {
                flush(section_start, offset, &headings_for_section, &mut out)?;
                path.retain(|(l, _)| *l < level);
                path.push((level, t[level..].trim().to_string()));
                headings_for_section = path.iter().map(|(_, h)| h.clone()).collect();
                section_start = offset;
            }
            offset += line.len();
        }
        flush(section_start, text.len(), &headings_for_section, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harpia_text::Lang;

    fn texts<'a>(t: &'a str, c: &[Chunk]) -> Vec<&'a str> {
        c.iter().map(|c| c.span.slice(t)).collect()
    }

    #[test]
    fn fixed_with_overlap() {
        let t = "abcdefghij";
        let c = FixedChunker { size: 4, overlap: 1 }.chunk(t).unwrap();
        assert_eq!(texts(t, &c), ["abcd", "defg", "ghij"]);
        assert!(FixedChunker { size: 2, overlap: 2 }.chunk(t).is_err());
    }

    #[test]
    fn sentences_grouped() {
        let t = "Um. Dois. Três. Quatro.";
        let c = SentenceChunker { max_chars: 9, overlap: 1, lang: Lang::PtBr.data() }.chunk(t).unwrap();
        assert_eq!(texts(t, &c), ["Um. Dois.", "Três.", "Quatro."]);
        let c = SentenceChunker { max_chars: 16, overlap: 1, lang: Lang::PtBr.data() }.chunk(t).unwrap();
        assert_eq!(texts(t, &c), ["Um. Dois. Três.", "Três. Quatro."]);
    }

    #[test]
    fn recursive_respects_limit_and_covers_text() {
        let t = "Paragraph one is here.\n\nParagraph two is a bit longer than the first one.\nWith a second line.";
        let c = RecursiveChunker::new(30).chunk(t).unwrap();
        assert!(c.iter().all(|x| x.span.slice(t).chars().count() <= 30));
        let joined: String = c.iter().map(|x| x.span.slice(t).replace(char::is_whitespace, "")).collect();
        assert_eq!(joined, t.replace(char::is_whitespace, ""));
    }

    #[test]
    fn markdown_headings() {
        let t = "# Intro\nHello.\n## Details\nMore text.\n```\n# not a heading\n```\n# Next\nEnd.";
        let c = MarkdownChunker { max_chars: 200 }.chunk(t).unwrap();
        assert_eq!(c[0].headings, ["Intro"]);
        assert_eq!(c[1].headings, ["Intro", "Details"]);
        assert!(c[1].span.slice(t).contains("# not a heading"));
        assert_eq!(c[2].headings, ["Next"]);
    }
}
