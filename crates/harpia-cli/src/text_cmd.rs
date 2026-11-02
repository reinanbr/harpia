//! Text subcommands: segment, stats, similarity, chunk, count-tokens, repair-json.

use crate::util::{OutputFormat, emit, lang, read_text};
use clap::{Args, ValueEnum};
use harpia::core::traits::{Chunker, Tokenizer};
use harpia::core::{HarpiaError, LineIndex, Result, Span};
use harpia::data::chunk::{FixedChunker, MarkdownChunker, RecursiveChunker, SentenceChunker, TokenChunker};
use harpia::data::tokenize::HfTokenizer;
use harpia::text::similarity::{containment, jaccard, longest_shared_span, shingle_set, shingles, tokens};
use harpia::text::{segment, stats};
use serde_json::{Value, json};
use std::process::ExitCode;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Unit {
    Words,
    Sentences,
    Paragraphs,
}

#[derive(Args)]
pub struct SegmentArgs {
    /// Input file (`-` for stdin).
    pub file: String,
    /// Segmentation unit.
    #[arg(long, value_enum, default_value_t = Unit::Sentences)]
    pub unit: Unit,
    /// Language rules (en, pt-br).
    #[arg(long, default_value = "en")]
    pub lang: String,
}

fn span_json(text: &str, idx: &LineIndex, s: Span) -> Value {
    let lc = idx.line_col(text, s.start);
    json!({
        "text": s.slice(text), "start": s.start, "end": s.end,
        "char_start": idx.char_offset(text, s.start), "char_end": idx.char_offset(text, s.end),
        "line": lc.line, "column": lc.column,
    })
}

pub fn segment(a: SegmentArgs, format: OutputFormat) -> Result<ExitCode> {
    let text = read_text(&a.file)?;
    let data = lang(&a.lang)?.data();
    let spans: Vec<Span> = match a.unit {
        Unit::Words => segment::words(&text).collect(),
        Unit::Sentences => segment::sentences(&text, &data).collect(),
        Unit::Paragraphs => segment::paragraphs(&text),
    };
    let idx = LineIndex::new(&text);
    let items: Vec<Value> = spans.iter().map(|s| span_json(&text, &idx, *s)).collect();
    emit(
        format,
        json!({ "unit": format!("{:?}", a.unit).to_lowercase(), "count": spans.len(), "spans": items }),
        || {
            spans
                .iter()
                .map(|s| {
                    let lc = idx.line_col(&text, s.start);
                    format!("{}:{}\t{}", lc.line, lc.column, s.slice(&text).replace('\n', " "))
                })
                .collect::<Vec<_>>()
                .join("\n")
        },
    );
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct StatsArgs {
    /// Input file (`-` for stdin).
    pub file: String,
    /// Language rules (en, pt-br).
    #[arg(long, default_value = "en")]
    pub lang: String,
}

pub fn stats(a: StatsArgs, format: OutputFormat) -> Result<ExitCode> {
    let text = read_text(&a.file)?;
    let data = lang(&a.lang)?.data();
    let s = stats::signals(&text, &data);
    let fl = stats::flesch(&text, &data);
    let mattr = stats::mattr(&text, 100);
    let burst = stats::burstiness(&text, &data);
    let sentences = segment::sentences(&text, &data).count();
    let payload = json!({
        "chars": s.chars, "words": s.words, "sentences": sentences, "lines": s.lines,
        "flesch": fl.map(|f| json!({"score": f.score, "words_per_sentence": f.asl, "syllables_per_word": f.asw})),
        "mattr_100": mattr, "burstiness": burst,
        "mean_word_len": s.mean_word_len, "alpha_ratio": s.alpha_ratio, "digit_ratio": s.digit_ratio,
        "upper_ratio": s.upper_ratio, "symbol_ratio": s.symbol_ratio, "stopword_ratio": s.stopword_ratio,
        "duplicate_line_ratio": s.duplicate_line_ratio, "bullet_line_ratio": s.bullet_line_ratio,
        "top_2gram_fraction": stats::top_ngram_fraction(&text, 2),
        "duplicate_5gram_fraction": stats::duplicate_ngram_fraction(&text, 5),
    });
    emit(format, payload.clone(), || {
        let mut lines =
            vec![format!("chars {}  words {}  sentences {}  lines {}", s.chars, s.words, sentences, s.lines)];
        match fl {
            Some(f) => {
                lines.push(format!("flesch {:.1}  (words/sentence {:.1}, syllables/word {:.2})", f.score, f.asl, f.asw))
            }
            None => lines.push("flesch n/a (needs ≥ 2 sentences and ≥ 30 words)".into()),
        }
        if let Some(m) = mattr {
            lines.push(format!("mattr(100) {m:.3}"));
        }
        if let Some(b) = burst {
            lines.push(format!("burstiness {b:.3}"));
        }
        lines.push(format!(
            "alpha {:.2}  symbol {:.2}  stopwords {:.2}  duplicate lines {:.2}",
            s.alpha_ratio, s.symbol_ratio, s.stopword_ratio, s.duplicate_line_ratio
        ));
        lines.join("\n")
    });
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct SimilarityArgs {
    /// First text file.
    pub a: String,
    /// Second text file.
    pub b: String,
    /// Word shingle size.
    #[arg(long, default_value_t = 5)]
    pub k: usize,
}

pub fn similarity(a: SimilarityArgs, format: OutputFormat) -> Result<ExitCode> {
    let ta = read_text(&a.a)?;
    let tb = read_text(&a.b)?;
    let sa = shingle_set(&ta, a.k);
    let sb = shingle_set(&tb, a.k);
    let toks = tokens(&ta);
    let shared = longest_shared_span(&shingles(&toks, a.k), &sb, &toks, a.k);
    let payload = json!({
        "k": a.k, "jaccard": jaccard(&sa, &sb), "containment": containment(&sa, &sb),
        "longest_shared": shared.map(|s| json!({"text": s.span.slice(&ta), "words": s.words, "start": s.span.start, "end": s.span.end})),
    });
    emit(format, payload, || {
        let mut out = format!("jaccard {:.3}  containment {:.3}", jaccard(&sa, &sb), containment(&sa, &sb));
        if let Some(s) = shared {
            out.push_str(&format!("\nlongest shared ({} words): {}", s.words, s.span.slice(&ta)));
        }
        out
    });
    Ok(ExitCode::SUCCESS)
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Strategy {
    Fixed,
    Sentence,
    Recursive,
    Markdown,
    Token,
}

#[derive(Args)]
pub struct ChunkArgs {
    /// Input file (`-` for stdin).
    pub file: String,
    /// Chunking strategy.
    #[arg(long, value_enum, default_value_t = Strategy::Recursive)]
    pub strategy: Strategy,
    /// Chunk size (characters; tokens for `token`).
    #[arg(long, default_value_t = 1000)]
    pub size: usize,
    /// Overlap (characters for `fixed`, sentences for `sentence`, tokens for `token`).
    #[arg(long, default_value_t = 0)]
    pub overlap: usize,
    /// Language rules for `sentence`.
    #[arg(long, default_value = "en")]
    pub lang: String,
    /// `tokenizer.json` for `token`.
    #[arg(long)]
    pub tokenizer: Option<String>,
}

pub fn chunk(a: ChunkArgs, format: OutputFormat) -> Result<ExitCode> {
    let text = read_text(&a.file)?;
    let chunker: Box<dyn Chunker> = match a.strategy {
        Strategy::Fixed => Box::new(FixedChunker { size: a.size, overlap: a.overlap }),
        Strategy::Sentence => {
            Box::new(SentenceChunker { max_chars: a.size, overlap: a.overlap, lang: lang(&a.lang)?.data() })
        }
        Strategy::Recursive => Box::new(RecursiveChunker::new(a.size)),
        Strategy::Markdown => Box::new(MarkdownChunker { max_chars: a.size }),
        Strategy::Token => {
            let path = a
                .tokenizer
                .as_deref()
                .ok_or_else(|| HarpiaError::invalid("--strategy token needs --tokenizer <tokenizer.json>"))?;
            Box::new(TokenChunker {
                tokenizer: Arc::new(HfTokenizer::from_file(path)?),
                max_tokens: a.size,
                overlap: a.overlap,
            })
        }
    };
    let chunks = chunker.chunk(&text)?;
    let items: Vec<Value> = chunks
        .iter()
        .map(|c| json!({"start": c.span.start, "end": c.span.end, "headings": c.headings, "text": c.span.slice(&text)}))
        .collect();
    emit(format, json!({"count": chunks.len(), "chunks": items}), || {
        chunks
            .iter()
            .enumerate()
            .map(|(i, c)| {
                format!(
                    "--- chunk {i} [{}..{}] {}\n{}",
                    c.span.start,
                    c.span.end,
                    c.headings.join(" > "),
                    c.span.slice(&text)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    });
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct CountTokensArgs {
    /// Input file (`-` for stdin).
    pub file: String,
    /// Path to `tokenizer.json`.
    #[arg(long)]
    pub tokenizer: String,
}

pub fn count_tokens(a: CountTokensArgs, format: OutputFormat) -> Result<ExitCode> {
    let text = read_text(&a.file)?;
    let tok = HfTokenizer::from_file(&a.tokenizer)?;
    let n = tok.count(&text)?;
    emit(format, json!({"tokens": n, "chars": text.chars().count()}), || n.to_string());
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct RepairArgs {
    /// Input file with LLM output (`-` for stdin).
    pub file: String,
    /// Pretty-print the repaired JSON.
    #[arg(long)]
    pub pretty: bool,
}

pub fn repair_json(a: RepairArgs, format: OutputFormat) -> Result<ExitCode> {
    let text = read_text(&a.file)?;
    let strict = harpia::structured::extract_json(&text).map(str::to_owned);
    let value = match &strict {
        Some(s) => serde_json::from_str::<Value>(s)?,
        None => match harpia::structured::repair::repair(&text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return Ok(ExitCode::from(1));
            }
        },
    };
    let repaired = strict.is_none();
    emit(format, json!({"repaired": repaired, "value": value}), || {
        if a.pretty { serde_json::to_string_pretty(&value).unwrap_or_default() } else { value.to_string() }
    });
    Ok(ExitCode::SUCCESS)
}
