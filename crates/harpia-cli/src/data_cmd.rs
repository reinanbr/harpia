//! `harpia data ...` subcommands.

use crate::util::{OutputFormat, emit, lang, read_records, write_json, write_records};
use clap::{Args, Subcommand};
use harpia::core::traits::{ChatTemplate, Tokenizer};
use harpia::core::{HarpiaError, Result};
use harpia::data::curate::{Pipeline, Stage};
use harpia::data::decontam::Decontaminator;
use harpia::data::dedup::{ExactDedup, LshParams, near_duplicates};
use harpia::data::filter::{LanguageFilter, QualityFilter, QualityThresholds};
use harpia::data::format::{Format, from_chat_sample, to_chat_sample};
use harpia::data::io::get_path;
use harpia::data::pack::{PackOptions, PackStrategy, TokenizedSample, pack, packing_efficiency};
use harpia::data::pii::PiiDetector;
use harpia::data::split::{Ratios, group_split, random_split, stratified_split};
use harpia::data::template::{CHATML, JinjaChatTemplate, LLAMA3, PLAIN, loss_mask};
use harpia::data::tokenize::HfTokenizer;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::process::ExitCode;

#[derive(Subcommand)]
pub enum DataCommand {
    /// Rows, fields and text-length statistics of a JSONL file.
    Inspect(InspectArgs),
    /// Convert between messages / sharegpt / alpaca / prompt-completion.
    Convert(ConvertArgs),
    /// Drop low-quality rows (heuristic presets or a TOML config, optional language filter).
    Filter(FilterArgs),
    /// Detect or redact PII and secrets.
    Pii(PiiArgs),
    /// Remove exact or near-duplicate rows.
    Dedup(DedupArgs),
    /// Drop rows that overlap with evaluation sets.
    Decontaminate(DecontamArgs),
    /// Split into train/val/test files.
    Split(SplitArgs),
    /// Render rows with a chat template and show assistant spans.
    Template(TemplateArgs),
    /// Tokenize, mask and pack chat rows into fixed-length sequences.
    Pack(PackArgs),
    /// Generate synthetic data with an LLM from a prompt template (resumable). Needs --online.
    Synth(crate::llm_cmd::SynthArgs),
}

pub fn run(cmd: DataCommand, format: OutputFormat) -> Result<ExitCode> {
    match cmd {
        DataCommand::Inspect(a) => inspect(a, format),
        DataCommand::Convert(a) => convert(a, format),
        DataCommand::Filter(a) => filter(a, format),
        DataCommand::Pii(a) => pii(a, format),
        DataCommand::Dedup(a) => dedup(a, format),
        DataCommand::Decontaminate(a) => decontaminate(a, format),
        DataCommand::Split(a) => split(a, format),
        DataCommand::Template(a) => template(a, format),
        DataCommand::Pack(a) => pack_cmd(a, format),
        DataCommand::Synth(a) => crate::llm_cmd::synth(a, format),
    }
}

fn text_field<'a>(r: &'a Value, field: &str) -> &'a str {
    get_path(r, field).and_then(Value::as_str).unwrap_or("")
}

#[derive(Args)]
pub struct InspectArgs {
    /// JSONL file (`-` for stdin).
    pub input: String,
    /// Text field to measure (dotted path).
    #[arg(long, default_value = "text")]
    pub field: String,
}

fn inspect(a: InspectArgs, format: OutputFormat) -> Result<ExitCode> {
    let rows = read_records(&a.input)?;
    let mut fields: BTreeMap<String, usize> = BTreeMap::new();
    for r in &rows {
        if let Some(o) = r.as_object() {
            for k in o.keys() {
                *fields.entry(k.clone()).or_default() += 1;
            }
        }
    }
    let mut lens: Vec<usize> =
        rows.iter().filter_map(|r| get_path(r, &a.field).and_then(Value::as_str)).map(|t| t.chars().count()).collect();
    lens.sort_unstable();
    let pct = |p: f64| lens.get(((lens.len().saturating_sub(1)) as f64 * p).round() as usize).copied();
    let detected = rows.first().and_then(Format::detect).map(|f| format!("{f:?}"));
    let payload = json!({
        "rows": rows.len(), "fields": fields, "format": detected,
        "text_field": a.field, "with_text": lens.len(),
        "chars": {"min": lens.first(), "p50": pct(0.5), "p90": pct(0.9), "max": lens.last()},
    });
    emit(format, payload, || {
        let mut s = format!(
            "rows {}  detected format {}\nfields:",
            rows.len(),
            detected.clone().unwrap_or_else(|| "unknown".into())
        );
        for (k, n) in &fields {
            s.push_str(&format!("\n  {k}: {n}"));
        }
        if !lens.is_empty() {
            s.push_str(&format!(
                "\n`{}` chars: min {} p50 {} p90 {} max {}",
                a.field,
                lens[0],
                pct(0.5).unwrap_or(0),
                pct(0.9).unwrap_or(0),
                lens[lens.len() - 1]
            ));
        }
        s
    });
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct ConvertArgs {
    /// JSONL file (`-` for stdin).
    pub input: String,
    /// Source format (auto-detected when omitted).
    #[arg(long)]
    pub from: Option<String>,
    /// Target format.
    #[arg(long)]
    pub to: String,
    /// Output file (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

fn convert(a: ConvertArgs, format: OutputFormat) -> Result<ExitCode> {
    let rows = read_records(&a.input)?;
    let to = Format::parse(&a.to)?;
    let from = a.from.as_deref().map(Format::parse).transpose()?;
    let mut out = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let f = from.or_else(|| Format::detect(r)).ok_or_else(|| HarpiaError::Record {
            path: None,
            line: i as u64 + 1,
            message: "cannot detect format; pass --from".into(),
        })?;
        let sample = to_chat_sample(r, f).map_err(|e| HarpiaError::Record {
            path: None,
            line: i as u64 + 1,
            message: e.to_string(),
        })?;
        out.push(from_chat_sample(&sample, to).map_err(|e| HarpiaError::Record {
            path: None,
            line: i as u64 + 1,
            message: e.to_string(),
        })?);
    }
    write_records(a.output.as_deref(), &out)?;
    if a.output.is_some() {
        emit(format, json!({"rows": out.len()}), || format!("converted {} rows", out.len()));
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct FilterArgs {
    /// JSONL file (`-` for stdin).
    pub input: String,
    /// Text field (dotted path).
    #[arg(long, default_value = "text")]
    pub field: String,
    /// Preset: gopher-like, c4-like, minimal.
    #[arg(long, default_value = "gopher-like")]
    pub preset: String,
    /// TOML file with thresholds (overrides --preset).
    #[arg(long)]
    pub config: Option<String>,
    /// Keep only these languages (ISO 639-3, comma-separated, e.g. `por,eng`).
    #[arg(long)]
    pub languages: Option<String>,
    /// Stopword language for the quality filter (en, pt-br).
    #[arg(long, default_value = "en")]
    pub lang: String,
    /// Output file (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
    /// Write the curation report (JSON) here.
    #[arg(long)]
    pub report: Option<String>,
}

fn filter(a: FilterArgs, format: OutputFormat) -> Result<ExitCode> {
    let rows = read_records(&a.input)?;
    let thresholds = match &a.config {
        Some(p) => QualityThresholds::from_toml(&std::fs::read_to_string(p).map_err(|e| HarpiaError::io(p, e))?)?,
        None => QualityThresholds::preset(&a.preset)?,
    };
    let mut pipe = Pipeline::new(&a.field);
    if let Some(l) = &a.languages {
        pipe = pipe.stage(Stage::Filter(Box::new(LanguageFilter::new(l.split(',').map(str::trim), 0.0))));
    }
    pipe = pipe.stage(Stage::Filter(Box::new(QualityFilter::new(thresholds, lang(&a.lang)?.data()))));
    let (kept, report) = pipe.run(rows)?;
    write_records(a.output.as_deref(), &kept)?;
    let rep = serde_json::to_value(&report)?;
    if let Some(p) = &a.report {
        write_json(p, &rep)?;
    }
    if a.output.is_some() {
        emit(format, json!({"report": rep}), || {
            let mut s = format!("kept {} of {} rows", report.output, report.input);
            for st in &report.stages {
                for (r, n) in &st.dropped {
                    s.push_str(&format!("\n  {}: {r} {n}", st.stage));
                }
            }
            s
        });
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct PiiArgs {
    /// JSONL file (`-` for stdin).
    pub input: String,
    /// Text field (dotted path).
    #[arg(long, default_value = "text")]
    pub field: String,
    /// Write redacted rows instead of only reporting.
    #[arg(long)]
    pub redact: bool,
    /// Output file for redacted rows (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

fn pii(a: PiiArgs, format: OutputFormat) -> Result<ExitCode> {
    let rows = read_records(&a.input)?;
    let det = PiiDetector::default();
    if a.redact {
        let (kept, report) = Pipeline::new(&a.field).stage(Stage::RedactPii(det)).run(rows)?;
        write_records(a.output.as_deref(), &kept)?;
        if a.output.is_some() {
            let modified = report.stages.first().map(|s| s.modified).unwrap_or(0);
            emit(format, json!({"rows": kept.len(), "redacted_rows": modified}), || {
                format!("redacted {modified} of {} rows", kept.len())
            });
        }
        return Ok(ExitCode::SUCCESS);
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut rows_with = 0;
    for r in &rows {
        let found = det.detect(text_field(r, &a.field));
        if !found.is_empty() {
            rows_with += 1;
        }
        for m in found {
            *counts.entry(serde_json::to_value(m.kind)?.as_str().unwrap_or("?").to_string()).or_default() += 1;
        }
    }
    emit(format, json!({"rows": rows.len(), "rows_with_pii": rows_with, "by_kind": counts}), || {
        let mut s = format!("{rows_with} of {} rows contain PII", rows.len());
        for (k, n) in &counts {
            s.push_str(&format!("\n  {k}: {n}"));
        }
        s
    });
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct DedupArgs {
    /// JSONL file (`-` for stdin).
    pub input: String,
    /// Text field (dotted path).
    #[arg(long, default_value = "text")]
    pub field: String,
    /// Near-duplicate (MinHash-LSH) instead of exact.
    #[arg(long)]
    pub minhash: bool,
    /// Similarity threshold for --minhash.
    #[arg(long, default_value_t = 0.8)]
    pub threshold: f64,
    /// Word shingle size for --minhash.
    #[arg(long, default_value_t = 5)]
    pub shingle: usize,
    /// Output file (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

fn dedup(a: DedupArgs, format: OutputFormat) -> Result<ExitCode> {
    let rows = read_records(&a.input)?;
    let keep: Vec<bool> = if a.minhash {
        let texts: Vec<&str> = rows.iter().map(|r| text_field(r, &a.field)).collect();
        near_duplicates(&texts, LshParams { shingle: a.shingle, ..LshParams::for_threshold(a.threshold) })
    } else {
        let mut d = ExactDedup::new();
        rows.iter().map(|r| d.is_new(text_field(r, &a.field))).collect()
    };
    let kept: Vec<Value> = rows.into_iter().zip(&keep).filter(|(_, k)| **k).map(|(r, _)| r).collect();
    write_records(a.output.as_deref(), &kept)?;
    if a.output.is_some() {
        let dropped = keep.len() - kept.len();
        emit(format, json!({"input": keep.len(), "kept": kept.len(), "dropped": dropped}), || {
            format!("kept {} of {} rows ({dropped} duplicates)", kept.len(), keep.len())
        });
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct DecontamArgs {
    /// JSONL training file (`-` for stdin).
    pub input: String,
    /// Text field of the training rows.
    #[arg(long, default_value = "text")]
    pub field: String,
    /// Evaluation JSONL files (benchmark name = file stem).
    #[arg(long, required = true, num_args = 1..)]
    pub against: Vec<String>,
    /// Text field of the evaluation rows.
    #[arg(long, default_value = "text")]
    pub eval_field: String,
    /// Word n-gram size.
    #[arg(long, default_value_t = 13)]
    pub n: usize,
    /// Minimum shared n-grams to flag a row.
    #[arg(long, default_value_t = 1)]
    pub min_shared: usize,
    /// Output file for clean rows (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

fn decontaminate(a: DecontamArgs, format: OutputFormat) -> Result<ExitCode> {
    let mut d = Decontaminator::new(a.n);
    for path in &a.against {
        let rows = read_records(path)?;
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path)
            .trim_end_matches(".jsonl")
            .to_string();
        d.add_benchmark(&name, rows.iter().map(|r| text_field(r, &a.eval_field).to_string()));
    }
    let rows = read_records(&a.input)?;
    let texts: Vec<&str> = rows.iter().map(|r| text_field(r, &a.field)).collect();
    let hits = d.check_all(&texts, a.min_shared);
    let mut by_bench: BTreeMap<String, usize> = BTreeMap::new();
    for h in hits.iter().flatten() {
        *by_bench.entry(h.benchmark.clone()).or_default() += 1;
    }
    let kept: Vec<Value> = rows.into_iter().zip(&hits).filter(|(_, h)| h.is_none()).map(|(r, _)| r).collect();
    write_records(a.output.as_deref(), &kept)?;
    if a.output.is_some() {
        emit(format, json!({"kept": kept.len(), "contaminated": by_bench}), || {
            let mut s = format!("kept {} rows", kept.len());
            for (b, n) in &by_bench {
                s.push_str(&format!("\n  {b}: {n} contaminated"));
            }
            s
        });
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct SplitArgs {
    /// JSONL file (`-` for stdin).
    pub input: String,
    /// Ratios train,val,test.
    #[arg(long, default_value = "0.8,0.1,0.1")]
    pub ratios: String,
    /// Seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Keep rows sharing this field's value in the same split.
    #[arg(long, conflicts_with = "stratify")]
    pub group_by: Option<String>,
    /// Preserve the proportions of this label field.
    #[arg(long)]
    pub stratify: Option<String>,
    /// Output directory (train.jsonl, val.jsonl, test.jsonl).
    #[arg(long)]
    pub out_dir: String,
}

fn split(a: SplitArgs, format: OutputFormat) -> Result<ExitCode> {
    let rows = read_records(&a.input)?;
    let r: Vec<f64> = a
        .ratios
        .split(',')
        .map(|x| x.trim().parse::<f64>().map_err(|_| HarpiaError::invalid(format!("bad ratio `{x}`"))))
        .collect::<Result<_>>()?;
    if r.len() != 3 {
        return Err(HarpiaError::invalid("--ratios needs three numbers, e.g. 0.8,0.1,0.1"));
    }
    let ratios = Ratios::new(r[0], r[1], r[2])?;
    let key = |f: &str| -> Vec<String> {
        rows.iter()
            .map(|row| {
                get_path(row, f)
                    .map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string()))
                    .unwrap_or_default()
            })
            .collect()
    };
    let s = match (&a.group_by, &a.stratify) {
        (Some(g), _) => group_split(&key(g), ratios, a.seed),
        (None, Some(l)) => stratified_split(&key(l), ratios, a.seed),
        _ => random_split(rows.len(), ratios, a.seed),
    };
    std::fs::create_dir_all(&a.out_dir).map_err(|e| HarpiaError::io(&a.out_dir, e))?;
    let mut counts = BTreeMap::new();
    for (name, idx) in [("train", &s.train), ("val", &s.val), ("test", &s.test)] {
        let part: Vec<Value> = idx.iter().map(|&i| rows[i].clone()).collect();
        write_records(Some(&format!("{}/{name}.jsonl", a.out_dir)), &part)?;
        counts.insert(name, part.len());
    }
    emit(format, json!({"splits": counts, "seed": a.seed}), || {
        format!("train {}  val {}  test {}  (seed {})", counts["train"], counts["val"], counts["test"], a.seed)
    });
    Ok(ExitCode::SUCCESS)
}

fn load_template(spec: &str) -> Result<JinjaChatTemplate> {
    match spec {
        "chatml" => JinjaChatTemplate::new(CHATML, "", "<|im_end|>"),
        "llama3" => JinjaChatTemplate::new(LLAMA3, "<|begin_of_text|>", "<|eot_id|>"),
        "plain" => JinjaChatTemplate::new(PLAIN, "", ""),
        path => JinjaChatTemplate::from_tokenizer_config(path),
    }
}

#[derive(Args)]
pub struct TemplateArgs {
    /// Chat JSONL (any supported format; `-` for stdin).
    pub input: String,
    /// `chatml`, `llama3`, `plain`, or a `tokenizer_config.json`.
    #[arg(long, default_value = "chatml")]
    pub template: String,
    /// Render at most this many rows.
    #[arg(long, default_value_t = 1)]
    pub limit: usize,
}

fn template(a: TemplateArgs, format: OutputFormat) -> Result<ExitCode> {
    let rows = read_records(&a.input)?;
    let t = load_template(&a.template)?;
    let mut out = Vec::new();
    for r in rows.iter().take(a.limit) {
        let f = Format::detect(r).ok_or_else(|| HarpiaError::invalid("unknown row format"))?;
        let sample = to_chat_sample(r, f)?;
        let rendered = t.render_with_spans(&sample.messages)?;
        out.push(json!({"text": rendered.text, "assistant_spans": rendered.assistant_spans}));
    }
    emit(format, json!({"rendered": out}), || {
        out.iter()
            .map(|r| {
                let text = r["text"].as_str().unwrap_or("");
                let spans: Vec<(usize, usize)> = r["assistant_spans"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|s| (s["start"].as_u64().unwrap_or(0) as usize, s["end"].as_u64().unwrap_or(0) as usize))
                    .collect();
                // mark trained-on spans with ⟦ ⟧
                let mut s = String::new();
                let mut last = 0;
                for (st, en) in spans {
                    s.push_str(&text[last..st]);
                    s.push('⟦');
                    s.push_str(&text[st..en]);
                    s.push('⟧');
                    last = en;
                }
                s.push_str(&text[last..]);
                s
            })
            .collect::<Vec<_>>()
            .join("\n----\n")
    });
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct PackArgs {
    /// Chat JSONL (any supported format; `-` for stdin).
    pub input: String,
    /// `tokenizer.json`.
    #[arg(long)]
    pub tokenizer: String,
    /// `chatml`, `llama3`, `plain`, or a `tokenizer_config.json`.
    #[arg(long, default_value = "chatml")]
    pub template: String,
    /// Row length in tokens.
    #[arg(long, default_value_t = 2048)]
    pub seq_len: usize,
    /// Padding token id.
    #[arg(long, default_value_t = 0)]
    pub pad_id: u32,
    /// Use best-fit decreasing (fewer rows; order not preserved).
    #[arg(long)]
    pub best_fit: bool,
    /// Train on every token, not only assistant turns.
    #[arg(long)]
    pub full_loss: bool,
    /// Output file (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

fn pack_cmd(a: PackArgs, format: OutputFormat) -> Result<ExitCode> {
    let rows = read_records(&a.input)?;
    let tok = HfTokenizer::from_file(&a.tokenizer)?;
    let t = load_template(&a.template)?;
    let mut samples = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let f = Format::detect(r).ok_or_else(|| HarpiaError::Record {
            path: None,
            line: i as u64 + 1,
            message: "unknown row format".into(),
        })?;
        let sample = to_chat_sample(r, f)?;
        let rendered = t.render_with_spans(&sample.messages)?;
        let enc = tok.encode(&rendered.text, false)?;
        let mask =
            if a.full_loss { vec![true; enc.ids.len()] } else { loss_mask(&enc.offsets, &rendered.assistant_spans) };
        samples.push(TokenizedSample { ids: enc.ids, loss_mask: mask });
    }
    let total_tokens: usize = samples.iter().map(|s| s.ids.len()).sum();
    let opts = PackOptions {
        strategy: if a.best_fit { PackStrategy::BestFitDecreasing } else { PackStrategy::Greedy },
        ..PackOptions::new(a.seq_len, a.pad_id)
    };
    let packed = pack(samples, opts)?;
    let eff = packing_efficiency(&packed);
    let out: Vec<Value> = packed.iter().map(|p| serde_json::to_value(p).expect("serializable")).collect();
    write_records(a.output.as_deref(), &out)?;
    if a.output.is_some() {
        emit(
            format,
            json!({"samples": rows.len(), "tokens": total_tokens, "rows": out.len(), "efficiency": eff}),
            || {
                format!(
                    "{} samples, {total_tokens} tokens → {} rows of {} (efficiency {:.1}%)",
                    rows.len(),
                    out.len(),
                    a.seq_len,
                    eff * 100.0
                )
            },
        );
    }
    Ok(ExitCode::SUCCESS)
}
