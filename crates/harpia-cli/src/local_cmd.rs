//! Local-model subcommands: classify, translate, generate, embed.

use crate::util::{OutputFormat, emit, read_records, read_text, write_records};
use clap::Args;
use harpia::core::traits::{Embedder, GenerationParams, PairClassifier, Translator};
use harpia::core::{HarpiaError, Result};
use harpia::data::JsonlCache;
use harpia::data::io::get_path;
use harpia::local::{DType, Device, EncoderEmbedder, MarianTranslator, Search, SequenceClassifier, TextGenerator};
use harpia::structured::constraint::{Choice, Constraint, JsonSchemaConstraint, RegexConstraint};
use serde_json::{Value, json};
use std::path::Path;
use std::process::ExitCode;

fn field<'a>(r: &'a Value, f: &str) -> Result<&'a str> {
    get_path(r, f)
        .and_then(Value::as_str)
        .ok_or_else(|| HarpiaError::invalid(format!("row without string field `{f}`")))
}

#[derive(Args)]
pub struct ClassifyArgs {
    /// Model directory (`*ForSequenceClassification`, BERT/RoBERTa/XLM-R).
    #[arg(long)]
    pub model: String,
    /// JSONL input (`-` for stdin).
    pub input: String,
    /// Premise field (pairs) or text field (single texts).
    #[arg(long, default_value = "premise")]
    pub premise: String,
    /// Hypothesis field; omit to classify single texts.
    #[arg(long)]
    pub hypothesis: Option<String>,
    /// Maximum tokens per input (longest-first truncation).
    #[arg(long, default_value_t = 512)]
    pub max_len: usize,
    /// JSON object of per-label thresholds; the predicted label is the most probable label that
    /// reaches its threshold, else `fallback`.
    #[arg(long)]
    pub thresholds: Option<String>,
    /// Label used when no threshold is reached.
    #[arg(long, default_value = "neutral")]
    pub fallback: String,
    /// Output file (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

pub fn classify(a: ClassifyArgs, format: OutputFormat) -> Result<ExitCode> {
    let clf = SequenceClassifier::load_with(Path::new(&a.model), &Device::Cpu, DType::F32, a.max_len)?;
    let mut rows = read_records(&a.input)?;
    let scores = match &a.hypothesis {
        Some(h) => {
            let pairs: Vec<(String, String)> = rows
                .iter()
                .map(|r| Ok((field(r, &a.premise)?.to_string(), field(r, h)?.to_string())))
                .collect::<Result<_>>()?;
            let refs: Vec<(&str, &str)> = pairs.iter().map(|(p, h)| (p.as_str(), h.as_str())).collect();
            clf.classify_pairs(&refs)?
        }
        None => {
            let texts: Vec<String> =
                rows.iter().map(|r| field(r, &a.premise).map(String::from)).collect::<Result<_>>()?;
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            clf.classify(&refs)?
        }
    };
    let thresholds: Option<serde_json::Map<String, Value>> = match &a.thresholds {
        Some(p) => Some(serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| HarpiaError::io(p, e))?)?),
        None => None,
    };
    for (r, s) in rows.iter_mut().zip(&scores) {
        let mut ranked: Vec<(&str, f32)> = s.iter().collect();
        ranked.sort_by(|x, y| y.1.total_cmp(&x.1));
        let label = match &thresholds {
            None => ranked[0].0.to_string(),
            Some(t) => ranked
                .iter()
                .find(|(l, p)| t.get(*l).and_then(Value::as_f64).is_some_and(|th| f64::from(*p) >= th))
                .map(|(l, _)| (*l).to_string())
                .unwrap_or_else(|| a.fallback.clone()),
        };
        if let Some(o) = r.as_object_mut() {
            o.insert(
                "scores".into(),
                json!(s.iter().map(|(l, p)| (l.to_string(), json!(p))).collect::<serde_json::Map<_, _>>()),
            );
            o.insert("label".into(), json!(label));
        }
    }
    write_records(a.output.as_deref(), &rows)?;
    if a.output.is_some() {
        emit(format, json!({"rows": rows.len()}), || format!("classified {} rows", rows.len()));
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct TranslateArgs {
    /// MarianMT model directory (needs `tokenizer.json`).
    #[arg(long)]
    pub model: String,
    /// JSONL input (`-` for stdin).
    pub input: String,
    /// Field to translate.
    #[arg(long, default_value = "text")]
    pub field: String,
    /// Field that receives the translation.
    #[arg(long, default_value = "translation")]
    pub to: String,
    /// Prefix for multilingual models, e.g. `>>por<<`.
    #[arg(long)]
    pub prefix: Option<String>,
    /// Beam width (1 = greedy).
    #[arg(long, default_value_t = 1)]
    pub beams: usize,
    /// Resumable cache (source text → translation).
    #[arg(long)]
    pub cache: Option<String>,
    /// Output file (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

pub fn translate(a: TranslateArgs, format: OutputFormat) -> Result<ExitCode> {
    let mut mt = MarianTranslator::load(Path::new(&a.model))?;
    mt.prefix = a.prefix.clone();
    mt.search = if a.beams > 1 { Search::Beam { width: a.beams } } else { Search::Greedy };
    let mut rows = read_records(&a.input)?;
    let mut cache: Option<JsonlCache<String, String>> = a.cache.as_ref().map(JsonlCache::open).transpose()?;
    let mut done = 0;
    for r in rows.iter_mut() {
        let src = field(r, &a.field)?.to_string();
        let out = match cache.as_ref().and_then(|c| c.get(&src).cloned()) {
            Some(t) => t,
            None => {
                let t = mt.translate(&[&src])?.remove(0);
                if let Some(c) = cache.as_mut() {
                    c.insert(src.clone(), t.clone())?;
                }
                done += 1;
                t
            }
        };
        if let Some(o) = r.as_object_mut() {
            o.insert(a.to.clone(), json!(out));
        }
    }
    write_records(a.output.as_deref(), &rows)?;
    if a.output.is_some() {
        emit(format, json!({"rows": rows.len(), "translated": done}), || {
            format!("{} rows, {done} newly translated", rows.len())
        });
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct GenerateArgs {
    /// Decoder model directory (Qwen2 / Llama families).
    #[arg(long)]
    pub model: String,
    /// Prompt text, or `-` for stdin.
    pub prompt: String,
    /// Render the prompt as a user message with the model's chat template.
    #[arg(long)]
    pub chat: bool,
    /// Constrain the output to JSON matching this JSON Schema file.
    #[arg(long, conflicts_with_all = ["regex", "choices"])]
    pub schema: Option<String>,
    /// Constrain the output to this regular expression.
    #[arg(long, conflicts_with = "choices")]
    pub regex: Option<String>,
    /// Constrain the output to one of these comma-separated strings.
    #[arg(long)]
    pub choices: Option<String>,
    /// Maximum new tokens.
    #[arg(long, default_value_t = 256)]
    pub max_tokens: u32,
    /// Sampling temperature (0 = greedy; default: the model's generation_config.json).
    #[arg(long)]
    pub temperature: Option<f32>,
    /// Nucleus sampling (default: generation_config.json).
    #[arg(long)]
    pub top_p: Option<f32>,
    /// Repetition penalty over prompt and output (default: generation_config.json).
    #[arg(long)]
    pub repetition_penalty: Option<f32>,
    /// Seed.
    #[arg(long)]
    pub seed: Option<u64>,
}

pub fn generate(a: GenerateArgs, format: OutputFormat) -> Result<ExitCode> {
    let lm = TextGenerator::load(Path::new(&a.model))?;
    let defaults = lm.default_params();
    let prompt = if a.prompt == "-" { read_text("-")? } else { a.prompt.clone() };
    // unset sampling flags fall back to the model's generation_config.json, as in transformers
    let params = GenerationParams {
        max_new_tokens: a.max_tokens,
        temperature: a.temperature.unwrap_or(defaults.temperature),
        top_p: a.top_p.unwrap_or(defaults.top_p),
        top_k: defaults.top_k,
        min_p: defaults.min_p,
        repetition_penalty: a.repetition_penalty.unwrap_or(defaults.repetition_penalty),
        seed: a.seed,
        ..Default::default()
    };
    let constraint: Option<Box<dyn Constraint>> = if let Some(p) = &a.schema {
        let s: Value = serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| HarpiaError::io(p, e))?)?;
        Some(Box::new(JsonSchemaConstraint::new(&s)?))
    } else if let Some(r) = &a.regex {
        Some(Box::new(RegexConstraint::new(r)?))
    } else {
        a.choices.as_ref().map(|c| Box::new(Choice::new(c.split(',').map(str::trim))) as Box<dyn Constraint>)
    };
    let (text, add_special) =
        if a.chat { (lm.render_chat(&[harpia::core::Message::user(prompt)])?, false) } else { (prompt, true) };
    let tokens = harpia::core::traits::Tokenizer::encode(lm.tokenizer(), &text, add_special)?.ids;
    let g = if format == OutputFormat::Text && constraint.is_none() {
        use std::io::Write;
        let g = lm.generate_with(&tokens, &params, None, |d| {
            print!("{d}");
            let _ = std::io::stdout().flush();
        })?;
        println!();
        let _ = g;
        return Ok(ExitCode::SUCCESS);
    } else {
        lm.generate_with(&tokens, &params, constraint.as_deref(), |_| {})?
    };
    emit(format, json!({"generation": g}), || g.text.clone());
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct EmbedArgs {
    /// Encoder model directory.
    #[arg(long)]
    pub model: String,
    /// JSONL input (`-` for stdin).
    pub input: String,
    /// Text field.
    #[arg(long, default_value = "text")]
    pub field: String,
    /// Output file (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

pub fn embed(a: EmbedArgs, format: OutputFormat) -> Result<ExitCode> {
    let emb = EncoderEmbedder::load(Path::new(&a.model))?;
    let mut rows = read_records(&a.input)?;
    let texts: Vec<String> = rows.iter().map(|r| field(r, &a.field).map(String::from)).collect::<Result<_>>()?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let vecs = emb.embed(&refs)?;
    for (r, v) in rows.iter_mut().zip(vecs) {
        if let Some(o) = r.as_object_mut() {
            o.insert("embedding".into(), json!(v));
        }
    }
    write_records(a.output.as_deref(), &rows)?;
    if a.output.is_some() {
        emit(format, json!({"rows": rows.len(), "dim": emb.dim()}), || {
            format!("embedded {} rows (dim {})", rows.len(), emb.dim())
        });
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct DownloadArgs {
    /// Repository id, e.g. `Qwen/Qwen2.5-0.5B-Instruct`.
    pub repo: String,
    /// Revision (branch, tag or commit).
    #[arg(long, default_value = "main")]
    pub revision: String,
    /// Cache directory (default: `$HARPIA_HOME/hub`, else `~/.cache/harpia/hub`).
    #[arg(long)]
    pub cache: Option<String>,
    /// File patterns to fetch (comma-separated; default: configs, safetensors, tokenizers).
    #[arg(long, value_delimiter = ',')]
    pub include: Vec<String>,
    /// Allow network access (required).
    #[arg(long)]
    pub online: bool,
}

pub fn download(a: DownloadArgs, format: OutputFormat) -> Result<ExitCode> {
    let cache = a.cache.clone().map(std::path::PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("HARPIA_HOME")
            .map(|h| std::path::PathBuf::from(h).join("hub"))
            .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache/harpia/hub")))
            .unwrap_or_else(|| "harpia-hub".into())
    });
    let policy = if a.online { harpia::core::NetworkPolicy::Online } else { harpia::core::NetworkPolicy::Offline };
    let hub = harpia::local::hub::Hub::new(policy, &cache)?;
    let pats: Vec<&str> = a.include.iter().map(String::as_str).collect();
    let dir = hub.snapshot(&a.repo, &a.revision, if pats.is_empty() { None } else { Some(&pats) })?;
    emit(format, json!({"path": dir}), || dir.display().to_string());
    Ok(ExitCode::SUCCESS)
}
