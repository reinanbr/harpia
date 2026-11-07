//! Evaluation subcommands: `eval run|judge|rag|perplexity|compare` and `calibrate`.

use crate::llm_cmd::{AnyProvider, ProviderArgs, runtime};
use crate::util::{OutputFormat, emit, lang, read_records, write_records};
use clap::{Args, Subcommand};
use harpia::core::traits::JudgeInput;
use harpia::core::{HarpiaError, Result, Scores};
use harpia::data::JsonlCache;
use harpia::eval::calibration::{
    Calibrated, DecisionRule, Objective, TestSet, ValidationSet, default_grid, grid_search,
};
use harpia::eval::harness::{ModelBackend, ProviderBackend, RunOptions, Task, TaskKind, TaskResult, run_task};
use harpia::eval::judge::{LlmJudge, Preference, agreement};
use harpia::eval::rag::{RagExample, faithfulness, retrieval_report};
use harpia::eval::report::{RunSummary, leaderboard};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::process::ExitCode;

#[derive(Subcommand)]
pub enum EvalCommand {
    /// Run tasks (TOML/JSON) against a model or provider.
    Run(RunArgs),
    /// Grade answers with an LLM judge (rubric scores or pairwise with position swap).
    Judge(JudgeArgs),
    /// Retrieval metrics and, with --classifier, answer faithfulness for RAG outputs.
    Rag(RagArgs),
    /// Perplexity / bits per byte of held-out text under a local decoder.
    Perplexity(PerplexityArgs),
    /// Markdown leaderboard from run summaries.
    Compare(CompareArgs),
}

pub fn run(c: EvalCommand, format: OutputFormat) -> Result<ExitCode> {
    match c {
        EvalCommand::Run(a) => run_tasks(a, format),
        EvalCommand::Judge(a) => judge(a, format),
        EvalCommand::Rag(a) => rag(a, format),
        EvalCommand::Perplexity(a) => perplexity(a, format),
        EvalCommand::Compare(a) => compare(a, format),
    }
}

#[derive(Args)]
pub struct RunArgs {
    /// Task files (TOML or JSON).
    #[arg(long = "task", required = true, num_args = 1..)]
    pub tasks: Vec<String>,
    #[command(flatten)]
    pub provider: ProviderArgs,
    /// Name shown in summaries and leaderboards (default: the model).
    #[arg(long)]
    pub name: Option<String>,
    /// Generation cache (JSONL); reruns skip cached prompts.
    #[arg(long)]
    pub cache: Option<String>,
    /// Concurrent requests.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// Write the run summary (JSON) here.
    #[arg(short, long)]
    pub output: Option<String>,
    /// Write per-row samples (JSONL) here.
    #[arg(long)]
    pub samples: Option<String>,
    /// Fail (exit 1) unless `task/metric>=value`, e.g. `capitals/exact_match=0.8`. Repeatable.
    #[arg(long = "min")]
    pub min: Vec<String>,
}

fn parse_min(spec: &str) -> Result<(String, String, f64)> {
    let (lhs, v) = spec
        .split_once('=')
        .ok_or_else(|| HarpiaError::invalid(format!("--min `{spec}`: expected task/metric=value")))?;
    let (t, m) = lhs
        .split_once('/')
        .ok_or_else(|| HarpiaError::invalid(format!("--min `{spec}`: expected task/metric=value")))?;
    let v: f64 = v.parse().map_err(|_| HarpiaError::invalid(format!("--min `{spec}`: `{v}` is not a number")))?;
    Ok((t.into(), m.into(), v))
}

fn run_tasks(a: RunArgs, format: OutputFormat) -> Result<ExitCode> {
    let mins: Vec<(String, String, f64)> = a.min.iter().map(|m| parse_min(m)).collect::<Result<_>>()?;
    let provider = a.provider.build()?;
    let rt = runtime()?;
    let mut results: Vec<TaskResult> = Vec::new();
    for path in &a.tasks {
        let task = Task::load(path)?;
        let mut opts =
            RunOptions { cache: a.cache.as_ref().map(JsonlCache::open).transpose()?, concurrency: a.concurrency };
        let needs_ll = matches!(task.kind, TaskKind::MultipleChoice | TaskKind::Perplexity);
        eprintln!("running {} ({:?})...", task.name, task.kind);
        let r = match (&provider, needs_ll) {
            (AnyProvider::Local(g), true) => rt.block_on(run_task(&task, &ModelBackend(g.as_ref()), &mut opts))?,
            (_, true) => {
                return Err(HarpiaError::invalid(format!(
                    "task `{}` needs log-likelihoods: use --provider local with a model directory",
                    task.name
                )));
            }
            (p, false) => rt.block_on(run_task(
                &task,
                &ProviderBackend { provider: p, model: a.provider.model.clone() },
                &mut opts,
            ))?,
        };
        results.push(r);
    }
    let name = a.name.clone().unwrap_or_else(|| a.provider.model.clone());
    let mut summary = RunSummary::new(&name, &results);
    summary.metadata.insert("provider".into(), json!(format!("{:?}", a.provider.provider).to_lowercase()));
    if let Some(p) = &a.output {
        summary.save(p)?;
    }
    if let Some(p) = &a.samples {
        let rows: Vec<Value> =
            results.iter().flat_map(|r| r.samples.iter().map(move |s| json!({"task": r.task, "sample": s}))).collect();
        write_records(Some(p), &rows)?;
    }
    let mut failed = Vec::new();
    for (t, m, v) in &mins {
        let got = summary.results.get(t).and_then(|x| x.get(m)).copied();
        if got.is_none_or(|g| g < *v) {
            failed.push(format!("{t}/{m} = {} < {v}", got.map_or("missing".into(), |g| format!("{g:.4}"))));
        }
    }
    let payload = json!({
        "model": name,
        "results": summary.results,
        "n": summary.n,
        "usage": results.iter().map(|r| (r.task.clone(), json!(r.usage))).collect::<BTreeMap<_, _>>(),
        "failed_thresholds": failed,
    });
    emit(format, payload, || {
        let mut s = String::new();
        for r in &results {
            s += &format!("{} ({} rows, {} cached, {:.1}s)\n", r.task, r.n, r.cached, r.seconds);
            for (k, v) in &r.metrics {
                s += &format!("  {k:<16} {v:.4}\n");
            }
        }
        for f in &failed {
            s += &format!("FAILED threshold: {f}\n");
        }
        s.trim_end().to_string()
    });
    Ok(if failed.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

#[derive(Args)]
pub struct JudgeArgs {
    /// JSONL with `question`, `answer` (and optional `reference`), or `answer_a`/`answer_b` with --pairwise.
    pub input: String,
    #[command(flatten)]
    pub provider: ProviderArgs,
    /// Rubric text, or `@file` to read it.
    #[arg(long)]
    pub rubric: String,
    /// Pairwise comparison of `answer_a` vs `answer_b` (both orders).
    #[arg(long)]
    pub pairwise: bool,
    /// Score range (`min,max`).
    #[arg(long, default_value = "1,10")]
    pub scale: String,
    /// Output JSONL with verdicts (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

fn field<'a>(row: &'a Value, k: &str) -> &'a str {
    row.get(k).and_then(Value::as_str).unwrap_or("")
}

fn judge(a: JudgeArgs, format: OutputFormat) -> Result<ExitCode> {
    let rubric = match a.rubric.strip_prefix('@') {
        Some(p) => std::fs::read_to_string(p).map_err(|e| HarpiaError::io(p, e))?,
        None => a.rubric.clone(),
    };
    let (lo, hi) = a
        .scale
        .split_once(',')
        .and_then(|(l, h)| Some((l.trim().parse().ok()?, h.trim().parse().ok()?)))
        .ok_or_else(|| HarpiaError::invalid("--scale must be `min,max`"))?;
    let rows = read_records(&a.input)?;
    let provider = a.provider.build()?;
    let mut j = LlmJudge::new(&provider, a.provider.model.clone());
    j.scale = (lo, hi);
    let rt = runtime()?;
    let mut out = Vec::with_capacity(rows.len());
    let mut summary = json!({});
    if a.pairwise {
        let (mut wa, mut wb, mut ties, mut inconsistent) = (0, 0, 0, 0);
        let (mut fwd, mut sw) = (Vec::new(), Vec::new());
        for row in rows {
            let v = rt.block_on(j.compare(
                field(&row, "question"),
                &rubric,
                field(&row, "answer_a"),
                field(&row, "answer_b"),
            ))?;
            match v.winner {
                Preference::A => wa += 1,
                Preference::B => wb += 1,
                Preference::Tie => ties += 1,
            }
            inconsistent += usize::from(!v.consistent);
            fwd.push(format!("{:?}", v.forward));
            sw.push(format!("{:?}", v.swapped));
            let mut r = row;
            r["verdict"] = json!(v);
            out.push(r);
        }
        let n = out.len().max(1) as f64;
        summary = json!({"rows": out.len(), "a_wins": wa, "b_wins": wb, "ties": ties,
            "a_win_rate": wa as f64 / n, "position_inconsistent": inconsistent,
            "order_agreement": agreement(&fwd, &sw)});
    } else {
        let mut scores = Vec::new();
        for row in rows {
            let input = JudgeInput {
                question: field(&row, "question").into(),
                answer: field(&row, "answer").into(),
                reference: row.get("reference").and_then(Value::as_str).map(String::from),
                rubric: rubric.clone(),
            };
            let v = rt.block_on(j.score(&input))?;
            scores.push(v.score);
            let mut r = row;
            r["verdict"] = json!(v);
            out.push(r);
        }
        let mean = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
        summary = json!({"rows": out.len(), "mean_score": mean, "scale": [lo, hi]});
    }
    write_records(a.output.as_deref(), &out)?;
    if a.output.is_some() {
        emit(format, summary.clone(), || serde_json::to_string_pretty(&summary).unwrap_or_default());
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct RagArgs {
    /// JSONL of RAG examples: `question`, `retrieved`, `relevant`, optional `contexts`, `answer`.
    pub input: String,
    /// Cut-off for recall/precision/nDCG.
    #[arg(long, default_value_t = 5)]
    pub k: usize,
    /// NLI cross-encoder directory for answer faithfulness (premise = context, hypothesis = answer).
    #[arg(long)]
    pub classifier: Option<String>,
    /// Classifier label meaning "supported".
    #[arg(long, default_value = "entailment")]
    pub entailment: String,
    /// Classifier label meaning "contradicted".
    #[arg(long, default_value = "contradiction")]
    pub contradiction: String,
    /// Classifier label meaning "neutral".
    #[arg(long, default_value = "neutral")]
    pub neutral: String,
    /// Calibrated thresholds (from `harpia calibrate`); default 0.5 for every label.
    #[arg(long)]
    pub thresholds: Option<String>,
    /// Language for sentence splitting.
    #[arg(long, default_value = "en")]
    pub lang: String,
    /// Fail (exit 1) when mean faithfulness is below this.
    #[arg(long)]
    pub min_faithfulness: Option<f64>,
    /// Write per-example faithfulness (JSONL) here.
    #[arg(short, long)]
    pub output: Option<String>,
}

fn rag(a: RagArgs, format: OutputFormat) -> Result<ExitCode> {
    let examples: Vec<RagExample> =
        read_records(&a.input)?.into_iter().map(serde_json::from_value).collect::<std::result::Result<_, _>>()?;
    let retrieval = retrieval_report(&examples, a.k);
    let mut faith = None;
    let mut rows = Vec::new();
    if let Some(dir) = &a.classifier {
        let clf = harpia::local::SequenceClassifier::load(std::path::Path::new(dir))?;
        let cal = match &a.thresholds {
            Some(p) => Calibrated::load(p)?,
            None => {
                let rule = DecisionRule::nli(&a.entailment, &a.contradiction, &a.neutral);
                Calibrated::fixed(rule.clone(), harpia::eval::calibration::Thresholds::uniform(&rule, 0.5))
            }
        };
        let l = lang(&a.lang)?.data();
        let mut total = 0.0;
        for e in examples.iter().filter(|e| !e.answer.is_empty()) {
            let f = faithfulness(&clf, &cal, &a.entailment, &e.answer, &e.contexts, &l)?;
            total += f.score;
            rows.push(json!({"question": e.question, "faithfulness": f}));
        }
        faith = Some(if rows.is_empty() { 0.0 } else { total / rows.len() as f64 });
    }
    if let Some(p) = &a.output {
        write_records(Some(p), &rows)?;
    }
    let failed = matches!((faith, a.min_faithfulness), (Some(f), Some(m)) if f < m);
    emit(format, json!({"retrieval": retrieval, "faithfulness": faith, "failed": failed}), || {
        let mut s = format!(
            "examples {}\nrecall@{k} {:.4}\nprecision@{k} {:.4}\nmrr {:.4}\nndcg@{k} {:.4}",
            retrieval.n,
            retrieval.recall,
            retrieval.precision,
            retrieval.mrr,
            retrieval.ndcg,
            k = retrieval.k
        );
        if let Some(f) = faith {
            s += &format!("\nfaithfulness {f:.4}");
        }
        s
    });
    Ok(if failed { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

#[derive(Args)]
pub struct PerplexityArgs {
    /// Local decoder model directory.
    #[arg(long)]
    pub model: String,
    /// Text file, or JSONL with --field (`-` for stdin).
    pub input: String,
    /// JSONL field holding the text (plain text input when omitted: one document per paragraph).
    #[arg(long)]
    pub field: Option<String>,
    /// Conditioning context before each text.
    #[arg(long, default_value = "\n")]
    pub context: String,
}

fn perplexity(a: PerplexityArgs, format: OutputFormat) -> Result<ExitCode> {
    let texts: Vec<String> = match &a.field {
        Some(f) => {
            read_records(&a.input)?.iter().filter_map(|r| r.get(f).and_then(Value::as_str).map(String::from)).collect()
        }
        None => crate::util::read_text(&a.input)?
            .split("\n\n")
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect(),
    };
    let dir = tempfile_dir()?;
    let ds = dir.join("ppl.jsonl");
    write_records(ds.to_str(), &texts.iter().map(|t| json!({"text": t})).collect::<Vec<_>>())?;
    let task: Task = toml::from_str(&format!(
        "name = \"perplexity\"\nkind = \"perplexity\"\ndataset = {:?}\ntarget = \"{{{{ text }}}}\"\nperplexity_context = {:?}",
        ds.display().to_string(),
        a.context
    ))
    .map_err(|e| HarpiaError::invalid(e.to_string()))?;
    let model = harpia::local::TextGenerator::load(std::path::Path::new(&a.model))?;
    let r = runtime()?.block_on(run_task(&task, &ModelBackend(&model), &mut RunOptions::default()))?;
    let _ = std::fs::remove_dir_all(&dir);
    emit(format, json!({"texts": r.n, "metrics": r.metrics}), || {
        format!(
            "texts {}\nbits_per_byte {:.4}\nword_perplexity {:.3}",
            r.n, r.metrics["bits_per_byte"], r.metrics["word_perplexity"]
        )
    });
    Ok(ExitCode::SUCCESS)
}

fn tempfile_dir() -> Result<std::path::PathBuf> {
    let d = std::env::temp_dir().join(format!("harpia-ppl-{}", std::process::id()));
    std::fs::create_dir_all(&d).map_err(|e| HarpiaError::io(&d, e))?;
    Ok(d)
}

#[derive(Args)]
pub struct CompareArgs {
    /// Run summaries written by `eval run -o`.
    #[arg(required = true, num_args = 1..)]
    pub summaries: Vec<String>,
    /// Only these metrics (comma-separated).
    #[arg(long)]
    pub metrics: Option<String>,
    /// Write the Markdown leaderboard here.
    #[arg(short, long)]
    pub output: Option<String>,
}

fn compare(a: CompareArgs, format: OutputFormat) -> Result<ExitCode> {
    let runs: Vec<RunSummary> = a.summaries.iter().map(RunSummary::load).collect::<Result<_>>()?;
    let filter: Option<Vec<String>> = a.metrics.as_ref().map(|m| m.split(',').map(|s| s.trim().to_string()).collect());
    let refs: Option<Vec<&str>> = filter.as_ref().map(|v| v.iter().map(String::as_str).collect());
    let md = leaderboard(&runs, refs.as_deref());
    if let Some(p) = &a.output {
        std::fs::write(p, &md).map_err(|e| HarpiaError::io(p, e))?;
    }
    emit(format, json!({"runs": runs.len(), "markdown": md}), || md.trim_end().to_string());
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct CalibrateArgs {
    /// Validation JSONL: a `scores` object (label → probability) and a gold `label` per row.
    #[arg(long)]
    pub val: String,
    /// Test JSONL (same shape); thresholds are applied frozen.
    #[arg(long)]
    pub test: Option<String>,
    /// Rule: `nli` (contradiction must beat entailment, else entailment, else neutral) or `argmax`.
    #[arg(long, default_value = "nli")]
    pub rule: String,
    /// Labels as `entailment,contradiction,neutral` (nli) or a list (argmax, last = fallback).
    #[arg(long, default_value = "entailment,contradiction,neutral")]
    pub labels: String,
    /// Objective: macro_f1, accuracy, weighted_f1.
    #[arg(long, default_value = "macro_f1")]
    pub objective: String,
    /// Field names.
    #[arg(long, default_value = "scores")]
    pub scores_field: String,
    /// Gold label field.
    #[arg(long, default_value = "label")]
    pub label_field: String,
    /// Write the fitted thresholds (JSON) here.
    #[arg(short, long)]
    pub output: Option<String>,
}

fn scored_rows(path: &str, sf: &str, lf: &str) -> Result<(Vec<Scores>, Vec<String>)> {
    let mut scores = Vec::new();
    let mut gold = Vec::new();
    for (i, r) in read_records(path)?.into_iter().enumerate() {
        let obj = r.get(sf).and_then(Value::as_object).ok_or_else(|| HarpiaError::Record {
            path: Some(path.into()),
            line: i as u64 + 1,
            message: format!("missing `{sf}` object"),
        })?;
        scores.push(Scores::new(obj.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0) as f32)).collect()));
        gold.push(r.get(lf).and_then(Value::as_str).unwrap_or("").to_string());
    }
    Ok((scores, gold))
}

pub fn calibrate(a: CalibrateArgs, format: OutputFormat) -> Result<ExitCode> {
    let labels: Vec<&str> = a.labels.split(',').map(str::trim).collect();
    let rule = match (a.rule.as_str(), labels.as_slice()) {
        ("nli", [e, c, n]) => DecisionRule::nli(e, c, n),
        ("argmax", [rest @ .., fallback]) if !rest.is_empty() => DecisionRule::argmax(rest, fallback),
        _ => return Err(HarpiaError::invalid("--rule nli needs 3 labels; argmax needs at least 2")),
    };
    let objective: Objective =
        serde_json::from_value(json!(a.objective)).map_err(|_| HarpiaError::invalid("unknown --objective"))?;
    let (vs, vg) = scored_rows(&a.val, &a.scores_field, &a.label_field)?;
    let cal = grid_search(&rule, &ValidationSet::new(vs, vg)?, &default_grid(), objective)?;
    if let Some(p) = &a.output {
        cal.save(p)?;
    }
    let test = match &a.test {
        Some(t) => {
            let (ts, tg) = scored_rows(t, &a.scores_field, &a.label_field)?;
            Some(cal.evaluate(&TestSet::new(ts, tg)?)?)
        }
        None => None,
    };
    emit(format, json!({"calibration": cal, "test": test}), || {
        let mut s = format!(
            "thresholds {:?}\nvalidation {:?} = {:.4} (n = {})",
            cal.thresholds.0, cal.objective, cal.validation_score, cal.validation_n
        );
        if let Some(t) = &test {
            s += &format!("\n\ntest (frozen thresholds)\n{}", t.to_text());
        }
        s
    });
    Ok(ExitCode::SUCCESS)
}
