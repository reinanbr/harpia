//! LLM evaluation harness: tasks defined in TOML/JSON over JSONL datasets.
//!
//! Task kinds:
//! - `generation`: render a prompt, generate, score with generation metrics;
//! - `multiple_choice`: score every choice by log-likelihood (accuracy and byte-length
//!   normalized accuracy, as lm-eval-harness reports `acc` / `acc_norm`);
//! - `perplexity`: log-likelihood of held-out text (bits per byte, word perplexity);
//! - `classification`: generate, map the reply to a label, report classification metrics.
//!
//! Prompts are `minijinja` templates over the row (`"Q: {{ question }}\nA:"`). Few-shot examples
//! are drawn with a seed from a separate pool (or from the dataset, never the row itself).
//! Generation runs against any [`Backend`]: an API or local [`Provider`] (chat-template aware:
//! the prompt becomes a user message) or a raw [`Model`] (needed for log-likelihood tasks).
//! Results can be cached in a [`JsonlCache`] so interrupted runs resume.
//!
//! ```toml
//! name = "capitals"
//! kind = "generation"
//! dataset = "capitals.jsonl"
//! prompt = "What is the capital of {{ country }}? Answer with one word."
//! target = "{{ capital }}"
//! metrics = ["exact_match", "token_f1"]
//! few_shot = 2
//! seed = 0
//! ```

use crate::classification::ClassificationReport;
use crate::metrics::GenMetric;
use harpia_core::traits::{ChatRequest, GenerationParams, Metric, Model, Provider};
use harpia_core::{HarpiaError, Message, Result, TokenUsage};
use harpia_data::cache::JsonlCache;
use harpia_data::rng::SplitMix64;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Kind of task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// Free generation scored with metrics.
    Generation,
    /// Log-likelihood ranking of choices.
    MultipleChoice,
    /// Log-likelihood of text.
    Perplexity,
    /// Generation mapped to labels.
    Classification,
}

/// A task definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    /// Task name.
    pub name: String,
    /// Kind.
    pub kind: TaskKind,
    /// JSONL dataset (relative paths resolve against the task file).
    pub dataset: PathBuf,
    /// Prompt template over the row (unused for perplexity).
    #[serde(default)]
    pub prompt: String,
    /// Target template (generation: reference answer; multiple choice: index or choice text;
    /// classification: gold label; perplexity: the text to score).
    #[serde(default)]
    pub target: String,
    /// Multiple choice: row field holding the list of choices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choices_field: Option<String>,
    /// Classification: allowed labels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    /// Generation metrics by name (default `exact_match`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metrics: Vec<String>,
    /// Number of few-shot examples.
    #[serde(default)]
    pub few_shot: usize,
    /// JSONL pool for few-shot examples (default: the dataset itself, excluding the row).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub few_shot_dataset: Option<PathBuf>,
    /// Separator between few-shot blocks.
    #[serde(default = "default_sep")]
    pub few_shot_separator: String,
    /// Seed for few-shot selection.
    #[serde(default)]
    pub seed: u64,
    /// Maximum rows to evaluate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    /// Maximum generated tokens.
    #[serde(default = "default_max_tokens")]
    pub max_new_tokens: u32,
    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// System prompt (chat backends).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Perplexity: context the text is conditioned on (local models need a non-empty one).
    #[serde(default = "default_ppl_context")]
    pub perplexity_context: String,
}

fn default_sep() -> String {
    "\n\n".into()
}
fn default_max_tokens() -> u32 {
    128
}
fn default_ppl_context() -> String {
    "\n".into()
}

impl Task {
    /// Load a `.toml` or `.json` task; relative dataset paths resolve against its directory.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| HarpiaError::io(path, e))?;
        let mut t: Task = if path.extension().is_some_and(|e| e == "json") {
            serde_json::from_str(&text)?
        } else {
            toml::from_str(&text).map_err(|e| HarpiaError::invalid(format!("{}: {e}", path.display())))?
        };
        let base = path.parent().unwrap_or(Path::new("."));
        for p in std::iter::once(&mut t.dataset).chain(t.few_shot_dataset.as_mut()) {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        }
        Ok(t)
    }

    fn metric_list(&self) -> Result<Vec<GenMetric>> {
        if self.metrics.is_empty() {
            return Ok(vec![GenMetric::ExactMatch]);
        }
        self.metrics
            .iter()
            .map(|m| GenMetric::parse(m).ok_or_else(|| HarpiaError::invalid(format!("unknown metric {m:?}"))))
            .collect()
    }
}

/// Something that can answer prompts, optionally with log-likelihoods.
pub trait Backend: Send + Sync {
    /// Identifier recorded in results and cache keys.
    fn id(&self) -> String;
    /// Generate a completion for `prompt`.
    fn generate(
        &self,
        system: Option<&str>,
        prompt: &str,
        params: &GenerationParams,
    ) -> impl Future<Output = Result<(String, TokenUsage)>> + Send;
    /// Log-likelihood of `continuation` after `context`; `None` when unsupported.
    fn loglikelihood(&self, context: &str, continuation: &str) -> Option<Result<f64>>;
}

/// A [`Provider`] as backend: prompts become user messages (so the provider's chat template
/// applies). No log-likelihoods.
#[derive(Debug)]
pub struct ProviderBackend<'a, P> {
    /// The provider.
    pub provider: &'a P,
    /// Model name sent with each request.
    pub model: String,
}

impl<P: Provider> Backend for ProviderBackend<'_, P> {
    fn id(&self) -> String {
        format!("{}:{}", self.provider.name(), self.model)
    }

    async fn generate(
        &self,
        system: Option<&str>,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<(String, TokenUsage)> {
        let mut messages = Vec::new();
        if let Some(s) = system {
            messages.push(Message::system(s));
        }
        messages.push(Message::user(prompt));
        let req = ChatRequest { model: self.model.clone(), messages, params: params.clone(), ..Default::default() };
        let r = self.provider.chat(&req).await?;
        Ok((r.message.content, r.usage))
    }

    fn loglikelihood(&self, _context: &str, _continuation: &str) -> Option<Result<f64>> {
        None
    }
}

/// A raw [`Model`] as backend: prompts are used verbatim (system prompt prepended).
#[derive(Debug)]
pub struct ModelBackend<'a, M>(pub &'a M);

impl<M: Model> Backend for ModelBackend<'_, M> {
    fn id(&self) -> String {
        self.0.id().to_string()
    }

    async fn generate(
        &self,
        system: Option<&str>,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<(String, TokenUsage)> {
        let full = match system {
            Some(s) => format!("{s}\n\n{prompt}"),
            None => prompt.to_string(),
        };
        let g = self.0.generate(&full, params)?;
        Ok((g.text, g.usage))
    }

    fn loglikelihood(&self, context: &str, continuation: &str) -> Option<Result<f64>> {
        Some(self.0.loglikelihood(context, continuation))
    }
}

/// One evaluated row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleResult {
    /// Row index in the dataset.
    pub index: usize,
    /// Full prompt sent (with few-shot examples).
    pub prompt: String,
    /// Model output (generation/classification), chosen choice (multiple choice) or empty.
    pub output: String,
    /// Reference.
    pub target: String,
    /// Per-metric score of this row.
    pub scores: BTreeMap<String, f64>,
}

/// Result of one task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task name.
    pub task: String,
    /// Kind.
    pub kind: TaskKind,
    /// Backend id.
    pub backend: String,
    /// Rows evaluated.
    pub n: usize,
    /// Aggregate metrics.
    pub metrics: BTreeMap<String, f64>,
    /// Classification report (classification tasks).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification: Option<ClassificationReport>,
    /// Token usage over uncached calls.
    pub usage: TokenUsage,
    /// Rows served from the cache.
    pub cached: usize,
    /// Wall-clock seconds.
    pub seconds: f64,
    /// Per-row results.
    pub samples: Vec<SampleResult>,
}

/// Run options.
#[derive(Default)]
pub struct RunOptions {
    /// Generation cache (key: hash of backend id, task, prompt and parameters).
    pub cache: Option<JsonlCache<String, String>>,
    /// Concurrent requests to the backend (default 1).
    pub concurrency: usize,
}

impl std::fmt::Debug for RunOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunOptions")
            .field("cache", &self.cache.as_ref().map(JsonlCache::len))
            .field("concurrency", &self.concurrency)
            .finish()
    }
}

fn render(env: &minijinja::Environment<'_>, template: &str, row: &Value) -> Result<String> {
    env.render_str(template, minijinja::Value::from(minijinja::value::Serde(row)))
        .map_err(|e| HarpiaError::Template(e.to_string()))
}

fn read_rows(path: &Path) -> Result<Vec<Value>> {
    harpia_data::io::read_jsonl(path)
}

fn cache_key(backend: &str, task: &Task, prompt: &str) -> String {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    for part in [backend, &task.name, prompt, &task.max_new_tokens.to_string(), &task.stop.join("\u{1f}")] {
        h.update(part.as_bytes());
        h.update(&[0x1e]);
    }
    if let Some(s) = &task.system {
        h.update(s.as_bytes());
    }
    format!("{:032x}", h.digest128())
}

/// Map a free-form reply to one of `labels`: exact (case-insensitive) match first, then the
/// label that appears earliest in the reply.
pub fn match_label<'a>(reply: &str, labels: &'a [String]) -> Option<&'a str> {
    let r = reply.trim().to_lowercase();
    let r = r.trim_matches(|c: char| !c.is_alphanumeric());
    labels.iter().find(|l| l.to_lowercase() == r).map(String::as_str).or_else(|| {
        labels
            .iter()
            .filter_map(|l| r.find(&l.to_lowercase()).map(|i| (i, l)))
            .min_by_key(|(i, _)| *i)
            .map(|(_, l)| l.as_str())
    })
}

/// Run a task against a backend.
pub async fn run_task<B: Backend>(task: &Task, backend: &B, opts: &mut RunOptions) -> Result<TaskResult> {
    let start = Instant::now();
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    let mut rows = read_rows(&task.dataset)?;
    if let Some(l) = task.limit {
        rows.truncate(l);
    }
    let pool = match &task.few_shot_dataset {
        Some(p) => Some(read_rows(p)?),
        None => None,
    };
    let backend_id = backend.id();
    let params = GenerationParams {
        max_new_tokens: task.max_new_tokens,
        stop: task.stop.clone(),
        ..GenerationParams::default()
    };

    // Prompts with seeded few-shot examples (row index mixed into the seed, so each row gets its
    // own deterministic sample independent of evaluation order).
    let mut prompts = Vec::with_capacity(rows.len());
    let mut targets = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let mut parts = Vec::new();
        if task.few_shot > 0 && task.kind != TaskKind::Perplexity {
            let (source, exclude) = match &pool {
                Some(p) => (p, None),
                None => (&rows, Some(i)),
            };
            let mut idx: Vec<usize> = (0..source.len()).filter(|&j| Some(j) != exclude).collect();
            SplitMix64::new(task.seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)).shuffle(&mut idx);
            for &j in idx.iter().take(task.few_shot) {
                let ex = &source[j];
                let answer = match task.kind {
                    TaskKind::MultipleChoice => choice_text(task, ex, &env)?,
                    _ => render(&env, &task.target, ex)?,
                };
                parts.push(format!("{} {}", render(&env, &task.prompt, ex)?, answer.trim()));
            }
        }
        if task.kind != TaskKind::Perplexity {
            parts.push(render(&env, &task.prompt, row)?);
        }
        prompts.push(parts.join(&task.few_shot_separator));
        targets.push(render(&env, &task.target, row)?);
    }

    let mut usage = TokenUsage::default();
    let mut cached = 0;
    let mut samples = Vec::with_capacity(rows.len());
    let mut metrics = BTreeMap::new();
    let mut classification = None;
    match task.kind {
        TaskKind::Generation | TaskKind::Classification => {
            let outputs =
                generate_all(task, backend, &backend_id, &prompts, &params, opts, &mut usage, &mut cached).await?;
            if task.kind == TaskKind::Generation {
                let ms = task.metric_list()?;
                for (i, out) in outputs.iter().enumerate() {
                    let scores: BTreeMap<String, f64> =
                        ms.iter().map(|m| (m.name().to_string(), m.score(out, &targets[i]))).collect();
                    samples.push(SampleResult {
                        index: i,
                        prompt: prompts[i].clone(),
                        output: out.clone(),
                        target: targets[i].clone(),
                        scores,
                    });
                }
                for m in &ms {
                    let mean = samples.iter().map(|s| s.scores[m.name()]).sum::<f64>() / samples.len().max(1) as f64;
                    metrics.insert(m.name().to_string(), mean);
                }
                let refs: Vec<&str> = targets.iter().map(String::as_str).collect();
                let hyps: Vec<&str> = outputs.iter().map(String::as_str).collect();
                if ms.contains(&GenMetric::Bleu) {
                    metrics.insert("corpus_bleu".into(), crate::metrics::corpus_bleu(&hyps, &refs));
                }
                if ms.contains(&GenMetric::Chrf) {
                    metrics.insert("corpus_chrf".into(), crate::metrics::corpus_chrf(&hyps, &refs));
                }
            } else {
                if task.labels.is_empty() {
                    return Err(HarpiaError::invalid(format!("classification task {:?} needs `labels`", task.name)));
                }
                let pred: Vec<String> =
                    outputs.iter().map(|o| match_label(o, &task.labels).unwrap_or("<unparsed>").to_string()).collect();
                for (i, out) in outputs.iter().enumerate() {
                    let ok = f64::from(u8::from(pred[i] == targets[i]));
                    samples.push(SampleResult {
                        index: i,
                        prompt: prompts[i].clone(),
                        output: out.clone(),
                        target: targets[i].clone(),
                        scores: BTreeMap::from([("correct".into(), ok)]),
                    });
                }
                let labels: Vec<&str> = task.labels.iter().map(String::as_str).collect();
                let r = ClassificationReport::new(&targets, &pred, &labels);
                metrics.insert("accuracy".into(), r.accuracy);
                metrics.insert("macro_f1".into(), r.macro_f1);
                metrics.insert("unparsed".into(), pred.iter().filter(|p| *p == "<unparsed>").count() as f64);
                classification = Some(r);
            }
        }
        TaskKind::MultipleChoice => {
            let field = task
                .choices_field
                .as_deref()
                .ok_or_else(|| HarpiaError::invalid("multiple_choice needs `choices_field`"))?;
            let (mut acc, mut acc_norm) = (0.0, 0.0);
            for (i, row) in rows.iter().enumerate() {
                let choices = choices(row, field)?;
                let gold = gold_index(&targets[i], &choices).ok_or_else(|| HarpiaError::Record {
                    path: Some(task.dataset.clone()),
                    line: i as u64 + 1,
                    message: format!("target {:?} is not a choice index or text", targets[i]),
                })?;
                let mut lls = Vec::with_capacity(choices.len());
                for c in &choices {
                    let ll = backend.loglikelihood(&prompts[i], &format!(" {c}")).ok_or_else(|| {
                        HarpiaError::invalid("multiple_choice needs a backend with log-likelihoods (a local Model)")
                    })??;
                    lls.push(ll);
                }
                let best = argmax(&lls);
                let normed: Vec<f64> = lls.iter().zip(&choices).map(|(ll, c)| ll / (c.len() + 1) as f64).collect();
                let best_norm = argmax(&normed);
                let (a, an) = (f64::from(u8::from(best == gold)), f64::from(u8::from(best_norm == gold)));
                acc += a;
                acc_norm += an;
                samples.push(SampleResult {
                    index: i,
                    prompt: prompts[i].clone(),
                    output: choices[best].clone(),
                    target: choices[gold].clone(),
                    scores: BTreeMap::from([("acc".into(), a), ("acc_norm".into(), an)]),
                });
            }
            let n = rows.len().max(1) as f64;
            metrics.insert("acc".into(), acc / n);
            metrics.insert("acc_norm".into(), acc_norm / n);
        }
        TaskKind::Perplexity => {
            let (mut ll_sum, mut bytes, mut words) = (0.0, 0usize, 0usize);
            for (i, text) in targets.iter().enumerate() {
                let ll = backend.loglikelihood(&task.perplexity_context, text).ok_or_else(|| {
                    HarpiaError::invalid("perplexity needs a backend with log-likelihoods (a local Model)")
                })??;
                ll_sum += ll;
                bytes += text.len();
                words += text.split_whitespace().count();
                samples.push(SampleResult {
                    index: i,
                    prompt: String::new(),
                    output: String::new(),
                    target: text.clone(),
                    scores: BTreeMap::from([("loglikelihood".into(), ll)]),
                });
            }
            metrics.insert("loglikelihood".into(), ll_sum);
            metrics.insert(
                "bits_per_byte".into(),
                if bytes == 0 { 0.0 } else { -ll_sum / std::f64::consts::LN_2 / bytes as f64 },
            );
            metrics.insert("word_perplexity".into(), if words == 0 { 0.0 } else { (-ll_sum / words as f64).exp() });
        }
    }
    Ok(TaskResult {
        task: task.name.clone(),
        kind: task.kind,
        backend: backend_id,
        n: rows.len(),
        metrics,
        classification,
        usage,
        cached,
        seconds: start.elapsed().as_secs_f64(),
        samples,
    })
}

#[allow(clippy::too_many_arguments)]
async fn generate_all<B: Backend>(
    task: &Task,
    backend: &B,
    backend_id: &str,
    prompts: &[String],
    params: &GenerationParams,
    opts: &mut RunOptions,
    usage: &mut TokenUsage,
    cached: &mut usize,
) -> Result<Vec<String>> {
    use futures_util::StreamExt;
    let keys: Vec<String> = prompts.iter().map(|p| cache_key(backend_id, task, p)).collect();
    let mut outputs: Vec<Option<String>> =
        keys.iter().map(|k| opts.cache.as_ref().and_then(|c| c.get(k).cloned())).collect();
    *cached += outputs.iter().filter(|o| o.is_some()).count();
    let todo: Vec<usize> = (0..prompts.len()).filter(|&i| outputs[i].is_none()).collect();
    let system = task.system.as_deref();
    let mut stream = futures_util::stream::iter(
        todo.into_iter().map(|i| async move { backend.generate(system, &prompts[i], params).await.map(|r| (i, r)) }),
    )
    .buffered(opts.concurrency.max(1));
    while let Some(res) = stream.next().await {
        let (i, (text, u)) = res?;
        *usage += u;
        if let Some(c) = opts.cache.as_mut() {
            c.insert(keys[i].clone(), text.clone())?;
        }
        outputs[i] = Some(text);
    }
    Ok(outputs.into_iter().map(|o| o.expect("every prompt answered")).collect())
}

fn choices(row: &Value, field: &str) -> Result<Vec<String>> {
    row.get(field)
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string())).collect())
        .ok_or_else(|| HarpiaError::invalid(format!("row has no array field {field:?}")))
}

fn gold_index(target: &str, choices: &[String]) -> Option<usize> {
    let t = target.trim();
    t.parse::<usize>().ok().filter(|&i| i < choices.len()).or_else(|| choices.iter().position(|c| c.trim() == t))
}

fn choice_text(task: &Task, row: &Value, env: &minijinja::Environment<'_>) -> Result<String> {
    let cs = choices(row, task.choices_field.as_deref().unwrap_or("choices"))?;
    let t = render(env, &task.target, row)?;
    Ok(gold_index(&t, &cs).map(|i| cs[i].clone()).unwrap_or(t))
}

fn argmax(v: &[f64]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0))).map_or(0, |(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{FnProvider, MockModel};
    use std::io::Write;

    fn block_on<F: Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
    }

    fn write_jsonl(dir: &Path, name: &str, rows: &[Value]) -> PathBuf {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        for r in rows {
            writeln!(f, "{r}").unwrap();
        }
        p
    }

    fn capitals(dir: &Path) {
        write_jsonl(
            dir,
            "capitals.jsonl",
            &[
                serde_json::json!({"country": "Brazil", "capital": "Brasília"}),
                serde_json::json!({"country": "France", "capital": "Paris"}),
                serde_json::json!({"country": "Japan", "capital": "Tokyo"}),
                serde_json::json!({"country": "Peru", "capital": "Lima"}),
            ],
        );
    }

    fn answer(prompt: &str) -> String {
        let last = prompt.rsplit("\n\n").next().unwrap();
        match () {
            _ if last.contains("Brazil") => "Brasília".into(),
            _ if last.contains("France") => "The answer is Paris.".into(),
            _ if last.contains("Japan") => "Tokyo".into(),
            _ => "Cusco".into(),
        }
    }

    #[test]
    fn generation_task_with_few_shot_and_cache() {
        let dir = tempfile::tempdir().unwrap();
        capitals(dir.path());
        std::fs::write(
            dir.path().join("t.toml"),
            r#"
name = "capitals"
kind = "generation"
dataset = "capitals.jsonl"
prompt = "Capital of {{ country }}?"
target = "{{ capital }}"
metrics = ["exact_match", "token_f1"]
few_shot = 2
seed = 7
"#,
        )
        .unwrap();
        let task = Task::load(dir.path().join("t.toml")).unwrap();
        let p = FnProvider::new(answer);
        let backend = ProviderBackend { provider: &p, model: "m".into() };
        let mut opts =
            RunOptions { cache: Some(JsonlCache::open(dir.path().join("cache.jsonl")).unwrap()), concurrency: 2 };
        let r = block_on(run_task(&task, &backend, &mut opts)).unwrap();
        assert_eq!(r.n, 4);
        assert_eq!(r.metrics["exact_match"], 0.5);
        assert!(r.metrics["token_f1"] > 0.5);
        // Few-shot: 2 examples, never the row itself, deterministic.
        let first = &r.samples[0].prompt;
        assert_eq!(first.matches("Capital of").count(), 3);
        assert!(first.ends_with("Capital of Brazil?"));
        assert!(!first[..first.len() - "Capital of Brazil?".len()].contains("Brazil"));
        let again = block_on(run_task(&task, &backend, &mut opts)).unwrap();
        assert_eq!(again.cached, 4);
        assert_eq!(again.samples[0].prompt, *first);
        assert_eq!(again.metrics, r.metrics);
    }

    #[test]
    fn classification_task() {
        let dir = tempfile::tempdir().unwrap();
        let ds = write_jsonl(
            dir.path(),
            "sent.jsonl",
            &[
                serde_json::json!({"text": "great", "label": "positive"}),
                serde_json::json!({"text": "awful", "label": "negative"}),
                serde_json::json!({"text": "fine", "label": "positive"}),
            ],
        );
        let task = Task {
            name: "sent".into(),
            kind: TaskKind::Classification,
            dataset: ds,
            prompt: "Sentiment of: {{ text }}".into(),
            target: "{{ label }}".into(),
            labels: vec!["positive".into(), "negative".into()],
            ..toml::from_str(
                r#"name="x"
kind="classification"
dataset="x""#,
            )
            .unwrap()
        };
        let p = FnProvider::new(
            |q: &str| if q.contains("awful") { "Negative.".into() } else { "I think it is negative".into() },
        );
        let r =
            block_on(run_task(&task, &ProviderBackend { provider: &p, model: "m".into() }, &mut RunOptions::default()))
                .unwrap();
        assert!((r.metrics["accuracy"] - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(r.classification.unwrap().confusion, vec![vec![0, 2], vec![0, 1]]);
        assert_eq!(match_label("POSITIVE!", &task.labels), Some("positive"));
        assert_eq!(match_label("meh", &task.labels), None);
    }

    #[test]
    fn multiple_choice_and_perplexity() {
        let dir = tempfile::tempdir().unwrap();
        let ds = write_jsonl(
            dir.path(),
            "mc.jsonl",
            &[
                serde_json::json!({"q": "Largest eagle?", "choices": ["harpy", "sparrow"], "answer": 0}),
                serde_json::json!({"q": "Bird that hoots?", "choices": ["eagle", "owl"], "answer": "owl"}),
            ],
        );
        let mut model = MockModel::new(|_: &str| String::new());
        model.preferred = vec![("Largest".into(), "harpy".into()), ("hoots".into(), "eagle".into())];
        let base: Task = toml::from_str("name=\"mc\"\nkind=\"multiple_choice\"\ndataset=\"x\"").unwrap();
        let task = Task {
            dataset: ds.clone(),
            prompt: "Q: {{ q }}\nA:".into(),
            target: "{{ answer }}".into(),
            choices_field: Some("choices".into()),
            ..base.clone()
        };
        let r = block_on(run_task(&task, &ModelBackend(&model), &mut RunOptions::default())).unwrap();
        assert_eq!(r.metrics["acc"], 0.5);
        assert_eq!(r.samples[1].output, "eagle");
        // Providers have no log-likelihoods.
        let p = FnProvider::new(|_: &str| String::new());
        assert!(
            block_on(run_task(&task, &ProviderBackend { provider: &p, model: "m".into() }, &mut RunOptions::default()))
                .is_err()
        );

        let ppl = Task { kind: TaskKind::Perplexity, dataset: ds, target: "{{ q }}".into(), ..base };
        let r = block_on(run_task(&ppl, &ModelBackend(&model), &mut RunOptions::default())).unwrap();
        let bytes = ("Largest eagle?".len() + "Bird that hoots?".len()) as f64;
        assert!((r.metrics["bits_per_byte"] - (0.1 * bytes) / std::f64::consts::LN_2 / bytes).abs() < 1e-9);
    }
}
