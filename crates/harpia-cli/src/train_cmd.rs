//! `harpia train ...`: classifier fine-tuning, LoRA SFT and DPO.

use crate::util::{OutputFormat, emit, read_records};
use clap::{Args, Subcommand};
use harpia::core::traits::{ChatTemplate, Tokenizer};
use harpia::core::{HarpiaError, Result};
use harpia::data::format::{Format, to_chat_sample, to_preference};
use harpia::data::io::get_path;
use harpia::data::pack::TokenizedSample;
use harpia::data::template::{CHATML, JinjaChatTemplate, LLAMA3, PLAIN, loss_mask};
use harpia::data::tokenize::HfTokenizer;
use harpia::train::{
    ClassifierConfig, Example, LoraConfig, LoraTrainConfig, PreferenceIds, train_classifier, train_lora_dpo,
    train_lora_sft,
};
use serde_json::{Value, json};
use std::path::Path;
use std::process::ExitCode;

#[derive(Subcommand)]
pub enum TrainCommand {
    /// Fine-tune a BERT/RoBERTa/XLM-R sequence classifier or NLI cross-encoder.
    Classifier(ClassifierArgs),
    /// LoRA supervised fine-tuning of a Qwen2/Llama decoder on chat data.
    Sft(SftArgs),
    /// LoRA DPO on preference data (experimental).
    Dpo(DpoArgs),
}

pub fn run(cmd: TrainCommand, format: OutputFormat) -> Result<ExitCode> {
    match cmd {
        TrainCommand::Classifier(a) => classifier(a, format),
        TrainCommand::Sft(a) => sft(a, format),
        TrainCommand::Dpo(a) => dpo(a, format),
    }
}

#[derive(Args)]
pub struct ClassifierArgs {
    /// Base model directory.
    #[arg(long)]
    pub base: String,
    /// Training JSONL.
    #[arg(long)]
    pub train: String,
    /// Validation JSONL (selects the best epoch by macro F1).
    #[arg(long)]
    pub val: String,
    /// Output directory.
    #[arg(long)]
    pub out: String,
    /// Field with the text or premise.
    #[arg(long, default_value = "premise")]
    pub premise: String,
    /// Field with the hypothesis (omit for single-text classification).
    #[arg(long)]
    pub hypothesis: Option<String>,
    /// Field with the gold label.
    #[arg(long, default_value = "label")]
    pub label: String,
    /// Output labels in order (default: sorted labels found in the training data).
    #[arg(long, value_delimiter = ',')]
    pub labels: Vec<String>,
    /// Reuse the base head: `new=base` pairs, e.g. `supports=entailment,neutral=neutral`.
    #[arg(long, value_delimiter = ',')]
    pub label_map: Vec<String>,
    /// Epochs.
    #[arg(long, default_value_t = 3)]
    pub epochs: usize,
    /// Batch size.
    #[arg(long, default_value_t = 16)]
    pub batch_size: usize,
    /// Peak learning rate.
    #[arg(long, default_value_t = 2e-5)]
    pub lr: f64,
    /// Maximum tokens per input.
    #[arg(long, default_value_t = 192)]
    pub max_len: usize,
    /// Seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Do not weight the loss by inverse class frequency.
    #[arg(long)]
    pub no_class_weights: bool,
}

fn examples(path: &str, a: &ClassifierArgs) -> Result<Vec<Example>> {
    let s = |r: &Value, f: &str| -> Result<String> {
        get_path(r, f)
            .and_then(Value::as_str)
            .map(String::from)
            .ok_or_else(|| HarpiaError::invalid(format!("{path}: row without `{f}`")))
    };
    read_records(path)?
        .iter()
        .map(|r| {
            Ok(Example {
                premise: s(r, &a.premise)?,
                hypothesis: a.hypothesis.as_ref().map(|h| s(r, h)).transpose()?,
                label: s(r, &a.label)?,
            })
        })
        .collect()
}

fn classifier(a: ClassifierArgs, format: OutputFormat) -> Result<ExitCode> {
    let train = examples(&a.train, &a)?;
    let val = examples(&a.val, &a)?;
    let labels = if a.labels.is_empty() {
        let mut l: Vec<String> = train.iter().map(|e| e.label.clone()).collect();
        l.sort();
        l.dedup();
        l
    } else {
        a.labels.clone()
    };
    let label_map = a
        .label_map
        .iter()
        .map(|p| {
            p.split_once('=')
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .ok_or_else(|| HarpiaError::invalid(format!("bad --label-map entry `{p}`")))
        })
        .collect::<Result<_>>()?;
    let cfg = ClassifierConfig {
        labels,
        label_map,
        epochs: a.epochs,
        batch_size: a.batch_size,
        lr: a.lr,
        max_len: a.max_len,
        seed: a.seed,
        class_weights: !a.no_class_weights,
        ..Default::default()
    };
    let every = (train.len().div_ceil(a.batch_size.max(1)) / 10).max(1);
    let report = train_classifier(Path::new(&a.base), Path::new(&a.out), &cfg, &train, &val, |p| {
        if p.step % every == 0 || p.step == p.total_steps {
            eprintln!(
                "epoch {} step {}/{} loss {:.4} lr {:.2e} ({:.0}s)",
                p.epoch, p.step, p.total_steps, p.loss, p.lr, p.elapsed
            );
        }
    })?;
    emit(format, json!({"report": report}), || {
        let mut s = String::new();
        for h in &report.history {
            s.push_str(&format!(
                "epoch {}: train loss {:.4}, val macro-F1 {:.4}\n",
                h.epoch, h.train_loss, h.val_macro_f1
            ));
        }
        s.push_str(&format!(
            "best val macro-F1 {:.4}; saved to {} ({:.0}s)",
            report.best_val_macro_f1, a.out, report.seconds
        ));
        s
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
pub struct LoraArgs {
    /// Base model directory.
    #[arg(long)]
    pub base: String,
    /// Output directory (PEFT adapter; `merged/` with --merge).
    #[arg(long)]
    pub out: String,
    /// Chat template: `chatml`, `llama3`, `plain`, or a `tokenizer_config.json` (default: the base
    /// model's own).
    #[arg(long)]
    pub template: Option<String>,
    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    pub r: usize,
    /// LoRA alpha.
    #[arg(long, default_value_t = 32.0)]
    pub alpha: f64,
    /// Epochs.
    #[arg(long, default_value_t = 1)]
    pub epochs: usize,
    /// Peak learning rate.
    #[arg(long)]
    pub lr: Option<f64>,
    /// Samples per optimizer step.
    #[arg(long, default_value_t = 1)]
    pub grad_accum: usize,
    /// Truncate samples to this many tokens.
    #[arg(long, default_value_t = 2048)]
    pub max_len: usize,
    /// Seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Also write merged full weights.
    #[arg(long)]
    pub merge: bool,
}

impl LoraArgs {
    fn config(&self, default_lr: f64) -> LoraTrainConfig {
        LoraTrainConfig {
            lora: LoraConfig { r: self.r, alpha: self.alpha, ..Default::default() },
            epochs: self.epochs,
            lr: self.lr.unwrap_or(default_lr),
            grad_accum: self.grad_accum,
            max_len: self.max_len,
            seed: self.seed,
            save_merged: self.merge,
            ..Default::default()
        }
    }

    fn template(&self) -> Result<JinjaChatTemplate> {
        match &self.template {
            Some(t) => load_template(t),
            None => JinjaChatTemplate::from_tokenizer_config(Path::new(&self.base).join("tokenizer_config.json")),
        }
    }
}

#[derive(Args)]
pub struct SftArgs {
    /// Chat JSONL (messages, sharegpt, alpaca or prompt-completion).
    #[arg(long)]
    pub data: String,
    #[command(flatten)]
    pub lora: LoraArgs,
}

fn sft(a: SftArgs, format: OutputFormat) -> Result<ExitCode> {
    let tok = HfTokenizer::from_file(Path::new(&a.lora.base).join("tokenizer.json"))?;
    let tmpl = a.lora.template()?;
    let mut samples = Vec::new();
    for (i, r) in read_records(&a.data)?.iter().enumerate() {
        let f = Format::detect(r).ok_or_else(|| HarpiaError::Record {
            path: None,
            line: i as u64 + 1,
            message: "unknown row format".into(),
        })?;
        let rendered = tmpl.render_with_spans(&to_chat_sample(r, f)?.messages)?;
        let enc = tok.encode(&rendered.text, false)?;
        samples.push(TokenizedSample { loss_mask: loss_mask(&enc.offsets, &rendered.assistant_spans), ids: enc.ids });
    }
    let report =
        train_lora_sft(Path::new(&a.lora.base), Path::new(&a.lora.out), &a.lora.config(2e-4), &samples, |s, t, l| {
            eprintln!("step {s}/{t} loss {l:.4}");
        })?;
    emit(format, json!({"report": report}), || {
        format!(
            "{} steps, loss {:.4} -> {:.4}; adapter in {} ({} weights, {} trainable params, {:.0}s)",
            report.steps,
            report.losses.first().copied().unwrap_or(f64::NAN),
            report.losses.last().copied().unwrap_or(f64::NAN),
            a.lora.out,
            report.adapted_weights,
            report.trainable_params,
            report.seconds
        )
    });
    Ok(ExitCode::SUCCESS)
}

#[derive(Args)]
pub struct DpoArgs {
    /// Preference JSONL (`prompt`, `chosen`, `rejected`).
    #[arg(long)]
    pub data: String,
    /// DPO temperature.
    #[arg(long, default_value_t = 0.1)]
    pub beta: f64,
    #[command(flatten)]
    pub lora: LoraArgs,
}

fn dpo(a: DpoArgs, format: OutputFormat) -> Result<ExitCode> {
    let tok = HfTokenizer::from_file(Path::new(&a.lora.base).join("tokenizer.json"))?;
    let tmpl = a.lora.template()?;
    let mut pairs = Vec::new();
    for r in read_records(&a.data)? {
        let p = to_preference(&r)?;
        let prompt = tmpl.render(&p.prompt, true)?;
        let full = |answer: &str| -> Result<Vec<u32>> {
            let mut msgs = p.prompt.clone();
            msgs.push(harpia::core::Message::assistant(answer));
            let text = tmpl.render(&msgs, false)?;
            let rest = text.strip_prefix(prompt.as_str()).unwrap_or(&text);
            Ok(tok.encode(rest, false)?.ids)
        };
        pairs.push(PreferenceIds {
            prompt: tok.encode(&prompt, false)?.ids,
            chosen: full(&p.chosen)?,
            rejected: full(&p.rejected)?,
        });
    }
    let mut cfg = a.lora.config(5e-5);
    cfg.beta = a.beta;
    let report = train_lora_dpo(Path::new(&a.lora.base), Path::new(&a.lora.out), &cfg, &pairs, |s, t, l| {
        eprintln!("step {s}/{t} loss {l:.4}")
    })?;
    emit(format, json!({"report": report}), || {
        format!("{} steps; adapter in {} ({:.0}s)", report.steps, a.lora.out, report.seconds)
    });
    Ok(ExitCode::SUCCESS)
}
