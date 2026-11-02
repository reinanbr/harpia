//! LLM subcommands: `chat` and `data synth`.

use crate::util::{OutputFormat, emit, read_records, read_text, write_records};
use clap::{Args, ValueEnum};
use futures_util::StreamExt;
use harpia::client::{Anthropic, BatchRunner, Ollama, OpenAiCompatible, Secret, SynthTemplate, synthesize};
use harpia::core::traits::{ChatRequest, ChatResponse, ChatStream, Provider};
use harpia::core::{HarpiaError, Message, NetworkPolicy, Result};
use harpia::http::{HttpClient, HttpConfig};
use serde_json::json;
use std::io::Write;
use std::process::ExitCode;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ProviderKind {
    /// OpenAI (OPENAI_API_KEY) or any compatible server with --base-url.
    Openai,
    /// Anthropic Messages API (ANTHROPIC_API_KEY).
    Anthropic,
    /// Local Ollama server.
    Ollama,
}

/// Provider selected at runtime.
pub enum AnyProvider {
    OpenAi(OpenAiCompatible),
    Anthropic(Anthropic),
    Ollama(Ollama),
}

impl Provider for AnyProvider {
    fn name(&self) -> &str {
        match self {
            AnyProvider::OpenAi(p) => p.name(),
            AnyProvider::Anthropic(p) => p.name(),
            AnyProvider::Ollama(p) => p.name(),
        }
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        match self {
            AnyProvider::OpenAi(p) => p.chat(request).await,
            AnyProvider::Anthropic(p) => p.chat(request).await,
            AnyProvider::Ollama(p) => p.chat(request).await,
        }
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        match self {
            AnyProvider::OpenAi(p) => p.chat_stream(request).await,
            AnyProvider::Anthropic(p) => p.chat_stream(request).await,
            AnyProvider::Ollama(p) => p.chat_stream(request).await,
        }
    }
}

#[derive(Args)]
pub struct ProviderArgs {
    /// Provider.
    #[arg(long, value_enum)]
    pub provider: ProviderKind,
    /// Model name at the provider (e.g. claude-opus-5-5, llama3.2, gpt-...).
    #[arg(long)]
    pub model: String,
    /// Base URL (OpenAI-compatible servers, Ollama, gateways).
    #[arg(long)]
    pub base_url: Option<String>,
    /// Allow network access. Without it every request is refused before any connection.
    #[arg(long)]
    pub online: bool,
}

impl ProviderArgs {
    pub fn build(&self) -> Result<AnyProvider> {
        let policy = if self.online { NetworkPolicy::Online } else { NetworkPolicy::Offline };
        let http = Arc::new(HttpClient::new(HttpConfig::new(policy))?);
        Ok(match self.provider {
            ProviderKind::Openai => match &self.base_url {
                Some(url) => AnyProvider::OpenAi(OpenAiCompatible::new(http, url, Secret::from_env("OPENAI_API_KEY"))),
                None => AnyProvider::OpenAi(OpenAiCompatible::openai(http)?),
            },
            ProviderKind::Anthropic => {
                let p = Anthropic::from_env(http)?;
                AnyProvider::Anthropic(match &self.base_url {
                    Some(u) => p.with_base_url(u),
                    None => p,
                })
            }
            ProviderKind::Ollama => AnyProvider::Ollama(Ollama::new(
                http,
                self.base_url.clone().unwrap_or_else(|| "http://localhost:11434".into()),
            )),
        })
    }
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| HarpiaError::Other(format!("cannot start runtime: {e}")))
}

#[derive(Args)]
pub struct ChatArgs {
    #[command(flatten)]
    pub provider: ProviderArgs,
    /// Prompt text, or `-` to read it from stdin.
    pub prompt: String,
    /// System prompt.
    #[arg(long)]
    pub system: Option<String>,
    /// Maximum tokens to generate.
    #[arg(long, default_value_t = 1024)]
    pub max_tokens: u32,
    /// Stream tokens as they arrive.
    #[arg(long)]
    pub stream: bool,
}

pub fn chat(a: ChatArgs, format: OutputFormat) -> Result<ExitCode> {
    let provider = a.provider.build()?;
    let prompt = if a.prompt == "-" { read_text("-")? } else { a.prompt.clone() };
    let mut messages = Vec::new();
    if let Some(s) = &a.system {
        messages.push(Message::system(s.clone()));
    }
    messages.push(Message::user(prompt));
    let mut req = ChatRequest { model: a.provider.model.clone(), messages, ..Default::default() };
    req.params.max_new_tokens = a.max_tokens;
    runtime()?.block_on(async {
        if a.stream && format == OutputFormat::Text {
            let mut s = provider.chat_stream(&req).await?;
            let mut out = std::io::stdout();
            while let Some(chunk) = s.next().await {
                let c = chunk?;
                print!("{}", c.delta);
                let _ = out.flush();
            }
            println!();
            return Ok(ExitCode::SUCCESS);
        }
        let resp = provider.chat(&req).await?;
        emit(format, json!({"response": resp}), || resp.message.content.clone());
        Ok(ExitCode::SUCCESS)
    })
}

#[derive(Args)]
pub struct SynthArgs {
    /// Input JSONL whose fields feed the template (`-` for stdin).
    pub input: String,
    #[command(flatten)]
    pub provider: ProviderArgs,
    /// TOML recipe (`user`, optional `system`, `schema`, `output_field`, `params`); its `model`
    /// is overridden by --model.
    #[arg(long)]
    pub template: String,
    /// Resumable cache of responses (JSONL).
    #[arg(long, default_value = "synth-cache.jsonl")]
    pub cache: String,
    /// Concurrent requests.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// Requests per minute budget.
    #[arg(long)]
    pub rpm: Option<f64>,
    /// Output file (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
}

pub fn synth(a: SynthArgs, format: OutputFormat) -> Result<ExitCode> {
    let provider = a.provider.build()?;
    let text = std::fs::read_to_string(&a.template).map_err(|e| HarpiaError::io(&a.template, e))?;
    let mut template = SynthTemplate::from_toml(&text)?;
    template.model = a.provider.model.clone();
    let rows = read_records(&a.input)?;
    let mut runner = BatchRunner::new(a.concurrency);
    if let Some(r) = a.rpm {
        runner = runner.requests_per_minute(r);
    }
    let (out, report) =
        runtime()?.block_on(synthesize(&provider, &template, &rows, &runner, std::path::Path::new(&a.cache)))?;
    write_records(a.output.as_deref(), &out)?;
    let failed = report.failed.len();
    if a.output.is_some() || failed > 0 {
        let rep = serde_json::to_value(&report)?;
        emit(format, json!({"report": rep}), || {
            let mut s = format!(
                "{} generated, {} from cache, {failed} failed; tokens in {} out {}",
                report.completed, report.from_cache, report.usage.input_tokens, report.usage.output_tokens
            );
            for (k, e) in report.failed.iter().take(5) {
                s.push_str(&format!("\n  {k}: {e}"));
            }
            s
        });
    }
    Ok(if failed > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS })
}
