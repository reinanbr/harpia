//! `harpia serve`: keep models loaded and expose them to other programs.

pub mod engine;
#[cfg(feature = "server")]
pub mod http;
pub mod stdio;

use crate::util::OutputFormat;
use clap::Args;
use engine::{Engine, EngineConfig};
use harpia::core::{HarpiaError, Result};
use std::process::ExitCode;

#[derive(Args)]
#[command(group(clap::ArgGroup::new("transport").required(true).args(["stdio", "http"])))]
#[command(after_help = "Examples:
  harpia serve --stdio --decoder models/qwen2.5-0.5b-instruct --embedder models/e5-small
  harpia serve --http --bind 127.0.0.1:8080 --decoder qwen=models/qwen2.5-0.5b-instruct
  HARPIA_API_KEY=secret harpia serve --http --api-key-env HARPIA_API_KEY --decoder models/qwen

Models are given as `[name=]dir` and may be repeated; the first of each kind is the default.
Protocol: docs/serve-protocol.md.")]
pub struct ServeArgs {
    /// Newline-delimited JSON-RPC 2.0 on stdin/stdout.
    #[arg(long)]
    pub stdio: bool,
    /// OpenAI-compatible HTTP server.
    #[arg(long)]
    pub http: bool,
    /// Address for `--http` (localhost by default).
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: String,
    /// Name of an environment variable holding the API key that `--http` clients must send
    /// as `Authorization: Bearer <key>`.
    #[arg(long)]
    pub api_key_env: Option<String>,
    /// Decoder model (`generate`, `chat`, `/v1/chat/completions`, `/v1/completions`).
    #[arg(long, value_name = "[NAME=]DIR")]
    pub decoder: Vec<String>,
    /// Embedding model (`embed`, `/v1/embeddings`, RAG queries).
    #[arg(long, value_name = "[NAME=]DIR")]
    pub embedder: Vec<String>,
    /// Sequence classifier (`classify`; NLI for `verify.*`).
    #[arg(long, value_name = "[NAME=]DIR")]
    pub classifier: Vec<String>,
    /// Cross-encoder reranker (`rerank`, `/v1/rerank`).
    #[arg(long, value_name = "[NAME=]DIR")]
    pub reranker: Vec<String>,
    /// MarianMT translator (`translate`).
    #[arg(long, value_name = "[NAME=]DIR")]
    pub translator: Vec<String>,
    /// RAG index saved by `RagPipeline::save` (`retrieve`); queried with the default embedder.
    #[arg(long, value_name = "[NAME=]DIR")]
    pub rag: Vec<String>,
    /// Rerank RAG hits with the default reranker.
    #[arg(long)]
    pub rag_rerank: bool,
    /// Trusted-source registry for `verify.*` (built-in registry when omitted).
    #[arg(long)]
    pub sources: Option<String>,
    /// Allow outbound network for `verify.*`.
    #[arg(long)]
    pub online: bool,
    /// Content store directory for `verify.*`.
    #[arg(long)]
    pub store: Option<String>,
    /// Content store mode: live, cached-first or frozen.
    #[arg(long, default_value = "live")]
    pub mode: String,
    /// Default language for filters and verification (en, pt-br).
    #[arg(long, default_value = "en")]
    pub lang: String,
}

pub fn run(a: ServeArgs, _format: OutputFormat) -> Result<ExitCode> {
    let mode = crate::verify_cmd::parse_mode(&a.mode)?;
    let api_key = match &a.api_key_env {
        Some(var) => Some(
            std::env::var(var)
                .ok()
                .filter(|k| !k.is_empty())
                .ok_or_else(|| HarpiaError::invalid(format!("environment variable `{var}` is unset or empty")))?,
        ),
        None => None,
    };
    let cfg = EngineConfig {
        decoders: a.decoder,
        embedders: a.embedder,
        classifiers: a.classifier,
        rerankers: a.reranker,
        translators: a.translator,
        rag: a.rag,
        rag_rerank: a.rag_rerank,
        sources: a.sources.map(Into::into),
        online: a.online,
        store: a.store.map(Into::into),
        mode,
        lang: a.lang,
    };
    let engine = Engine::new(cfg)?;
    if a.stdio {
        if api_key.is_some() {
            return Err(HarpiaError::invalid("--api-key-env applies to --http only"));
        }
        eprintln!("harpia: serving JSON-RPC 2.0 on stdio");
        let stdin = std::io::stdin();
        stdio::serve(&engine, stdin.lock(), std::io::stdout().lock())?;
        return Ok(ExitCode::SUCCESS);
    }
    serve_http(engine, &a.bind, api_key)
}

#[cfg(feature = "server")]
fn serve_http(engine: Engine, bind: &str, api_key: Option<String>) -> Result<ExitCode> {
    let addr: std::net::SocketAddr = bind.parse().map_err(|e| HarpiaError::invalid(format!("--bind `{bind}`: {e}")))?;
    http::serve(engine, addr, api_key)?;
    Ok(ExitCode::SUCCESS)
}

#[cfg(not(feature = "server"))]
fn serve_http(_engine: Engine, _bind: &str, _api_key: Option<String>) -> Result<ExitCode> {
    Err(HarpiaError::invalid("this build has no HTTP server; rebuild harpia-cli with the `server` feature"))
}
