//! The `harpia` command line.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod data_cmd;
mod eval_cmd;
mod llm_cmd;
mod local_cmd;
mod rag_cmd;
mod serve;
mod text_cmd;
mod train_cmd;
mod util;
mod verify_cmd;

use clap::{Parser, Subcommand};
use std::process::ExitCode;

/// Sharp-eyed LLM tooling in pure Rust.
#[derive(Parser)]
#[command(name = "harpia", version, about, long_about = None, after_help = EXAMPLES)]
struct Cli {
    /// Output format.
    #[arg(long, global = true, value_enum, default_value_t = util::OutputFormat::Text)]
    format: util::OutputFormat,

    #[command(subcommand)]
    command: Command,
}

const EXAMPLES: &str = "Examples:
  harpia segment paper.txt --unit sentences --lang pt-br
  harpia stats paper.txt --lang en --format json
  harpia similarity a.txt b.txt --k 5
  harpia chunk doc.md --strategy markdown --size 1200
  harpia repair-json broken.json
  harpia chat --provider anthropic --model claude-opus-5-5 --online 'Summarize RAG in one line'
  harpia generate --model models/qwen2.5-0.5b-instruct --chat --schema answer.json 'Is the sky blue?'
  harpia classify --model models/nli pairs.jsonl --premise evidence --hypothesis claim -o scored.jsonl
  harpia data synth topics.jsonl --provider ollama --model llama3.2 --template qa.toml --online -o qa.jsonl
  harpia data filter raw.jsonl --preset gopher-like -o clean.jsonl --report report.json
  harpia data pack chat.jsonl --tokenizer tokenizer.json --template chatml --seq-len 2048 -o packed.jsonl
  harpia eval run --task capitals.toml --provider ollama --model llama3.2 --online --min capitals/exact_match=0.8 -o run.json
  harpia eval rag outputs.jsonl --k 5 --classifier models/nli --thresholds th.json --min-faithfulness 0.7
  harpia calibrate --val nli_val.jsonl --test nli_test.jsonl -o th.json
  harpia rag index docs.jsonl --embedder models/e5-small -o idx && harpia rag query idx 'harpy eagle diet' --embedder models/e5-small
  harpia verify answer answer.txt --classifier models/nli --sources trusted.toml --online
  harpia verify reference --bib refs.bib --online
  harpia serve --http --decoder models/qwen2.5-0.5b-instruct --embedder models/e5-small

Exit codes: 0 ok, 1 a check failed, 2 usage or environment error.";

#[derive(Subcommand)]
enum Command {
    /// Split text into words, sentences or paragraphs with byte offsets and line:column.
    Segment(text_cmd::SegmentArgs),
    /// Readability and quality signals of a text.
    Stats(text_cmd::StatsArgs),
    /// Shingle similarity between two texts.
    Similarity(text_cmd::SimilarityArgs),
    /// Chunk a document for RAG.
    Chunk(text_cmd::ChunkArgs),
    /// Count tokens with a `tokenizer.json`.
    CountTokens(text_cmd::CountTokensArgs),
    /// Extract and repair JSON from LLM output.
    RepairJson(text_cmd::RepairArgs),
    /// Chat with an LLM provider (OpenAI-compatible, Anthropic, Ollama need --online; `local` runs a model directory).
    Chat(llm_cmd::ChatArgs),
    /// Generate text with a local decoder model, optionally constrained (JSON Schema, regex, choices).
    Generate(local_cmd::GenerateArgs),
    /// Classify texts or (premise, hypothesis) pairs with a local encoder model.
    Classify(local_cmd::ClassifyArgs),
    /// Translate a JSONL field with a local MarianMT model (resumable cache).
    Translate(local_cmd::TranslateArgs),
    /// Add sentence embeddings to JSONL rows with a local encoder model.
    Embed(local_cmd::EmbedArgs),
    /// Rerank documents for a query with a local cross-encoder.
    Rerank(local_cmd::RerankArgs),
    /// Build and query RAG indexes (chunk, embed, hybrid retrieval, rerank, cited context).
    #[command(subcommand)]
    Rag(rag_cmd::RagCommand),
    /// Ground answers, claims, datasets, references and quotes against trusted sources.
    #[command(subcommand)]
    Verify(verify_cmd::VerifyCommand),
    /// Inspect and check trusted-source registries.
    #[command(subcommand)]
    Sources(verify_cmd::SourcesCommand),
    /// Download a model snapshot from the Hugging Face Hub (needs --online).
    Download(local_cmd::DownloadArgs),
    /// Fine-tune models (classifier, LoRA SFT, DPO).
    #[command(subcommand)]
    Train(train_cmd::TrainCommand),
    /// Dataset tools (inspect, convert, filter, pii, dedup, decontaminate, split, template, pack).
    #[command(subcommand)]
    Data(data_cmd::DataCommand),
    /// Evaluate models and outputs (task harness, LLM judge, RAG metrics, perplexity, leaderboards).
    #[command(subcommand)]
    Eval(eval_cmd::EvalCommand),
    /// Fit decision thresholds for classifier scores on a validation set; report frozen test metrics.
    Calibrate(eval_cmd::CalibrateArgs),
    /// Serve loaded models over JSON-RPC on stdio or an OpenAI-compatible HTTP API.
    Serve(serve::ServeArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Segment(a) => text_cmd::segment(a, cli.format),
        Command::Stats(a) => text_cmd::stats(a, cli.format),
        Command::Similarity(a) => text_cmd::similarity(a, cli.format),
        Command::Chunk(a) => text_cmd::chunk(a, cli.format),
        Command::CountTokens(a) => text_cmd::count_tokens(a, cli.format),
        Command::RepairJson(a) => text_cmd::repair_json(a, cli.format),
        Command::Chat(a) => llm_cmd::chat(a, cli.format),
        Command::Generate(a) => local_cmd::generate(a, cli.format),
        Command::Classify(a) => local_cmd::classify(a, cli.format),
        Command::Translate(a) => local_cmd::translate(a, cli.format),
        Command::Embed(a) => local_cmd::embed(a, cli.format),
        Command::Rerank(a) => local_cmd::rerank(a, cli.format),
        Command::Rag(c) => rag_cmd::run(c, cli.format),
        Command::Verify(c) => verify_cmd::run(c, cli.format),
        Command::Sources(c) => verify_cmd::sources(c, cli.format),
        Command::Download(a) => local_cmd::download(a, cli.format),
        Command::Data(c) => data_cmd::run(c, cli.format),
        Command::Train(c) => train_cmd::run(c, cli.format),
        Command::Eval(c) => eval_cmd::run(c, cli.format),
        Command::Calibrate(a) => eval_cmd::calibrate(a, cli.format),
        Command::Serve(a) => serve::run(a, cli.format),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
