//! Core abstractions shared by data pipelines, local models and API providers.
//!
//! Everything that produces or consumes model outputs goes through these traits, so a local
//! model, an API provider or a mock can be swapped without touching the pipeline.

use crate::Result;
use crate::span::Span;
use crate::types::{Label, Message, Scores, TokenUsage, ToolCall};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

// ---------------------------------------------------------------- tokenization

/// Output of a tokenizer.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Encoding {
    /// Token ids.
    pub ids: Vec<u32>,
    /// Byte span of each token in the input text (empty span for special tokens).
    pub offsets: Vec<Span>,
    /// For pair encodings: 0 for tokens of the first sequence, 1 for the second.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub type_ids: Vec<u32>,
    /// `true` for special tokens (BOS, EOS, SEP, ...).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub special: Vec<bool>,
}

impl Encoding {
    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// A text tokenizer.
pub trait Tokenizer: Send + Sync {
    /// Encode `text`, optionally adding the model's special tokens.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding>;

    /// Encode a pair of texts (e.g. premise and hypothesis for a cross-encoder).
    fn encode_pair(&self, first: &str, second: &str, add_special_tokens: bool) -> Result<Encoding>;

    /// Decode ids back to text.
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Vocabulary size including added tokens.
    fn vocab_size(&self) -> usize;

    /// Id of a token string, if present.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Count tokens of `text` without special tokens.
    fn count(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text, false)?.len())
    }
}

// ---------------------------------------------------------------- chat templates

/// A rendered conversation with the byte spans of assistant content (for loss masks).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RenderedChat {
    /// Full prompt text.
    pub text: String,
    /// Byte spans of the text that the model must learn to produce (assistant turns,
    /// including the end-of-turn marker emitted by the template).
    pub assistant_spans: Vec<Span>,
}

/// Turns a conversation into model input text.
pub trait ChatTemplate: Send + Sync {
    /// Render `messages`; with `add_generation_prompt`, append the assistant header so the model
    /// continues as the assistant.
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String>;

    /// Render and locate assistant spans (for assistant-only loss masks).
    fn render_with_spans(&self, messages: &[Message]) -> Result<RenderedChat>;
}

// ---------------------------------------------------------------- generation

/// Sampling / decoding parameters, shared by local models and API providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationParams {
    /// Maximum tokens to generate.
    pub max_new_tokens: u32,
    /// Softmax temperature; `0.0` means greedy.
    pub temperature: f32,
    /// Keep only the `k` most likely tokens (0 = disabled).
    pub top_k: u32,
    /// Nucleus sampling mass (1.0 = disabled).
    pub top_p: f32,
    /// Min-p filtering relative to the top token (0.0 = disabled).
    pub min_p: f32,
    /// Multiplicative repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Additive frequency penalty.
    pub frequency_penalty: f32,
    /// Additive presence penalty.
    pub presence_penalty: f32,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// RNG seed for sampling.
    pub seed: Option<u64>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop: Vec::new(),
            seed: None,
        }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    /// End-of-sequence token or natural end.
    Stop,
    /// `max_new_tokens` reached.
    Length,
    /// The model requested tool calls.
    ToolCalls,
    /// Blocked by a provider filter.
    ContentFilter,
}

/// Output of a local generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Generation {
    /// Generated text.
    pub text: String,
    /// Generated token ids.
    pub token_ids: Vec<u32>,
    /// Log-probability of each generated token, when requested.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logprobs: Vec<f32>,
    /// Why it stopped.
    pub finish_reason: FinishReason,
    /// Token accounting.
    pub usage: TokenUsage,
}

/// A local text-generation model.
pub trait Model: Send + Sync {
    /// Model identifier.
    fn id(&self) -> &str;
    /// Generate a continuation of `prompt`.
    fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<Generation>;
    /// Sum of token log-probabilities of `continuation` given `context` (for eval/perplexity).
    fn loglikelihood(&self, context: &str, continuation: &str) -> Result<f64>;
}

// ---------------------------------------------------------------- providers

/// A tool definition offered to a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// What the tool does.
    pub description: String,
    /// JSON Schema of the arguments.
    pub parameters: serde_json::Value,
}

/// A chat request, independent of the provider.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Model name at the provider (or local model id).
    pub model: String,
    /// Conversation so far.
    pub messages: Vec<Message>,
    /// Decoding parameters.
    #[serde(default)]
    pub params: GenerationParams,
    /// Ask for JSON output matching this schema, when supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
    /// Tools the model may call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpec>,
}

/// A complete chat response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Model that answered.
    pub model: String,
    /// The assistant message.
    pub message: Message,
    /// Why it stopped.
    pub finish_reason: FinishReason,
    /// Token accounting.
    pub usage: TokenUsage,
}

/// One streamed chunk.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatChunk {
    /// Text delta.
    pub delta: String,
    /// Completed tool calls carried by this chunk.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Final usage (usually only on the last chunk).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// Set on the last chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// A boxed stream of chunks.
pub type ChatStream = Pin<Box<dyn futures_core::Stream<Item = Result<ChatChunk>> + Send>>;

/// An LLM provider (remote API or local model behind the same interface).
pub trait Provider: Send + Sync {
    /// Provider name (e.g. `"anthropic"`, `"openai-compatible"`, `"local"`).
    fn name(&self) -> &str;

    /// Complete a chat.
    fn chat(&self, request: &ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send;

    /// Stream a chat completion.
    fn chat_stream(&self, request: &ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send;
}

// ---------------------------------------------------------------- encoders

/// Produces dense vector embeddings.
pub trait Embedder: Send + Sync {
    /// Embedding dimension.
    fn dim(&self) -> usize;
    /// Embed a batch of texts.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// Scores documents for relevance to a query (cross-encoder reranking).
pub trait Reranker: Send + Sync {
    /// One relevance score per document (higher is more relevant).
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;
}

/// Classifies `(premise, hypothesis)` pairs, e.g. an NLI cross-encoder.
///
/// The order matters: the *premise* is the evidence (a source passage) and the *hypothesis*
/// is the claim being checked.
pub trait PairClassifier: Send + Sync {
    /// Labels in model order.
    fn labels(&self) -> &[Label];
    /// Score a batch of `(premise, hypothesis)` pairs.
    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Scores>>;
}

/// Translates text.
pub trait Translator: Send + Sync {
    /// Translate a batch of texts.
    fn translate(&self, texts: &[&str]) -> Result<Vec<String>>;
}

// ---------------------------------------------------------------- data & retrieval

/// A chunk of a document.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Chunk {
    /// Byte span in the source text.
    pub span: Span,
    /// Heading path (for Markdown-aware chunking), outermost first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headings: Vec<String>,
}

/// Splits text into chunks.
pub trait Chunker: Send + Sync {
    /// Chunk `text`. Spans are byte ranges into `text`.
    fn chunk(&self, text: &str) -> Result<Vec<Chunk>>;
}

/// One retrieval hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    /// Document/chunk id.
    pub id: String,
    /// Score (higher is better; scale depends on the retriever).
    pub score: f32,
    /// Span in the source document, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
}

/// Retrieves documents for a query.
pub trait Retriever: Send + Sync {
    /// Top-`k` hits for `query`.
    fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Hit>>;
}

/// Decision of a data filter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterDecision {
    /// Keep the record.
    Keep,
    /// Drop the record, with a short machine-friendly reason (e.g. `"too_short"`).
    Drop(String),
}

impl FilterDecision {
    /// `true` for [`FilterDecision::Keep`].
    pub fn is_keep(&self) -> bool {
        matches!(self, Self::Keep)
    }
}

/// A text-level data filter.
pub trait DataFilter: Send + Sync {
    /// Short name used in curation reports.
    fn name(&self) -> &str;
    /// Decide whether to keep `text`.
    fn decide(&self, text: &str) -> FilterDecision;
}

/// A scalar metric comparing a prediction to a reference.
pub trait Metric: Send + Sync {
    /// Metric name.
    fn name(&self) -> &str;
    /// Score of one prediction (usually in `0.0..=1.0`).
    fn score(&self, prediction: &str, reference: &str) -> f64;
}

/// Input to an LLM judge.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct JudgeInput {
    /// Question or task.
    pub question: String,
    /// Answer to judge.
    pub answer: String,
    /// Reference answer, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// Grading rubric.
    pub rubric: String,
}

/// A judge's verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// Score on the rubric's scale.
    pub score: f64,
    /// Short rationale.
    pub rationale: String,
}

/// Grades answers (LLM-as-judge or a programmatic grader).
pub trait Judge: Send + Sync {
    /// Grade one answer.
    fn judge(&self, input: &JudgeInput) -> impl Future<Output = Result<Verdict>> + Send;
}

/// Scores text for some property (e.g. machine-generated text likelihood), in `0.0..=1.0`.
pub trait Detector: Send + Sync {
    /// Detector name.
    fn name(&self) -> &str;
    /// Whether scoring sends text over the network (then it must only run when online).
    fn uses_network(&self) -> bool;
    /// Score `text`.
    fn score(&self, text: &str) -> Result<f32>;
}

/// A key/value cache with optional TTL.
pub trait Cache: Send + Sync {
    /// Fetch a fresh value.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    /// Store a value, expiring after `ttl` if given.
    fn put(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()>;
}
