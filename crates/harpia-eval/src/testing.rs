//! Mock providers and models for tests and examples (no network, deterministic).

use harpia_core::traits::{
    ChatChunk, ChatRequest, ChatResponse, ChatStream, FinishReason, Generation, GenerationParams, Model, Provider,
};
use harpia_core::{Message, Result, TokenUsage};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

fn usage(prompt: &str, reply: &str) -> TokenUsage {
    TokenUsage {
        input_tokens: prompt.split_whitespace().count() as u64,
        output_tokens: reply.split_whitespace().count() as u64,
        cached_input_tokens: 0,
    }
}

fn last_user(req: &ChatRequest) -> String {
    req.messages.iter().rev().find(|m| m.role == harpia_core::Role::User).map(|m| m.content.clone()).unwrap_or_default()
}

fn stream_of(text: String, usage: TokenUsage) -> ChatStream {
    let mut chunks: Vec<Result<ChatChunk>> =
        text.split_inclusive(' ').map(|w| Ok(ChatChunk { delta: w.to_string(), ..Default::default() })).collect();
    chunks.push(Ok(ChatChunk { usage: Some(usage), finish_reason: Some(FinishReason::Stop), ..Default::default() }));
    Box::pin(futures_util::stream::iter(chunks))
}

/// Replies with a fixed script, one entry per call (the last entry repeats when it runs out).
#[derive(Debug)]
pub struct ScriptedProvider {
    replies: Mutex<VecDeque<String>>,
    last: Mutex<String>,
    calls: AtomicUsize,
}

impl ScriptedProvider {
    /// New provider with the given replies.
    pub fn new<I, S>(replies: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            replies: Mutex::new(replies.into_iter().map(Into::into).collect()),
            last: Mutex::default(),
            calls: AtomicUsize::new(0),
        }
    }

    /// Number of chat calls made.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn next(&self) -> String {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut q = self.replies.lock().expect("script lock");
        let mut last = self.last.lock().expect("script lock");
        if let Some(r) = q.pop_front() {
            *last = r;
        }
        last.clone()
    }
}

impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        "scripted"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let reply = self.next();
        Ok(ChatResponse {
            model: request.model.clone(),
            usage: usage(&last_user(request), &reply),
            message: Message::assistant(reply),
            finish_reason: FinishReason::Stop,
        })
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let reply = self.next();
        let u = usage(&last_user(request), &reply);
        Ok(stream_of(reply, u))
    }
}

/// Answers with a function of the last user message.
pub struct FnProvider<F> {
    f: F,
}

impl<F> std::fmt::Debug for FnProvider<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FnProvider")
    }
}

impl<F: Fn(&str) -> String + Send + Sync> FnProvider<F> {
    /// New provider.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F: Fn(&str) -> String + Send + Sync> Provider for FnProvider<F> {
    fn name(&self) -> &str {
        "fn"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let prompt = last_user(request);
        let reply = (self.f)(&prompt);
        Ok(ChatResponse {
            model: request.model.clone(),
            usage: usage(&prompt, &reply),
            message: Message::assistant(reply),
            finish_reason: FinishReason::Stop,
        })
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let prompt = last_user(request);
        let reply = (self.f)(&prompt);
        let u = usage(&prompt, &reply);
        Ok(stream_of(reply, u))
    }
}

/// A [`Model`] whose log-likelihood is `-(edit distance to a preferred continuation) - 0.1 *
/// length`, and whose generation returns a function of the prompt. Good enough to test
/// multiple-choice and perplexity plumbing.
pub struct MockModel<F> {
    gen_fn: F,
    /// Continuations that should score best, per context substring.
    pub preferred: Vec<(String, String)>,
}

impl<F> std::fmt::Debug for MockModel<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockModel").field("preferred", &self.preferred).finish_non_exhaustive()
    }
}

impl<F: Fn(&str) -> String + Send + Sync> MockModel<F> {
    /// New mock model.
    pub fn new(gen_fn: F) -> Self {
        Self { gen_fn, preferred: Vec::new() }
    }
}

impl<F: Fn(&str) -> String + Send + Sync> Model for MockModel<F> {
    fn id(&self) -> &str {
        "mock"
    }

    fn generate(&self, prompt: &str, _params: &GenerationParams) -> Result<Generation> {
        let text = (self.gen_fn)(prompt);
        Ok(Generation {
            usage: usage(prompt, &text),
            text,
            token_ids: Vec::new(),
            logprobs: Vec::new(),
            finish_reason: FinishReason::Stop,
        })
    }

    fn loglikelihood(&self, context: &str, continuation: &str) -> Result<f64> {
        let target = self.preferred.iter().find(|(c, _)| context.contains(c.as_str())).map(|(_, t)| t.as_str());
        let base = -0.1 * continuation.len() as f64;
        Ok(match target {
            Some(t) => base - crate::metrics::edit_distance(continuation.trim(), t) as f64,
            None => base,
        })
    }
}
