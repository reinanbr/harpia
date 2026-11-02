//! Request/response hooks around any [`Provider`].

use harpia_core::traits::{ChatRequest, ChatResponse, ChatStream, Provider};
use harpia_core::{Result, TokenUsage};
use std::sync::{Arc, Mutex};

/// A hook run around provider calls.
pub trait Middleware: Send + Sync {
    /// Inspect or modify the request; an error aborts the call.
    fn before(&self, _request: &mut ChatRequest) -> Result<()> {
        Ok(())
    }
    /// Observe a completed (non-streaming) response.
    fn after(&self, _request: &ChatRequest, _response: &ChatResponse) {}
}

/// A provider wrapped with middleware layers (applied in order).
///
/// ```
/// use harpia_client::{UsageTracker, WithMiddleware};
/// use std::sync::Arc;
/// # fn wrap<P: harpia_core::traits::Provider>(p: P) {
/// let tracker = Arc::new(UsageTracker::default());
/// let provider = WithMiddleware::new(p).layer(tracker.clone());
/// # }
/// ```
pub struct WithMiddleware<P> {
    inner: P,
    layers: Vec<Arc<dyn Middleware>>,
}

impl<P> WithMiddleware<P> {
    /// Wrap `inner`.
    pub fn new(inner: P) -> Self {
        Self { inner, layers: Vec::new() }
    }

    /// Add a layer.
    pub fn layer(mut self, m: Arc<dyn Middleware>) -> Self {
        self.layers.push(m);
        self
    }

    /// The wrapped provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<P: Provider> Provider for WithMiddleware<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut req = request.clone();
        for l in &self.layers {
            l.before(&mut req)?;
        }
        let resp = self.inner.chat(&req).await?;
        for l in &self.layers {
            l.after(&req, &resp);
        }
        Ok(resp)
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let mut req = request.clone();
        for l in &self.layers {
            l.before(&mut req)?;
        }
        self.inner.chat_stream(&req).await
    }
}

/// Middleware that sums token usage and counts calls.
#[derive(Debug, Default)]
pub struct UsageTracker {
    state: Mutex<(u64, TokenUsage)>,
}

impl UsageTracker {
    /// `(calls, usage)` so far.
    pub fn totals(&self) -> (u64, TokenUsage) {
        *self.state.lock().expect("usage lock")
    }
}

impl Middleware for UsageTracker {
    fn after(&self, _request: &ChatRequest, response: &ChatResponse) {
        let mut s = self.state.lock().expect("usage lock");
        s.0 += 1;
        s.1 += response.usage;
    }
}

/// Middleware that sets a default system prompt when the request has none.
#[derive(Debug, Clone)]
pub struct DefaultSystemPrompt(pub String);

impl Middleware for DefaultSystemPrompt {
    fn before(&self, request: &mut ChatRequest) -> Result<()> {
        if !request.messages.iter().any(|m| m.role == harpia_core::Role::System) {
            request.messages.insert(0, harpia_core::Message::system(self.0.clone()));
        }
        Ok(())
    }
}
