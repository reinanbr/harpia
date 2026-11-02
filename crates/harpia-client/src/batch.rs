//! Dataset-scale calls: bounded concurrency, RPM/TPM budgets, resumable results.

use futures_util::StreamExt;
use harpia_core::traits::{ChatRequest, ChatResponse, Provider};
use harpia_core::{Result, TokenUsage};
use harpia_data::JsonlCache;
use harpia_http::ratelimit::TokenBucket;
use serde::{Deserialize, Serialize};

/// Summary of a [`BatchRunner::run`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BatchReport {
    /// Items answered in this run.
    pub completed: usize,
    /// Items skipped because the cache already had them.
    pub from_cache: usize,
    /// Items that failed, with the error message (they are retried on the next run).
    pub failed: Vec<(String, String)>,
    /// Usage of the calls made in this run.
    pub usage: TokenUsage,
}

/// Runs many chat requests with bounded concurrency and optional rate budgets, writing each
/// success to a [`JsonlCache`] immediately so an interrupted job resumes where it stopped.
///
/// ```no_run
/// # async fn run<P: harpia_core::traits::Provider>(provider: P, rows: Vec<(String, harpia_core::traits::ChatRequest)>) -> harpia_core::Result<()> {
/// use harpia_client::BatchRunner;
/// use harpia_data::JsonlCache;
/// let mut cache = JsonlCache::open("labels.jsonl")?;
/// let report = BatchRunner::new(8).requests_per_minute(500.0).tokens_per_minute(200_000.0)
///     .run(&provider, rows, &mut cache).await?;
/// println!("{} done, {} cached, {} failed", report.completed, report.from_cache, report.failed.len());
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct BatchRunner {
    max_in_flight: usize,
    rpm: Option<TokenBucket>,
    tpm: Option<TokenBucket>,
}

impl BatchRunner {
    /// At most `max_in_flight` concurrent requests.
    pub fn new(max_in_flight: usize) -> Self {
        Self { max_in_flight: max_in_flight.max(1), rpm: None, tpm: None }
    }

    /// Requests-per-minute budget.
    pub fn requests_per_minute(mut self, n: f64) -> Self {
        self.rpm = Some(TokenBucket::per_minute(n));
        self
    }

    /// Tokens-per-minute budget (estimated as prompt chars / 4 + `max_new_tokens`).
    pub fn tokens_per_minute(mut self, n: f64) -> Self {
        self.tpm = Some(TokenBucket::per_minute(n));
        self
    }

    fn estimate(req: &ChatRequest) -> f64 {
        let chars: usize = req.messages.iter().map(|m| m.content.len()).sum();
        (chars / 4) as f64 + f64::from(req.params.max_new_tokens)
    }

    /// Run `items` (`(key, request)`), skipping keys already in `cache`.
    pub async fn run<P: Provider>(
        &self,
        provider: &P,
        items: Vec<(String, ChatRequest)>,
        cache: &mut JsonlCache<String, ChatResponse>,
    ) -> Result<BatchReport> {
        let mut report = BatchReport::default();
        let mut todo = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (k, r) in items {
            if cache.contains(&k) {
                report.from_cache += 1;
            } else if seen.insert(k.clone()) {
                todo.push((k, r));
            }
        }
        let mut results = futures_util::stream::iter(todo)
            .map(|(key, req)| async move {
                if let Some(b) = &self.rpm {
                    b.acquire(1.0).await;
                }
                if let Some(b) = &self.tpm {
                    b.acquire(Self::estimate(&req)).await;
                }
                (key, provider.chat(&req).await)
            })
            .buffer_unordered(self.max_in_flight);
        while let Some((key, res)) = results.next().await {
            match res {
                Ok(resp) => {
                    report.usage += resp.usage;
                    report.completed += 1;
                    cache.insert(key, resp)?;
                }
                Err(e) => report.failed.push((key, e.to_string())),
            }
        }
        Ok(report)
    }
}
