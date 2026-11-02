//! Cost from [`TokenUsage`] with a configurable price table, and `tracing` spans with latency,
//! time-to-first-token and throughput for every LLM call.
//!
//! Prices are user data: Harpia ships no price list, because prices change. A table looks like
//!
//! ```toml
//! currency = "USD"
//! [models."my-model"]
//! input_per_mtok = 0.50
//! output_per_mtok = 1.50
//! cached_input_per_mtok = 0.05
//! ```

use harpia_core::traits::{ChatRequest, ChatResponse, Provider};
use harpia_core::{Cost, HarpiaError, Result, TokenUsage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Price of one model, per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelPrice {
    /// Input tokens.
    pub input_per_mtok: f64,
    /// Output tokens.
    pub output_per_mtok: f64,
    /// Cached input tokens (defaults to the input price).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_per_mtok: Option<f64>,
}

/// Prices by model name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceTable {
    /// ISO currency code.
    #[serde(default = "usd")]
    pub currency: String,
    /// Model → price. Lookups fall back to the longest key that prefixes the model name, so
    /// `"my-model"` also prices `"my-model-2026-01-01"`.
    #[serde(default)]
    pub models: BTreeMap<String, ModelPrice>,
}

fn usd() -> String {
    "USD".into()
}

impl Default for PriceTable {
    fn default() -> Self {
        Self { currency: usd(), models: BTreeMap::new() }
    }
}

impl PriceTable {
    /// Parse TOML.
    pub fn from_toml(text: &str) -> Result<Self> {
        toml::from_str(text).map_err(|e| HarpiaError::invalid(format!("price table: {e}")))
    }

    /// Price for `model` (exact name, else longest prefix).
    pub fn price(&self, model: &str) -> Option<&ModelPrice> {
        self.models.get(model).or_else(|| {
            self.models
                .iter()
                .filter(|(k, _)| model.starts_with(k.as_str()))
                .max_by_key(|(k, _)| k.len())
                .map(|(_, v)| v)
        })
    }

    /// Cost of `usage` for `model`; `None` when the model is not priced (unknown is not free).
    pub fn cost(&self, model: &str, usage: &TokenUsage) -> Option<Cost> {
        let p = self.price(model)?;
        let cached = usage.cached_input_tokens.min(usage.input_tokens);
        let fresh = usage.input_tokens - cached;
        let amount = (fresh as f64 * p.input_per_mtok
            + cached as f64 * p.cached_input_per_mtok.unwrap_or(p.input_per_mtok)
            + usage.output_tokens as f64 * p.output_per_mtok)
            / 1e6;
        Some(Cost { amount, currency: self.currency.clone() })
    }
}

/// Measurements of one LLM call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallRecord {
    /// Provider name.
    pub provider: String,
    /// Model name.
    pub model: String,
    /// Wall-clock latency.
    #[serde(with = "secs")]
    pub latency: Duration,
    /// Time to first streamed token, for streaming calls.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_secs")]
    pub time_to_first_token: Option<Duration>,
    /// Token accounting.
    pub usage: TokenUsage,
    /// Output tokens per second over the whole call.
    pub tokens_per_second: f64,
    /// Cost, when the model is priced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
}

mod secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(d.as_secs_f64())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_secs_f64(f64::deserialize(d)?))
    }
}

mod opt_secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        match d {
            Some(d) => s.serialize_f64(d.as_secs_f64()),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.map(Duration::from_secs_f64))
    }
}

/// Accumulates call records (thread-safe) and prices them.
#[derive(Debug, Default)]
pub struct CostTracker {
    prices: PriceTable,
    calls: Mutex<Vec<CallRecord>>,
}

/// Totals over all tracked calls.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CostSummary {
    /// Number of calls.
    pub calls: usize,
    /// Summed usage.
    pub usage: TokenUsage,
    /// Summed cost over priced calls.
    pub cost: Cost,
    /// Calls whose model had no price.
    pub unpriced_calls: usize,
    /// Mean latency in seconds.
    pub mean_latency_s: f64,
    /// Mean time to first token in seconds, over streaming calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_ttft_s: Option<f64>,
}

impl CostTracker {
    /// New tracker with a price table.
    pub fn new(prices: PriceTable) -> Self {
        Self { prices, calls: Mutex::new(Vec::new()) }
    }

    /// Record a finished call (computing its cost and throughput) and emit a `tracing` event.
    pub fn record(
        &self,
        provider: &str,
        model: &str,
        usage: TokenUsage,
        latency: Duration,
        time_to_first_token: Option<Duration>,
    ) -> CallRecord {
        let secs = latency.as_secs_f64();
        let rec = CallRecord {
            provider: provider.into(),
            model: model.into(),
            latency,
            time_to_first_token,
            usage,
            tokens_per_second: if secs > 0.0 { usage.output_tokens as f64 / secs } else { 0.0 },
            cost: self.prices.cost(model, &usage),
        };
        tracing::info!(
            target: "harpia::llm",
            provider,
            model,
            latency_s = secs,
            ttft_s = time_to_first_token.map(|d| d.as_secs_f64()),
            input_tokens = usage.input_tokens,
            output_tokens = usage.output_tokens,
            tokens_per_second = rec.tokens_per_second,
            cost = rec.cost.as_ref().map(|c| c.amount),
            "llm call"
        );
        self.calls.lock().expect("cost tracker lock").push(rec.clone());
        rec
    }

    /// All records so far.
    pub fn records(&self) -> Vec<CallRecord> {
        self.calls.lock().expect("cost tracker lock").clone()
    }

    /// Totals.
    pub fn summary(&self) -> CostSummary {
        let calls = self.calls.lock().expect("cost tracker lock");
        let mut s = CostSummary { calls: calls.len(), ..Default::default() };
        s.cost.currency = self.prices.currency.clone();
        let mut ttft = Vec::new();
        for c in calls.iter() {
            s.usage += c.usage;
            match &c.cost {
                Some(cost) => s.cost.amount += cost.amount,
                None => s.unpriced_calls += 1,
            }
            s.mean_latency_s += c.latency.as_secs_f64();
            ttft.extend(c.time_to_first_token.map(|d| d.as_secs_f64()));
        }
        if !calls.is_empty() {
            s.mean_latency_s /= calls.len() as f64;
        }
        if !ttft.is_empty() {
            s.mean_ttft_s = Some(ttft.iter().sum::<f64>() / ttft.len() as f64);
        }
        s
    }

    /// Run a chat through `provider` inside a `tracing` span and record it.
    pub async fn chat<P: Provider>(&self, provider: &P, request: &ChatRequest) -> Result<(ChatResponse, CallRecord)> {
        use tracing::Instrument;
        let span = tracing::info_span!("llm.chat", provider = provider.name(), model = %request.model);
        async {
            let start = Instant::now();
            let resp = provider.chat(request).await?;
            let rec = self.record(provider.name(), &resp.model, resp.usage, start.elapsed(), None);
            Ok((resp, rec))
        }
        .instrument(span)
        .await
    }

    /// Stream a chat, collecting the text and measuring time to first token.
    pub async fn chat_streamed<P: Provider>(
        &self,
        provider: &P,
        request: &ChatRequest,
    ) -> Result<(String, CallRecord)> {
        use futures_util::StreamExt;
        use tracing::Instrument;
        let span = tracing::info_span!("llm.chat_stream", provider = provider.name(), model = %request.model);
        async {
            let start = Instant::now();
            let mut stream = provider.chat_stream(request).await?;
            let mut text = String::new();
            let mut ttft = None;
            let mut usage = TokenUsage::default();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                if ttft.is_none() && !chunk.delta.is_empty() {
                    ttft = Some(start.elapsed());
                }
                text.push_str(&chunk.delta);
                if let Some(u) = chunk.usage {
                    usage = u;
                }
            }
            let rec = self.record(provider.name(), &request.model, usage, start.elapsed(), ttft);
            Ok((text, rec))
        }
        .instrument(span)
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prices_and_prefix_lookup() {
        let t = PriceTable::from_toml(
            r#"
currency = "USD"
[models."m"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[models."m-large"]
input_per_mtok = 10.0
output_per_mtok = 20.0
cached_input_per_mtok = 1.0
"#,
        )
        .unwrap();
        let u = TokenUsage { input_tokens: 1_000_000, output_tokens: 500_000, cached_input_tokens: 0 };
        assert!((t.cost("m-2026", &u).unwrap().amount - 2.0).abs() < 1e-12);
        let cached = TokenUsage { cached_input_tokens: 500_000, ..u };
        assert!((t.cost("m-large-x", &cached).unwrap().amount - (5.0 + 0.5 + 10.0)).abs() < 1e-9);
        assert!(t.cost("other", &u).is_none());
    }

    #[test]
    fn tracker_summary() {
        let tr = CostTracker::new(PriceTable::default());
        let u = TokenUsage { input_tokens: 10, output_tokens: 20, cached_input_tokens: 0 };
        let r = tr.record("mock", "x", u, Duration::from_secs(2), Some(Duration::from_millis(500)));
        assert_eq!(r.tokens_per_second, 10.0);
        tr.record("mock", "x", u, Duration::from_secs(4), None);
        let s = tr.summary();
        assert_eq!((s.calls, s.unpriced_calls, s.usage.total()), (2, 2, 60));
        assert_eq!(s.mean_latency_s, 3.0);
        assert_eq!(s.mean_ttft_s, Some(0.5));
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<CallRecord>(&json).unwrap(), r);
    }
}
