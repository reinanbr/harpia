//! Rate limiting: per-host minimum intervals and token buckets.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Spaces requests to each host by a minimum interval. Slots are reserved under a lock and the
/// wait happens outside it, so concurrent tasks are queued fairly.
#[derive(Debug, Default)]
pub struct HostLimiter {
    intervals: HashMap<String, Duration>,
    default: Duration,
    next: Mutex<HashMap<String, Instant>>,
}

impl HostLimiter {
    /// New limiter.
    pub fn new(intervals: HashMap<String, Duration>, default: Duration) -> Self {
        Self { intervals, default, next: Mutex::new(HashMap::new()) }
    }

    /// Delay to wait before the next request to `host` (reserving the slot).
    pub fn reserve(&self, host: &str) -> Duration {
        let interval = self.intervals.get(host).copied().unwrap_or(self.default);
        if interval.is_zero() {
            return Duration::ZERO;
        }
        let now = Instant::now();
        let mut next = self.next.lock().expect("limiter lock");
        let at = next.get(host).copied().filter(|t| *t > now).unwrap_or(now);
        next.insert(host.to_string(), at + interval);
        at - now
    }

    /// Wait for the next slot of `host`.
    pub async fn wait(&self, host: &str) {
        let d = self.reserve(host);
        if !d.is_zero() {
            tokio::time::sleep(d).await;
        }
    }
}

/// A token bucket, e.g. for requests-per-minute or tokens-per-minute budgets of LLM APIs.
///
/// ```
/// use harpia_http::ratelimit::TokenBucket;
/// let b = TokenBucket::per_minute(60.0);
/// assert!(b.try_acquire(10.0));
/// assert!(!b.try_acquire(100.0));
/// ```
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    refill_per_sec: f64,
    state: Mutex<(f64, Instant)>,
}

impl TokenBucket {
    /// Bucket holding `capacity` tokens refilled at `refill_per_sec`, starting full.
    pub fn new(capacity: f64, refill_per_sec: f64) -> Self {
        Self { capacity, refill_per_sec, state: Mutex::new((capacity, Instant::now())) }
    }

    /// `n` tokens per minute, burst of one minute.
    pub fn per_minute(n: f64) -> Self {
        Self::new(n, n / 60.0)
    }

    fn refill(&self, s: &mut (f64, Instant)) {
        let now = Instant::now();
        s.0 = (s.0 + now.duration_since(s.1).as_secs_f64() * self.refill_per_sec).min(self.capacity);
        s.1 = now;
    }

    /// Take `n` tokens if available now.
    pub fn try_acquire(&self, n: f64) -> bool {
        let mut s = self.state.lock().expect("bucket lock");
        self.refill(&mut s);
        if s.0 >= n {
            s.0 -= n;
            true
        } else {
            false
        }
    }

    /// Wait until `n` tokens are available and take them (`n` above capacity is clamped).
    pub async fn acquire(&self, n: f64) {
        let n = n.min(self.capacity);
        loop {
            let wait = {
                let mut s = self.state.lock().expect("bucket lock");
                self.refill(&mut s);
                if s.0 >= n {
                    s.0 -= n;
                    return;
                }
                Duration::from_secs_f64((n - s.0) / self.refill_per_sec.max(1e-9))
            };
            tokio::time::sleep(wait.min(Duration::from_secs(60))).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_slots_are_spaced() {
        let l = HostLimiter::new([("a".to_string(), Duration::from_millis(100))].into(), Duration::ZERO);
        assert_eq!(l.reserve("a"), Duration::ZERO);
        let d = l.reserve("a");
        assert!(d > Duration::from_millis(90) && d <= Duration::from_millis(100), "{d:?}");
        assert_eq!(l.reserve("other"), Duration::ZERO);
    }

    #[tokio::test]
    async fn bucket_refills() {
        let b = TokenBucket::new(2.0, 100.0);
        assert!(b.try_acquire(2.0));
        assert!(!b.try_acquire(1.0));
        b.acquire(1.0).await; // ~10 ms
    }
}
