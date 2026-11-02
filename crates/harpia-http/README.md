# harpia-http

The polite HTTP layer of Harpia, used by `harpia-client` and by anything that calls public
APIs at dataset scale:

- **offline by default**: with `NetworkPolicy::Offline` no socket is ever opened;
- per-host minimum intervals and token buckets (requests and tokens per minute);
- retries on transport errors, 429 and 5xx with exponential backoff and jitter, honoring
  `Retry-After` (seconds or HTTP date);
- a per-host circuit breaker with a status report ("source unavailable, results incomplete");
- an optional TTL response cache in a pure-Rust embedded store (`redb`);
- response size caps and capped, content-addressed binary downloads;
- an async API and a blocking facade.

Feature `tls` (default) enables HTTPS through rustls + `ring`. `ring` contains C/assembly; build
with `--no-default-features` for a pure-Rust, `http://`-only client.
