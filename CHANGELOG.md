# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `harpia-http`: offline-by-default polite client (async + blocking): per-host intervals,
  token buckets for RPM/TPM, retries on transport errors/429/5xx with jittered backoff honoring
  `Retry-After` (seconds or HTTP date), per-host circuit breaker with `status()`, `redb` TTL
  cache keyed on method/URL/sorted params/body hash with excluded params, body size caps,
  capped content-addressed downloads, streaming responses, and `testing::ConnectionCounter`
  to prove that offline code opens no sockets.
- `harpia-client`: `OpenAiCompatible`, `Anthropic` (raw HTTP, API version `2026-10-21`) and
  `Ollama` providers with chat, SSE/NDJSON streaming, tool calls and embeddings; `chat_json` /
  `chat_typed` structured output with validation and corrective retries; middleware
  (`UsageTracker`, `DefaultSystemPrompt`); `BatchRunner` with bounded concurrency, RPM/TPM
  budgets and `JsonlCache` resume; `synthesize` + `SynthTemplate` for synthetic data with
  provenance.
- CLI: `harpia chat` and `harpia data synth` (both require `--online`).
- Umbrella features `http`, `client` and `tls`.
- `harpia-local`: candle-based `SequenceClassifier` (BERT/RoBERTa/XLM-R, pairs, label aliases,
  length-sorted batching, `PairClassifier` + `Reranker`), `EncoderEmbedder`, `TextGenerator`
  (Qwen2/Llama: KV cache, sampling, stop sequences, constrained decoding, log-probs,
  log-likelihood, chat templates, `Model` + `Provider`), experimental `MarianTranslator`,
  `ModelCache`, and `testing` helpers that write tiny random-weight models. Parity with
  transformers on errei-cita-v1: max |Δlogit| 1.0e-5.
- CLI: `classify`, `generate`, `translate`, `embed`, and `chat --provider local`.
- Umbrella features `local`, `cuda`, `metal` (the last two pull native code).

### Changed

- MSRV raised to 1.94: `redb` 4 needs 1.90 and `candle-core` 0.11 needs 1.94 (measured: it fails
  to build on 1.93).
- `harpia-data` and `harpia-http` are workspace dependencies without default features; enable
  `langid` / `tls` explicitly.
- HTTPS (`tls` feature) uses rustls + `ring`, which contains C/assembly: the only opt-in
  exception to the no-C-toolchain rule. Everything else, including `full`, builds with
  `CC=false CXX=false`.

## [0.1.0] - 2026-11-02

### Added

- `harpia-core`: `HarpiaError`, `NetworkPolicy` (offline by default), `Span`/`LineIndex`,
  chat/sample/document types and the core traits (`Provider`, `Model`, `Tokenizer`,
  `ChatTemplate`, `Embedder`, `Reranker`, `PairClassifier`, `Translator`, `Chunker`,
  `Retriever`, `DataFilter`, `Metric`, `Judge`, `Detector`, `Cache`).
- `harpia-text`: Unicode normalization, word/sentence/paragraph segmentation with byte spans
  and en/pt-BR abbreviation lists, Flesch (English and pt-BR), MATTR, burstiness, repetition
  and character-ratio signals, Aho-Corasick phrase matching, language identification,
  shingles/Jaccard/containment/longest shared span, fingerprint index, MinHash,
  rapidfuzz-style fuzzy ratios and BM25.
- `harpia-data`: streaming JSONL/CSV with gzip/zstd, `JsonlCache`, sharded datasets with
  manifests, quality filters with gopher-like/c4-like/minimal presets and TOML config, language
  filter, PII/secret redaction (incl. CPF/CNPJ), exact and MinHash-LSH dedup, decontamination,
  seeded/grouped/stratified splits, a curation pipeline with reports, dataset-format
  converters, Jinja chat templates with assistant spans and loss masks, Hugging Face tokenizer
  backend, sequence packing, RAG chunkers and dataset cards.
- `harpia-structured`: JSON extraction and repair, incremental JSON, choice/regex/JSON-Schema
  constraints with token masks, tool-call parsing, JSON Schema validation and derivation.
- `harpia-cli`: `segment`, `stats`, `similarity`, `chunk`, `count-tokens`, `repair-json`, and
  `data inspect|convert|filter|pii|dedup|decontaminate|split|template|pack`.
- Name reservations with documented scope for `harpia-http`, `harpia-client`, `harpia-local`,
  `harpia-train`, `harpia-eval` and `harpia-verify`.
