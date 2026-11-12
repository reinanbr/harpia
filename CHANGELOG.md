# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added (milestone 7: serve)

- `harpia serve --stdio`: newline-delimited JSON-RPC 2.0 (batches, notifications, streaming
  deltas as `stream` notifications) exposing `info`, `generate`, `chat`, `embed`, `rerank`,
  `classify`, `translate`, `retrieve`, `verify.claim`, `verify.answer`, `verify.reference`,
  `data.filter`, `structured.repair` and `shutdown`. Models registered as `[name=]dir` load at
  startup and stay loaded; stdio clients may also name model directories.
- `harpia serve --http` (feature `server`, axum, pure Rust, on by default in `harpia-cli`):
  OpenAI-compatible `/v1/chat/completions` (SSE streaming, `stream_options.include_usage`,
  `response_format` JSON schema, tool-call parsing), `/v1/completions`, `/v1/embeddings`
  (float and base64), `/v1/rerank`, `/v1/models`, `/health`, and JSON-RPC at `/rpc`.
  Localhost by default, optional bearer API key read from an environment variable, only
  registered models.
- `docs/serve-protocol.md`; integration tests that drive both transports with tiny models.

### Fixed

- `TextGenerator` and `MarianTranslator` stop at the model's `max_position_embeddings` (finish
  reason `length`) instead of failing inside candle when `max_new_tokens` exceeds the position
  table. Inputs that do not fit at all return a clear error.

### Added (milestone 6: verify)

- `harpia-verify`: TOML `SourceRegistry` (built-in Crossref, OpenAlex, Semantic Scholar, arXiv,
  Europe PMC, DOAJ, Wikipedia en/pt, Wikidata, open-access full texts; extend / restrict /
  replace), `Fetcher` (offline by default, HTTPS only, per-source host allowlist checked on
  every redirect hop, size and content-type caps, `robots.txt` for HTML, `Disclosure` log,
  content store with `Live` / `CachedFirst` / `Frozen` modes), adapters behind features, the
  `SourceAdapter` trait for custom sources, `ClaimVerifier` (keyword queries, BM25 passages,
  NLI windows, calibrated decisions, tier-aware aggregation that reports disagreement),
  answer and dataset grounding, `ReferenceVerifier` (DOI / arXiv lookups, search, weighted
  fuzzy matching, retraction and correction flags), quote checks by page, a BibTeX reader,
  trusted documents for `RagPipeline`, optional LLM assistance (`llm`).
- Replay parity with errei's reference classification on its recorded HTTP fixtures.
- `harpia-http`: `RedirectPolicy` (follow, none, or checked per hop), `Response::content_type`
  and final `url`.
- `harpia-data`: feature `html` (dependency-free HTML to text, main-content extraction, link
  extraction, entity decoding).
- Umbrella features `verify`, `verify-*` per adapter, `verify-all`, `verify-llm`.

### Added (milestone 5: eval and RAG retrieval)

- `harpia-eval`: evaluation harness (TOML/JSON tasks: generation, multiple choice by
  log-likelihood, perplexity / bits per byte, classification; seeded few-shot; `Provider` or
  `Model` backends; concurrency; `JsonlCache` resume), `LlmJudge` (rubric scores, pairwise with
  position swap, corrective retries) with Cohen's kappa / Pearson agreement, RAG metrics
  (recall@k, precision@k, MRR, nDCG, context recall, faithfulness via `PairClassifier`, answer
  relevance), generation metrics (EM, token F1, BLEU, chrF, ROUGE-N/L, edit distance),
  classification reports, calibration (decision rules as data, grid search on validation,
  frozen test evaluation that refuses the tuning data, best-of-N windows with evidence),
  dataset contamination / label-noise reports, leaderboards, model cards, price tables and a
  cost tracker with `tracing` spans (latency, time to first token, tokens/s).
- `harpia-data::vector`: exact `FlatIndex` (parallel, batched) and deterministic `HnswIndex`,
  both with incremental add/replace/delete, filters and a `.hvec` file format with a 64-byte
  aligned vector block.
- `harpia-data::rag`: `RagPipeline` (chunk → embed → index → BM25/dense retrieval → RRF or
  weighted fusion → optional reranking → context within a token budget with citations to
  document spans, sources and licenses), metadata filters, save/load.
- `harpia-train`: `train.json` records a data hash and hardware; every run writes a model card
  (`README.md`) through `harpia-eval`.
- Umbrella feature `eval`.

### Added (pending-items pass)

- `harpia_data::spm`: SentencePiece `ModelProto` reader (pure Rust protobuf) that builds a Unigram
  tokenizer with the model's precompiled normalization map. `MarianTranslator` uses it to load
  `source.spm` + `vocab.json` straight from a Hub snapshot like `MarianTokenizer`; token ids and
  greedy translations match transformers on `opus-mt-tc-big-en-pt`. MarianMT is now stable.

- `harpia-http::HttpClient::download_to_file` (streaming, atomic, capped); `harpia-local` feature
  `hub` (`Hub::snapshot`) and `harpia download <repo> --online`.
- Parity with transformers on a real decoder (Qwen2.5-0.5B-Instruct): template, tokens, logits,
  greedy decoding. LoRA export cross-checked against the PEFT merge formula.
- `TextGenerator::default_params` from `generation_config.json`; the CLI uses it for unset flags.
- Training dropout (from `config.json`, seeded) and LoRA dropout; per-epoch resumable checkpoints
  (`resume`, `stop_after`) that reproduce uninterrupted runs bit for bit.
- `cargo deny` and `cargo hack` run clean locally (44 feature combinations).

### Fixed

- Repetition penalty now covers prompt tokens, as in transformers.
- Training histories round-trip exactly through JSON (`serde_json/float_roundtrip`).

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
- `harpia-train`: `train_classifier` (full fine-tuning with head reuse/remap, class-weighted
  loss, AdamW, warmup + decay, clipping, macro-F1 model selection, HF-format checkpoints),
  `train_lora_sft` and experimental `train_lora_dpo` with PEFT-format adapters, differentiable
  `nn` forward passes for XLM-R/BERT/Qwen2/Llama, and a fused AdamW.
- `harpia-local`: `CausalLm`, `TextGenerator::load_with_adapter` (PEFT LoRA merge),
  `SequenceClassifier::{from_parts, batch_logits, encode_input}`.
- CLI: `harpia train classifier|sft|dpo`.
- Umbrella feature `train`.

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
