# Harpia

**Sharp-eyed LLM tooling in pure Rust.**

Harpia is named after the harpy eagle (*Harpia harpyja*), the apex bird of prey of the
Brazilian rainforest: big, precise and with famously sharp eyes. The toolkit aims for the same
qualities in LLM work: see your data clearly, then strike precisely.

Harpia covers the full lifecycle of **data and large language models**:

1. **Data**: clean, filter, deduplicate, decontaminate, format and pack datasets for training,
   evaluation and RAG.
2. **Use**: run models locally or through APIs, with chat templates, structured/constrained
   output, embeddings, reranking, RAG and grounding.
3. **Train**: fine-tune LLMs (LoRA SFT, preference optimization) and the models around them.
4. **Evaluate**: benchmarks, LLM-as-judge, RAG faithfulness, calibration, cost and latency.

## Why Harpia

- **Pure Rust, no Python.** No PyO3, no interpreter, no `pyproject.toml`. Builds need no C/C++
  toolchain either (CI builds with `CC=false CXX=false`). The one, opt-in exception is HTTPS:
  the `tls` feature uses rustls with the `ring` crypto provider, which contains C/assembly,
  because no pure-Rust TLS crypto provider is production-ready yet. Without `tls`, the HTTP
  stack is pure Rust and talks to `http://` servers (Ollama, vLLM, llama.cpp, gateways).
- **Modular.** Use one crate or the umbrella `harpia` crate; heavy dependencies compile only
  behind their feature.
- **Fast and parallel.** Zero-copy text processing with byte spans, `rayon` everywhere it pays.
- **Offline by default.** Nothing opens a socket unless you pass `NetworkPolicy::Online`.
- **Reproducible.** Every shuffle, split and sample takes an explicit seed and gives the same
  result on every platform.

## Crates

| crate | what it does | status |
|---|---|---|
| `harpia-core` | shared types, errors, spans, traits | **stable** |
| `harpia-text` | normalization, segmentation with offsets, quality signals, similarity, fuzzy matching, BM25 | **stable** |
| `harpia-data` | streaming IO, curation (filters, PII, dedup, decontamination), splits, chat templates with loss masks, packing, chunking, dataset cards, RAG retrieval (flat and HNSW vector indexes, BM25, hybrid fusion, reranking, filters, persistence, `RagPipeline` with cited context) | **stable** |
| `harpia-structured` | JSON extraction and repair, streaming JSON, constrained-decoding constraints, tool-call parsing, JSON Schema | **stable** |
| `harpia-http` | polite HTTP: offline guard, per-host rate limits, token buckets, retries honoring `Retry-After`, circuit breaker, `redb` TTL cache, size caps, blocking facade | **stable** |
| `harpia-client` | OpenAI-compatible, Anthropic and Ollama providers with streaming and tool calls; structured output with validation and retries; middleware; RPM/TPM batch runner; synthetic data generation | **stable** |
| `harpia` | umbrella crate re-exporting the above | **stable** |
| `harpia-cli` | the `harpia` binary | **stable** (text, data and LLM commands) |
| `harpia-local` | candle inference: sequence/pair classifiers and rerankers (BERT, RoBERTa, XLM-R; parity with transformers 1e-5), embeddings, Qwen2/Llama generation with constrained decoding and log-probs, MarianMT straight from Hub SentencePiece files (parity with transformers) | **stable** |
| `harpia-train` | classifier / cross-encoder fine-tuning (verified on a real XLM-R model; checkpoints load in transformers), LoRA SFT with PEFT export, DPO (experimental); every run writes `train.json` and a model card | **stable** |
| `harpia-eval` | eval harness (generation, multiple choice, perplexity, classification; few-shot, cached), LLM-as-judge with position swap and agreement, RAG metrics and faithfulness, generation and classification metrics, threshold calibration applied frozen, best-of-N windows, leaderboards, model cards, cost tracking | **stable** |
| `harpia-verify` | grounding and fact-checking against user-trusted sources: TOML source registry with hard host allowlist, offline-by-default fetcher (HTTPS only, redirect checks, `robots.txt`, size/type caps, disclosure log, frozen replay), adapters for Crossref, OpenAlex, Semantic Scholar, arXiv, Europe PMC, DOAJ, Wikipedia/Wikidata and allowlisted web pages, claim verification with NLI and tier-aware aggregation, answer and dataset grounding, reference and quote checks, BibTeX | **stable** |

"Stable" means implemented, documented and tested in this release; the API may still change
before 1.0.

## Quick start

### Library

```toml
[dependencies]
harpia = { version = "0.1", features = ["full"] }
```

```rust,ignore
use harpia::core::traits::ChatTemplate;
use harpia::core::types::Message;
use harpia::data::curate::{Pipeline, Stage};
use harpia::data::filter::{QualityFilter, QualityThresholds};
use harpia::data::template::{JinjaChatTemplate, CHATML};
use harpia::text::Lang;

// 1. curate raw text
let rows = vec![serde_json::json!({"text": "Contact ana@example.com about the harpy eagle survey results."})];
let pipe = Pipeline::new("text")
    .stage(Stage::RedactPii(Default::default()))
    .stage(Stage::Filter(Box::new(QualityFilter::new(QualityThresholds::minimal(), Lang::En.data()))))
    .stage(Stage::ExactDedup);
let (clean, report) = pipe.run(rows)?;

// 2. render chat data with the model's template and find the tokens to train on
let template = JinjaChatTemplate::new(CHATML, "", "<|im_end|>")?;
let rendered = template.render_with_spans(&[Message::user("Hi"), Message::assistant("Hello!")])?;
assert_eq!(rendered.assistant_spans[0].slice(&rendered.text), "Hello!<|im_end|>\n");

// 3. get JSON out of a chatty model reply
#[derive(serde::Deserialize)]
struct Verdict { label: String, score: f64 }
let v: Verdict = harpia::structured::parse_llm_json("Sure! {label: 'supports', score: 0.9,}")?;
Ok::<(), harpia::HarpiaError>(())
```

### CLI

```bash
cargo install harpia-cli

harpia segment paper.txt --unit sentences --lang pt-br --format json
harpia stats paper.txt --lang pt-br
harpia data inspect raw.jsonl
harpia data filter raw.jsonl --preset gopher-like --languages eng,por -o clean.jsonl --report report.json
harpia data pii clean.jsonl --redact -o safe.jsonl
harpia data dedup safe.jsonl --minhash --threshold 0.8 -o dedup.jsonl
harpia data decontaminate dedup.jsonl --against evals/mmlu.jsonl evals/gsm8k.jsonl -o train.jsonl
harpia data convert sharegpt.jsonl --to messages -o chat.jsonl
harpia data template chat.jsonl --template tokenizer_config.json
harpia data pack chat.jsonl --tokenizer tokenizer.json --template tokenizer_config.json --seq-len 4096 -o packed.jsonl
harpia chunk handbook.md --strategy markdown --size 1200
harpia repair-json model_output.txt

# LLM calls need --online (offline by default); keys come from OPENAI_API_KEY / ANTHROPIC_API_KEY
harpia chat --provider anthropic --model claude-opus-5-5 --online "Explain sequence packing in two sentences"
harpia chat --provider ollama --model llama3.2 --online --stream "Olá!"
harpia chat --provider openai --base-url http://localhost:8000/v1 --model qwen2.5 --online "hi"   # vLLM / llama.cpp
harpia classify --model models/nli pairs.jsonl --premise evidence --hypothesis claim --thresholds t.json -o scored.jsonl
harpia generate --model models/qwen2.5-0.5b-instruct --chat --schema answer.json "Is the sky blue?"
harpia chat --provider local --model models/qwen2.5-0.5b-instruct "Olá!"
harpia train classifier --base models/mnli --train pairs.jsonl --val val.jsonl --out models/my-nli --hypothesis claim
harpia train sft --base models/qwen2.5-0.5b-instruct --data chat.jsonl --out adapters/sft --merge
harpia data synth topics.jsonl --provider anthropic --model claude-opus-5-5 --template qa.toml --online -o qa.jsonl
```

### Serve: use Harpia from any language

```bash
# drop-in OpenAI-compatible endpoint (localhost only by default)
harpia serve --http --bind 127.0.0.1:8080 --decoder qwen=models/qwen2.5-0.5b-instruct --embedder e5=models/e5-small
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Olá!"}], "stream": true}'

# newline-delimited JSON-RPC 2.0 for subprocess use; models stay loaded
echo '{"jsonrpc":"2.0","id":1,"method":"structured.repair","params":{"text":"{a: 1,}"}}' | harpia serve --stdio
```

From Python, with no bindings (documentation only; the repository has no Python files):

```python
import json, subprocess
p = subprocess.Popen(["harpia", "serve", "--stdio", "--embedder", "models/e5-small"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "embed", "params": {"texts": ["olá"]}}) + "\n")
p.stdin.flush()
print(json.loads(p.stdout.readline())["result"]["dim"])
```

`serve --stdio` exposes `generate`, `chat`, `embed`, `rerank`, `classify`, `translate`,
`retrieve`, `verify.claim|answer|reference`, `data.filter` and `structured.repair`; `serve --http`
serves `/v1/chat/completions` (SSE streaming), `/v1/completions`, `/v1/embeddings`,
`/v1/rerank` and the same methods at `/rpc`. See [`docs/serve-protocol.md`](docs/serve-protocol.md).

`data synth` renders a TOML recipe (`user`, optional `system`, `schema`, `output_field`,
`params`) per row, caches every response in `--cache` so an interrupted run resumes, and adds
the output plus a `_provenance` object (provider, model, prompt hash, seed, usage) to each row.

Every subcommand accepts `--format json` (stable output with `"schema": 1`) and `-` for stdin.
Exit codes: `0` success, `1` a check failed (e.g. no JSON could be recovered), `2` usage or
environment error.

## Feature flags (umbrella crate)

| feature | default | enables | native code |
|---|---|---|---|
| `text` | yes | `harpia::text` | no |
| `data` | yes | `harpia::data` | no |
| `structured` | yes | `harpia::structured` | no |
| `langid` | no | language identification (`whatlang`) and the language filter | no |
| `tokenizers` | no | Hugging Face `tokenizers` (built with `fancy-regex`, without `onig`/`esaxx` C++) | no |
| `schema` | no | JSON Schema validation (`jsonschema`) and derivation (`schemars`) | no |
| `http` | no | `harpia::http` (polite HTTP, `http://` only without `tls`) | no |
| `client` | no | `harpia::client` (OpenAI-compatible, Anthropic, Ollama, batch runner, synthesis) | no |
| `local` | no | `harpia::local` (candle, CPU) | no |
| `train` | no | `harpia::train` (fine-tuning) | no |
| `eval` | no | `harpia::eval` (harness, judges, RAG metrics, calibration, reports, cost) | no |
| `verify` | no | `harpia::verify` core (registry, fetcher, claims, references, BibTeX) | no |
| `verify-crossref`, `verify-openalex`, `verify-semantic-scholar`, `verify-arxiv`, `verify-europepmc`, `verify-doaj`, `verify-wikipedia`, `verify-web` | no | one trusted-source adapter each (`verify-all` for every one) | no |
| `verify-llm` | no | LLM assistance for verification (claim decomposition, query rewriting, summaries) | no |
| `full` | no | all of the above | no |
| `cuda` | no | NVIDIA GPU inference | **yes** (CUDA toolkit) |
| `metal` | no | Apple GPU inference | **yes** (Metal) |
| `tls` | no | HTTPS for `http`/`client`/`verify` via rustls + `ring` (verification needs it for real sources) | **yes** (`ring`: C/assembly) |

The `harpia-cli` binary enables `tls` and `server` (the `serve --http` server, axum, pure Rust)
by default; `cargo install harpia-cli --no-default-features --features server` builds a
pure-Rust binary for local `http://` servers.

See [`docs/models.md`](docs/models.md) for supported architectures, verification status and limits.

## Replacing a Python stack

| Python | Harpia | status |
|---|---|---|
| `datasets` / `datatrove` (IO, filters, dedup, decontamination) | `harpia-data` | available |
| `transformers` `apply_chat_template` + assistant masks | `harpia-data::template` | available |
| `trl` packing | `harpia-data::pack` | available |
| `json_repair`, `partial-json-parser` | `harpia-structured` | available |
| `outlines` / `guidance` (constrained decoding) | `harpia-structured::constraint` + `harpia-local` | available |
| `rapidfuzz` | `harpia-text::fuzzy` | available |
| `langchain` text splitters | `harpia-data::chunk` | available |
| `transformers` sequence classification, cross-encoders, `sentence-transformers` embeddings | `harpia-local` | available (XLM-R verified against transformers) |
| `transformers` text generation + `outlines` constrained decoding | `harpia-local::TextGenerator` | available (Qwen2.5 verified against transformers; ~2.8× faster than `generate()` on CPU) |
| `transformers` MarianMT | `harpia-local::MarianTranslator` | stable |
| `transformers` Trainer for sequence classification | `harpia-train::train_classifier` | available (about 1.7× slower than PyTorch on CPU, see benchmarks) |
| `peft` LoRA SFT, `trl` DPO | `harpia-train::{train_lora_sft, train_lora_dpo}` | available (verified on Qwen2.5-0.5B; DPO experimental) |
| `huggingface_hub` snapshot downloads | `harpia-local::hub`, `harpia download` | available |
| `lm-eval-harness` / `ragas` | `harpia-eval` | available |
| `faiss` / `hnswlib` / vector stores, `rank_bm25`, `langchain` retrievers | `harpia-data::{vector, rag}` | available |
| `sacrebleu`, `rouge-score`, `scikit-learn` classification metrics | `harpia-eval::{metrics, classification}` | available |
| `httpx` + sqlite cache, `tenacity` retries | `harpia-http` | available |
| `openai`, `anthropic`, `ollama` clients (chat, streaming, tools, embeddings) | `harpia-client` | available |
| `instructor` (structured output with retries) | `harpia-client::chat_json`, `chat_typed` | available |
| `distilabel`-style synthetic data | `harpia-client::synthesize`, `harpia data synth` | available |

## Grounding against trusted sources

`harpia-verify` checks claims against **sources you trust** wherever a pipeline needs it:
grounding LLM answers claim by claim, web-grounded RAG (trusted sources become `RagPipeline`
documents with citations), filtering synthetic or scraped training rows whose claims are
contradicted, and checking that cited works exist and match their metadata.

**Registry and allowlist.** Sources live in a TOML registry: id, kind, allowed hosts (exact or
`*.suffix`), trust tier (1 = primary/peer-reviewed, 2 = curated reference, 3 = general),
politeness interval, contact/API-key environment variables and adapter. Every request and
every redirect hop must stay on an allowed host; only HTTPS; bodies are size- and type-capped;
HTML pages are fetched only when `robots.txt` allows; no cookies, JavaScript, logins or paywall
circumvention; APIs are used through their documented endpoints only.

**Built-in sources** (each behind a `verify-*` feature): Crossref, OpenAlex, Semantic Scholar,
arXiv, Europe PMC, DOAJ, Wikipedia (en, pt) and Wikidata, open-access full texts reported by
those APIs, and `GenericAllowlistedWeb` for your own domains. Implement `SourceAdapter` to add
private document stores without forking.

```toml
# trusted.toml: extends (or, with `restrict`, replaces) the built-in list
[[sources]]
id = "handbook"
name = "Engineering handbook"
kind = "docs"
domains = ["handbook.example.org"]
tier = 1
adapter = "web"
search_url = "https://handbook.example.org/search?q={query}"

[[sources]]
id = "semantic-scholar"          # same id: replaces the built-in entry
name = "Semantic Scholar (keyed)"
kind = "scholarly-api"
domains = ["api.semanticscholar.org"]
tier = 1
min_interval_ms = 300
api_key_env = "SEMANTIC_SCHOLAR_API_KEY"
adapter = "semantic-scholar"
```

**Privacy.** Nothing leaves the machine unless the network policy is `Online` (`--online` in the
CLI); offline, every check returns `Skipped { reason }` without opening a socket. Queries are
keywords only by default (`QueryMode::FullText` is opt-in), and every outgoing request is listed
in a `Disclosure` log returned with the results (contact e-mails redacted). A content store keeps
every fetched response with URL, time and SHA-256; `Mode::Frozen` answers only from it, offline,
for reproducible dataset builds and evaluations.

**Results are indicators, not verdicts.** Claims come back as `Supported`, `Contradicted`,
`NotEnoughInfo` or `Unavailable` with confidence and evidence (source, tier, URL, passage, byte
span, scores, retrieval time, content hash). Aggregation across sources is explicit: a tier-1
contradiction is never hidden by tier-3 support, and disagreement is reported, not averaged.
References come back as `Confirmed`, `Divergent { fields }`, `NotFound` or `Unavailable`, with
retraction and correction flags; quoted passages can be located in open full text by page.
An optional LLM (`verify-llm`) can decompose claims, rewrite queries and summarize evidence, but
never sets a status, and its use is recorded.

**Limitations.** "Not found" is not "false": many works and facts are simply not indexed.
`NotEnoughInfo` is the normal outcome when sources do not cover the point. NLI classifiers
degrade out of domain; calibrate thresholds on your own validation data (`harpia-eval`). Which
sources are trustworthy is your configuration, not Harpia's judgment.

## Built on Harpia

The first downstream consumer is [errei](docs/errei-mapping.md), a scientific-article auditor
whose ML, NLP and HTTP needs (NLI cross-encoder inference and fine-tuning, MarianMT dataset
translation, threshold calibration, segmentation, shingles, fuzzy matching, polite cached HTTP,
reference checking) are mapped to Harpia APIs in `docs/errei-mapping.md`.

## Roadmap

1. **core + text + data + structured**: done in 0.1.
2. **http + client**: done: polite HTTP layer, providers, structured output, batch runner,
   synthetic data generation.
3. **local**: done: encoders (classifiers, rerankers, embeddings), Qwen2/Llama decoders with chat
   templates, constrained decoding and log-probs, MarianMT. Still to come: GGUF
   quantized models, LoRA adapter loading (with train), Hub downloads.
4. **train**: done: classifier fine-tuning, LoRA SFT with PEFT export, DPO (experimental). Still
   to come: embedding fine-tuning, QLoRA, optimizer-state checkpoints.
5. **eval**: done: harness, judges, RAG metrics, calibration, model cards, cost tracking; RAG
   retrieval (flat/HNSW, BM25, hybrid, rerank, `RagPipeline`) in `harpia-data`.
6. **verify**: done: trusted-source registry and allowlist enforcement, adapters, claim, answer
   and dataset verification, references and quotes.
7. **serve**: done: `harpia serve --stdio` (JSON-RPC 2.0 with streaming) and an
   OpenAI-compatible `serve --http` (chat with SSE, completions, embeddings, rerank, `/rpc`).
   Still to come in this milestone: the remaining examples, criterion benches and CLI commands
   (`verify`, `sources`, `rag`, `rerank`).

Also planned for `harpia-data`: Parquet/Arrow, HTML and PDF readers, Hugging Face Hub
downloads and model-based filters (perplexity and classifier filters need milestone 3).

## Benchmarks

Criterion benchmarks live in `crates/*/benches` (`cargo bench -p harpia-text`,
`cargo bench -p harpia-structured`). Comparisons against the Python reference libraries are
**TODO**: they will be published only when measured on the same machine, with hardware and
commit stated. Raw Harpia-only numbers and the comparison protocol are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## MSRV

Rust **1.94** (required by `candle-core` 0.11, which fails on 1.93; verified by building the whole
workspace, all targets, with `harpia/full,harpia/tls` on the 1.94 toolchain). Raising it is a minor-version change.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
