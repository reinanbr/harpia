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
| `harpia-data` | streaming IO, curation (filters, PII, dedup, decontamination), splits, chat templates with loss masks, packing, chunking, dataset cards | **stable** |
| `harpia-structured` | JSON extraction and repair, streaming JSON, constrained-decoding constraints, tool-call parsing, JSON Schema | **stable** |
| `harpia-http` | polite HTTP: offline guard, per-host rate limits, token buckets, retries honoring `Retry-After`, circuit breaker, `redb` TTL cache, size caps, blocking facade | **stable** |
| `harpia-client` | OpenAI-compatible, Anthropic and Ollama providers with streaming and tool calls; structured output with validation and retries; middleware; RPM/TPM batch runner; synthetic data generation | **stable** |
| `harpia` | umbrella crate re-exporting the above | **stable** |
| `harpia-cli` | the `harpia` binary | **stable** (text, data and LLM commands) |
| `harpia-local` | candle inference: decoders, encoders, rerankers, MarianMT | planned (milestone 3) |
| `harpia-train` | LoRA SFT, DPO, encoder and embedding fine-tuning | planned (milestone 4) |
| `harpia-eval` | eval harness, judges, RAG metrics, calibration, model cards | planned (milestone 5) |
| `harpia-verify` | grounding against trusted sources | planned (milestone 6) |

"Stable" means implemented, documented and tested in this release; the API may still change
before 1.0. Planned crates reserve their names and document their scope but expose no API.

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
harpia data synth topics.jsonl --provider anthropic --model claude-opus-5-5 --template qa.toml --online -o qa.jsonl
```

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
| `full` | no | all of the above | no |
| `tls` | no | HTTPS for `http`/`client` via rustls + `ring` | **yes** (`ring`: C/assembly) |

The `harpia-cli` binary enables `tls` by default; `cargo install harpia-cli --no-default-features`
builds a pure-Rust binary for local `http://` servers.

GPU backends (`cuda`, `metal`) and accelerated BLAS (`mkl`, `accelerate`) will arrive with
`harpia-local` as opt-in features that pull native code; they will be labeled as such here.

## Replacing a Python stack

| Python | Harpia | status |
|---|---|---|
| `datasets` / `datatrove` (IO, filters, dedup, decontamination) | `harpia-data` | available |
| `transformers` `apply_chat_template` + assistant masks | `harpia-data::template` | available |
| `trl` packing | `harpia-data::pack` | available |
| `json_repair`, `partial-json-parser` | `harpia-structured` | available |
| `outlines` / `guidance` (constrained decoding) | `harpia-structured::constraint` (+ `harpia-local`) | constraints available; decoding loop planned |
| `rapidfuzz` | `harpia-text::fuzzy` | available |
| `langchain` text splitters | `harpia-data::chunk` | available |
| `transformers` inference, `sentence-transformers` | `harpia-local` | planned |
| `peft` / `trl` training | `harpia-train` | planned |
| `lm-eval-harness` / `ragas` | `harpia-eval` | planned |
| `httpx` + sqlite cache, `tenacity` retries | `harpia-http` | available |
| `openai`, `anthropic`, `ollama` clients (chat, streaming, tools, embeddings) | `harpia-client` | available |
| `instructor` (structured output with retries) | `harpia-client::chat_json`, `chat_typed` | available |
| `distilabel`-style synthetic data | `harpia-client::synthesize`, `harpia data synth` | available |

## Grounding against trusted sources

`harpia-verify` (planned, milestone 6) will check LLM answers, RAG outputs and synthetic
training rows against **sources you trust**: an allowlist registry in TOML (HTTPS only,
redirects must stay on allowed hosts, `robots.txt` respected), adapters for Crossref,
OpenAlex, Semantic Scholar, arXiv, Europe PMC, Wikipedia/Wikidata and user-added domains, a
keywords-only query default and a `Disclosure` log of everything sent. Results are indicators
with evidence and provenance, never verdicts.

## Built on Harpia

The first downstream consumer is [errei](docs/errei-mapping.md), a scientific-article auditor
whose ML, NLP and HTTP needs (NLI cross-encoder inference and fine-tuning, MarianMT dataset
translation, threshold calibration, segmentation, shingles, fuzzy matching, polite cached HTTP,
reference checking) are mapped to Harpia APIs in `docs/errei-mapping.md`.

## Roadmap

1. **core + text + data + structured**: done in 0.1.
2. **http + client**: done: polite HTTP layer, providers, structured output, batch runner,
   synthetic data generation.
3. **local**: decoders with chat templates, constrained decoding and log-probs; encoders
   (classifiers, embeddings, rerankers); MarianMT.
4. **train**: LoRA SFT with PEFT-compatible export, encoder and embedding fine-tuning, then
   DPO/QLoRA.
5. **eval**: harness, judges, RAG metrics, calibration, model cards; RAG retrieval in
   `harpia-data`.
6. **verify**: trusted-source registry, adapters, claim and reference verification.
7. **serve**: `harpia serve --stdio` (JSON-RPC) and an OpenAI-compatible `serve --http`.

Also planned for `harpia-data`: Parquet/Arrow, HTML and PDF readers, Hugging Face Hub
downloads and model-based filters (perplexity and classifier filters need milestone 3).

## Benchmarks

Criterion benchmarks live in `crates/*/benches` (`cargo bench -p harpia-text`,
`cargo bench -p harpia-structured`). Comparisons against the Python reference libraries are
**TODO**: they will be published only when measured on the same machine, with hardware and
commit stated. Raw Harpia-only numbers and the comparison protocol are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## MSRV

Rust **1.90** (required by `redb`; verified by building the whole workspace, all targets and all
features, with the 1.90 toolchain). Raising it is a minor-version change.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
