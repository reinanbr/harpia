# Serve protocol

`harpia serve` loads models once and exposes them to programs written in any language, with no
bindings:

- `harpia serve --stdio`: newline-delimited JSON-RPC 2.0 on stdin/stdout. Use this when your
  program starts Harpia as a subprocess.
- `harpia serve --http`: an OpenAI-compatible HTTP API, so existing OpenAI clients work by
  changing the base URL. It also exposes the JSON-RPC methods at `POST /rpc`.

Both transports share one engine. Each model handles one request at a time.

## Models

You register models on the command line as `[name=]dir`. Each flag can be repeated, and the
first model of each kind is the default. Without a `name=` prefix, the directory's last path
component is the name.

| flag | kind | used by |
|---|---|---|
| `--decoder` | Qwen2/Llama decoder | `generate`, `chat`, `/v1/chat/completions`, `/v1/completions` |
| `--embedder` | BERT/XLM-R sentence encoder | `embed`, `/v1/embeddings`, RAG queries |
| `--classifier` | `*ForSequenceClassification` | `classify`, NLI for `verify.*` |
| `--reranker` | cross-encoder | `rerank`, `/v1/rerank`, `--rag-rerank` |
| `--translator` | MarianMT | `translate` |
| `--rag` | index saved with `RagPipeline::save` | `retrieve`, queried with the default embedder |

Registered models load at startup, so a bad path fails right away with exit code 2.

A request picks a model with `"model"` (or `"index"` for `retrieve`):

- If it is omitted, `""` or `"default"`, the default model is used.
- If it is a registered name, that model is used.
- Over stdio only, it can also be a model directory. The directory loads on first use and
  stays cached. The HTTP server never loads a path a client sends.

Other options:

| flag | default | meaning |
|---|---|---|
| `--online` | off | allow outbound network for `verify.*`; nothing else uses the network |
| `--sources trusted.toml` | built-in registry | trusted-source registry for `verify.*` |
| `--store DIR`, `--mode live\|cached-first\|frozen` | `live` | verification content store |
| `--lang en\|pt-br` | `en` | default language for `data.filter` and `verify.*` |
| `--bind ADDR` | `127.0.0.1:8080` | HTTP listen address (`:0` picks a free port) |
| `--api-key-env VAR` | none | HTTP clients must send `Authorization: Bearer $VAR` |

## stdio: JSON-RPC 2.0

Each line on stdin is one request or one batch (a JSON array). Each response is one line on
stdout, written in request order. Requests without an `id` are notifications: they run but get
no response. Logs go to stderr only, so stdout stays machine-readable.

```json
{"jsonrpc": "2.0", "id": 1, "method": "embed", "params": {"texts": ["the harpy eagle"]}}
{"jsonrpc": "2.0", "id": 1, "result": {"model": "e5-small", "dim": 384, "embeddings": [[0.01, ...]]}}
```

**Streaming.** With `"stream": true`, `generate` and `chat` first emit `stream` notifications,
then the normal response:

```json
{"jsonrpc": "2.0", "method": "stream", "params": {"id": 7, "delta": "The harpy"}}
{"jsonrpc": "2.0", "method": "stream", "params": {"id": 7, "delta": " eagle"}}
{"jsonrpc": "2.0", "id": 7, "result": {"message": {"role": "assistant", "content": "The harpy eagle"}, ...}}
```

**Stopping.** The server stops at EOF on stdin or after answering `shutdown`.

### Methods

Every `params` value is an object. Text lists accept either a single string or an array.

| method | params | result |
|---|---|---|
| `info` | none | `name`, `version`, `methods`, `models` (names per kind), `network` |
| `generate` | `prompt`, `chat?` (wrap the prompt as a user message with the chat template), sampling, constraint, `stream?`, `logprobs?` | `text`, `finish_reason`, `usage`, plus `token_ids` and `logprobs` when `logprobs` is set |
| `chat` | `messages`, sampling, constraint, `tools?`, `stream?` | `message` (with `tool_calls` when tools were offered), `finish_reason`, `usage` |
| `embed` | `texts` | `dim`, `embeddings` |
| `rerank` | `query`, `documents` (strings or `{"text"}`), `top_n?` | `results`: `[{index, score}]`, best first |
| `classify` | `texts`, or `pairs: [[premise, hypothesis], ...]`; `thresholds?` (label → minimum probability), `fallback?` (default `neutral`) | `labels`, `results`: `[{label, scores}]` |
| `translate` | `texts`, `prefix?` (e.g. `>>por<<`), `beams?` | `translations` |
| `retrieve` | `query`, `index?`, `k?`, `filter?` (`MetadataFilter`), `budget?`, `tokenizer?` (decoder name) | `hits`, plus `context` (`text` with `[n]` markers, `citations`) when `budget` is set |
| `verify.claim` | `claim`, `queries?`, `classifier?`, `lang?`, `sources?` (source ids) | `verification`, `disclosure` |
| `verify.answer` | `answer`, `classifier?`, `lang?`, `sources?` | `claims`, `counts`, `disclosure` |
| `verify.reference` | `reference` or `references` (`ReferenceQuery`: `key`, `title`, `authors`, `year`, `doi`, ...) | `results`, `disclosure` |
| `data.filter` | `texts`, `preset?` (`minimal`, `gopher-like`, `c4-like`) or `thresholds?` (a `QualityThresholds` object), `lang?` | `decisions`: `[{keep, reason?}]`, `kept`, `dropped` |
| `structured.repair` | `text` | `value` (the repaired JSON); error `-32004` when no JSON is found |
| `shutdown` | none | `null`, then the server exits (stdio only) |

**Sampling.** These parameters apply to `generate`, `chat` and the HTTP endpoints:
`max_tokens` (also `max_new_tokens` or `max_completion_tokens`), `temperature`, `top_p`,
`top_k`, `min_p`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `seed`, and
`stop` (a string or an array). Unset fields use the model's `generation_config.json`, as in
transformers. The same seed with the same input gives the same output on CPU.

**Constraints.** Give at most one: `schema` (JSON Schema), `regex`, `choices` (an array of
strings), or OpenAI's `response_format` (`{"type": "json_schema", "json_schema": {"schema":
...}}` or `{"type": "json_object"}`). A constrained output stops only once it is complete.
If it hits `max_tokens` first, `finish_reason` is `length` and the text is a valid prefix.

**Messages.** `role` is `system`, `developer` (treated as `system`), `user`, `assistant` or
`tool`. `content` is a string or OpenAI content parts; text parts are concatenated. Assistant
`tool_calls` and tool `tool_call_id` follow OpenAI's shape.

**Length limits.** Generation never runs past the model's position table
(`max_position_embeddings`): it stops with `finish_reason: "length"`. A prompt that does not
fit at all is an error.

### Errors

Errors are standard JSON-RPC error objects. Harpia errors carry `data.kind` (`invalid_argument`,
`model`, `io`, ...).

| code | meaning |
|---|---|
| -32700 | invalid JSON |
| -32600 | invalid request (missing `jsonrpc: "2.0"`, `method`, bad `id`, empty batch) |
| -32601 | unknown method |
| -32602 | bad or missing parameters |
| -32000 | other Harpia error |
| -32001 | network access refused (offline) |
| -32002 | a model failed to load or run |
| -32003 | model, index or feature not available on this server |
| -32004 | `structured.repair` found no JSON |

### Using stdio from another language

The repository contains no Python; this snippet is documentation only:

```python
import json, subprocess

p = subprocess.Popen(
    ["harpia", "serve", "--stdio", "--decoder", "models/qwen2.5-0.5b-instruct"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
)

def call(method, params, id=[0]):
    id[0] += 1
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": id[0], "method": method, "params": params}) + "\n")
    p.stdin.flush()
    while True:
        msg = json.loads(p.stdout.readline())
        if msg.get("id") == id[0]:
            return msg  # skip `stream` notifications

print(call("chat", {"messages": [{"role": "user", "content": "Olá!"}], "max_tokens": 64}))
```

## HTTP: OpenAI-compatible

```bash
harpia serve --http --bind 127.0.0.1:8080 --decoder qwen=models/qwen2.5-0.5b-instruct --embedder e5=models/e5-small
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Olá!"}]}'
```

| endpoint | notes |
|---|---|
| `GET /health` | `{"status": "ok"}`, no authentication |
| `GET /v1/models` | every registered model, with a Harpia-specific `kind` |
| `POST /v1/chat/completions` | `messages`, sampling, `response_format`, `tools` (calls are parsed from the output), `stream` with SSE and `stream_options.include_usage` |
| `POST /v1/completions` | `prompt` as a string or array (one choice per prompt); `stream` needs a single prompt; `echo` is not supported |
| `POST /v1/embeddings` | `input` as a string or array of strings; `encoding_format` `float` or `base64`; `dimensions` must equal the model's dimension; `usage` is reported as zeros |
| `POST /v1/rerank` | `query`, `documents`, `top_n`, `return_documents` (Cohere/Jina shape: `results[].relevance_score`) |
| `POST /rpc` | one JSON-RPC request; any method except `shutdown`; model paths are refused |

`model` must be a registered name or `"default"`. An unknown name returns 404 with the list of
available models, so a client asking for `gpt-4o` fails loudly instead of silently getting a
different model. `n > 1` is rejected.

Streaming follows OpenAI's format:

1. a first chunk with `delta.role`;
2. content deltas;
3. a final chunk with `finish_reason` (and `delta.tool_calls` when tools were called);
4. with `include_usage`, a chunk with `choices: []` and `usage`;
5. `data: [DONE]`.

Errors use OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`:

| status | when |
|---|---|
| 400 | bad request (`invalid_request_error`) |
| 401 | missing or wrong API key |
| 403 | network refused |
| 404 | unknown model |
| 500 | model failure |

**Security.** The server binds to localhost by default. Binding to another address without
`--api-key-env` prints a warning. The key is read from an environment variable, so it does not
show up in `ps`. Request bodies are capped at 16 MiB.

### Using the HTTP server from an OpenAI client

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
r = client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "Olá!"}])
```
