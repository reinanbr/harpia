# harpia-cli

The `harpia` binary. Every subcommand accepts `--format text|json` (JSON output carries
`"schema": 1`) and reads standard input when the path is `-`.

```bash
cargo install harpia-cli
harpia segment paper.txt --unit sentences --lang pt-br
harpia data filter raw.jsonl --preset gopher-like --field text -o clean.jsonl --report report.json
harpia data pack chat.jsonl --tokenizer tokenizer.json --template tokenizer_config.json --seq-len 4096 -o packed.jsonl
harpia repair-json broken.json

# keep models loaded and serve them to other programs
harpia serve --stdio --decoder models/qwen2.5-0.5b-instruct --embedder models/e5-small
harpia serve --http --bind 127.0.0.1:8080 --decoder qwen=models/qwen2.5-0.5b-instruct
```

`serve --stdio` speaks newline-delimited JSON-RPC 2.0. `serve --http` is an OpenAI-compatible
API (`/v1/chat/completions` with streaming, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`).
Both are specified in [`docs/serve-protocol.md`](../../docs/serve-protocol.md).

Features: `tls` (HTTPS for hosted APIs; `ring` needs a C compiler) and `server` (the HTTP
server; pure Rust). Both are on by default.

Exit codes: `0` success, `1` the command ran but reported a failing check (e.g. invalid JSON
that could not be repaired), `2` usage or environment error.
