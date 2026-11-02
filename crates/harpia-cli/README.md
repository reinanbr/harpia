# harpia-cli

The `harpia` binary. Every subcommand accepts `--format text|json` (JSON output carries
`"schema": 1`) and reads standard input when the path is `-`.

```bash
cargo install harpia-cli
harpia segment paper.txt --unit sentences --lang pt-br
harpia data filter raw.jsonl --preset gopher-like --field text -o clean.jsonl --report report.json
harpia data pack chat.jsonl --tokenizer tokenizer.json --template tokenizer_config.json --seq-len 4096 -o packed.jsonl
harpia repair-json broken.json
```

Exit codes: `0` success, `1` the command ran but reported a failing check (e.g. invalid JSON
that could not be repaired), `2` usage or environment error.
