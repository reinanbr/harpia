# Dataset formats

`harpia-data` reads JSONL (optionally `.gz`/`.zst`), CSV/TSV and sharded directories with a
`manifest.json`. Chat rows are auto-detected (`harpia data inspect` shows the guess):

| format | shape | notes |
|---|---|---|
| `messages` | `{"messages": [{"role": "user", "content": "..."}]}` | OpenAI style; tool calls and multimodal text parts supported |
| `sharegpt` | `{"conversations": [{"from": "human", "value": "..."}]}` | roles `human`/`gpt`/`system`/`tool` |
| `alpaca` | `{"instruction": "...", "input": "...", "output": "...", "system": "..."}` | `input` and `system` optional |
| `prompt-completion` | `{"prompt": "...", "completion": "..."}` | single turn |
| preference | `{"prompt": ..., "chosen": ..., "rejected": ...}` | strings or message lists |

Extra fields are preserved as metadata through conversions.

## Packed rows

`harpia data pack` writes one JSON object per packed row:

```json
{"ids": [...], "loss_mask": [...], "position_ids": [...], "cu_seqlens": [0, 812, 1990]}
```

`position_ids` restart at 0 for each packed sample and `cu_seqlens` gives sample boundaries, so
variable-length attention never crosses samples. Padding (`--pad-id`) has `loss_mask = false`.

## Shard manifests

```json
{"version": 1, "shards": [{"file": "train-00000.jsonl", "rows": 100000, "bytes": 52428800, "xxh3": "9f0c..."}]}
```
