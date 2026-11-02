# Local models (`harpia-local`)

Models load from a local Hugging Face directory: `config.json`, `model.safetensors` (or the
shards of `model.safetensors.index.json`), `tokenizer.json` and, for decoders,
`tokenizer_config.json` (chat template, EOS) and `generation_config.json`. Weights are read into
memory (no memory mapping), so the crate contains no `unsafe`. CPU is the default; the `cuda` and
`metal` features enable GPUs and pull native code.

| type | architectures (`model_type`) | status | verified against transformers |
|---|---|---|---|
| `SequenceClassifier` (pairs, single texts, reranking) | `xlm-roberta`, `roberta`, `camembert`, `bert` | stable | XLM-R: `errei-cita-v1`, max \|Δlogit\| 1.0e-5 on 50 pairs, identical tokenization |
| `EncoderEmbedder` (mean/CLS pooling, L2 norm) | same encoders | stable | not yet (tiny-model tests only) |
| `TextGenerator` (generation, constrained decoding, log-probs, chat, `Provider`) | `qwen2`, `llama`, `mistral` | stable | not yet: no real decoder checkpoint was available offline; tested on tiny random models |
| `MarianTranslator` (greedy, beam) | `marian` | **experimental** | not yet |

## Notes and limits

- **Pair order** is `(premise, hypothesis)`: evidence first, claim second, encoded with the
  model's pair template (`<s> A </s></s> B </s>` for RoBERTa models).
- **Truncation** is longest-first at `max_len` (default 512; errei uses 192).
- **Padding** never changes results: tests compare batched and one-by-one logits.
- **Decoders**: Qwen2 scores whole sequences in one pass; Llama scoring feeds tokens one at a
  time because candle's Llama returns only the last position (correct, slower).
  `Provider::chat_stream` yields the pieces after generation finishes; use
  `TextGenerator::generate_with` for live token callbacks.
- **Constrained decoding** checks candidate tokens in logit order against a
  `harpia_structured::constraint` (JSON Schema subset, regex, choices); end-of-sequence is allowed
  only when the output is complete.
- **MarianMT** needs `tokenizer.json` (and optionally `tokenizer-target.json`). Hugging Face ships
  Marian tokenizers as SentencePiece files; converting them is a one-time step outside Harpia.
  Sentences are translated one at a time because candle's Marian decoder has no encoder padding
  mask. Status stays experimental until verified on a real `opus-mt` model.
- **Not yet supported**: GGUF/quantized models, LoRA adapter loading (arrives with
  `harpia-train`), Hugging Face Hub download.
