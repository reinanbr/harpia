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
| `TextGenerator` (generation, constrained decoding, log-probs, chat, `Provider`) | `qwen2`, `llama`, `mistral` | stable | Qwen2.5-0.5B-Instruct: identical chat template and tokens, max \|Δlogit\| 4.3e-4, 20 greedy tokens identical to `generate()`; Llama: tiny models only |
| `MarianTranslator` (greedy, beam) | `marian` | stable | tokens + greedy output (opus-mt-tc-big-en-pt) |

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
- **MarianMT** loads the SentencePiece files Hugging Face ships (`source.spm` + `vocab.json`)
  directly, mirroring transformers' `MarianTokenizer`: a leading `>>lang<<` code is looked up in
  the vocabulary, the rest is split by the source model (Unigram, precompiled normalization map)
  and `</s>` is appended (`harpia_data::spm` parses the protobuf, no Python). A `tokenizer.json`
  (plus optional `tokenizer-target.json`) wins when present. Sentences are translated one at a
  time because candle's Marian decoder has no encoder padding mask. Token ids and greedy
  translations match transformers on `Helsinki-NLP/opus-mt-tc-big-en-pt` (see `docs/parity.md`).
- **Generation defaults** come from `generation_config.json` (`TextGenerator::default_params`),
  as in transformers; the repetition penalty covers prompt and output tokens (HF semantics),
  frequency/presence penalties only the output (OpenAI semantics).
- **LoRA adapters** in PEFT format load with `TextGenerator::load_with_adapter`.
- **Hub downloads** (feature `hub`): `harpia download Qwen/Qwen2.5-0.5B-Instruct --online`
  writes `<cache>/<owner>--<name>/<revision>/` atomically; `HF_TOKEN` is sent when set.
- **Not yet supported**: GGUF/quantized models.
