# Parity tests

Harpia contains **no Python**. Where a result must match a reference implementation (logits of
a model, a tokenizer, a chat template, a metric), the reference values are produced **once, by
a human**, outside this repository, and checked in as JSON golden files under
`crates/<crate>/tests/golden/`. Parity tests are `#[ignore]` by default and opt-in through
environment variables, so CI never needs the reference stack or large downloads.

## Protocol

1. Pin the reference versions (e.g. `transformers==X.Y.Z`, `tokenizers==X.Y.Z`) and record
   them in the golden file's `"reference"` field together with the date and hardware.
2. Produce the values with a short script kept **outside** the repository; paste the script
   into the pull request description for review.
3. Store inputs and expected outputs in the golden JSON with an explicit tolerance
   (`1e-4` absolute for f32 logits unless stated otherwise).
4. Write a Rust test that reads the golden file and compares, marked
   `#[ignore = "parity: needs <ENV_VAR>"]`.

## Planned parity suites

| suite | env var | compares | milestone |
|---|---|---|---|
| tokenizer | `HARPIA_PARITY_TOKENIZERS` | ids and offsets on a multilingual sample | 3 |
| chat templates | `HARPIA_PARITY_TEMPLATES` | rendering vs. `apply_chat_template` for bundled templates | 3 |
| decoder/encoder logits | `HARPIA_PARITY_MODELS` | logits within `1e-4` (f32) | 3 |
| LoRA SFT step | `HARPIA_PARITY_TRAIN` | loss and adapter gradients vs. peft | 4 |
| errei | `HARPIA_ERREI_DIR` | `errei-cita-v1` logits and verdicts, segmentation, metrics, reference classes replayed from `tests/fixtures/http/` | 3–6 |

errei's model weights (`treino/modelos/errei-cita-v1`, CC BY-NC 2.0) and datasets are read in
place from `HARPIA_ERREI_DIR` and are never copied into this repository.

## Already covered without golden files

- `harpia_text::stats::flesch` uses the published coefficients (Martins et al., 1996; Flesch,
  1948); tests check the formula exactly.
- `harpia_data::rng::SplitMix64` is checked against the reference first output for seed 0.
- `harpia_text::fuzzy::ratio` follows rapidfuzz's normalized Indel definition; tests use
  hand-computed values (e.g. `kitten`/`sitting` = 8/13).
