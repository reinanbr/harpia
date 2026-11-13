# Learning Harpia

Lessons written as tests: every example here compiles and runs in CI, so it stays correct
as the library changes. Read them in order and run them with `--nocapture` to see what each step
prints.

```sh
cargo test -p harpia-learn -- --nocapture                         # all lessons
cargo test -p harpia-learn --test 02_tokenization -- --nocapture  # one lesson
```

| lesson | what you learn |
|---|---|
| [`01_text`](lessons/01_text.rs) | Unicode normalization, words and sentences with byte spans (pt-BR and English), Flesch and quality signals, fuzzy matching, shingles/containment/MinHash for copy detection |
| [`02_tokenization`](lessons/02_tokenization.rs) | a WordPiece `tokenizer.json` built by hand, ids and offsets, pair encoding, chat templates from `tokenizer_config.json`, assistant-only loss masks, sequence packing |
| [`03_models`](lessons/03_models.rs) | reading a model directory (config, weights, labels), Qwen2/Llama generation with seeds and log-probs, constrained decoding, BERT/XLM-R classifiers and cross-encoders, embeddings, MarianMT translation |
| [`04_model_review`](lessons/04_model_review.rs) | grading models: EM/F1/BLEU/chrF/ROUGE, classification reports, the TOML eval harness and a leaderboard; then correcting a model with LoRA and measuring again |
| [`05_article_review`](lessons/05_article_review.rs) | loading a LaTeX project (`\input`, bibliography, clean text mapped to `file:line:col`) and reviewing it with [`src/review.rs`](src/review.rs) |

## Models

The lessons use *tiny random-weight models* that `harpia::local::testing` writes to a temporary
directory, so nothing is downloaded and everything runs offline in about a second. Their
outputs are gibberish, but every call is the real one: point the same code at a real Hugging
Face directory (`Qwen2.5-0.5B-Instruct`, an XLM-R NLI cross-encoder, `opus-mt-*`) and it does
real work.

## The article reviewer

[`src/review.rs`](src/review.rs) is a short errei-style reviewer built only on Harpia's
public API. It checks:

- broken `\input`s;
- missing sections or abstract;
- undefined or unused citations;
- BibTeX syntax, required fields and DOI shape;
- long sentences, repeated words and filler phrases;
- low Flesch readability;
- paragraphs copied from a reference corpus.

Every finding has a source location. The real errei (`errei-rs`, a separate project) goes much
further; this module is a teaching example and is not needed by it.

To review your own LaTeX project:

```sh
HARPIA_ARTICLE_DIR=/path/to/thesis cargo test -p harpia-learn --test 05_article_review -- --ignored --nocapture
# English text: add HARPIA_ARTICLE_LANG=en
```

The fixture in [`fixtures/article`](fixtures/article) is a short pt-BR paper with deliberate
mistakes; [`fixtures/corpus`](fixtures/corpus) is the reference text it partly copies.

This crate is `publish = false`: it ships nothing to crates.io.
