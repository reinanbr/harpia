# harpia-eval

Evaluation for LLMs and datasets, part of [Harpia](../../README.md).

- **Harness** (`harness`): tasks in TOML/JSON over JSONL: generation scored with metrics,
  multiple choice by log-likelihood (`acc`, `acc_norm`), perplexity / bits per byte, and
  classification. Seeded few-shot prompts, `minijinja` templates, any `Provider` (API or local,
  chat-template aware) or local `Model`, bounded concurrency and a resumable `JsonlCache`.
- **LLM-as-judge** (`judge`): rubric scores and pairwise comparison with position swap,
  verdicts parsed and repaired with `harpia-structured` (corrective retries), Cohen's kappa and
  Pearson agreement between judges.
- **RAG** (`rag`): recall@k, precision@k, MRR, nDCG, context recall, answer faithfulness with
  a `PairClassifier` (best window per sentence, with evidence spans), answer relevance with an
  `Embedder`.
- **Metrics** (`metrics`, `classification`): exact match, token F1, BLEU (sentence and corpus),
  chrF, ROUGE-N/L, edit distance; accuracy, per-class P/R/F1, macro/weighted F1, confusion matrix.
- **Calibration** (`calibration`): decision rules as data, threshold grid search on a
  `ValidationSet`, frozen application to a `TestSet` (evaluating on the tuning data is refused),
  best-of-N sentence windows with evidence.
- **Reports** (`report`): JSON run summaries, Markdown leaderboards, model cards with metrics
  vs. a baseline, data licenses and limitations.
- **Cost** (`cost`): user-supplied price tables, a thread-safe tracker, and `tracing` spans with
  latency, time to first token and tokens per second for every call.
- **Datasets** (`dataset`): benchmark contamination, label-noise and duplicate statistics.

`testing` provides scripted mock providers for tests and examples.
