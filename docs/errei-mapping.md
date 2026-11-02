# errei → Harpia mapping

errei (local path `~/Desktop/std/plugins/python/errei`) is a
Python auditor of pt-BR LaTeX scientific articles. It is Harpia's real-world acceptance target:
Harpia must be able to replace every ML, NLP and HTTP dependency it has. errei's own domain
logic (pt-BR grammar rules, LaTeX parsing, ABNT checks, Qualis, reports) stays in errei and is
**not** ported.

Status legend: **done** (in 0.1), **planned (Mn)** (milestone n), **gap** (not yet designed).

| errei module | what it needs | Harpia API | status |
|---|---|---|---|
| `text/segment.py` `words`, `alpha_words`, `sentences` | word/sentence spans, pt-BR abbreviations, initials, decimals, lowercase continuation | `harpia_text::segment::{words, alpha_words, sentences}`, `Lang::PtBr` | done |
| `text/segment.py` `strip_accents`, `load_list` | accent folding, word lists | `harpia_text::normalize::strip_accents`, `LangData::with_*`, `lang::parse_list` | done |
| `text/metrics.py` `syllables`, `flesch_pt` | pt-BR syllable heuristic, Martins et al. (1996) Flesch | `harpia_text::stats::{syllables, flesch}` with `SyllableRule::Portuguese` (same hiatus rules) | done |
| `text/metrics.py` `mattr`, `cv` | MATTR (window 100), coefficient of variation | `stats::mattr`, `stats::coefficient_of_variation` | done |
| `text/metrics.py` `connectives` | case-insensitive phrase list matching | `harpia_text::phrases::PhraseMatcher` | done |
| `text/similarity.py` | word k-gram shingles, Jaccard, containment, longest shared span, sampled fingerprint index | `harpia_text::similarity::{shingles, jaccard, containment, longest_shared_span, FingerprintIndex}` | done (hash is XXH3 instead of BLAKE2b; spans are bytes instead of chars, convert with `LineIndex`) |
| `checkers/plagio.py` | corpus similarity, PDF corpus files | similarity: done; PDF text: `harpia-data` feature `pdf` | PDF planned (M2) |
| `checkers/ia.py` | burstiness, repetition, diversity, phrase lists, external detector | `stats::{burstiness, top_ngram_fraction, mattr}`, `PhraseMatcher`, `harpia_core::traits::Detector` | done |
| `checkers/refs.py` `_ratio`, `partial_ratio` | rapidfuzz `token_set_ratio`, `partial_ratio` | `harpia_text::fuzzy::{token_set_ratio, partial_ratio}` | done |
| `net/client.py` | per-host rate limits, `Retry-After`, retries with jitter, circuit breaker, size caps, offline guard | `harpia_http::{HttpClient, blocking::HttpClient, HttpConfig::host_interval}`, `status()`, `download` | done |
| `net/cache.py` | SQLite TTL cache | `harpia_http::CacheConfig` (`redb`, `mailto` excluded from keys by default) | done |
| `net/sources.py` | Crossref, OpenAlex, Semantic Scholar, arXiv adapters | `harpia-verify` adapters | planned (M6) |
| `checkers/refs.py` reference matching | fuzzy title/author/year match → confirmed/divergent/not found, retractions | `harpia-verify` reference verification | planned (M6) |
| `citecheck.py` `Classificador` | XLM-RoBERTa pair classification, label aliases, length-limited batches, load once per process | `harpia-local` `AutoSequenceClassifier` + `PairClassifier` | planned (M3) |
| `citecheck.py` `julgar_resumo` | sentence + 2-sentence windows, max per label with evidence | `harpia-eval` "best of N windows" | planned (M5) |
| `citecheck.py` thresholds file | calibrated per-label thresholds | `harpia-eval` calibration | planned (M5) |
| `treino/traduzir.py` | MarianMT batched translation, resumable cache | `harpia-local` MarianMT + `harpia_data::JsonlCache` | cache done; MarianMT planned (M3) |
| `treino/preparar.py`, `comum.py` | JSONL IO, seeded splits | `harpia_data::{read_jsonl, JsonlWriter}`, `split::{random_split, group_split}`, `rng::SplitMix64` | done |
| `treino/treinar.py` | cross-encoder fine-tuning: AdamW, warmup/linear decay, clipping, class weights, macro-F1 per epoch, best checkpoint | `harpia-train`; class weights: `harpia_data::split::class_weights` | weights done; training planned (M4) |
| `treino/avaliar.py`, `finalizar.py` | per-class P/R/F1, macro F1, threshold grid search on val, frozen test, baseline, model card | `harpia-eval` | planned (M5) |
| `tests/test_acceptance.py` "no network unless `--online`" | offline guarantee | `NetworkPolicy::Offline` default, enforced in `harpia-http`; `harpia_http::testing::ConnectionCounter` | done |
| `tests/test_performance.py` 20 files < 5 s | text throughput | `harpia-text` benches: segmentation + signals + Flesch + MATTR + shingles on 128 KB take ≈ 7 ms on an i5-12450H (see `benchmarks.md`) | done |

## Gaps

- **Offsets.** errei reports Python character offsets; Harpia uses byte spans. Use
  `LineIndex::char_offset` / `line_col` at the boundary.
- **Hash compatibility.** Shingle hashes differ from errei's BLAKE2b, so fingerprint indexes
  are not interchangeable with ones built by errei (similarity values are identical).
- **LanguageTool** (Java) is outside Harpia's scope.
