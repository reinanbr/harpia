# Benchmarks

Run:

```bash
cargo bench -p harpia-text
cargo bench -p harpia-structured
cargo bench -p harpia-data          # curation, dedup, template+pack, flat/HNSW retrieval
cargo bench -p harpia-local         # tiny models unless HARPIA_BENCH_{DECODER,ENCODER,EMBEDDER} are set
cargo bench -p harpia-train         # tiny decoder unless HARPIA_BENCH_DECODER is set
```

The local and train benches run on the tiny random-weight models from `harpia_local::testing`
by default; those figures measure plumbing overhead only and are not reported here. Point the
`HARPIA_BENCH_*` variables at real Hugging Face model directories for real numbers.

## Harpia-only measurements

Measured with `cargo bench -p harpia-text --bench text -- --warm-up-time 1 --measurement-time 3`
on an Intel Core i5-12450H (12 threads, Linux 7.0, rustc 1.97.1), commit: initial import
(v0.1.0). Input: a ~128 KB pt-BR text (one paragraph repeated 400 times).

| benchmark | median time | throughput |
|---|---:|---:|
| `text/words` | 315 µs | 397 MiB/s |
| `text/sentences_pt` | 401 µs | 312 MiB/s |
| `text/signals` | 1.31 ms | 95 MiB/s |
| `text/flesch_pt` | 1.89 ms | 66 MiB/s |
| `text/mattr_100` | 1.45 ms | 87 MiB/s |
| `text/shingles_k5` | 1.93 ms | 65 MiB/s |
| `text/minhash_128` | 1.93 ms | 65 MiB/s |
| `fuzzy/partial_ratio_8k` (55-char quote in 8 KB) | 1.06 ms | – |
| `fuzzy/token_set_ratio` | 2.6 µs | – |

The repeated-paragraph input has few distinct shingles, so `minhash_128` is dominated by
shingling here; real corpora with more distinct shingles will cost more per byte.

## Data, retrieval and constrained decoding

Measured with `cargo bench -p harpia-data --bench data` and `cargo bench -p harpia-structured
--bench repair` (`--warm-up-time 1 --measurement-time 3`) on the same i5-12450H, rustc 1.97.1,
milestone 7 (after commit becd6ef). Inputs are synthetic and seeded (see the bench sources).

| benchmark | input | median time | throughput |
|---|---|---:|---:|
| `data/curate` (normalize + quality filter + PII) | 2 000 English docs, 1.3 MiB | 9.5 ms | 139 MiB/s |
| `data/minhash_dedup` (5-shingles, LSH for 0.8) | same docs, 20% near-copies | 23.4 ms | 56 MiB/s |
| `data/template_mask_pack` (ChatML + spans + loss mask + best-fit pack to 4096) | 500 chats | 6.4 ms | 69 M byte-tokens/s |
| `retrieval/flat_query_100000` (cosine, top 10) | 100k × 128-d random vectors | 7.6 ms / query | 132 queries/s |
| `retrieval/hnsw_query_100000` (default params, top 10) | same | 0.43 ms / query | 2 350 queries/s |
| `structured/token_mask_32k_vocab` (JSON Schema mask, one decoding step) | 32k-entry vocabulary | 4.5 ms | – |

Building the indexes for the retrieval bench took 46 ms (flat) and 107 s (HNSW, sequential
inserts, about 1 ms per vector): HNSW construction is the slow path today. Token ids in
`template_mask_pack` are bytes, so real tokenizers add their own cost on top. The 4.5 ms mask
is per generated token under a constraint and is paid on top of the forward pass.

## Comparisons with Python

**TODO.** Comparisons with errei's Python implementation, rapidfuzz, `datasets`/`datatrove`,
`json_repair` and transformers will be added only when run on the same machine, with the
versions, hardware and commit recorded here. Protocol: same input files, warm caches, median
of at least 10 runs, single thread unless stated.

## errei-cita-v1 (XLM-R cross-encoder, 107 M parameters, f32, CPU)

Measured on the same machine (i5-12450H) on the 50 parity pairs (`max_length` 192, batches of
32, length-sorted), release build, rustc 1.97.1. PyTorch: best of 3 runs, 8 threads.

| | model load | throughput |
|---|---:|---:|
| Harpia (`SequenceClassifier`, candle 0.11) | 1.6–1.9 s | 18–19 pairs/s |
| PyTorch 2.15.0+cpu / transformers 5.22.1 | 3.1 s | 20.9 pairs/s |

Harpia loads faster and runs about 10% slower on this workload; the logits agree to 1e-5.

## Classifier fine-tuning (multilingual-MiniLMv2-L6-mnli-xnli, 117 M parameters, f32, CPU)

One epoch over 320 errei/SciFact training pairs (20 steps of 16, `max_length` 192), validation on
200 pairs, same machine, dropout off in both, errei's hyperparameters.

| | wall time | peak RSS | val macro-F1 after 1 epoch |
|---|---:|---:|---:|
| Harpia (`harpia train classifier`, release) | 152 s total (load, train, eval, save) | 8.4 GB | 0.488 |
| PyTorch 2.15.0+cpu (training loop only) | 88 s | 3.6 GB | 0.458 |

The base model scores 0.519 on this validation slice before training; one epoch on 320 pairs is
a smoke test, not a quality claim. Harpia is roughly 1.7× slower and uses 2.3× the memory: the
backward pass of candle's CPU autodiff dominates (≈4 s of ≈6 s per step); the fused AdamW brought
the optimizer step from 3.3 s to 0.7 s. The saved checkpoint loads in transformers and gives the
same probabilities to 6 decimals.

## Text generation (Qwen2.5-0.5B-Instruct, f32, CPU)

Greedy decoding on the same 33-token chat prompt, same machine.

| | throughput |
|---|---:|
| Harpia `TextGenerator` (candle 0.11; candle's own forward is 0.235 s/token) | 3.3–3.6 tok/s |
| transformers 5.22.1 `generate()` (40 tokens incl. prefill), torch 2.15.0+cpu, 8 threads | 1.3 tok/s |

## LoRA SFT (Qwen2.5-0.5B-Instruct, r=8, f32, CPU)

Six optimizer steps over three short chat samples (~50 tokens each): 43 s (≈7 s per step),
peak RSS 8.5 GB; 168 adapted matrices, 4.4 M trainable parameters. No PyTorch comparison yet.
