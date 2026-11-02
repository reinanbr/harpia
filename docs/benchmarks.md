# Benchmarks

Run:

```bash
cargo bench -p harpia-text
cargo bench -p harpia-structured
```

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

## Comparisons with Python

**TODO.** Comparisons with errei's Python implementation, rapidfuzz, `datasets`/`datatrove`,
`json_repair` and transformers will be added only when run on the same machine, with the
versions, hardware and commit recorded here. Protocol: same input files, warm caches, median
of at least 10 runs, single thread unless stated.
