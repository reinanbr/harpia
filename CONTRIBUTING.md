# Contributing to Harpia

Thanks for helping! A few rules keep the project healthy:

## Ground rules

- **No Python.** No Python files, PyO3, maturin or `pyproject.toml`. Golden values for parity
  tests are produced once by a human following `docs/parity.md` and checked in as JSON.
- **Pure Rust by default.** A default or `full` build must not need a C/C++ toolchain. Before
  adding a dependency, check its features with `cargo tree -e features` and make sure no
  `*-sys` crate appears. Native acceleration may only arrive as an opt-in, documented feature.
- **Offline by default.** Code that can touch the network takes a `NetworkPolicy` and does
  nothing when it is `Offline`. Tests never use the real network (use recorded fixtures).
- **Deterministic.** Anything random takes an explicit seed (`harpia_data::rng::SplitMix64`).
- **No invented numbers.** Benchmarks and accuracy figures are published only when measured,
  with hardware and commit.
- **No hidden `todo!()`.** Unfinished work is marked experimental or planned in docs and the
  README; stable APIs must not panic with `todo!()`/`unimplemented!()`.

## Before opening a pull request

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo build --workspace --no-default-features
cargo deny check            # licenses, advisories, bans
```

Every public item needs docs (`#![warn(missing_docs)]`) and, where practical, a doc example.
Bug fixes come with a regression test; new parsers and repairers come with property tests.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat(data): ...`,
`fix(structured): ...`, `docs: ...`, `perf(text): ...`, `test: ...`, `chore: ...`.

## License

By contributing you agree that your contributions are dual-licensed under MIT OR Apache-2.0,
without additional terms.
