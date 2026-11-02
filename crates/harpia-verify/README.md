# harpia-verify

Grounding and fact-checking for LLM pipelines against **user-trusted** sources, part of
[Harpia](../../README.md). See the workspace README section "Grounding against trusted sources".

- `SourceRegistry`: TOML registry (built-in list + your own), hard host allowlist, tiers.
- `Fetcher`: offline by default, HTTPS only, redirect hops checked, `robots.txt`, size and
  content-type caps, `Disclosure` log, content store with `Frozen` replay.
- Adapters (features): `crossref`, `openalex`, `semantic-scholar`, `arxiv`, `europepmc`,
  `doaj`, `wikipedia` (Wikipedia + Wikidata), `web` (allowlisted pages); `SourceAdapter` for
  your own.
- `ClaimVerifier`: claims → keyword queries → passages → NLI `PairClassifier` → calibrated
  decision → tier-aware aggregation; `verify_answer`, `dataset::verify_rows`.
- `ReferenceVerifier`: DOI / arXiv lookups and search, weighted fuzzy matching, retractions;
  `reference::check_quote`; `bib::parse` for BibTeX.
- `rag::trusted_documents`: trusted sources as `RagPipeline` documents.
- Feature `llm`: claim decomposition, query rewriting and evidence summaries (never decides).
- Feature `tls`: HTTPS (needed for real sources).

Results are indicators with evidence, never verdicts.
