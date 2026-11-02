# harpia-data

Dataset tooling for LLM training, evaluation and RAG:

- streaming JSONL/CSV (gzip and zstd transparently), sharded datasets with manifests, a
  resumable `JsonlCache` for annotation/translation/LLM-labeling jobs;
- curation: cleaning, heuristic quality filters with presets, language filters, PII
  redaction (incl. CPF/CNPJ), exact and MinHash-LSH dedup, benchmark decontamination,
  seeded splits and a curation report;
- training formats: OpenAI messages, ShareGPT, Alpaca, prompt/completion, preferences;
  chat templates rendered with the model's own Jinja template and assistant-only loss masks;
  sequence packing;
- chunking for RAG with byte offsets.
