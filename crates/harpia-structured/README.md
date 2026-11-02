# harpia-structured

Structured output for LLM pipelines:

- `extract_json` / `parse_llm_json`: pull JSON out of chatty model output (fences, prose);
- `repair`: fix malformed JSON (missing quotes, commas, brackets, trailing commas, single
  quotes, Python literals, comments, raw newlines);
- `IncrementalJson`: best-effort values while tokens stream in;
- `constraint`: JSON-Schema, regex and choice constraints that compute token masks for
  constrained decoding;
- `toolcall`: parse OpenAI-, Anthropic- and Hermes/Qwen-style tool calls;
- feature `schema`: JSON Schema validation and schema derivation from Rust types.
