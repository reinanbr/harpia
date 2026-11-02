# harpia-client

Typed async clients for LLM APIs, all implementing `harpia_core::traits::Provider`, built on the
polite `harpia-http` layer (offline by default, rate limits, retries, circuit breaker):

| provider | feature | endpoint |
|---|---|---|
| `OpenAiCompatible` | `openai-compatible` | `/chat/completions`, `/embeddings` (OpenAI, vLLM, llama.cpp, LM Studio, OpenRouter...) |
| `Anthropic` | `anthropic` | `/v1/messages` with `anthropic-version: 2026-10-21` |
| `Ollama` | `ollama` | `/api/chat`, `/api/embed` |

Also: SSE/NDJSON streaming into typed chunks, tool calls in both directions, structured output
(`chat_json`: native JSON-schema modes plus repair, validation and one corrective retry),
middleware hooks, and a concurrency- and RPM/TPM-limited `BatchRunner` that resumes from a
`JsonlCache`. API keys are read from the environment or passed explicitly and are redacted in
`Debug` output.
