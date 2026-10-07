# Kyro Architecture

## Request Lifecycle (async flow)

```
HTTP client
  │  POST /v1/chat/completions (messages | prompt)
  ▼
api::openai::chat_completions
  │  1. Validate model name, max_tokens, temperature, top_p,
  │     message count, prompt byte size (ApiError on violation)
  │  2. LuminaTokenizer::encode(prompt)
  │  3. Build scheduler::Request, push to Scheduler (Mutex)
  ▼
Worker::run_loop            (scheduler / device boundary)
  │  Phase 1: schedule() → prefill/decode work items (lock held briefly)
  │  Phase 2: LlamaModel|QuantizedLlama::forward on token-id tensors,
  │            sample from last-position logits (lock released)
  │  Phase 3: push tokens into per-request mpsc channel, advance
  │            prefill cursors, finish completed requests (lock re-held)
  ▼
Token channel (tokio mpsc)
  │  u32 token IDs per generated step
  ▼
api::openai (non-streaming)        api::openai (streaming SSE)
  decode(ids) → JSON response       decode incrementally → sse::Event chunks,
                                    final chunk carries finish_reason
```

Errors propagate as `ApiError` → structured JSON; worker loop failures are
logged with `tracing::error!` and do not panic the server.

## Metrics

`GET /metrics` exposes Prometheus counters/gauges from `EngineMetrics`:
requests total, requests by model label, queue depth, prefix-cache
hits/misses, token throughput, TTFT/TBT histograms, KV-cache usage.

## Configuration

`config::AppConfig` centralizes CLI flag + environment variable parsing and
is validated at startup (model path existence, non-empty model name, positive
caps). See README "Model Configuration".

## Safety & deployment

- Runtime Docker image runs as non-root `kyro` user with no login shell.
- CI runs `cargo fmt`, `cargo clippy -D warnings`, tests, release build, and
  `cargo audit` for supply-chain vulnerability checks.
