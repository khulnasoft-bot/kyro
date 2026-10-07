# Troubleshooting Guide

Common issues when running Kyro in development or production.

## Startup Failures

### `model path does not exist`
- **Cause:** `KYRO_MODEL_PATH` or `--model-path` points to a non-existent directory/file.
- **Fix:** Verify the path. For GGUF, point directly at the `.gguf` file. For Safetensors, point at the directory containing `config.json` and weight shards.

### `Failed to load tokenizer`
- **Cause:** `KYRO_TOKENIZER_PATH` is missing or the tokenizer file is corrupt.
- **Fix:** Download a HuggingFace `tokenizer.json` for your model. If you don't need detokenization, unset the path — Kyro will return raw token IDs.

### Port already in use
- **Cause:** Another process is bound to `KYRO_PORT` (default 3000).
- **Fix:** Set `KYRO_PORT` to a free port or stop the conflicting process.

## Request Failures

### 503 "Engine is not ready"
- **Cause:** Request arrived before model loading completed.
- **Fix:** Poll `GET /ready` until it returns 200 before sending traffic. Use this as a Kubernetes readiness probe.

### 400 "Unsupported model"
- **Cause:** Request `model` field does not match `KYRO_MODEL_NAME` (default `kyro`).
- **Fix:** Set `KYRO_MODEL_NAME` to match your requests, or send the configured name.

### 504 "Request timed out"
- **Cause:** Non-streaming request exceeded `KYRO_REQUEST_TIMEOUT_SECS` (default 600).
- **Fix:** Increase the timeout, reduce `max_tokens`, or switch to streaming mode.

### Streaming produces no output
- **Cause:** Client disconnected or network issue; the scheduler request may still be running.
- **Fix:** Check server logs for panics. The worker loop continues on errors and logs them.

## Performance Issues

### High TTFT (Time to First Token)
- **Cause:** Large prompts, cold prefix cache, or scheduler token budget too small.
- **Fix:** Increase `KYRO_MAX_TOKENS_CAP` for larger batches. Ensure common system prompts are stable to hit the Radix cache.

### Low throughput
- **Cause:** Single-GPU bottleneck, small batch, or KV cache thrashing.
- **Fix:** Increase `num_gpu_blocks` in `BlockManager::new` (main.rs) if memory allows. Use GGUF quantization to reduce memory pressure.

### KV cache eviction storms
- **Cause:** Cache capacity (`num_gpu_blocks / 3`) is too small for working set.
- **Fix:** Increase total GPU blocks or reduce concurrent long-context requests.

## Observability

### Metrics endpoint returns 503
- **Cause:** Prometheus registry not configured (only happens if `with_metrics` was not called).
- **Fix:** This should not occur in normal operation; check that `main.rs` wires metrics.

### Grafana shows no data
- **Cause:** Prometheus not scraping `GET /metrics` on port 3000.
- **Fix:** Add a scrape target: `kyro:3000/metrics`.

## Crash Recovery

### Worker loop panic
- **Cause:** OOM, device error, or model forward pass failure.
- **Fix:** The worker loop logs the error and continues. Check logs with `RUST_LOG=info`. For OOM, reduce `max_tokens` or model size.

### Graceful shutdown not draining
- **Cause:** SIGINT received but in-flight requests are still processing.
- **Fix:** The server waits for in-flight requests to complete. Force-kill with SIGTERM only if needed.
