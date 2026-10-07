# Kyro Implementation Status Audit

**Date:** October 7, 2026
**Scope:** Verification of advertised features against actual code paths.

## Summary

| Feature | Advertised | Actual Status |
|---------|-----------|---------------|
| Continuous Batching | Yes | **Implemented** — `src/scheduler/continuous_batching.rs` |
| PagedAttention / Block Manager | Yes | **Implemented** — `src/scheduler/block_manager.rs` |
| Prefix Caching (Radix Cache) | Yes | **Implemented** — `src/scheduler/radix_cache.rs` |
| Chunked Prefill | Yes | **Implemented** — `PREFILL_CHUNK_SIZE` in scheduler |
| GGUF Quantization | Yes | **Implemented** — `src/model/quantized.rs` via candle |
| AWQ (4-bit) | Yes (README) | **Stub** — `src/model/quantization/awq.rs` is a no-op cast |
| FP8 (Hopper) | Yes (README) | **Stub** — `src/model/quantization/fp8.rs` casts without a real kernel |
| Tensor Parallelism | Yes (README) | **Stub** — `src/distributed.rs` holds rank/world_size only |
| Pipeline Parallelism | Yes (README) | **Stub** — no layer assignment or activation checkpointing |
| Multi-LoRA | Yes (README) | **Partial** — `src/model/lora.rs` has math but no loader, no API, no scheduler integration |
| Speculative Decoding | Yes (README) | **Partial** — `src/speculative.rs` has draft loop but no verification, no API, not called by worker |
| Vision Models | — | **Stub** — `src/model/vision.rs` |
| MoE | — | **Stub** — `src/model/moe.rs` |
| Prometheus Metrics | Yes | **Implemented** — `src/metrics.rs` |
| OpenAI-compatible API | Yes | **Implemented** — `src/api/openai.rs` |
| Grammar-constrained decoding | Yes | **Implemented** — `src/api/grammar.rs` |
| Readiness probe | — | **Implemented** — `GET /ready` |
| Request timeout | — | **Implemented** — `KYRO_REQUEST_TIMEOUT_SECS` env var |
| Graceful shutdown | — | **Implemented** — SIGINT handler in `main.rs` |

## Detailed Notes

### GGUF (Working)
- `ModelLoader` auto-detects `.gguf` files (`src/model/loader.rs:26`).
- `QuantizedLlama` wraps `candle_transformers::models::quantized_llama::ModelWeights`.
- Worker loop dispatches to `QuantizedLlama::forward` for quantized models.

### AWQ / FP8 (Stubs)
- `src/model/quantization/awq.rs`: `unpack_weights()` is a no-op cast with comment "Mock unpacking 4-bit to F16".
- `src/model/quantization/fp8.rs`: casts activations to `F8E4M3` without a real dequant kernel.
- Both modules are `#![allow(dead_code)]` and never invoked by the loader or forward pass.
- **Action:** Remove from README feature list or mark as "planned".

### Distributed Inference (Stub)
- `src/distributed.rs` (307 bytes): `DistributedContext` has `rank: u32` and `world_size: u32` fields only.
- `DistributedContext::new()` always returns `rank: 0, world_size: 1`.
- Threaded through `ModelLoader` and `LlamaModel` but never used for communication.
- NCCL is listed in architecture docs but not integrated.
- **Action:** Implement NCCL init + AllReduce, or remove from README.

### LoRA (Partial)
- `src/model/lora.rs` (1276 bytes): `LoraLinear` with `forward()` applying `x @ A.T @ B.T * (alpha/rank)`.
- No weight loading from disk.
- No API endpoint or request field for `lora_id`.
- Scheduler does not track active LoRA per request.
- **Action:** Add loader + API parameter + scheduler tracking, or remove from README.

### Speculative Decoding (Partial)
- `src/speculative.rs` (1386 bytes): `SpeculativeDecoder` with draft model loop.
- No target-model verification step.
- Not called from worker loop.
- No API parameter to enable/disable.
- **Action:** Integrate into worker loop with verification, or remove from README.

## Recommended Priority

1. **Retract or implement** AWQ/FP8 claims in README (1 day).
2. **Retract or implement** distributed inference claims (2–4 weeks if implementing).
3. **Retract or implement** LoRA integration (1–2 weeks if implementing).
4. **Retract or implement** speculative decoding (1–2 weeks if implementing).
5. Add unit tests for `LoraLinear::forward` and `SpeculativeDecoder::step` (2 days).
