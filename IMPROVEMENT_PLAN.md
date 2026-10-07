# Kyro LLM Engine: Gap Detection & Improvement Plan

**Date:** October 7, 2026  
**Repository:** nrelab/kyro  
**Analysis Scope:** Code structure, architecture, feature completeness, testing, and observability

---

## Executive Summary

**Kyro** is a high-throughput LLM serving engine written in Rust (60.3% of codebase) with complementary Jupyter Notebook tutorials (35.8%) and Python tooling (2.9%). The engine implements state-of-the-art inference optimizations: continuous batching, PagedAttention, prefix caching (Radix cache), and chunked prefill. Speculative decoding, distributed inference, and multi-quantization support are advertised but incomplete (see status tracker below).

**Overall Maturity:** Early production (v0.1.1, created April 2026). The core serving infrastructure is solid, but significant feature, observability, and testing gaps exist.

---

## Implementation Status Tracker (updated October 7, 2026)

| Plan Item | Status | Notes |
|-----------|--------|-------|
| Tier 1 #1 — Distributed Inference (TP/PP) | ❌ Open | `src/distributed.rs` still a stub; requires multi-GPU hardware |
| Tier 1 #2 — Expand Test Suite | ✅ Done | Unit tests for scheduler/block_manager/radix_cache + all model/API modules; 73 tests pass; coverage 70.40% enforced by CI gate (≥70%); concurrency integration test added |
| Tier 1 #3 — Quantization Paths | 🔶 Partial | GGUF loads real models + new `gguf_demo` example; AWQ/FP8 remain stubs, retracted from README; status audited in `docs/implementation_status.md` |
| Tier 2 #4 — LoRA Integration | 🔶 Partial | `LoraLinear::forward` now unit-tested; loader, API parameter, and scheduler tracking still missing |
| Tier 2 #5 — Speculative Decoding | 🔶 Partial | `SpeculativeDecoder::step` now unit-tested; verification loop and worker/API integration still missing |
| Tier 2 #6 — Model Ecosystem | ❌ Open | Only Llama implemented |
| Tier 2 #7 — Observability | 🔶 Partial | Grafana dashboard (`deploy/grafana-dashboard.json`) + SLO/alerting guide (`docs/slos.md`) added; no OTLP/tracing export yet |
| Tier 3 #8 — Error Handling & Resilience | 🔶 Partial | Readiness probe, graceful SIGINT shutdown, request timeout added; no circuit breaker/retry yet |
| Tier 3 #9 — Deployment Guide | 🔶 Partial | `docs/troubleshooting.md` + SLO guide added; Docker/K8s manifests already existed |
| Tier 3 #10 — API Compatibility | 🔶 Partial | Request cancellation (`POST /v1/cancel` + `X-Request-Id` header) added; no tools/functions or priority queues yet |

**Coverage:** 52.77% → **70.40%** (plan target was >70% for critical modules; scheduler modules are >94%).

---

## Section 1: Gap Analysis

### 1.1 **Testing & Validation**

**Current State:**
- `tests/integration.rs` contains 14 integration tests covering end-to-end request flows (non-streaming, SSE streaming, metrics endpoint, validation errors, tokenizer loading).
- 9 unit tests exist inline in `src/`; `benches/scheduler_bench.rs` provides a scheduler benchmark; `benchmarks/stress_test.py` exists for load testing.
- CI runs `cargo test` (stable + nightly), llvm-cov coverage, bench smoke test, examples build, notebook execution, and Docker build — but no enforced coverage threshold.

**Gaps:**
- ❌ No unit tests for scheduler internals (`continuous_batching.rs`, `block_manager.rs`, `radix_cache.rs`).
- ❌ No performance regression tests or SLO validation in CI.
- ❌ No test coverage metrics gate or badge in README (coverage is reported but not threshold-enforced).
- ⚠️ Stress test tooling exists in Python but has no Rust-side equivalent run in CI.

**Impact:** Moderate — a test harness exists, but coverage of the most failure-prone modules (scheduler, KV cache) is still missing, leaving silent-regression risk there.

---

### 1.2 **Quantization Implementation**

**Current State (audited Oct 7, 2026):**
- README advertises **FP8 (Hopper), AWQ (4-bit), GGUF** support.
- **GGUF is real**: `QuantizedLlama` (src/model/quantized.rs) loads via `candle_transformers::models::quantized_llama::ModelWeights`; `ModelLoader` auto-detects `.gguf` (src/model/loader.rs:26).
- **AWQ and FP8 are mocks**: `src/model/quantization/awq.rs` `unpack_weights()` is a no-op cast ("Mock unpacking 4-bit to F16"); `fp8.rs` casts activations to `F8E4M3` without a real kernel. Both are `#![allow(dead_code)]` and never invoked by the loader or forward pass.

**Gaps:**
- ❌ AWQ and FP8 are simulations, not implementations; README should not advertise them as supported.
- ⚠️ No examples demonstrating GGUF loading end-to-end (only the loader path exists).
- ⚠️ No benchmarks comparing throughput/accuracy tradeoffs across quantization types.

**Impact:** GGUF users are served; AWQ/FP8 claims are misleading and should be retracted or implemented.

---

### 1.3 **Distributed Inference (Multi-GPU / Multi-Node)**

**Current State:**
- README claims **Tensor Parallelism (TP) and Pipeline Parallelism (PP)** support.
- Code has `src/distributed.rs` (307 bytes—confirmed stub: `rank`/`world_size` fields only).
- `DistributedContext::new()` is threaded through `ModelLoader` and `LlamaModel` but always `rank: 0, world_size: 1`.
- NCCL listed in architecture but not integrated into active code paths.

**Gaps:**
- ❌ `src/distributed.rs` appears to be a placeholder; full TP/PP implementation missing.
- ❌ No multi-GPU scheduling or weight sharding logic visible.
- ❌ No documentation on how to configure or run distributed inference.
- ❌ No examples or tests for multi-node setups.
- ❌ Worker loop runs single-GPU inference only.

**Impact:** **Critical** — distributed inference is advertised but not implemented. Users expecting multi-GPU support will fail.

---

### 1.4 **LoRA (Low-Rank Adaptation) Support**

**Current State:**
- README advertises **Multi-LoRA Support: Dynamic loading and switching of task-specific adapters**.
- Code has `src/model/lora.rs` (1276 bytes).
- LoRA is mentioned in the schema but not integrated into model forward passes.

**Gaps:**
- ⚠️ LoRA module exists but integration into model inference is unclear.
- ❌ No API endpoint for dynamic LoRA switching.
- ❌ No documentation on how to load LoRA weights or enable multi-LoRA mode.
- ❌ No examples demonstrating LoRA inference.
- ❌ Scheduler and worker do not track or apply LoRA state.

**Impact:** Advertised feature is non-functional; users cannot use LoRA adapters.

---

### 1.5 **Speculative Decoding**

**Current State:**
- README advertises speculative decoding; **"Accelerates generation by 2x using a lightweight draft model"**.
- Code has `src/speculative.rs` (1386 bytes) and tutorial 08.
- Worker loop does not invoke speculative decoding.

**Gaps:**
- ⚠️ Module exists but not integrated into the main inference loop.
- ❌ No draft model loading mechanism.
- ❌ No API parameter to enable/disable speculative decoding.
- ❌ Worker sampling is deterministic; no parallel verification logic.

**Impact:** Advertised performance feature is non-functional.

---

### 1.6 **Observability & Monitoring**

**Current State:**
- Prometheus metrics endpoint exists (`GET /metrics`).
- `EngineMetrics` (src/metrics.rs) tracks counters plus TTFT and time-between-tokens histograms and KV cache usage gauge, so the core metric set is largely implemented.
- Architecture mentions queue depth and cache hit/miss gauges, which exist.

**Gaps:**
- ⚠️ No tracing/span integration for distributed tracing (e.g., Jaeger, OTLP).
- ⚠️ No structured logging for request lifecycle events.
- ⚠️ No dashboards or Grafana examples provided.
- ⚠️ No SLO definitions or alerting guidelines.
- ❌ Error rates and latency percentiles not documented.

**Impact:** Production deployments will struggle to debug performance issues (partial — core metrics exist, but tracing/alerting are missing).

---

### 1.7 **API & Request Validation**

**Current State:**
- OpenAI-compatible `/v1/chat/completions` endpoint.
- Request validation in place (model name, max_tokens, temperature, message count, prompt size).
- Grammar-constrained decoding (JSON mode) implemented.

**Gaps:**
- ⚠️ No OpenAI API versioning or backwards-compatibility strategy.
- ⚠️ JSON schema validation not fully explored (grammar module exists but integration unclear).
- ⚠️ No support for `functions` or `tools` parameters (common in OpenAI API).
- ⚠️ No batching hints or priority queues for requests.
- ⚠️ No timeout or cancellation mechanism for long-running requests.

**Impact:** API is incomplete compared to vLLM/TGI; limited use cases.

---

### 1.8 **Model Support & Compatibility**

**Current State:**
- Primary model: **Llama** (LlamaModel, LlamaConfig).
- Code supports Safetensors and GGUF loading.
- Vision model stub exists (`src/model/vision.rs`).
- MoE (Mixture of Experts) stub exists (`src/model/moe.rs`).

**Gaps:**
- ⚠️ Only Llama architecture fully implemented; no other transformers (Mistral, Mixtral, Qwen, etc.).
- ⚠️ Vision models are stubs; no multimodal support.
- ⚠️ MoE support incomplete; no router logic or sparse computation.
- ❌ No documentation on adding new model architectures.
- ❌ No compatibility matrix (which model versions work?).

**Impact:** Severely limits model ecosystem; users with non-Llama models cannot use Kyro.

---

### 1.9 **Error Handling & Recovery**

**Current State:**
- `error.rs` exists; ApiError type defined.
- Worker loop logs errors but does not panic; continues on errors.
- Configuration validation at startup.

**Gaps:**
- ⚠️ No circuit breaker or graceful degradation (e.g., if scheduler fails, what happens?).
- ⚠️ No retry logic for transient failures.
- ⚠️ No health check beyond liveness; readiness probe not implemented.
- ⚠️ OOM or device memory errors may crash worker silently.
- ⚠️ No error budget or SLO tracking.

**Impact:** Production deployments may experience silent failures or cascading errors.

---

### 1.10 **Documentation & Examples**

**Current State:**
- **Comprehensive tutorials** (14 modules covering pattern matching → serving).
- README with feature overview and getting-started instructions.
- Architecture, API, limits, and deployment docs exist.
- Rust examples for core LLM primitives.

**Gaps:**
- ⚠️ Tutorials are excellent but server-side features (LoRA, speculative decoding, distributed) not covered.
- ⚠️ No troubleshooting guide.
- ⚠️ No production deployment checklist.
- ⚠️ No performance tuning guide (batch size, KV cache size, etc.).
- ⚠️ No cost/throughput analysis.

**Impact:** Users will struggle to deploy and optimize Kyro in production.

---

## Section 2: Prioritized Improvement Plan

### **Priority Tier 1: Critical (Blocks Production Deployment)**

#### 1. **Implement & Validate Distributed Inference (TP/PP)**
- **Effort:** 4–6 weeks
- **Owner:** Distributed Systems Team
- **Tasks:**
  - Flesh out `src/distributed.rs` with NCCL initialization and AllReduce primitives.
  - Implement tensor parallelism: weight sharding, forward/backward splits, allreduce at layer boundaries.
  - Implement pipeline parallelism: layer assignment to ranks, activation checkpointing, bubble minimization.
  - Add unit tests for weight sharding and communication patterns.
  - Add integration test for 2–4 GPU setup (local or CI-friendly).
  - Benchmark single-GPU vs. multi-GPU throughput & latency.
  - **Acceptance Criteria:**
    - TP+PP work end-to-end on 2+ GPUs.
    - Throughput scales >80% with 2 GPUs.
    - Integration test passes in CI.

#### 2. **Expand Unit & Integration Test Suite**
- **Effort:** 2–3 weeks
- **Owner:** QA/Testing Team
- **Tasks:**
   - Add unit tests for `scheduler/continuous_batching.rs` (prefill scheduling, token generation).
   - Add unit tests for `scheduler/block_manager.rs` (block allocation/freeing, fragmentation).
   - Add unit tests for `scheduler/radix_cache.rs` (prefix matching, eviction).
   - Existing API-level integration tests (streaming, validation, metrics) should be kept; extend with multi-request concurrency and cancellation cases.
   - Add coverage threshold gate to CI; add badge to README.
  - **Acceptance Criteria:**
    - >70% code coverage for critical modules.
    - All tests pass in CI.
    - Stress test runs without panics.

#### 3. **Implement & Document Quantization Paths**
- **Effort:** 2–3 weeks
- **Owner:** Model Optimization Team
- **Tasks:**
  - Audit `src/model/quantization/` structure; document which formats are complete vs. stubs.
  - Implement missing quantization loaders (FP8, AWQ, GGUF if not already present).
  - Add examples demonstrating how to load quantized models.
  - Add benchmarks comparing throughput/latency across quantization types.
  - Document in README and deployment guide.
  - **Acceptance Criteria:**
    - All three quantization types (FP8, AWQ, GGUF) can load real models.
    - Examples run without errors.
    - Benchmarks show expected throughput gains.

---

### **Priority Tier 2: High (Major Feature Gaps)**

#### 4. **Integrate LoRA Support**
- **Effort:** 2–3 weeks
- **Owner:** Model Adaptation Team
- **Tasks:**
  - Complete `src/model/lora.rs`: LoRA weight loading, merging with base model.
  - Extend model forward pass to apply LoRA projections.
  - Add scheduler support for tracking active LoRA per request.
  - Add API endpoint or parameter to specify LoRA ID (`lora_id` in request).
  - Add examples and tests.
  - **Acceptance Criteria:**
    - LoRA weights load and apply correctly.
    - Multi-request handling with different LoRA configs works.
    - Example runs end-to-end.

#### 5. **Integrate Speculative Decoding**
- **Effort:** 2–3 weeks
- **Owner:** Performance Optimization Team
- **Tasks:**
  - Load and manage draft model separate from target model.
  - Implement parallel verification in worker loop.
  - Add scheduler logic to decide when to use speculative decoding.
  - Add API parameter to enable/disable (`use_speculative: bool`).
  - Add benchmarks showing 2x speedup claim.
  - **Acceptance Criteria:**
    - Speculative decoding produces correct outputs.
    - Benchmarks show speedup (target: >1.5x).
    - API parameter works end-to-end.

#### 6. **Expand Model Ecosystem**
- **Effort:** 3–4 weeks
- **Owner:** Model Support Team
- **Tasks:**
  - Add support for Mistral 7B/Mixtral architecture.
  - Add support for Qwen architecture.
  - Create model architecture abstraction/trait to ease future additions.
  - Add per-model config examples.
  - Document compatibility matrix (model → Kyro version).
  - **Acceptance Criteria:**
    - 3+ model architectures work end-to-end.
    - Examples provided for each.
    - Architecture trait adopted for all models.

#### 7. **Enhanced Observability & Monitoring**
- **Effort:** 2–3 weeks
- **Owner:** DevOps/Observability Team
- **Tasks:**
  - Complete Prometheus metrics (histograms for TTFT/TBT, gauge for KV cache).
  - Add structured logging with `tracing::span!` for request lifecycle.
  - Add optional OpenTelemetry exporter (OTLP).
  - Provide Grafana dashboard JSON example.
  - Document SLOs and alerting rules.
  - **Acceptance Criteria:**
    - Metrics endpoint exposes all promised metrics.
    - Logs include trace IDs and structured fields.
    - Grafana dashboard shows key insights.

---

### **Priority Tier 3: Medium (Nice-to-Have, UX Improvements)**

#### 8. **Improve Error Handling & Resilience**
- **Effort:** 1–2 weeks
- **Tasks:**
  - Add circuit breaker for scheduler failures.
  - Implement graceful shutdown (drain in-flight requests).
  - Add readiness probe (checks model load, scheduler health).
  - Add retry logic for transient errors.
  - Add detailed error codes/messages for debugging.

#### 9. **Production Deployment Guide & Checklist**
- **Effort:** 1 week
- **Tasks:**
  - Document performance tuning (batch size, KV cache, device memory).
  - Provide Docker Compose or Kubernetes manifests.
  - Add security best practices (secrets, TLS, rate limiting).
  - Troubleshooting guide for common issues.

#### 10. **Enhanced API Compatibility**
- **Effort:** 2–3 weeks
- **Tasks:**
   - Add support for `functions` / `tools` parameters.
   - Add request queuing with priority/SLA hints.
   - Add request cancellation (cancel by ID). — **DONE**: `POST /v1/cancel` + `X-Request-Id` response header.
   - Add timeout handling. — **DONE**: `KYRO_REQUEST_TIMEOUT_SECS` → 504 on expiry.

---

## Section 3: Short-Term Action Items (Next 4 Weeks)

1. **Create GitHub Issues** for each Priority Tier 1 & 2 item (10 issues).
2. **Tighten Test CI:** — **DONE**: CI enforces `--fail-under-lines 70` (measured 70.40%); scheduler/block-manager unit tests in place.
3. **Audit Quantization & Distributed Code:** — **DONE**: `docs/implementation_status.md` documents complete vs. stub per feature; README claims retracted accordingly.
4. **Kick Off Parallel Efforts:** — testing (Tier 1 #2) and documentation (Tier 3 #9) tracks completed this session.

---

## Section 4: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Distributed inference fails at scale | Medium | Critical | Early multi-GPU testing in CI; load test with 4+ GPUs |
| Quantization regressions undetected | Medium | High | Quantization-specific unit & benchmark tests |
| Undocumented API changes break users | Low | Medium | Semantic versioning; changelog enforcement |
| Silent worker loop failures | Medium | High | Enhanced logging; health probes; SLO dashboards |
| LoRA/speculative features conflict | Low | Medium | Integration tests covering feature combinations |

---

## Section 5: Success Metrics

By EOQ (end of quarter):
- ✅ Test coverage >70% (Tier 1 #2) — **achieved 70.40%, CI-gated**.
- ⬜ Distributed inference works on 2+ GPUs (Tier 1 #1) — blocked on multi-GPU hardware.
- 🔶 Quantization paths fully documented & working (Tier 1 #3) — GGUF done; AWQ/FP8 stubs documented in `docs/implementation_status.md`.
- 🔶 LoRA and speculative decoding integrated (Tier 2 #4, #5) — unit-tested; integration pending.
- 🔶 Production deployment guide published (Tier 3 #9) — troubleshooting + SLO/alerting guides added.
- ⬜ 3+ model architectures supported (Tier 2 #6).
- 🔶 Comprehensive monitoring/alerting setup (Tier 2 #7) — Grafana dashboard + SLO rules added; OTLP pending.

---

## Section 6: Recommended Roadmap

```
Week 1–2:   Audit code; create GitHub issues; set up CI for tests.
Week 3–4:   Start Tier 1 #2 (tests) and Tier 3 #9 (deployment guide) in parallel.
Week 5–8:   Focus Tier 1 #1 (distributed) and #3 (quantization).
Week 9–12:  Tier 2 items (#4, #5, #6, #7) in parallel.
Week 13+:   Polish, optimization, and community feedback.
```

---

## Conclusion

**Kyro is a well-architected early-stage LLM serving engine with significant potential.** The core serving infrastructure (continuous batching, prefix caching, chunked prefill) is solid and production-ready for single-GPU Llama deployments. Testing gaps are now closed (70.40% coverage, CI-gated); **advertised features (distributed inference, LoRA, speculative decoding, AWQ/FP8 quantization) remain incomplete or missing** and have been re-labeled in the README to match reality.

**Immediate priorities** are:
1. ~~Close testing gaps (unit + integration).~~ ✅ Done — 73 tests, 70.40% coverage, CI gate at 70%.
2. Complete distributed inference (TP/PP).
3. Validate & document quantization. — GGUF done; AWQ/FP8 audit documented.
4. Integrate LoRA and speculative decoding.
5. Expand model support.

**Success will require 12–16 weeks of focused engineering effort** across distributed systems, model optimization, and DevOps. The team should adopt semantic versioning and maintain a public roadmap to manage user expectations around feature completeness.

