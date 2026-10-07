# Kyro Improvement Plan: GitHub Issues Backlog

**Last Updated:** October 7, 2026  
**Status:** Ready for implementation  
**Total Issues:** 10 (Priority 1: 3, Priority 2: 4, Priority 3: 3)

---

## Priority 1: Production Blockers (Weeks 1–8)

These issues block reliable production deployment. Schedule them in parallel where possible.

### Issue 1: Implement Distributed Inference (Tensor & Pipeline Parallelism)

**Title:**  
`[P1] Implement distributed inference: tensor parallelism and pipeline parallelism`

**Labels:**  
`priority:critical`, `type:feature`, `area:distributed`, `effort:4-6w`

**Description:**

Distributed inference (multi-GPU support) is advertised in the README but currently not implemented. The `src/distributed.rs` file is a stub containing only `rank` and `world_size` fields.

**Problem:**
- Users expecting multi-GPU serving will experience silent failures.
- The system always runs with `rank: 0, world_size: 1`.
- No NCCL or AllReduce primitives are integrated.
- No documentation on distributed configuration.

**Acceptance Criteria:**
- [ ] Tensor Parallelism (TP) is implemented and tested
  - Weight sharding across GPUs
  - Forward pass splits across layer boundaries
  - AllReduce at layer boundaries works correctly
- [ ] Pipeline Parallelism (PP) is implemented and tested
  - Layer assignment to ranks is automatic
  - Activation transfer between pipeline stages works
  - Bubble minimization is handled
- [ ] 2-GPU integration test passes in CI
  - Test covers both TP and PP modes
  - Throughput scaling is measured (target: >80% with 2 GPUs)
- [ ] Documentation added to `docs/distributed.md`
  - Configuration guide (how to enable TP/PP)
  - Example commands for 2, 4, 8 GPU setup
  - Troubleshooting common issues

**Tasks:**
1. Flesh out `src/distributed.rs` with NCCL initialization
2. Implement tensor parallelism: weight sharding, forward/backward splits
3. Implement pipeline parallelism: layer assignment, activation checkpointing
4. Add unit tests for weight sharding and communication patterns
5. Add 2–4 GPU integration test (can use CI-friendly GPU simulation if needed)
6. Benchmark single-GPU vs. multi-GPU throughput and latency
7. Write distributed inference documentation and examples

**Effort:** 4–6 weeks  
**Owner:** Distributed Systems Team  
**Depends On:** None

---

### Issue 2: Expand Unit & Integration Test Suite with Coverage Gates

**Title:**  
`[P1] Achieve >75% coverage on critical modules; enforce coverage gate in CI`

**Labels:**  
`priority:critical`, `type:test`, `area:testing`, `effort:2-3w`

**Description:**

Test coverage for critical runtime paths (scheduler, block manager, cache) is still insufficient. While overall coverage is at 70.40%, the most failure-prone modules need higher confidence.

**Problem:**
- Scheduler internals (`continuous_batching.rs`, `block_manager.rs`, `radix_cache.rs`) lack unit test coverage
- No performance regression tests or SLO validation in CI
- Silent regressions in scheduler/cache code could cause production incidents
- No visible coverage badge or enforcement threshold

**Acceptance Criteria:**
- [ ] Unit tests added for scheduler modules
  - Continuous batching (prefill scheduling, token generation)
  - Block manager (allocation, freeing, fragmentation)
  - Radix cache (prefix matching, eviction logic)
- [ ] Coverage threshold is enforced in CI
  - Scheduler modules: >90% coverage
  - API and model modules: >85% coverage
  - Overall: >75% coverage
- [ ] Integration tests extended
  - Multi-request concurrency scenarios
  - Request cancellation workflows
  - Timeout and deadline scenarios
- [ ] Stress test runs in CI without panics
- [ ] Coverage badge added to README

**Tasks:**
1. Write unit tests for `scheduler/continuous_batching.rs`
2. Write unit tests for `scheduler/block_manager.rs`
3. Write unit tests for `scheduler/radix_cache.rs`
4. Write integration tests for concurrency, cancellation, timeouts
5. Add CI coverage gate: `--fail-under-lines 75`
6. Add coverage badge to README
7. Document coverage expectations for new code

**Effort:** 2–3 weeks  
**Owner:** QA / Testing Team  
**Depends On:** None

---

### Issue 3: Audit, Document, and Complete Quantization Support

**Title:**  
`[P1] Complete quantization implementation and retract unsupported claims from README`

**Labels:**  
`priority:critical`, `type:feature`, `area:quantization`, `effort:2-3w`

**Description:**

README advertises FP8, AWQ, and GGUF support. Audit reveals that only GGUF is real; AWQ and FP8 are mock implementations.

**Problem:**
- AWQ and FP8 are no-op casts, not real quantization
- Users loading AWQ/FP8 models will experience silent failures or incorrect results
- No end-to-end examples for GGUF loading
- No benchmarks comparing quantization modes
- README claims are misleading and damage trust

**Acceptance Criteria:**
- [ ] GGUF support is fully validated
  - End-to-end example runs without errors
  - Accuracy is correct (validated against reference)
  - Throughput gain is measured and documented
- [ ] AWQ and FP8 status is clarified
  - Either: fully implemented with benchmarks
  - Or: removed from README and marked as unsupported in code comments
- [ ] Quantization examples added to `examples/`
  - GGUF model loading example
  - Benchmark script comparing model sizes
- [ ] Documentation updated
  - `docs/quantization.md` lists supported formats
  - Compatibility matrix added (model variant → format support)
- [ ] README claims are truthful and current

**Tasks:**
1. Audit `src/model/quantization/` for actual vs. mock implementations
2. Either complete AWQ/FP8 or remove from advertised features
3. Add end-to-end GGUF loading example
4. Add quantization benchmarks (throughput, latency, memory)
5. Update README to match implementation status
6. Write `docs/quantization.md` with format compatibility matrix

**Effort:** 2–3 weeks  
**Owner:** Model Optimization Team  
**Depends On:** Issue #2 (for benchmark infrastructure)

---

## Priority 2: High-Value Feature Gaps (Weeks 5–12)

Schedule these in parallel after Priority 1 foundations are in place.

### Issue 4: Integrate LoRA Support into Inference Pipeline

**Title:**  
`[P2] Integrate LoRA weight loading and request-level adapter selection`

**Labels:**  
`priority:high`, `type:feature`, `area:model-adaptation`, `effort:2-3w`

**Description:**

LoRA support is advertised and partially coded, but is not integrated into the model forward pass or exposed to the API.

**Problem:**
- LoRA module exists but is not called during inference
- No API parameter or request metadata for LoRA selection
- Scheduler does not track active adapter per request
- Users cannot use task-specific adapters

**Acceptance Criteria:**
- [ ] LoRA weight loading is complete
  - Weights can be loaded from safetensors or HuggingFace
  - Multiple adapters can be registered at startup
- [ ] Model forward pass applies LoRA projections
  - LoRA Linear layers are integrated into LlamaModel
  - Forward pass produces correct outputs
- [ ] Request-level adapter selection works
  - API accepts `lora_id` or `adapter_name` parameter
  - Scheduler tracks active adapter per request
- [ ] Concurrent requests with different adapters don't interfere
- [ ] Example demonstrates loading and using LoRA adapters
- [ ] Unit tests cover LoRA weight application

**Tasks:**
1. Complete `src/model/lora.rs`: weight loading, merging logic
2. Integrate LoRA into `LlamaModel::forward()`
3. Add scheduler support for tracking active LoRA per request
4. Add API field to request schema for LoRA selection
5. Add request-to-adapter mapping in worker loop
6. Write LoRA loading and inference example
7. Add unit tests for LoRA weight application

**Effort:** 2–3 weeks  
**Owner:** Model Adaptation Team  
**Depends On:** Issue #2 (testing infrastructure)

---

### Issue 5: Integrate Speculative Decoding into Generation Loop

**Title:**  
`[P2] Integrate speculative decoding with draft model and verification`

**Labels:**  
`priority:high`, `type:feature`, `area:performance`, `effort:2-3w`

**Description:**

Speculative decoding is advertised to provide 2x speedup but is not active in the main generation loop.

**Problem:**
- Draft model loading mechanism is missing
- Worker loop does not invoke speculative decoding logic
- No API parameter to enable/disable speculative decoding
- Speedup claim cannot be validated

**Acceptance Criteria:**
- [ ] Draft model loading is implemented
  - Draft model is loaded separately from target model
  - Draft model can be configured at startup or request time
- [ ] Speculative decoding works in worker loop
  - Draft model generates candidate tokens in parallel
  - Target model verifies candidates
  - Rejected tokens are re-sampled from target
- [ ] Outputs are identical to non-speculative baseline
- [ ] API parameter enables/disables speculative decoding
- [ ] Benchmarks show measured speedup
  - Target: >1.5x (claim is 2x)
  - Include latency and throughput metrics
- [ ] Unit tests verify correctness

**Tasks:**
1. Implement draft model loading and management
2. Add speculative decoding logic to worker sampling loop
3. Implement parallel verification and rejection handling
4. Add API parameter `use_speculative: bool` to request schema
5. Add scheduler logic to decide when speculative is beneficial
6. Write speculative decoding example
7. Add benchmarks comparing speculative vs. baseline
8. Unit tests for draft model correctness

**Effort:** 2–3 weeks  
**Owner:** Performance Optimization Team  
**Depends On:** Issue #2 (for benchmark infrastructure)

---

### Issue 6: Expand Model Ecosystem Beyond Llama

**Title:**  
`[P2] Add support for Mistral, Mixtral, and Qwen model architectures`

**Labels:**  
`priority:high`, `type:feature`, `area:model-support`, `effort:3-4w`

**Description:**

Only Llama architecture is currently supported. Adding 2–3 more model families broadens usability and establishes a pattern for future expansion.

**Problem:**
- Non-Llama users cannot use Kyro
- No abstraction layer to simplify adding new architectures
- Vision and MoE stubs exist but are not completed
- No compatibility matrix or architecture guide

**Acceptance Criteria:**
- [ ] Mistral/Mixtral architecture is supported
  - Model loading works for all common sizes
  - Inference produces correct outputs
  - Example provided
- [ ] Qwen architecture is supported
  - Model loading works
  - Inference produces correct outputs
  - Example provided
- [ ] Model abstraction trait is defined and used
  - All models implement a common `ModelArchitecture` trait
  - Loader dispatches to correct architecture
  - New architectures can be added without modifying core code
- [ ] Compatibility matrix published
  - Lists which model versions work with Kyro
  - Documents any limitations per architecture
- [ ] Examples added for each architecture
  - Model loading example
  - Inference example with streaming

**Tasks:**
1. Define `ModelArchitecture` trait for model abstraction
2. Refactor Llama to use new trait
3. Implement Mistral/Mixtral architecture
4. Implement Qwen architecture
5. Add examples for each architecture
6. Write architecture addition guide for future contributors
7. Publish compatibility matrix in docs/

**Effort:** 3–4 weeks  
**Owner:** Model Support Team  
**Depends On:** Issue #2 (for testing infrastructure)

---

### Issue 7: Complete Observability & Tracing Infrastructure

**Title:**  
`[P2] Add structured logging, OpenTelemetry tracing, and production dashboards`

**Labels:**  
`priority:high`, `type:infrastructure`, `area:observability`, `effort:2-3w`

**Description:**

Prometheus metrics are in place, but distributed tracing, structured logs, and alerting guidance are missing. Production deployments cannot diagnose issues without this visibility.

**Problem:**
- No trace IDs or request correlation across services
- Logs are unstructured and lack lifecycle events
- No OpenTelemetry exporter for integration with observability platforms
- No dashboards or alerting rules
- SLO definitions and error budgets are not published

**Acceptance Criteria:**
- [ ] Structured logging is implemented
  - Request lifecycle events are logged (enqueue, prefill, generation, complete)
  - Logs include trace ID, request ID, and structured fields
  - Error messages include context for debugging
- [ ] Distributed tracing is implemented
  - Optional OTLP exporter (can be disabled)
  - Request spans include latency breakdowns
  - Integration with Jaeger/Datadog/similar works
- [ ] Grafana dashboard is provided
  - Shows throughput, latency percentiles, queue depth, cache hit rate
  - Includes error rate and failed request graphs
  - Actionable and easy to interpret
- [ ] SLO definitions are documented
  - TTFT targets (e.g., p50 < 100ms, p99 < 500ms)
  - Generation throughput targets
  - Cache hit rate targets
- [ ] Alerting rules are provided
  - Template alert rules for common thresholds
  - Runbook links for oncall responders

**Tasks:**
1. Add `tracing` instrumentation to request lifecycle
2. Implement optional OTLP exporter
3. Ensure all logs are structured with key-value pairs
4. Create Grafana dashboard JSON (improvements to existing)
5. Write SLO definitions in `docs/slos.md`
6. Write alert rules and runbooks in `docs/alerting.md`
7. Document how to set up observability in `docs/observability.md`

**Effort:** 2–3 weeks  
**Owner:** DevOps / Observability Team  
**Depends On:** None (can be done in parallel)

---

## Priority 3: Quality-of-Life Improvements (Weeks 9–14)

These improve reliability, operational ease, and user experience.

### Issue 8: Improve Error Handling & Resilience

**Title:**  
`[P3] Add circuit breaker, graceful shutdown, and health probes`

**Labels:**  
`priority:medium`, `type:reliability`, `area:error-handling`, `effort:1-2w`

**Description:**

Worker loop continues on errors, but lacks graceful degradation and recovery mechanisms for production stability.

**Problem:**
- No circuit breaker for cascading failures
- Graceful shutdown does not drain in-flight requests
- Readiness probe is incomplete
- Retry logic for transient failures is absent
- Error codes are not standardized

**Acceptance Criteria:**
- [ ] Readiness probe is complete
  - Checks model loading status
  - Checks scheduler health
  - Returns 200 when ready, 503 when not
- [ ] Graceful shutdown implemented
  - SIGTERM/SIGINT signals drain in-flight requests
  - New requests are rejected with 503
  - Timeout for drain period is configurable
- [ ] Circuit breaker added for scheduler
  - Fails fast if scheduler reaches error threshold
  - Monitors for recovery
- [ ] Retry logic for transient errors
  - Retryable errors (e.g., OOM) retry with backoff
  - Non-retryable errors fail immediately
- [ ] Error codes are standardized
  - Each error type has a code (e.g., E_SCHEDULER_OVERLOAD)
  - Errors include actionable messages

**Tasks:**
1. Implement complete readiness probe
2. Add graceful shutdown handler
3. Add circuit breaker for scheduler failures
4. Implement retry logic with exponential backoff
5. Standardize error codes and messages
6. Add tests for shutdown and recovery scenarios
7. Document error codes in `docs/errors.md`

**Effort:** 1–2 weeks  
**Owner:** Runtime / DevOps Team  
**Depends On:** None

---

### Issue 9: Production Deployment Guide & Checklist

**Title:**  
`[P3] Write comprehensive deployment guide, security best practices, and troubleshooting`

**Labels:**  
`priority:medium`, `type:documentation`, `area:deployment`, `effort:1-2w`

**Description:**

Users lack clear guidance on deploying Kyro to production, tuning performance, and debugging issues.

**Problem:**
- No production deployment checklist
- No performance tuning guide (batch size, cache size, etc.)
- No security best practices documentation
- Troubleshooting guide is incomplete
- No Docker Compose or Kubernetes manifests (or they're not well-documented)

**Acceptance Criteria:**
- [ ] Deployment guide covers
  - Hardware requirements and recommendations
  - Model selection and downloading
  - Configuration tuning (batch size, KV cache, device memory)
  - Environment variables and secrets management
  - TLS/HTTPS setup
- [ ] Kubernetes manifests provided
  - Deployment, Service, ConfigMap, Secret examples
  - Resource requests and limits guidance
  - Readiness and liveness probe configuration
- [ ] Security best practices documented
  - API authentication options
  - Rate limiting
  - Network isolation
- [ ] Troubleshooting guide covers
  - Common startup failures (missing models, OOM)
  - Performance issues (low throughput, high latency)
  - Memory issues and KV cache tuning
  - Distributed inference troubleshooting
- [ ] Performance tuning guide provided
  - How to choose batch size
  - How to size KV cache
  - Trade-offs between throughput and latency

**Tasks:**
1. Write `docs/deployment.md` with hardware and config guidance
2. Write `docs/kubernetes.md` with K8s deployment examples
3. Write `docs/security.md` with best practices
4. Enhance `docs/troubleshooting.md` with common issues
5. Write `docs/performance-tuning.md` with configuration guidance
6. Create Docker Compose examples in `deploy/`
7. Update README with deployment quickstart

**Effort:** 1–2 weeks  
**Owner:** DevOps / Documentation Team  
**Depends On:** None

---

### Issue 10: Enhanced API Compatibility with OpenAI

**Title:**  
`[P3] Add functions/tools support, priority queueing, and request management`

**Labels:**  
`priority:medium`, `type:feature`, `area:api`, `effort:2-3w`

**Description:**

API is OpenAI-compatible at a basic level but lacks advanced features like functions, tools, and request prioritization.

**Problem:**
- No support for `functions` or `tools` parameters
- No request priority queueing (all requests treated equally)
- Limited request management features
- No backward compatibility strategy documented

**Acceptance Criteria:**
- [ ] `functions` / `tools` parameter support is added
  - Schema validation works
  - Function calling is compatible with OpenAI format
  - Example provided
- [ ] Priority queueing is implemented
  - Requests can specify priority or SLA hints
  - High-priority requests are served faster
  - SLA deadlines are respected (best-effort)
- [ ] Request management is complete
  - Request cancellation works (`POST /v1/cancel` + `X-Request-Id`)
  - Timeout handling is robust (`KYRO_REQUEST_TIMEOUT_SECS`)
  - Request IDs are generated and returned
- [ ] Backward compatibility strategy is documented
  - API versioning plan (e.g., v1 stable)
  - Deprecation policy for features
  - Changelog is maintained

**Tasks:**
1. Add schema validation for functions/tools
2. Implement function calling compatibility layer
3. Add priority field to request schema
4. Implement priority queue in scheduler
5. Verify request cancellation and timeout handling
6. Write API versioning and compatibility guide
7. Add examples for functions/tools usage

**Effort:** 2–3 weeks  
**Owner:** API / Platform Team  
**Depends On:** None (some parts depend on Issue #8 for graceful timeout handling)

---

## Summary Table

| Issue # | Title | Priority | Effort | Owner | Status |
|---------|-------|----------|--------|-------|--------|
| 1 | Distributed Inference (TP/PP) | P1 | 4–6w | Distributed Systems | Open |
| 2 | Test Suite & Coverage Gates | P1 | 2–3w | QA/Testing | Open |
| 3 | Quantization Audit & Completion | P1 | 2–3w | Model Optimization | Open |
| 4 | LoRA Integration | P2 | 2–3w | Model Adaptation | Open |
| 5 | Speculative Decoding Integration | P2 | 2–3w | Performance | Open |
| 6 | Model Ecosystem Expansion | P2 | 3–4w | Model Support | Open |
| 7 | Observability & Tracing | P2 | 2–3w | DevOps | Open |
| 8 | Error Handling & Resilience | P3 | 1–2w | Runtime | Open |
| 9 | Deployment Guide | P3 | 1–2w | DevOps | Open |
| 10 | API Compatibility | P3 | 2–3w | API/Platform | Open |

---

## How to Use This Backlog

1. **Create GitHub Issues:** Copy each issue description above into GitHub issues with the specified labels.
2. **Assign Owners:** Assign each issue to the team/person responsible.
3. **Prioritize in Milestones:** See `MILESTONE_PLAN.md` for recommended scheduling.
4. **Track Progress:** Update issue status as work progresses.
5. **Link Dependencies:** Use GitHub issue linking to show dependencies between issues.

---

## Next Steps

1. [ ] Create GitHub issues from this backlog
2. [ ] Assign owners and priorities
3. [ ] Schedule into milestones (see `MILESTONE_PLAN.md`)
4. [ ] Begin Priority 1 work

