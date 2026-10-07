# Kyro Milestone Plan

**Date:** October 7, 2026  
**Repository:** nrelab/kyro  
**Objective:** Sequence the work from production blockers to feature expansion and operational hardening.

---

## Milestone 1: Production Blockers

**Target Window:** Week 1–8  
**Theme:** Stabilize the runtime and remove the highest-risk product blockers.

### Goals
- Make the serving stack reliable for production deployment
- Validate the runtime under real concurrency and failure scenarios
- Remove misleading product claims about unsupported features
- Establish a trustworthy baseline before feature expansion

### Included Issues
- Issue 1: Implement Distributed Inference (TP/PP)
- Issue 2: Expand Unit & Integration Test Suite with Coverage Gates
- Issue 3: Audit, Document, and Complete Quantization Support

### Deliverables
- Multi-GPU serving path validated on at least 2 GPUs
- CI-enforced test coverage for critical modules
- Quantization support claims aligned with actual capabilities
- Runtime documentation for distributed and quantized deployment

### Exit Criteria
- Distributed inference has a working end-to-end path on 2+ GPUs
- Critical module coverage exceeds the agreed threshold
- All advertised quantization paths are either implemented or explicitly marked unsupported
- Engineering team can deploy a stable single-node baseline without blocking issues

### Owner
- Distributed Systems Team
- QA / Testing Team
- Model Optimization Team

### Suggested Date Range
- Start: 2026-10-12
- End: 2026-11-27

---

## Milestone 2: High-Value Features

**Target Window:** Week 5–12  
**Theme:** Unlock the most important user-facing capabilities that expand product value.

### Goals
- Add the feature set most likely to drive adoption and differentiation
- Convert partially implemented modules into active runtime features
- Expand the model ecosystem beyond Llama
- Improve the engineer and user experience with stronger observability

### Included Issues
- Issue 4: Integrate LoRA Support into Inference Pipeline
- Issue 5: Integrate Speculative Decoding into Generation Loop
- Issue 6: Expand Model Ecosystem Beyond Llama
- Issue 7: Complete Observability & Tracing Infrastructure

### Deliverables
- Working LoRA support and request-level adapter selection
- Functional speculative decoding path with verified speedup
- Support for at least 3 model families beyond the initial Llama baseline
- Structured logs, trace IDs, SLO documentation, and dashboard coverage

### Exit Criteria
- At least 3 model families run end-to-end in production-like scenarios
- LoRA workloads are stable under mixed request conditions
- Speculative decoding produces correct outputs with measurable throughput gain
- Request traces and metrics are useful for runtime debugging and incident response

### Owner
- Model Adaptation Team
- Performance Optimization Team
- Model Support Team
- DevOps / Observability Team

### Suggested Date Range
- Start: 2026-11-16
- End: 2027-01-08

---

## Milestone 3: Production Hardening and Ops

**Target Window:** Week 9–16  
**Theme:** Improve resilience, deployment readiness, and operational maturity.

### Goals
- Make Kyro production-safe for real-world operations
- Provide a predictable API and deployment story
- Ensure graceful failure behavior and easier troubleshooting
- Reduce operational risk for teams running the engine in production

### Included Issues
- Issue 8: Improve Error Handling & Resilience
- Issue 9: Production Deployment Guide & Checklist
- Issue 10: Enhanced API Compatibility with OpenAI

### Deliverables
- Readiness probes, graceful shutdown, retry logic, and circuit breakers
- Production deployment checklist and operational runbooks
- Kubernetes and Docker deployment examples
- API compatibility improvements for function/tool use and request controls

### Exit Criteria
- Production deployment documentation is complete and usable by operators
- Graceful degradation and shutdown behavior are tested and documented
- API supports the key compatibility features that users expect
- Common production failures are covered in troubleshooting and SLO guidance

### Owner
- Runtime / DevOps Team
- Documentation Team
- API / Platform Team

### Suggested Date Range
- Start: 2027-01-04
- End: 2027-02-05

---

## Dependency View

Milestone 1 provides the stable foundation for Milestone 2. Milestone 2 adds the feature depth that drives product adoption. Milestone 3 closes the operational gaps needed for reliability and scale in production.

- Milestone 1 blocks Milestone 2's success
- Milestone 2 informs the design for Milestone 3
- Milestone 3 is the final hardening layer before broader production rollout

---

## Recommended Execution Pattern

- Parallelize work within each milestone where dependencies allow
- Use issue-level tracking and milestone boards for progress visibility
- Treat readiness and exit criteria as mandatory before moving to the next milestone
- Keep feature claims aligned with implementation status throughout all milestones

---

## Summary

| Milestone | Focus | Target Window | Key Outcomes |
|-----------|-------|--------------|--------------|
| Milestone 1 | Production blockers | Weeks 1–8 | Stable runtime, working multi-GPU baseline, truthful quantization story |
| Milestone 2 | High-value features | Weeks 5–12 | LoRA, speculative decoding, model ecosystem expansion, observability |
| Milestone 3 | Production hardening and ops | Weeks 9–16 | Resilience, deployment maturity, API and operations readiness |

