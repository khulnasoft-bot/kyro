# Module 0 — Onboarding & Repo Grounding

## Learning Objectives
- Clone and build the Kyro repo locally
- Run the Kyro server and verify it responds to `/health` and `/v1/chat/completions`
- Understand the high-level architecture
- Run the benchmarking script

## Architecture Overview

Kyro is a high-throughput LLM serving engine written in Rust, inspired by vLLM and TGI.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Axum API   │────▶│  Scheduler   │────▶│   Candle Model   │
│  (openai.rs) │     │(continuous_  │     │  (llama.rs)      │
│              │     │  batching)   │     │                  │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │                      │
                    ┌──────▼──────┐        ┌──────▼──────┐
                    │ BlockManager│        │  Worker Loop │
                    │ (paged attn)│        │ (worker.rs)  │
                    └─────────────┘        └─────────────┘
```

Key components:
- **Frontend** (`src/api/openai.rs`): Axum HTTP server with SSE streaming
- **Scheduler** (`src/scheduler/continuous_batching.rs`): Continuous batching with prefix caching
- **Model** (`src/model/llama.rs`): Llama decoder with PagedAttention via candle
- **KV Cache** (`src/scheduler/block_manager.rs`): PagedAttention + Radix Cache
- **Worker** (`src/worker.rs`): Async inference loop with sampling
- **Distributed** (`src/distributed.rs`): Pipeline/Tensor parallelism context

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/nrelab/kyro.git
cd kyro
```

### 2. Build with Cargo

```bash
# CPU build (fast, works everywhere)
cargo build --release

# GPU build (requires CUDA toolkit)
cargo build --release --features cuda
```

### 3. Run the Server

```bash
cargo run --release
```

The server starts on `http://localhost:3000`.

### 4. Verify Health Endpoint

```bash
curl http://localhost:3000/health
# Expected output: OK
```

### 5. Test Chat Completions

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kyro-llama3",
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "max_tokens": 10,
    "stream": false
  }'
```

### 6. Run the Benchmark

```bash
python benchmarks/stress_test.py
```

## Project Structure

```
kyro/
├── Cargo.toml              # Rust package manifest
├── src/
│   ├── main.rs             # Entry point
│   ├── api/
│   │   ├── openai.rs       # HTTP server & chat completions
│   │   ├── tokenizer.rs    # LuminaTokenizer wrapper
│   │   └── grammar.rs      # Constrained decoding
│   ├── model/
│   │   ├── llama.rs        # Llama model implementation
│   │   ├── config.rs       # Model configuration
│   │   ├── loader.rs       # Model loading (GGUF/safetensors)
│   │   └── pipeline.rs     # Pipeline parallelism context
│   ├── scheduler/
│   │   ├── continuous_batching.rs  # Request scheduling
│   │   ├── block_manager.rs        # PagedAttention block management
│   │   └── radix_cache.rs          # Prefix caching
│   ├── worker.rs            # Inference worker loop
│   ├── distributed.rs       # Distributed context
│   └── speculative.rs       # Speculative decoding
├── benchmarks/
│   └── stress_test.py       # Load testing script
├── tutorials/               # Interactive Jupyter notebooks
└── examples/                # Rust example programs
```

## Docker / Devcontainer

See [`../devcontainer.json`](../../devcontainer.json) for a reproducible development environment.

## What's Next?

- **Module 1**: Pattern matching & n-grams — understand statistical language modeling foundations
- **Module 2**: Tokenization — learn how BPE tokenizers work and connect to Kyro's tokenizer
- **Module 3**: Embeddings — map tokens to vectors and understand the first stage of the LLM pipeline

Each module includes a runnable Python notebook and a Rust example that mirrors Kyro's internals.
