# Kyro Interactive Tutorials

A staged, interactive learning walkthrough that takes you from simple pattern matching through training a transformer from scratch, all tied into the Kyro codebase.

## Table of Contents

| Module | Topic | Duration |
|--------|-------|----------|
| [00](00_onboarding.md) | Onboarding & Repo Grounding | 1–2 days |
| [01](01_pattern_matching.ipynb) | Pattern Matching & N-grams | 1–2 days |
| [02](02_tokenization.ipynb) | Tokenization (BPE, SentencePiece) | 2–3 days |
| [03](03_embeddings.ipynb) | Embeddings & Representation Basics | 2 days |
| [04](04_attention.ipynb) | Attention: Intuition & Small Implementation | 3–4 days |
| [05](05_transformer_block.ipynb) | Transformer Block & Decoder-Only Model | 4–5 days |
| [06](06_decoding.ipynb) | Autoregressive Decoding & Sampling | 3 days |
| [07](07_kv_cache.ipynb) | KV Cache, Prefix Caching, Chunked Prefill | 3 days |
| [08](08_speculative.ipynb) | Speculative Decoding | 3 days |
| [09](09_quantization.ipynb) | Quantization & Efficient Inference | 4 days |
| [10](10_data_pipeline.ipynb) | Dataset Pipeline & Preprocessing | 2–3 days |
| [11](11_training_loop.ipynb) | Loss, Optimization & Training Loop | 4–6 days |
| [12](12_scale_training.ipynb) | Scaling & Distributed Training | 2–3 weeks |
| [13](13_serving.md) | Serving the Trained Model with Kyro | 2–3 days |

## Prerequisites

- Rust toolchain (for building Kyro and running Rust examples)
- Python 3.10+ with `jupyter`, `numpy`, `matplotlib`
- CUDA (optional, for GPU acceleration)
- See [devcontainer.json](../devcontainer.json) for a complete environment spec

## Running the Notebooks

```bash
# Start Jupyter
jupyter notebook tutorials/

# Or use the run script
python scripts/run_tutorial.py 01_pattern_matching
```

## Running the Rust Examples

```bash
# From the examples/ directory
cargo run --example pattern_matching
cargo run --example tokenization_demo
cargo run --example attention_demo
```

## Prerequisites (Python packages)

```bash
pip install numpy matplotlib jupyter torch tokenizers tqdm
```
