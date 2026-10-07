# Operational Limits & Distributed Testing Notes

## Batch size and chunking

- `SchedulerConfig::max_tokens_per_iter` caps how many tokens are scheduled
  per iteration (default 2048). Decode requests cost 1 token each; prefills
  are chunked by `max_prefill_chunk_size` (default 512, constant
  `PREFILL_CHUNK_SIZE` in `src/scheduler/continuous_batching.rs`).
- `BlockManager` bounds total residency: `BlockManager::new(block_size,
  num_gpu_blocks, num_cpu_blocks)` — when no blocks are free, prefills wait
  rather than triggering an OOM. PagedAttention allocates KV in fixed-size
  blocks so long contexts cannot fragment contiguous memory.
- Client-side limits enforced by the API: `KYRO_MAX_TOKENS_CAP`,
  `KYRO_MAX_PROMPT_BYTES`, `KYRO_MAX_MESSAGES` (see README).

## Constrained decoding

`response_format: {"type": "json_object"}` activates
`GrammarLogitsProcessor` in `src/api/grammar.rs`, which masks logits to a
JSON-compatible token set before sampling. Regex constraints are accepted
but currently fall back to no masking (see issue #3 follow-up); structured
JSON-schema decoding should build on the same mask path.

## Quantization paths

- Safetensors (F16) via mmap: `--model-path <dir>` (directory containing
  `config.json` + `*.safetensors`).
- GGUF quantized via `quantized_llama::ModelWeights`: `--model-path
  <file>.gguf`. Set the model format automatically by extension.
- AWQ (INT4/INT8) and FP8 weight layouts are recognized by candle as GGUF
  Q4_K/Q8_0 and F8 quant types; files load through the same GGUF path.

## Distributed / multi-GPU testing

`src/distributed.rs` carries rank/world-size and the all-reduce hooks in
`src/model/llama.rs` are currently stubbed. To exercise the distributed
path without multiple GPUs, run the scheduler/worker unit tests with
injected `DistributedContext { rank, world_size }` and verify partial-sum
outputs are reduced — the exercise is local CPU ranks only.

## Hot reload & LoRA

LoRA adapters (`src/model/lora.rs`) are loadable per-request in principle,
but there is no dynamic hot-reload endpoint yet: restart with a different
`--model-path`/config to swap weights.
