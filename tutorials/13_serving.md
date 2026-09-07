# Module 13: Serving the Trained Model with Kyro

## Learning Objectives
- Export a trained model into Kyro-compatible format
- Load the model into Kyro's ModelLoader
- Verify end-to-end inference through the Kyro API

## Steps

### 1. Train and Export
```python
# After training in Module 11
import torch
model = TinyLM(vocab_size=50, d_model=16)
# ... training code ...
torch.save(model.state_dict(), 'tiny_model.pt')
```

### 2. Convert to Kyro Format
Kyro supports:
- safetensors (native format for `ModelLoader`)
- GGUF (for quantized models)

### 3. Start Kyro with the Model
```bash
cargo run --release -- --model-path ./tiny_model/
```

### 4. Verify Serving
```bash
curl http://localhost:3000/health
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tiny-llama", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}'
```

## Acceptance Criteria
- Kyro loads the trained model without errors
- `/health` returns OK
- `/v1/chat/completions` returns generated tokens
- End-to-end test passes

## Kyro Integration Points
- `src/model/loader.rs::ModelLoader::load()` — loads safetensors/GGUF
- `src/model/llama.rs::LlamaModel` — model architecture
- `src/api/openai.rs` — serves completions
- `src/worker.rs` — runs inference loop
