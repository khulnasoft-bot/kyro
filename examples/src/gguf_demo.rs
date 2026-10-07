//! Demonstrates loading a GGUF-quantized model and running a forward pass.
//!
//! Usage:
//!   cargo run --example gguf_demo -- /path/to/model.gguf
//!
//! Download a GGUF model first, e.g.:
//!   huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf

use candle_core::{DType, Device, Tensor};
use kyro::distributed::DistributedContext;
use kyro::model::loader::{LoadedModel, ModelLoader};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| std::env::var("KYRO_MODEL_PATH").unwrap_or_default());

    if model_path.is_empty() {
        anyhow::bail!("Usage: gguf_demo <path-to-model.gguf>");
    }
    if !model_path.ends_with(".gguf") {
        anyhow::bail!("Model path must point to a .gguf file");
    }

    let device = Device::Cpu;
    let dist = Arc::new(DistributedContext::new());

    println!("Loading GGUF model from {} ...", model_path);
    let loader = ModelLoader::new(&model_path)?;
    assert!(loader.is_gguf, "loader should detect GGUF format");
    let model = loader.load(&device, dist)?;
    println!("Model loaded: {:?}", std::mem::discriminant(&model));

    // Run a small forward pass to verify the model works.
    let prompt = Tensor::new(&[1u32, 15043, 3186, 29892], &device)?.unsqueeze(0)?;
    let logits = match &model {
        LoadedModel::Standard(m) => m.forward(&prompt, 0)?,
        LoadedModel::Quantized(q) => q.forward(&prompt, 0)?,
    };
    println!("Forward pass OK. logits shape: {:?}", logits.dims());

    let next_token = logits
        .squeeze(1)?
        .squeeze(0)?
        .argmax(0)?
        .to_scalar::<u32>()?;
    println!("Sampled next token id: {}", next_token);

    // Keep dtype usage explicit so the import is not flagged when refactored.
    let _ = DType::F16;
    Ok(())
}
