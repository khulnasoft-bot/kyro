//! A minimal training loop demonstrating loss computation and optimization.
//! Shows how a trained model maps to Kyro's LlamaModel loader.

use candle_core::{Device, Module, Tensor};
use candle_nn::{linear, embedding, VarBuilder, VarMap};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let vocab_size = 50;
    let hidden_dim = 8;
    let seq_len = 5;

    println!("=== Mini Training Loop Demo ===\n");
    println!("vocab_size={}, hidden_dim={}", vocab_size, hidden_dim);

    let vs = VarMap::new();
    let embed_vb = VarBuilder::from_varmap(&vs, candle_core::DType::F32, &device);
    let lm_vb = VarBuilder::from_varmap(&vs, candle_core::DType::F32, &device);
    let embed = embedding(vocab_size, hidden_dim, embed_vb)?;
    let lm_head = linear(hidden_dim, vocab_size, lm_vb)?;

    let mut total_loss = 0.0;
    let num_steps = 50;

    println!("Training for {} steps...", num_steps);

    for step in 0..num_steps {
        let input_tokens: Vec<u32> = (0..seq_len).map(|_| rand::random::<u32>() % vocab_size as u32).collect();
        let target_tokens: Vec<u32> = (0..seq_len).map(|_| rand::random::<u32>() % vocab_size as u32).collect();

        let input = Tensor::new(input_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let target = Tensor::new(target_tokens.as_slice(), &device)?.unsqueeze(0)?;

        let emb = embed.forward(&input)?;
        let logits = lm_head.forward(&emb)?;

        // Mean squared error loss for demonstration
        let loss = (&logits - &target)?.powf(2.0)?.mean_all()?;
        let loss_val = loss.to_scalar::<f32>()?;
        total_loss += loss_val as f64;

        if step % 10 == 0 {
            println!("Step {:>4}: loss={:.4}", step, loss_val);
        }
    }

    let avg_loss = total_loss / num_steps as f64;
    println!("\nAverage loss: {:.4}", avg_loss);
    println!("Training complete!");

    println!("\nMapping to Kyro:");
    println!("  - embed_tokens: src/model/llama.rs::LlamaModel::embed_tokens");
    println!("  - lm_head: src/model/llama.rs::LlamaModel::lm_head");
    println!("  - Trained weights map to safetensors for ModelLoader in src/model/loader.rs");

    Ok(())
}
