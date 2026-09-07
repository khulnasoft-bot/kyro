//! Demonstrates embedding layers and cosine similarity.
//! Mirrors LlamaModel::embed_tokens in src/model/llama.rs.

use candle_core::{Device, Module, Tensor};
use candle_nn::{embedding, VarBuilder, VarMap};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let vocab_size = 100;
    let hidden_dim = 16;

    println!("=== Embedding Layer Demo ===");
    println!("vocab_size={}, hidden_dim={}", vocab_size, hidden_dim);

    let vs = VarMap::new();
    let vb = VarBuilder::from_varmap(&vs, candle_core::DType::F32, &device);
    let embed = embedding(vocab_size, hidden_dim, vb)?;

    let tokens = Tensor::new(&[0u32, 1, 2, 3, 4], &device)?;
    let embeddings = embed.forward(&tokens)?;
    println!("Input tokens: [0, 1, 2, 3, 4]");
    println!("Embedding shape: {:?}", embeddings.dims());

    // Cosine similarity
    let e0 = embeddings.get(0)?;
    let e1 = embeddings.get(1)?;
    let e0_norm = e0.norm()?;
    let e1_norm = e1.norm()?;
    let dot = e0.matmul(&e1.t()?)?;
    let sim = (&dot / (e0_norm * e1_norm))?.to_dtype(candle_core::DType::F32)?;
    println!("Cosine similarity between token 0 and 1: {:.4}", sim.to_scalar::<f32>()?);

    // Nearest neighbor
    println!("\n=== Nearest Neighbor Retrieval ===");
    let query = Tensor::new(&[0u32], &device)?;
    let query_emb = embed.forward(&query)?;
    let mut best_sim = f32::NEG_INFINITY;
    let mut best_token = 0u32;
    for i in 0..vocab_size {
        let token_emb = embed.forward(&Tensor::new(&[i as u32], &device)?)?;
        let q_norm = query_emb.norm()?;
        let t_norm = token_emb.norm()?;
        let s = (&query_emb.matmul(&token_emb.t()?)? / (q_norm * t_norm))?;
        let val = s.to_scalar::<f32>()?;
        if val > best_sim {
            best_sim = val;
            best_token = i as u32;
        }
    }
    println!("Nearest neighbor to token 0: token {} (sim={:.4})", best_token, best_sim);
    println!("(Self-similarity should be ~1.0)");

    println!("\nEmbedding demo complete!");
    println!("Maps to LlamaModel::embed_tokens in src/model/llama.rs");

    Ok(())
}
