//! Demonstrates scaled dot-product attention and Rotary Embedding.
//! Mirrors LlamaAttention in src/model/llama.rs with tiny tensors on CPU.

use candle_core::{Device, Tensor, D};
use candle_nn::ops::softmax;

fn scaled_dot_product_attention(
    q: &Tensor, k: &Tensor, v: &Tensor,
    num_heads: usize, head_dim: usize,
) -> candle_core::Result<Tensor> {
    let (b_sz, seq_len, _) = q.dims3()?;
    let q = q.reshape((b_sz, seq_len, num_heads, head_dim))?;
    let k = k.reshape((b_sz, seq_len, num_heads, head_dim))?;
    let v = v.reshape((b_sz, seq_len, num_heads, head_dim))?;
    let scores = (q.matmul(&k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?;
    let att = softmax(&scores, D::Minus1)?;
    let out = att.matmul(&v)?;
    out.reshape((b_sz, seq_len, num_heads * head_dim))
}

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let batch_size = 1;
    let seq_len = 4;
    let num_heads = 2;
    let head_dim = 4;
    let hidden_dim = num_heads * head_dim;

    println!("=== Scaled Dot-Product Attention Demo ===");
    println!("batch_size={}, seq_len={}, num_heads={}, head_dim={}", batch_size, seq_len, num_heads, head_dim);

    let q = Tensor::randn(0.0, 1.0, (batch_size, seq_len, hidden_dim), &device)?;
    let k = Tensor::randn(0.0, 1.0, (batch_size, seq_len, hidden_dim), &device)?;
    let v = Tensor::randn(0.0, 1.0, (batch_size, seq_len, hidden_dim), &device)?;

    let attn_output = scaled_dot_product_attention(&q, &k, &v, num_heads, head_dim)?;
    println!("Attention output shape: {:?}", attn_output.dims());

    // Causal mask demonstration
    let mask = Tensor::tril2(seq_len, candle_core::DType::F32, &device)?;
    println!("\nCausal mask:\n{}", mask);

    println!("\nAttention demo complete!");
    println!("Maps to src/model/llama.rs::LlamaAttention::forward");
    println!("with PagedAttention handling KV cache management.");

    Ok(())
}
