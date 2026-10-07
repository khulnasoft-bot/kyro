#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};
use std::collections::HashMap;

pub struct LoraAdapter {
    pub id: String,
    pub a: Tensor, // [rank, hidden_in]
    pub b: Tensor, // [hidden_out, rank]
    pub alpha: f64,
    pub rank: usize,
}

pub struct LoraLinear {
    pub base: Linear,
    pub adapters: HashMap<String, LoraAdapter>,
}

impl LoraLinear {
    pub fn new(base: Linear) -> Self {
        Self {
            base,
            adapters: HashMap::new(),
        }
    }

    pub fn add_adapter(&mut self, adapter: LoraAdapter) {
        self.adapters.insert(adapter.id.clone(), adapter);
    }

    pub fn forward(&self, x: &Tensor, adapter_id: Option<&str>) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;

        if let Some(id) = adapter_id {
            if let Some(adapter) = self.adapters.get(id) {
                // lora_out = base_out + (x @ A.T @ B.T) * (alpha / rank)
                let lora_x = x.matmul(&adapter.a.t()?)?;
                let lora_out = lora_x.matmul(&adapter.b.t()?)?;
                let scaling = adapter.alpha / (adapter.rank as f64);
                return base_out.broadcast_add(&(lora_out * scaling)?);
            }
        }

        Ok(base_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn test_linear() -> Linear {
        Linear::new(
            candle_core::Tensor::zeros((4, 4), candle_core::DType::F32, &Device::Cpu).unwrap(),
            None,
        )
    }

    #[test]
    fn forward_without_adapter_returns_base() {
        let layer = LoraLinear::new(test_linear());
        let x = Tensor::ones((1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let out = layer.forward(&x, None).unwrap();
        assert_eq!(out.dims(), &[1, 4]);
    }

    #[test]
    fn forward_with_unknown_adapter_returns_base() {
        let layer = LoraLinear::new(test_linear());
        let x = Tensor::ones((1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let out = layer.forward(&x, Some("missing")).unwrap();
        assert_eq!(out.dims(), &[1, 4]);
    }

    #[test]
    fn forward_with_adapter_adds_lora_term() {
        let mut layer = LoraLinear::new(test_linear());
        let a = Tensor::ones((2, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::ones((4, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        layer.add_adapter(LoraAdapter {
            id: "test".into(),
            a,
            b,
            alpha: 4.0,
            rank: 2,
        });
        let x = Tensor::ones((1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let out = layer.forward(&x, Some("test")).unwrap();
        // base is zero, lora = x @ A.T @ B.T * (alpha/rank) = 4 * 2 * (4/2) = 16
        let val = out
            .get(0)
            .unwrap()
            .get(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((val - 16.0).abs() < 1e-4, "expected ~16.0, got {}", val);
    }
}
