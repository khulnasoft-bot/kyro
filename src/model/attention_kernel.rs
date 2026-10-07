#![allow(dead_code)]

use candle_core::{Device, Result, Tensor};

pub struct PagedAttention {
    pub block_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl PagedAttention {
    pub fn new(block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            block_size,
            num_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        query: &Tensor,
        _key_cache: &Tensor,
        _value_cache: &Tensor,
        _block_table: &Tensor,
        _context_lens: &Tensor,
    ) -> Result<Tensor> {
        let device = query.device();

        match device {
            Device::Cuda(_) => self.software_paged_attention(query),
            _ => self.software_paged_attention(query),
        }
    }

    fn software_paged_attention(&self, query: &Tensor) -> Result<Tensor> {
        Ok(query.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn forward_returns_query_shape() {
        let pa = PagedAttention::new(16, 32, 128);
        let q = Tensor::zeros((1, 4, 32, 128), candle_core::DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::zeros((1, 4, 32, 128), candle_core::DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 4, 32, 128), candle_core::DType::F32, &Device::Cpu).unwrap();
        let bt = Tensor::zeros((1, 1), candle_core::DType::U32, &Device::Cpu).unwrap();
        let cl = Tensor::zeros((1,), candle_core::DType::U32, &Device::Cpu).unwrap();
        let out = pa.forward(&q, &k, &v, &bt, &cl).unwrap();
        assert_eq!(out.dims(), q.dims());
    }
}
