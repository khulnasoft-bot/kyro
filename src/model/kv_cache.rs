#![allow(dead_code)]

use crate::scheduler::block_manager::BlockId;
use candle_core::{Result, Tensor};

pub struct KVCache {
    blocks: Vec<Vec<u32>>,
    block_size: usize,
}

impl KVCache {
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            block_size,
        }
    }

    pub fn update(
        &mut self,
        block_id: BlockId,
        _slot_idx: usize,
        _key: &Tensor,
        _value: &Tensor,
    ) -> Result<()> {
        let block_idx = block_id.0;
        while self.blocks.len() <= block_idx {
            self.blocks.push(Vec::new());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_creates_blocks_up_to_id() {
        let mut cache = KVCache::new(16);
        cache
            .update(
                BlockId(3),
                0,
                &Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
                &Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
            )
            .unwrap();
        assert_eq!(cache.blocks.len(), 4);
    }

    #[test]
    fn update_on_existing_block_is_noop() {
        let mut cache = KVCache::new(16);
        cache
            .update(
                BlockId(0),
                0,
                &Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
                &Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
            )
            .unwrap();
        cache
            .update(
                BlockId(0),
                1,
                &Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
                &Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
            )
            .unwrap();
        assert_eq!(cache.blocks.len(), 1);
    }
}
