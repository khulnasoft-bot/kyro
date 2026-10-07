#![allow(dead_code)]

use candle_core::{Device, Result, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::Path;

pub struct QuantizedLlama {
    pub inner: ModelWeights,
    device: Device,
}

impl QuantizedLlama {
    pub fn load_gguf<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let model = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let inner = ModelWeights::from_gguf(model, &mut file, device)?;
        Ok(Self {
            inner,
            device: device.clone(),
        })
    }

    pub fn forward(&mut self, x: &Tensor, index: usize) -> Result<Tensor> {
        // Quantized models often take slightly different forward signatures.
        // We adapt it here.
        self.inner.forward(x, index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_gguf_fails_on_nonexistent_file() {
        let result = QuantizedLlama::load_gguf("/nonexistent/model.gguf", &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn load_gguf_fails_on_invalid_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, b"not a gguf file").unwrap();
        let result = QuantizedLlama::load_gguf(&path, &Device::Cpu);
        assert!(result.is_err());
    }
}
