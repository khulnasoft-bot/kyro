#![allow(dead_code)]

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl LlamaConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let config: Self = serde_json::from_reader(file)?;
        Ok(config)
    }

    pub fn llama_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            vocab_size: 32000,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_7b_has_expected_dims() {
        let cfg = LlamaConfig::llama_7b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.vocab_size, 32000);
    }

    #[test]
    fn from_file_parses_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            r#"{"hidden_size":512,"intermediate_size":1024,"num_hidden_layers":4,"num_attention_heads":8,"num_key_value_heads":8,"vocab_size":1000,"rms_norm_eps":1e-5,"rope_theta":500000.0}"#,
        )
        .unwrap();
        let cfg = LlamaConfig::from_file(&path).unwrap();
        assert_eq!(cfg.hidden_size, 512);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(cfg.rope_theta, 500000.0);
    }
}
