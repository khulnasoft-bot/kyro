#![allow(dead_code)]

use candle_core::{Result, Tensor};
use std::collections::HashSet;

#[derive(Default, Clone)]
#[allow(dead_code)]
pub enum GrammarConstraint {
    #[default]
    None,
    Json,
    Regex(String),
}

#[derive(Default)]
#[allow(dead_code)]
pub struct GrammarState {
    pub stack: Vec<String>,
    pub current_text: String,
}

#[allow(dead_code)]
pub struct GrammarLogitsProcessor {
    pub constraint: GrammarConstraint,
    pub state: GrammarState,
}

impl GrammarLogitsProcessor {
    pub fn new(constraint: GrammarConstraint) -> Self {
        Self {
            constraint,
            state: GrammarState::default(),
        }
    }

    pub fn apply_grammar_mask(&mut self, logits: &Tensor, vocab_size: usize) -> Result<Tensor> {
        let valid_tokens = self.get_valid_tokens(vocab_size);
        let mut mask_data = vec![f32::NEG_INFINITY; vocab_size];
        for &token_id in &valid_tokens {
            if token_id < vocab_size {
                mask_data[token_id] = 0.0;
            }
        }
        let mask = Tensor::from_slice(&mask_data, (vocab_size,), logits.device())?
            .to_dtype(logits.dtype())?;
        logits.broadcast_add(&mask)
    }

    fn get_valid_tokens(&self, vocab_size: usize) -> HashSet<usize> {
        match &self.constraint {
            GrammarConstraint::None => (0..vocab_size).collect(),
            GrammarConstraint::Json => {
                let valid = vec![0, 10, 32, 34, 91, 93, 123, 125];
                valid.into_iter().collect()
            }
            GrammarConstraint::Regex(_) => (0..vocab_size).collect(),
        }
    }

    pub fn advance(&mut self, token_text: &str) {
        self.state.current_text.push_str(token_text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn json_constraint_masks_invalid_tokens() {
        let mut proc = GrammarLogitsProcessor::new(GrammarConstraint::Json);
        let logits = Tensor::zeros((16,), candle_core::DType::F32, &Device::Cpu).unwrap();
        let masked = proc.apply_grammar_mask(&logits, 16).unwrap();
        let vals: Vec<f32> = masked.to_vec1().unwrap();
        // Token 0 is valid for JSON; token 1 is not.
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[1], f32::NEG_INFINITY);
    }

    #[test]
    fn none_constraint_passes_all_tokens() {
        let mut proc = GrammarLogitsProcessor::new(GrammarConstraint::None);
        let logits = Tensor::zeros((8,), candle_core::DType::F32, &Device::Cpu).unwrap();
        let masked = proc.apply_grammar_mask(&logits, 8).unwrap();
        let vals: Vec<f32> = masked.to_vec1().unwrap();
        assert!(vals.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn advance_appends_text() {
        let mut proc = GrammarLogitsProcessor::new(GrammarConstraint::Json);
        proc.advance("hello");
        proc.advance(" world");
        assert_eq!(proc.state.current_text, "hello world");
    }
}
