#![allow(dead_code)]

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct LuminaTokenizer {
    tokenizer: Tokenizer,
}

impl LuminaTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let decoded = self
            .tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokenizer() -> LuminaTokenizer {
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        LuminaTokenizer::from_file(&path).unwrap()
    }

    #[test]
    fn encode_returns_token_ids() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world").unwrap();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn decode_roundtrips() {
        let tok = make_tokenizer();
        let text = tok.decode(&[1, 2]).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn from_file_fails_on_missing_path() {
        assert!(LuminaTokenizer::from_file("/nonexistent/tokenizer.json").is_err());
    }
}
