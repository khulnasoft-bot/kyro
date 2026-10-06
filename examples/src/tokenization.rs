//! Demonstrates BPE tokenization fundamentals.
//! Mirrors Kyro's LuminaTokenizer (src/api/tokenizer.rs).

use std::collections::HashMap;

struct MinimalBPE {
    vocab: Vec<String>,
    vocab_map: HashMap<String, usize>,
}

impl MinimalBPE {
    fn new(vocab: Vec<String>) -> Self {
        let vocab_map = vocab
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        Self { vocab, vocab_map }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let text = text.to_lowercase();
        let mut tokens = Vec::new();
        let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Simple greedy matching from longest to shortest
        let mut i = 0;
        while i < chars.len() {
            let mut matched = false;
            for length in (1..=3).rev() {
                if i + length <= chars.len() {
                    let substr: String = chars[i..i + length].iter().cloned().collect();
                    if let Some(&idx) = self.vocab_map.get(&substr) {
                        tokens.push(idx);
                        i += length;
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                let c = chars[i].clone();
                if let Some(&idx) = self.vocab_map.get(&c) {
                    tokens.push(idx);
                } else {
                    tokens.push(0); // UNK
                }
                i += 1;
            }
        }
        tokens
    }

    fn decode(&self, token_ids: &[usize]) -> String {
        token_ids
            .iter()
            .map(|&id| self.vocab.get(id).cloned().unwrap_or_default())
            .collect()
    }
}

fn main() {
    let vocab = vec![
        "h".to_string(),
        "e".to_string(),
        "l".to_string(),
        "o".to_string(),
        "he".to_string(),
        "ll".to_string(),
        "lo".to_string(),
        "hel".to_string(),
        "ell".to_string(),
        "llo".to_string(),
        "hello".to_string(),
        "wor".to_string(),
        "orl".to_string(),
        "ld".to_string(),
        "worl".to_string(),
        "orld".to_string(),
        "world".to_string(),
        " ".to_string(),
    ];

    let bpe = MinimalBPE::new(vocab);

    let text = "hello world";
    let tokens = bpe.encode(text);
    println!("Text: '{}'", text);
    println!("Token IDs: {:?}", tokens);
    println!("Detokenized: '{}'", bpe.decode(&tokens));
    println!("Token count: {}", tokens.len());

    // Compare character-level vs subword
    let char_count = text.chars().count();
    println!("\nCharacter-level tokens: {}", char_count);
    println!("Subword tokens: {}", tokens.len());
    println!(
        "Compression ratio: {:.2}x",
        char_count as f64 / tokens.len() as f64
    );
}
