//! Demonstrates n-gram language models and MLE training.
//! This example mirrors the statistical language modeling concepts from Module 1.

use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;

struct NgramModel {
    n: usize,
    counts: HashMap<Vec<String>, HashMap<String, usize>>,
    totals: HashMap<Vec<String>, usize>,
}

impl NgramModel {
    fn new(n: usize) -> Self {
        Self {
            n,
            counts: HashMap::new(),
            totals: HashMap::new(),
        }
    }

    fn train(&mut self, corpus: &str) {
        let tokens: Vec<String> = corpus
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        for window in tokens.windows(self.n) {
            let context = window[..self.n - 1].to_vec();
            let next = window[self.n - 1].clone();
            *self
                .counts
                .entry(context.clone())
                .or_default()
                .entry(next)
                .or_insert(0) += 1;
            *self.totals.entry(context).or_insert(0) += 1;
        }
    }

    fn probability(&self, context: &[String], next: &str) -> f64 {
        let total = self.totals.get(context).copied().unwrap_or(0);
        if total == 0 {
            return 1e-10;
        }
        let count = self
            .counts
            .get(context)
            .and_then(|c| c.get(next))
            .copied()
            .unwrap_or(0);
        count as f64 / total as f64
    }

    fn perplexity(&self, corpus: &str) -> f64 {
        let tokens: Vec<String> = corpus
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        let mut log_prob_sum = 0.0;
        let mut n = 0;
        for window in tokens.windows(self.n) {
            let context = window[..self.n - 1].to_vec();
            let next = &window[self.n - 1];
            let p = self.probability(&context, next).max(1e-10);
            log_prob_sum += p.ln();
            n += 1;
        }
        (-log_prob_sum / n as f64).exp()
    }

    fn generate(&self, prefix: &str, max_tokens: usize) -> String {
        let mut tokens: Vec<String> = prefix
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        let mut rng = rng();
        for _ in 0..max_tokens {
            let start = tokens.len().saturating_sub(self.n - 1);
            let context = tokens[start..].to_vec();
            if let Some(nexts) = self.counts.get(&context) {
                let total: usize = nexts.values().sum();
                let r: usize = rng.random_range(0..total);
                let mut cumsum = 0;
                for (token, &count) in nexts {
                    cumsum += count;
                    if cumsum > r {
                        tokens.push(token.clone());
                        break;
                    }
                }
            } else {
                break;
            }
        }
        tokens.join(" ")
    }
}

fn main() {
    let corpus = "the cat sat on the mat the dog ran on the grass the cat slept on the mat";
    println!("Training n-gram model on corpus:\n{}\n", corpus);

    for n in [2, 3] {
        let mut model = NgramModel::new(n);
        model.train(corpus);
        println!("=== {}-gram Model ===", n);
        println!("Perplexity: {:.4}", model.perplexity(corpus));
        println!("Generate from 'the cat': {}", model.generate("the cat", 8));
        println!("Generate from 'the dog': {}", model.generate("the dog", 8));
        println!();
    }

    // Smoothing demo
    println!("=== Laplace Smoothing (2-gram) ===");
    let mut model = NgramModel::new(2);
    model.train(corpus);
    let vocab_size = 10;
    let context = vec!["cat".to_string()];
    let next = "sat";
    let raw_prob = model.probability(&context, next);
    let smoothed_prob = (model
        .counts
        .get(&context)
        .and_then(|c| c.get(next))
        .copied()
        .unwrap_or(0)
        + 1) as f64
        / (model.totals.get(&context).copied().unwrap_or(0) + vocab_size) as f64;
    println!("Raw P('{}' | {:?}): {:.6}", next, context, raw_prob);
    println!(
        "Smoothed P('{}' | {:?}): {:.6}",
        next, context, smoothed_prob
    );
}
