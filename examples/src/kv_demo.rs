//! Demonstrates KV cache and prefix caching concepts.
//! Mirrors BlockManager and RadixCache in src/scheduler/.

use std::collections::HashMap;

struct SimpleKVCache {
    cache: HashMap<u64, Vec<f32>>,
    max_size: usize,
}

impl SimpleKVCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }

    fn insert(&mut self, req_id: u64, tokens: usize) {
        self.cache.insert(req_id, vec![0.0; tokens * 64]);
        if self.cache.len() > self.max_size {
            let oldest = self.cache.keys().next().copied();
            if let Some(k) = oldest {
                self.cache.remove(&k);
            }
        }
    }

    fn get(&self, req_id: u64) -> Option<&Vec<f32>> {
        self.cache.get(&req_id)
    }
}

struct RadixCache {
    nodes: HashMap<Vec<u32>, usize>,
    capacity: usize,
}

impl RadixCache {
    fn new(capacity: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            capacity,
        }
    }

    fn match_prefix(&self, tokens: &[u32]) -> (usize, usize) {
        let mut best_len = 0;
        let mut best_blocks = 0;
        for len in (1..=tokens.len()).rev() {
            let prefix = tokens[..len].to_vec();
            if let Some(&blocks) = self.nodes.get(&prefix) {
                best_len = len;
                best_blocks = blocks;
                break;
            }
        }
        (best_len, best_blocks)
    }

    fn insert(&mut self, tokens: &[u32], num_blocks: usize) {
        self.nodes.insert(tokens.to_vec(), num_blocks);
        if self.nodes.len() > self.capacity {
            let oldest = self.nodes.keys().next().cloned();
            if let Some(k) = oldest {
                self.nodes.remove(&k);
            }
        }
    }
}

fn main() {
    println!("=== KV Cache and Prefix Caching Demo ===\n");

    // KV Cache
    println!("--- Simple KV Cache ---");
    let mut kv_cache = SimpleKVCache::new(3);
    kv_cache.insert(1, 10);
    kv_cache.insert(2, 5);
    kv_cache.insert(3, 8);
    println!("Cache entries: {}", kv_cache.cache.len());
    println!("Request 1 cached: {:?}", kv_cache.get(1).is_some());
    println!("Request 4 cached: {:?}", kv_cache.get(4).is_none());

    // Prefix caching
    println!("\n--- Radix Cache (Prefix Matching) ---");
    let mut radix = RadixCache::new(10);
    let prompt_a = vec![1u32, 2, 3, 4, 5];
    let prompt_b = vec![1u32, 2, 3, 6, 7];
    radix.insert(&prompt_a, 2);
    radix.insert(&prompt_b, 2);

    let (cached_len, blocks) = radix.match_prefix(&[1u32, 2, 3, 8, 9]);
    println!("Prompt [1,2,3,8,9] matched prefix length: {}", cached_len);
    println!("Blocks reused: {}", blocks);

    // TTFT improvement
    println!("\n--- Prefix Cache Benefit ---");
    let prompt_tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let cached_len = 3;
    let remaining = prompt_tokens.len() - cached_len;
    println!(
        "Full prompt: {} tokens, Cached: {} tokens",
        prompt_tokens.len(),
        cached_len
    );
    println!("Tokens to process: {}", remaining);
    println!(
        "TTFT improvement: {:.1}% reduction",
        (1.0 - remaining as f64 / prompt_tokens.len() as f64) * 100.0
    );

    println!("\nKV cache demo complete!");
    println!("Maps to:");
    println!("  - src/scheduler/block_manager.rs (PagedAttention)");
    println!("  - src/scheduler/radix_cache.rs (Prefix matching)");
    println!("  - src/scheduler/continuous_batching.rs (cached_prefix_len)");
}
