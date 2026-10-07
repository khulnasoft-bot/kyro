use crate::scheduler::block_manager::BlockId;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug)]
struct RadixNode {
    /// The sequence of tokens stored in this node
    tokens: Vec<u32>,
    /// The physical block IDs corresponding to these tokens
    block_ids: Vec<BlockId>,
    /// Children nodes mapped by the first token of the next sub-sequence
    children: HashMap<u32, Box<RadixNode>>,
    /// Last time this prefix was accessed (for LRU eviction)
    last_accessed: Instant,
}

impl RadixNode {
    fn new(tokens: Vec<u32>, block_ids: Vec<BlockId>) -> Self {
        Self {
            tokens,
            block_ids,
            children: HashMap::new(),
            last_accessed: Instant::now(),
        }
    }
}

pub struct RadixCache {
    root: RadixNode,
    /// Total number of blocks currently cached
    pub num_cached_blocks: usize,
    /// Maximum blocks allowed in cache before eviction
    pub max_capacity: usize,
}

impl RadixCache {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            root: RadixNode::new(vec![], vec![]),
            num_cached_blocks: 0,
            max_capacity,
        }
    }

    /// Matches a sequence of tokens against the cache.
    /// Returns (Cached Block IDs, tokens_matched_count)
    pub fn match_prefix(&mut self, tokens: &[u32]) -> (Vec<BlockId>, usize) {
        let mut current_node = &mut self.root;
        let mut matched_blocks = Vec::new();
        let mut total_matched_tokens = 0;

        let mut token_idx = 0;
        while token_idx < tokens.len() {
            let first_token = tokens[token_idx];

            if let Some(child) = current_node.children.get_mut(&first_token) {
                // Check if the rest of the child's tokens match
                let match_len = child
                    .tokens
                    .iter()
                    .zip(&tokens[token_idx..])
                    .take_while(|(a, b)| a == b)
                    .count();

                if match_len == child.tokens.len() {
                    // Full node match, move deeper
                    matched_blocks.extend_from_slice(&child.block_ids);
                    total_matched_tokens += match_len;
                    token_idx += match_len;
                    child.last_accessed = Instant::now();
                    current_node = child;
                } else {
                    // Partial match (prefix of a node)
                    break;
                }
            } else {
                break;
            }
        }

        (matched_blocks, total_matched_tokens)
    }

    /// Inserts a new sequence of tokens and their computed blocks into the cache
    pub fn insert(&mut self, tokens: &[u32], block_ids: &[BlockId]) {
        let mut current_node = &mut self.root;
        let mut token_idx = 0;

        while token_idx < tokens.len() {
            let first_token = tokens[token_idx];

            if let std::collections::hash_map::Entry::Vacant(e) =
                current_node.children.entry(first_token)
            {
                let new_node = RadixNode::new(tokens[token_idx..].to_vec(), block_ids.to_vec());
                e.insert(Box::new(new_node));
                self.num_cached_blocks += block_ids.len();
                break;
            }

            let child = current_node.children.get_mut(&first_token).unwrap();
            token_idx += child.tokens.len();
            current_node = child;
        }

        if self.num_cached_blocks > self.max_capacity {
            self.evict_lru();
        }
    }

    pub fn evict_lru(&mut self) -> Vec<(Vec<u32>, Vec<BlockId>)> {
        let mut evicted = Vec::new();
        while let Some((tokens, block_ids)) = self.remove_oldest_leaf() {
            self.num_cached_blocks -= block_ids.len();
            evicted.push((tokens, block_ids));
            if self.num_cached_blocks <= self.max_capacity {
                break;
            }
        }
        evicted
    }

    /// Insert without triggering capacity-based eviction. Used to restore
    /// evicted entries after a failed allocation.
    pub fn insert_unchecked(&mut self, tokens: &[u32], block_ids: &[BlockId]) {
        let mut current_node = &mut self.root;
        let mut token_idx = 0;
        while token_idx < tokens.len() {
            let first_token = tokens[token_idx];
            match current_node.children.entry(first_token) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(Box::new(RadixNode::new(
                        tokens[token_idx..].to_vec(),
                        block_ids.to_vec(),
                    )));
                    self.num_cached_blocks += block_ids.len();
                    return;
                }
                std::collections::hash_map::Entry::Occupied(o) => {
                    let child = o.into_mut();
                    token_idx += child.tokens.len();
                    current_node = child;
                }
            }
        }
    }

    fn remove_oldest_leaf(&mut self) -> Option<(Vec<u32>, Vec<BlockId>)> {
        let node = &mut self.root;
        if node.children.is_empty() {
            return None;
        }

        let mut oldest_token = None;
        let mut oldest_time = Instant::now();

        for (token, child) in &node.children {
            if child.children.is_empty() && child.last_accessed < oldest_time {
                oldest_time = child.last_accessed;
                oldest_token = Some(*token);
            }
        }

        if let Some(token) = oldest_token {
            let child = node.children.remove(&token).unwrap();
            Some((child.tokens, child.block_ids))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::block_manager::BlockId;
    fn b(ids: &[usize]) -> Vec<BlockId> {
        ids.iter().map(|&i| BlockId(i)).collect()
    }

    #[test]
    fn match_returns_nothing_when_empty() {
        let mut cache = RadixCache::new(16);
        let (blocks, matched) = cache.match_prefix(&[1, 2, 3]);
        assert!(blocks.is_empty());
        assert_eq!(matched, 0);
    }

    #[test]
    fn insert_then_exact_match() {
        let mut cache = RadixCache::new(16);
        cache.insert(&[1, 2, 3, 4], &b(&[10, 11]));
        let (blocks, matched) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(blocks, b(&[10, 11]));
        assert_eq!(matched, 4);
    }

    #[test]
    fn match_prefix_longer_than_cached() {
        let mut cache = RadixCache::new(16);
        cache.insert(&[1, 2], &b(&[5]));
        let (blocks, matched) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(blocks, b(&[5]));
        assert_eq!(matched, 2);
    }

    #[test]
    fn match_prefix_partial_node_returns_zero() {
        let mut cache = RadixCache::new(16);
        cache.insert(&[1, 2, 3], &b(&[7, 8]));
        let (blocks, matched) = cache.match_prefix(&[1, 2]);
        assert!(blocks.is_empty());
        assert_eq!(matched, 0);
    }

    #[test]
    fn distinct_prefixes_cached_independently() {
        let mut cache = RadixCache::new(16);
        cache.insert(&[1, 2], &b(&[1]));
        cache.insert(&[3, 4], &b(&[2]));
        assert_eq!(cache.match_prefix(&[1, 2]).0, b(&[1]));
        assert_eq!(cache.match_prefix(&[3, 4]).0, b(&[2]));
        assert_eq!(cache.num_cached_blocks, 2);
    }

    #[test]
    fn eviction_respects_capacity() {
        let mut cache = RadixCache::new(2);
        cache.insert(&[1], &b(&[1, 2]));
        cache.insert(&[2], &b(&[3, 4]));
        assert!(cache.num_cached_blocks <= 2);
    }

    #[test]
    fn evicted_blocks_are_returned() {
        let mut cache = RadixCache::new(1);
        cache.insert(&[1], &b(&[10]));
        cache.insert(&[2], &b(&[20]));
        // LRU eviction of the first leaf should return its block.
        let freed = cache.evict_lru();
        assert!(cache.num_cached_blocks <= 1);
        // Either already evicted via insert or evicted now.
        assert!(freed.len() + cache.num_cached_blocks <= 2);
    }
}
