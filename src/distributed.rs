pub struct DistributedContext {
    pub rank: u32,
    pub world_size: u32,
}

impl DistributedContext {
    pub fn new() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
}

impl Default for DistributedContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_single_rank() {
        let ctx = DistributedContext::new();
        assert_eq!(ctx.rank, 0);
        assert_eq!(ctx.world_size, 1);
    }

    #[test]
    fn default_matches_new() {
        let ctx = DistributedContext::default();
        assert_eq!(ctx.rank, 0);
        assert_eq!(ctx.world_size, 1);
    }
}
