#![allow(dead_code)]

pub struct PipelineContext {
    pub rank: usize,
    pub world_size: usize,
    pub start_layer: usize,
    pub end_layer: usize,
}

impl PipelineContext {
    pub fn new(rank: usize, world_size: usize, total_layers: usize) -> Self {
        let layers_per_gpu = total_layers.div_ceil(world_size);
        let start_layer = rank * layers_per_gpu;
        let end_layer = std::cmp::min(start_layer + layers_per_gpu, total_layers);

        Self {
            rank,
            world_size,
            start_layer,
            end_layer,
        }
    }

    pub fn is_first_stage(&self) -> bool {
        self.rank == 0
    }

    pub fn is_last_stage(&self) -> bool {
        self.rank == self.world_size - 1
    }

    pub fn should_process_layer(&self, index: usize) -> bool {
        index >= self.start_layer && index < self.end_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_rank_processes_all_layers() {
        let ctx = PipelineContext::new(0, 1, 32);
        assert!(ctx.is_first_stage());
        assert!(ctx.is_last_stage());
        assert_eq!(ctx.start_layer, 0);
        assert_eq!(ctx.end_layer, 32);
        assert!(ctx.should_process_layer(0));
        assert!(ctx.should_process_layer(31));
    }

    #[test]
    fn two_ranks_split_layers_evenly() {
        let r0 = PipelineContext::new(0, 2, 32);
        let r1 = PipelineContext::new(1, 2, 32);
        assert!(r0.is_first_stage());
        assert!(!r0.is_last_stage());
        assert!(r1.is_last_stage());
        assert!(!r1.is_first_stage());
        assert_eq!(r0.start_layer, 0);
        assert_eq!(r0.end_layer, 16);
        assert_eq!(r1.start_layer, 16);
        assert_eq!(r1.end_layer, 32);
        assert!(!r0.should_process_layer(16));
        assert!(r1.should_process_layer(16));
    }

    #[test]
    fn uneven_split_clamps_to_total() {
        let ctx = PipelineContext::new(2, 3, 10);
        // layers_per_gpu = ceil(10/3) = 4; rank 2 -> start=8, end=min(12,10)=10
        assert_eq!(ctx.start_layer, 8);
        assert_eq!(ctx.end_layer, 10);
        assert!(ctx.should_process_layer(9));
        assert!(!ctx.should_process_layer(10));
    }
}
