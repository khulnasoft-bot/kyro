#![allow(dead_code)]

use anyhow::Result;
use prometheus::{Counter, Gauge, Histogram, Registry};
use std::sync::Arc;

pub struct EngineMetrics {
    pub total_requests: Counter,
    pub requests_by_model: prometheus::CounterVec,
    pub queue_depth: Gauge,
    pub cache_hits: Gauge,
    pub cache_misses: Gauge,
    pub total_tokens_generated: Counter,
    pub token_latency: Histogram,
    pub time_to_first_token: Histogram,
    pub time_between_tokens: Histogram,
    pub kv_cache_usage: Gauge,
}

impl EngineMetrics {
    pub fn new(registry: &Registry) -> Result<Arc<Self>> {
        let total_requests = prometheus::register_counter_with_registry!(
            "kyro_requests_total",
            "Total number of requests processed",
            registry
        )?;

        let total_tokens_generated = prometheus::register_counter_with_registry!(
            "kyro_tokens_total",
            "Total number of tokens generated",
            registry
        )?;

        let token_latency = prometheus::register_histogram_with_registry!(
            "kyro_token_latency_seconds",
            "Latency per token generation",
            registry
        )?;

        let time_to_first_token = prometheus::register_histogram_with_registry!(
            "kyro_ttft_ms",
            "Time to first token",
            registry
        )?;

        let time_between_tokens = prometheus::register_histogram_with_registry!(
            "kyro_tbt_ms",
            "Time between tokens",
            registry
        )?;

        let kv_cache_usage = prometheus::register_gauge_with_registry!(
            "kyro_kv_cache_usage_percent",
            "KV cache utilization percentage",
            registry
        )?;

        let requests_by_model = prometheus::register_counter_vec_with_registry!(
            "kyro_requests_by_model_total",
            "Total requests, labeled by served model name",
            &["model"],
            registry
        )?;

        let queue_depth = prometheus::register_gauge_with_registry!(
            "kyro_queue_depth",
            "Number of requests waiting or running in the scheduler",
            registry
        )?;

        let cache_hits = prometheus::register_gauge_with_registry!(
            "kyro_prefix_cache_hits_total",
            "Number of requests that reused a cached prefix",
            registry
        )?;

        let cache_misses = prometheus::register_gauge_with_registry!(
            "kyro_prefix_cache_misses_total",
            "Number of requests that required a cold prefill",
            registry
        )?;

        Ok(Arc::new(Self {
            total_requests,
            requests_by_model,
            queue_depth,
            cache_hits,
            cache_misses,
            total_tokens_generated,
            token_latency,
            time_to_first_token,
            time_between_tokens,
            kv_cache_usage,
        }))
    }
}
