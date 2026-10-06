use kyro::distributed::DistributedContext;
use kyro::model::loader::{LoadedModel, ModelLoader};
use kyro::scheduler::block_manager::BlockManager;
use kyro::scheduler::continuous_batching::Scheduler;
use kyro::worker::Worker;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(Level::INFO.into()),
        )
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting Kyro LLM Engine...");

    // 1. Hardware detection
    let device = kyro::device::get_device()?;
    info!("Using device: {:?}", device);

    // 2. Initialize Distributed Context, Block Manager, Scheduler, and Metrics
    let dist = Arc::new(DistributedContext::new());
    info!(
        "Distributed context initialized (Rank: {}, World Size: {})",
        dist.rank, dist.world_size
    );

    let block_manager = BlockManager::new(16, 1024, 256);
    let scheduler_cfg = kyro::scheduler::continuous_batching::SchedulerConfig::default();
    let scheduler = Arc::new(Mutex::new(Scheduler::new(block_manager, scheduler_cfg)));
    let notify = Arc::new(Notify::new());

    let registry = prometheus::Registry::new();
    let metrics = kyro::metrics::EngineMetrics::new(&registry)?;
    info!("Scheduler and Metrics initialized.");

    // 3. Resolve startup configuration (CLI args take precedence over env vars)
    let model_path = arg_value("--model-path").or_else(|| std::env::var("KYRO_MODEL_PATH").ok());
    let tokenizer_path =
        arg_value("--tokenizer-path").or_else(|| std::env::var("KYRO_TOKENIZER_PATH").ok());
    let model_name = arg_value("--model-name")
        .or_else(|| std::env::var("KYRO_MODEL_NAME").ok())
        .unwrap_or_else(|| "kyro".to_string());

    let tokenizer = match &tokenizer_path {
        Some(path) => {
            let tok = kyro::api::tokenizer::LuminaTokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", path, e))?;
            info!("Tokenizer loaded from {}", path);
            Some(Arc::new(tok))
        }
        None => {
            info!("No tokenizer path configured; raw token IDs will be returned.");
            None
        }
    };

    // 4. Load model through ModelLoader when a path is configured; otherwise use the dummy model.
    let loaded_model = match &model_path {
        Some(path) => {
            let loader = ModelLoader::new(path)
                .map_err(|e| anyhow::anyhow!("Invalid model path '{}': {}", path, e))?;
            let model = loader.load(&device, dist).map_err(|e| {
                anyhow::anyhow!("Failed to load model from '{}': {}", path, e)
            })?;
            info!("Model loaded from {}", path);
            model
        }
        None => {
            info!("No model path configured; running with dummy model.");
            let cfg = kyro::model::config::LlamaConfig::llama_7b();
            LoadedModel::Standard(kyro::model::llama::LlamaModel::dummy(&cfg)?)
        }
    };

    // 5. Start Worker Loop
    let mut worker = Worker::new(loaded_model, scheduler.clone(), device, metrics);
    let worker_notify = notify.clone();
    tokio::spawn(async move {
        if let Err(e) = worker.run_loop(worker_notify).await {
            tracing::error!("Worker loop failed: {:?}", e);
        }
    });

    // 6. Start API Server
    let app_state = Arc::new(kyro::api::openai::AppState::new(
        scheduler,
        notify,
        tokenizer,
        model_name,
    ));
    let app = kyro::api::openai::app(app_state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("Kyro API serving on http://localhost:3000");

    axum::serve(listener, app).await?;

    Ok(())
}

fn arg_value(flag: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == flag {
            return args.next();
        }
    }
    None
}
