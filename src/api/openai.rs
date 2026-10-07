#![allow(dead_code)]

use crate::api::tokenizer::LuminaTokenizer;
use crate::error::ApiError;
use crate::metrics::EngineMetrics;
use crate::scheduler::continuous_batching::{Request, Scheduler};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::{Mutex, Notify};

pub struct AppState {
    pub scheduler: Arc<Mutex<Scheduler>>,
    pub notify: Arc<Notify>,
    pub tokenizer: Option<Arc<LuminaTokenizer>>,
    pub model_name: String,
    pub metrics: Option<Arc<EngineMetrics>>,
    pub registry: Option<Arc<prometheus::Registry>>,
    pub max_tokens_cap: usize,
    pub max_prompt_bytes: usize,
    pub max_messages: usize,
}

impl AppState {
    pub fn new(
        scheduler: Arc<Mutex<Scheduler>>,
        notify: Arc<Notify>,
        tokenizer: Option<Arc<LuminaTokenizer>>,
        model_name: String,
    ) -> Self {
        Self {
            scheduler,
            notify,
            tokenizer,
            model_name,
            metrics: None,
            registry: None,
            max_tokens_cap: 4096,
            max_prompt_bytes: 64 * 1024,
            max_messages: 256,
        }
    }

    pub fn with_metrics(
        mut self,
        metrics: Arc<EngineMetrics>,
        registry: Arc<prometheus::Registry>,
    ) -> Self {
        self.metrics = Some(metrics);
        self.registry = Some(registry);
        self
    }

    pub fn with_limits(
        mut self,
        max_tokens_cap: usize,
        max_prompt_bytes: usize,
        max_messages: usize,
    ) -> Self {
        self.max_tokens_cap = max_tokens_cap;
        self.max_prompt_bytes = max_prompt_bytes;
        self.max_messages = max_messages;
        self
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub messages: Option<Vec<Message>>,
    pub prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    MultiModal(Vec<ContentItem>),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ContentItem {
    #[serde(rename = "type")]
    pub item_type: String,
    pub text: Option<String>,
    pub image_url: Option<ImageUrl>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    pub content: Option<String>,
}

/// Render OpenAI-style chat messages into a single prompt string.
pub fn messages_to_prompt(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        let text = match &msg.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::MultiModal(items) => items
                .iter()
                .filter_map(|item| item.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        out.push_str(&format!("{}: {}\n", msg.role, text));
    }
    out
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if payload.model != state.model_name {
        return ApiError::bad_request(format!("Unsupported model: {}", payload.model))
            .into_response();
    }

    if let Some(m) = &payload.messages {
        if m.len() > state.max_messages {
            return ApiError::bad_request(format!(
                "Too many messages (max {})",
                state.max_messages
            ))
            .into_response();
        }
    }
    if let Some(mt) = payload.max_tokens {
        if mt == 0 || mt > state.max_tokens_cap {
            return ApiError::bad_request(format!(
                "max_tokens must be in 1..={}",
                state.max_tokens_cap
            ))
            .into_response();
        }
    }
    if let Some(t) = payload.temperature {
        if !(0.0..=2.0).contains(&t) {
            return ApiError::bad_request("temperature must be in [0, 2]").into_response();
        }
    }
    if let Some(p) = payload.top_p {
        if !(0.0..=1.0).contains(&p) || p == 0.0 {
            return ApiError::bad_request("top_p must be in (0, 1]").into_response();
        }
    }
    if let Some(metrics) = &state.metrics {
        metrics
            .requests_by_model
            .with_label_values(&[&state.model_name])
            .inc();
    }

    let prompt_text = match (&payload.prompt, &payload.messages) {
        (Some(p), _) => p.clone(),
        (None, Some(msgs)) => messages_to_prompt(msgs),
        (None, None) => {
            return ApiError::bad_request("Request must include either 'messages' or 'prompt'")
                .into_response();
        }
    };

    if prompt_text.len() > state.max_prompt_bytes {
        return ApiError::bad_request(format!("Prompt exceeds {} bytes", state.max_prompt_bytes))
            .into_response();
    }

    let prompt_tokens = match &state.tokenizer {
        Some(tok) => match tok.encode(&prompt_text) {
            Ok(ids) if !ids.is_empty() => ids,
            Ok(_) => {
                return ApiError::bad_request("Prompt produced no tokens").into_response();
            }
            Err(e) => {
                return ApiError::bad_request(format!("Failed to encode prompt: {}", e))
                    .into_response();
            }
        },
        None => {
            // No tokenizer configured (e.g. raw development mode): treat each
            // whitespace-separated word as a single token id derived from its hash.
            prompt_text
                .split_whitespace()
                .map(|w| (w.len() as u32) % 50 + 1)
                .collect()
        }
    };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
    let request_id = rand::random::<u64>();

    let request = Request {
        id: request_id,
        prompt_tokens,
        generated_tokens: Vec::new(),
        max_tokens: payload.max_tokens.unwrap_or(50),
        is_prefill: true,
        cached_prefix_len: 0,
        prefill_cursor: 0,
        temperature: payload.temperature.unwrap_or(1.0),
        top_p: payload.top_p.unwrap_or(1.0),
        token_sender: Some(tx),
        grammar_processor: match &payload.response_format {
            Some(rf) if rf.format_type == "json_object" => {
                Some(crate::api::grammar::GrammarLogitsProcessor::new(
                    crate::api::grammar::GrammarConstraint::Json,
                ))
            }
            _ => None,
        },
    };

    // Add request to scheduler
    {
        let mut sched = state.scheduler.lock().await;
        sched.add_request(request);
        if let Some(metrics) = &state.metrics {
            metrics
                .queue_depth
                .set((sched.waiting_queue.len() + sched.running_queue.len()) as f64);
        }
    }
    state.notify.notify_one();

    let model_name = state.model_name.clone();

    if payload.stream.unwrap_or(false) {
        let tokenizer = state.tokenizer.clone();
        let stream = async_stream::stream! {
            let mut ids: Vec<u32> = Vec::new();
            let mut prev_text = String::new();
            while let Some(token) = rx.recv().await {
                ids.push(token);
                let delta = match &tokenizer {
                    Some(tok) => {
                        let text = tok.decode(&ids).unwrap_or_default();
                        let delta = if text.starts_with(&prev_text) {
                            text[prev_text.len()..].to_string()
                        } else {
                            String::new()
                        };
                        prev_text = text;
                        delta
                    }
                    None => format!("token_{}", token),
                };
                let chunk = ChatCompletionStreamResponse {
                    id: format!("chatcmpl-{}", request_id),
                    object: "chat.completion.chunk".to_string(),
                    created: 1677652288,
                    model: model_name.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: Delta {
                            content: Some(delta),
                        },
                        finish_reason: None,
                    }],
                };
                yield Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()));
            }
            let final_chunk = ChatCompletionStreamResponse {
                id: format!("chatcmpl-{}", request_id),
                object: "chat.completion.chunk".to_string(),
                created: 1677652288,
                model: model_name.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: Delta { content: None },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            yield Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
        };

        Sse::new(stream).into_response()
    } else {
        let mut ids: Vec<u32> = Vec::new();
        while let Some(token) = rx.recv().await {
            ids.push(token);
        }

        let full_content = match &state.tokenizer {
            Some(tok) => tok.decode(&ids).unwrap_or_else(|_| {
                ids.iter()
                    .map(|t| format!("token_{}", t))
                    .collect::<Vec<_>>()
                    .join("")
            }),
            None => ids
                .iter()
                .map(|t| format!("token_{}", t))
                .collect::<Vec<_>>()
                .join(""),
        };

        Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", request_id),
            object: "chat.completion".to_string(),
            created: 1677652288,
            model: state.model_name.clone(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(full_content),
                },
                finish_reason: "stop".to_string(),
            }],
        })
        .into_response()
    }
}

pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match &state.registry {
        Some(registry) => {
            if let Some(metrics) = &state.metrics {
                let sched = state.scheduler.lock().await;
                metrics.cache_hits.set(sched.cache_hits as f64);
                metrics.cache_misses.set(sched.cache_misses as f64);
                metrics
                    .queue_depth
                    .set((sched.waiting_queue.len() + sched.running_queue.len()) as f64);
                drop(sched);
            }
            use prometheus::Encoder;
            let mut buffer = Vec::new();
            let encoder = prometheus::TextEncoder::new();
            match encoder.encode(&registry.gather(), &mut buffer) {
                Ok(_) => (
                    StatusCode::OK,
                    [("content-type", prometheus::TEXT_FORMAT)],
                    String::from_utf8(buffer).unwrap_or_default(),
                )
                    .into_response(),
                Err(e) => ApiError::internal(format!("Failed to encode metrics: {}", e))
                    .into_response(),
            }
        }
        None => ApiError::unavailable("Metrics registry not configured").into_response(),
    }
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": 1677652288u64,
            "owned_by": "kyro",
        }]
    }))
}

pub fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(|| async { "OK" }))
        .route("/v1/models", get(list_models))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}
