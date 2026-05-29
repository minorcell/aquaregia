//! Provider adapter traits and concrete provider implementations for Aquaregia.
//!
//! This module defines the adapter abstraction that allows Aquaregia to support
//! multiple LLM providers through a unified interface:
//!
//! - [`ModelAdapter<P>`]: Trait for provider-specific request/response handling
//! - Provider implementations:
//!   - `openai::OpenAiAdapter`: OpenAI API adapter
//!   - `anthropic::AnthropicAdapter`: Anthropic API adapter
//!   - `google::GoogleAdapter`: Google Generative AI API adapter
//!   - `openai_compatible::OpenAiCompatibleAdapter`: OpenAI-compatible endpoints
//!
//! ## Adapter Architecture
//!
//! Each provider adapter implements the [`ModelAdapter`] trait which defines:
//! - `generate_text`: Non-streaming text generation
//! - `stream_text`: Streaming text generation with SSE parsing
//!
//! ## Creating Adapters
//!
//! Adapters are typically created through [`crate::ClientBuilder`] which handles
//! the configuration and HTTP client setup:
//!
//! ```rust,no_run
//! use aquaregia::LlmClient;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // OpenAI adapter
//! let openai_client = LlmClient::openai().api_key("api-key").build()?;
//!
//! // Anthropic adapter
//! let anthropic_client = LlmClient::anthropic().api_key("api-key").build()?;
//!
//! // Google adapter
//! let google_client = LlmClient::google().api_key("api-key").build()?;
//!
//! // OpenAI-compatible adapter (e.g., DeepSeek, local LLMs)
//! let compatible_client = LlmClient::openai_compatible().base_url("https://api.example.com")
//!     .api_key("api-key")
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::Response;

use crate::error::Error;
use crate::types::{GenerateTextRequest, GenerateTextResponse, TextStream};

/// Anthropic provider adapter implementation.
pub mod anthropic;
/// Google provider adapter implementation.
pub mod google;
/// OpenAI provider adapter implementation.
pub mod openai;
/// OpenAI-compatible provider adapter implementation.
pub mod openai_compatible;

/// Provider adapter contract used by [`crate::BoundClient`].
#[async_trait]
pub trait ModelAdapter: Send + Sync {
    async fn generate_text(&self, req: &GenerateTextRequest)
    -> Result<GenerateTextResponse, Error>;
    async fn stream_text(&self, req: &GenerateTextRequest) -> Result<TextStream, Error>;
}

pub(crate) async fn check_response_status(
    provider_id: &str,
    response: Response,
) -> Result<Response, Error> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let headers = response.headers();
    let request_id = headers
        .get("x-request-id")
        .or_else(|| headers.get("request-id"))
        .and_then(|v| v.to_str().ok())
        .map(ToOwned::to_owned);

    let retry_after_secs = headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());

    let body = response.text().await.ok();
    Err(crate::error::provider_http_error(
        provider_id,
        status.as_u16(),
        body,
        request_id,
        retry_after_secs,
    ))
}

pub(crate) fn map_send_error(provider_id: &str, err: reqwest::Error) -> Error {
    crate::error::transport_error(provider_id, err)
}

pub(crate) fn base64_encode(data: &[u8]) -> String {
    STANDARD.encode(data)
}

/// Merges the slug-keyed block of a `provider_options` JSON object into
/// `target`. Non-object blocks and missing slugs are silently skipped, matching
/// the opaque-passthrough contract documented for `provider_options`.
pub(crate) fn merge_provider_options(
    target: &mut serde_json::Map<String, serde_json::Value>,
    source: Option<&serde_json::Value>,
    slug: &str,
) {
    if let Some(obj) = source
        .and_then(|v| v.get(slug))
        .and_then(serde_json::Value::as_object)
    {
        for (key, value) in obj {
            target.insert(key.clone(), value.clone());
        }
    }
}

/// Builds an `InvalidRequest` error for a `FilePart` whose `media_type` the
/// adapter cannot translate. Subtype-level support is decided by the upstream
/// API at request time; this helper is only for top-level categories the
/// adapter has no representation for (e.g. PDFs in a chat-completions-only
/// path, or arbitrary types in an image-only path).
pub(crate) fn unsupported_media_type(slug: &str, media_type: &str) -> crate::error::Error {
    crate::error::Error::new(
        crate::error::ErrorCode::InvalidRequest,
        format!(
            "{} adapter does not support media_type {} for file parts",
            slug, media_type
        ),
    )
}
