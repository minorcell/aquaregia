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
//! let openai_client = LlmClient::openai("api-key").build()?;
//!
//! // Anthropic adapter
//! let anthropic_client = LlmClient::anthropic("api-key").build()?;
//!
//! // Google adapter
//! let google_client = LlmClient::google("api-key").build()?;
//!
//! // OpenAI-compatible adapter (e.g., DeepSeek, local LLMs)
//! let compatible_client = LlmClient::openai_compatible("https://api.example.com")
//!     .api_key("api-key")
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use reqwest::Response;

use crate::error::Error;
use crate::types::{GenerateTextRequest, GenerateTextResponse, ProviderMarker, TextStream};

/// Anthropic provider adapter implementation.
pub mod anthropic;
/// Google provider adapter implementation.
pub mod google;
/// OpenAI provider adapter implementation.
pub mod openai;
/// OpenAI-compatible provider adapter implementation.
pub mod openai_compatible;
pub(crate) mod think_tag_parser;

/// Provider adapter contract used by [`crate::BoundClient`].
///
/// End users typically do not implement this trait directly unless integrating
/// a custom in-tree provider adapter.
#[async_trait]
pub trait ModelAdapter<P: ProviderMarker>: Send + Sync {
    /// Executes a non-streaming text generation call.
    async fn generate_text(
        &self,
        req: &GenerateTextRequest<P>,
    ) -> Result<GenerateTextResponse, Error>;

    /// Executes a streaming text generation call.
    async fn stream_text(&self, req: &GenerateTextRequest<P>) -> Result<TextStream, Error>;
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
