use std::sync::Arc;

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

/// Shared adapter object used internally by provider-bound clients.
pub type SharedAdapter<P> = Arc<dyn ModelAdapter<P>>;

pub(crate) async fn check_response_status(
    provider_id: &str,
    response: Response,
) -> Result<Response, Error> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let request_id = response
        .headers()
        .get("x-request-id")
        .or_else(|| response.headers().get("request-id"))
        .and_then(|v| v.to_str().ok())
        .map(ToOwned::to_owned);

    let body = response.text().await.ok();
    Err(crate::error::provider_http_error(
        provider_id,
        status.as_u16(),
        body,
        request_id,
    ))
}

pub(crate) fn map_send_error(provider_id: &str, err: reqwest::Error) -> Error {
    crate::error::transport_error(provider_id, err)
}
