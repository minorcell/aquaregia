//! Unified error types and HTTP-to-error mapping helpers for Aquaregia.
//!
//! This module provides a stable error categorization system that normalizes
//! errors from different providers into consistent [`ErrorCode`] variants.
//!
//! ## Error Handling Pattern
//!
//! ```rust,no_run
//! use aquaregia::{ErrorCode, GenerateTextRequest, LlmClient};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = LlmClient::openai("key").build()?;
//!
//! match client.generate(GenerateTextRequest::from_user_prompt("gpt-4o", "hello")).await {
//!     Ok(out) => println!("{}", out.output_text),
//!     Err(err) => match err.code {
//!         ErrorCode::RateLimited => eprintln!("rate limited; retry later"),
//!         ErrorCode::AuthFailed  => eprintln!("check API key"),
//!         ErrorCode::Cancelled   => eprintln!("request was cancelled"),
//!         _ => eprintln!("request failed: {}", err),
//!     },
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Retryable Errors
//!
//! The [`Error::retryable`] field indicates whether retrying the operation
//! might succeed. Errors like rate limits, server errors, and timeouts are
//! generally retryable, while authentication and validation errors are not.

use serde::{Deserialize, Serialize};

/// Stable error categories exposed by the SDK.
///
/// This enum provides a unified classification of errors that can occur
/// during LLM operations, normalizing provider-specific error conditions
/// into consistent categories.
///
/// ## Error Categories
///
/// - **Client Errors**: `InvalidRequest`, `AuthFailed`, `InvalidToolArgs`
/// - **Server Errors**: `ProviderServerError`, `RateLimited`, `Timeout`
/// - **Agent Errors**: `UnknownTool`, `ToolExecutionFailed`, `MaxStepsExceeded`
/// - **Protocol Errors**: `StreamProtocol`, `InvalidResponse`
/// - **Control Flow**: `Cancelled`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    /// Request shape or arguments are invalid before/after provider call.
    ///
    /// This error indicates a client-side validation failure, such as:
    /// - Empty messages or model name
    /// - Invalid temperature or top_p values
    /// - Malformed tool definitions
    InvalidRequest,
    /// Provider rejected authentication/authorization.
    ///
    /// This error indicates invalid or missing API credentials.
    /// Check your API key and provider configuration.
    AuthFailed,
    /// Provider rate-limited the request.
    ///
    /// This error indicates too many requests. Implement exponential backoff
    /// before retrying.
    RateLimited,
    /// Provider returned a 5xx class failure.
    ///
    /// This error indicates a server-side issue at the provider.
    /// Retrying after a delay may succeed.
    ProviderServerError,
    /// Network/transport failure not mapped to timeout.
    ///
    /// This error indicates connectivity issues, DNS failures, or
    /// other network-level problems.
    Transport,
    /// Request timed out.
    ///
    /// This error indicates the request exceeded the configured timeout.
    /// Consider increasing the timeout or retrying.
    Timeout,
    /// Streaming protocol payload was malformed or incomplete.
    ///
    /// This error indicates the provider's SSE stream did not conform
    /// to the expected format.
    StreamProtocol,
    /// Model requested a tool that is not registered.
    ///
    /// This error indicates the model tried to call a tool that wasn't
    /// provided in the agent's tool registry.
    UnknownTool,
    /// Tool arguments failed schema validation or deserialization.
    ///
    /// This error indicates the model generated invalid tool arguments
    /// that don't match the tool's JSON Schema.
    InvalidToolArgs,
    /// Tool execution failed.
    ///
    /// This error indicates a tool returned an error during execution.
    /// Handling depends on the configured [`ToolErrorPolicy`](crate::types::ToolErrorPolicy).
    ToolExecutionFailed,
    /// Agent exceeded configured max tool-loop steps.
    ///
    /// This error indicates the agent ran for more steps than allowed
    /// without producing a final answer.
    MaxStepsExceeded,
    /// Provider response shape could not be parsed/normalized.
    ///
    /// This error indicates the provider returned an unexpected response
    /// format that couldn't be parsed.
    InvalidResponse,
    /// Request was cancelled via cancellation token.
    ///
    /// This error indicates the operation was explicitly cancelled by
    /// the caller using a [`CancellationToken`](crate::CancellationToken).
    Cancelled,
}

/// Rich error payload returned by all fallible SDK operations.
///
/// This struct provides detailed error information including:
/// - Stable error category ([`ErrorCode`])
/// - Human-readable diagnostic message
/// - Provider identification
/// - HTTP status code (when applicable)
/// - Retryability hint
/// - Request tracing information
///
/// # Example
///
/// ```rust
/// use aquaregia::Error;
///
/// let err = Error::new(aquaregia::ErrorCode::RateLimited, "API rate limit exceeded");
/// println!("Error: {}", err);
/// println!("Retryable: {}", err.retryable);
/// ```
#[derive(Debug, thiserror::Error, Clone, Serialize, Deserialize)]
#[error("{code:?}: {message}")]
pub struct Error {
    /// High-level error category.
    pub code: ErrorCode,
    /// Human-readable diagnostic message.
    pub message: String,
    /// Provider slug when the error originated from a provider call.
    pub provider: Option<String>,
    /// HTTP status code, when applicable.
    pub status: Option<u16>,
    /// Indicates whether retrying might succeed.
    pub retryable: bool,
    /// Provider request id when present in response headers.
    pub request_id: Option<String>,
    /// Raw provider error body (best effort).
    pub raw_body: Option<String>,
    /// Seconds to wait before retrying, from a Retry-After header.
    pub retry_after_secs: Option<u64>,
}

impl Error {
    /// Creates a new error and derives `retryable` from [`ErrorCode`].
    ///
    /// This is the primary constructor for creating errors. The `retryable`
    /// field is automatically determined based on the error code.
    ///
    /// # Arguments
    ///
    /// * `code` - The error category
    /// * `message` - Human-readable diagnostic message
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        let code_value = code;
        Self {
            code,
            message: message.into(),
            provider: None,
            status: None,
            retryable: is_retryable(code_value),
            request_id: None,
            raw_body: None,
            retry_after_secs: None,
        }
    }

    /// Attaches the provider identifier to this error.
    ///
    /// This method is used to add context about which provider caused the error,
    /// useful for debugging multi-provider applications.
    ///
    /// # Arguments
    ///
    /// * `provider` - Provider slug (e.g., `"openai"`, `"anthropic"`)
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Attaches HTTP status metadata.
    ///
    /// This method adds the HTTP status code from the provider response,
    /// useful for debugging and error classification.
    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    /// Attaches provider request id metadata.
    ///
    /// This method adds the request ID from provider response headers,
    /// useful for tracing and support inquiries with the provider.
    pub(crate) fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = request_id;
        self
    }

    /// Attaches raw provider response body metadata.
    ///
    /// This method preserves the original provider error response body
    /// for debugging and diagnostics.
    pub(crate) fn with_raw_body(mut self, raw_body: Option<String>) -> Self {
        self.raw_body = raw_body;
        self
    }
}

/// Maps HTTP status codes to stable SDK [`ErrorCode`] values.
///
/// This function provides a consistent mapping from HTTP status codes
/// to SDK error categories, normalizing provider-specific HTTP responses.
///
/// # Arguments
///
/// * `status` - HTTP status code from provider response
///
/// # Mapping Rules
///
/// - `401`, `403` → [`ErrorCode::AuthFailed`]
/// - `429` → [`ErrorCode::RateLimited`]
/// - `500-599` → [`ErrorCode::ProviderServerError`]
/// - `400-499` → [`ErrorCode::InvalidRequest`]
/// - Others → [`ErrorCode::Transport`]
pub(crate) fn classify_http_error(status: u16) -> ErrorCode {
    match status {
        401 | 403 => ErrorCode::AuthFailed,
        429 => ErrorCode::RateLimited,
        500..=599 => ErrorCode::ProviderServerError,
        400..=499 => ErrorCode::InvalidRequest,
        _ => ErrorCode::Transport,
    }
}

/// Returns whether errors in this category are generally safe to retry.
///
/// This function determines if an error is transient and might succeed
/// on retry. Rate limits, server errors, transport issues, and timeouts
/// are considered retryable.
///
/// # Arguments
///
/// * `code` - Error code to check
pub(crate) fn is_retryable(code: ErrorCode) -> bool {
    matches!(
        code,
        ErrorCode::RateLimited
            | ErrorCode::ProviderServerError
            | ErrorCode::Transport
            | ErrorCode::Timeout
    )
}

/// Creates an error from a provider HTTP response.
///
/// This internal helper constructs a rich error with all available
/// metadata from the provider response.
///
/// # Arguments
///
/// * `provider_id` - Provider slug
/// * `status` - HTTP status code
/// * `body` - Optional response body
/// * `request_id` - Optional request ID from headers
pub(crate) fn provider_http_error(
    provider_id: &str,
    status: u16,
    body: Option<String>,
    request_id: Option<String>,
    retry_after_secs: Option<u64>,
) -> Error {
    let code = classify_http_error(status);
    let message = body
        .clone()
        .unwrap_or_else(|| format!("provider returned HTTP status {}", status));
    let mut err = Error::new(code, message)
        .with_provider(provider_id)
        .with_status(status)
        .with_raw_body(body)
        .with_request_id(request_id);
    err.retry_after_secs = retry_after_secs;
    err
}

/// Creates an error from a transport-level failure.
///
/// This internal helper constructs an error from reqwest client errors,
/// classifying timeouts separately from other transport issues.
///
/// # Arguments
///
/// * `provider_id` - Provider slug
/// * `err` - The reqwest error
pub(crate) fn transport_error(provider_id: &str, err: reqwest::Error) -> Error {
    let code = if err.is_timeout() {
        ErrorCode::Timeout
    } else {
        ErrorCode::Transport
    };
    Error::new(code, err.to_string()).with_provider(provider_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Error construction ──────────────────────────────────────────────

    #[test]
    fn error_new_derives_retryable() {
        let err = Error::new(ErrorCode::RateLimited, "too many requests");
        assert!(err.retryable);
        assert_eq!(err.code, ErrorCode::RateLimited);
        assert_eq!(err.message, "too many requests");
        assert!(err.provider.is_none());
        assert!(err.status.is_none());
        assert!(err.request_id.is_none());
        assert!(err.raw_body.is_none());
        assert!(err.retry_after_secs.is_none());
    }

    #[test]
    fn error_new_non_retryable() {
        let err = Error::new(ErrorCode::InvalidRequest, "bad request");
        assert!(!err.retryable);
    }

    #[test]
    fn error_with_provider() {
        let err = Error::new(ErrorCode::AuthFailed, "unauthorized").with_provider("openai");
        assert_eq!(err.provider, Some("openai".into()));
    }

    #[test]
    fn error_with_status() {
        let err = Error::new(ErrorCode::AuthFailed, "unauthorized").with_status(401);
        assert_eq!(err.status, Some(401));
    }

    #[test]
    fn error_with_request_id_some() {
        let err =
            Error::new(ErrorCode::Transport, "timeout").with_request_id(Some("req-123".into()));
        assert_eq!(err.request_id, Some("req-123".into()));
    }

    #[test]
    fn error_with_request_id_none() {
        let err = Error::new(ErrorCode::Transport, "timeout").with_request_id(None);
        assert_eq!(err.request_id, None);
    }

    #[test]
    fn error_with_raw_body_some() {
        let err =
            Error::new(ErrorCode::ProviderServerError, "error").with_raw_body(Some("body".into()));
        assert_eq!(err.raw_body, Some("body".into()));
    }

    #[test]
    fn error_with_raw_body_none() {
        let err = Error::new(ErrorCode::ProviderServerError, "error").with_raw_body(None);
        assert_eq!(err.raw_body, None);
    }

    #[test]
    fn error_display_format() {
        let err = Error::new(ErrorCode::RateLimited, "rate limit exceeded");
        let display = format!("{}", err);
        assert!(display.contains("RateLimited"));
        assert!(display.contains("rate limit exceeded"));
    }

    #[test]
    fn error_clone() {
        let err = Error::new(ErrorCode::Timeout, "timeout")
            .with_provider("google")
            .with_status(504);
        let cloned = err.clone();
        assert_eq!(cloned.code, ErrorCode::Timeout);
        assert_eq!(cloned.provider, Some("google".into()));
        assert_eq!(cloned.status, Some(504));
    }

    #[test]
    fn error_serialization_roundtrip() {
        let mut err = Error::new(ErrorCode::RateLimited, "rate limited")
            .with_provider("openai")
            .with_status(429)
            .with_request_id(Some("req-abc".into()))
            .with_raw_body(Some("{}".into()));
        err.retry_after_secs = Some(30);
        let json = serde_json::to_string(&err).unwrap();
        let back: Error = serde_json::from_str(&json).unwrap();
        assert_eq!(back.code, ErrorCode::RateLimited);
        assert_eq!(back.message, "rate limited");
        assert_eq!(back.provider, Some("openai".into()));
        assert_eq!(back.status, Some(429));
        assert_eq!(back.retry_after_secs, Some(30));
        assert!(back.retryable);
    }

    // ─── classify_http_error ─────────────────────────────────────────────

    #[test]
    fn classify_http_401_is_auth_failed() {
        assert_eq!(classify_http_error(401), ErrorCode::AuthFailed);
    }

    #[test]
    fn classify_http_403_is_auth_failed() {
        assert_eq!(classify_http_error(403), ErrorCode::AuthFailed);
    }

    #[test]
    fn classify_http_429_is_rate_limited() {
        assert_eq!(classify_http_error(429), ErrorCode::RateLimited);
    }

    #[test]
    fn classify_http_500_is_provider_server_error() {
        assert_eq!(classify_http_error(500), ErrorCode::ProviderServerError);
    }

    #[test]
    fn classify_http_502_is_provider_server_error() {
        assert_eq!(classify_http_error(502), ErrorCode::ProviderServerError);
    }

    #[test]
    fn classify_http_503_is_provider_server_error() {
        assert_eq!(classify_http_error(503), ErrorCode::ProviderServerError);
    }

    #[test]
    fn classify_http_400_is_invalid_request() {
        assert_eq!(classify_http_error(400), ErrorCode::InvalidRequest);
    }

    #[test]
    fn classify_http_404_is_invalid_request() {
        assert_eq!(classify_http_error(404), ErrorCode::InvalidRequest);
    }

    #[test]
    fn classify_http_422_is_invalid_request() {
        assert_eq!(classify_http_error(422), ErrorCode::InvalidRequest);
    }

    #[test]
    fn classify_http_200_is_transport() {
        assert_eq!(classify_http_error(200), ErrorCode::Transport);
    }

    #[test]
    fn classify_http_300_is_transport() {
        assert_eq!(classify_http_error(302), ErrorCode::Transport);
    }

    #[test]
    fn classify_http_418_is_invalid_request() {
        assert_eq!(classify_http_error(418), ErrorCode::InvalidRequest);
    }

    // ─── is_retryable ────────────────────────────────────────────────────

    #[test]
    fn retryable_rate_limited() {
        assert!(is_retryable(ErrorCode::RateLimited));
    }

    #[test]
    fn retryable_provider_server_error() {
        assert!(is_retryable(ErrorCode::ProviderServerError));
    }

    #[test]
    fn retryable_transport() {
        assert!(is_retryable(ErrorCode::Transport));
    }

    #[test]
    fn retryable_timeout() {
        assert!(is_retryable(ErrorCode::Timeout));
    }

    #[test]
    fn not_retryable_invalid_request() {
        assert!(!is_retryable(ErrorCode::InvalidRequest));
    }

    #[test]
    fn not_retryable_auth_failed() {
        assert!(!is_retryable(ErrorCode::AuthFailed));
    }

    #[test]
    fn not_retryable_cancelled() {
        assert!(!is_retryable(ErrorCode::Cancelled));
    }

    #[test]
    fn not_retryable_tool_execution_failed() {
        assert!(!is_retryable(ErrorCode::ToolExecutionFailed));
    }

    #[test]
    fn not_retryable_max_steps_exceeded() {
        assert!(!is_retryable(ErrorCode::MaxStepsExceeded));
    }

    // ─── provider_http_error ─────────────────────────────────────────────

    #[test]
    fn provider_http_error_with_body() {
        let err = provider_http_error("openai", 429, Some("rate limited".into()), None, None);
        assert_eq!(err.code, ErrorCode::RateLimited);
        assert_eq!(err.provider, Some("openai".into()));
        assert_eq!(err.status, Some(429));
        assert_eq!(err.raw_body, Some("rate limited".into()));
    }

    #[test]
    fn provider_http_error_without_body_uses_default_message() {
        let err = provider_http_error("google", 500, None, None, None);
        assert_eq!(err.code, ErrorCode::ProviderServerError);
        assert!(err.message.contains("500"));
    }

    #[test]
    fn provider_http_error_with_retry_after() {
        let err = provider_http_error(
            "anthropic",
            429,
            Some("rate limited".into()),
            Some("req-1".into()),
            Some(60),
        );
        assert_eq!(err.retry_after_secs, Some(60));
        assert_eq!(err.request_id, Some("req-1".into()));
    }

    #[test]
    fn provider_http_error_400_invalid_request() {
        let err = provider_http_error("openai", 400, Some("bad request".into()), None, None);
        assert_eq!(err.code, ErrorCode::InvalidRequest);
        assert!(!err.retryable);
    }

    // ─── transport_error ─────────────────────────────────────────────────

    #[tokio::test]
    async fn transport_error_regular() {
        let result = reqwest::Client::new()
            .get("http://0.0.0.0:0")
            .send()
            .await
            .err()
            .unwrap();
        let err = transport_error("openai", result);
        assert_eq!(err.code, ErrorCode::Transport);
        assert_eq!(err.provider, Some("openai".into()));
        assert!(err.retryable);
    }

    // ─── ErrorCode serialization ─────────────────────────────────────────

    #[test]
    fn error_code_serialization_roundtrip() {
        let codes = [
            ErrorCode::InvalidRequest,
            ErrorCode::AuthFailed,
            ErrorCode::RateLimited,
            ErrorCode::ProviderServerError,
            ErrorCode::Transport,
            ErrorCode::Timeout,
            ErrorCode::StreamProtocol,
            ErrorCode::UnknownTool,
            ErrorCode::InvalidToolArgs,
            ErrorCode::ToolExecutionFailed,
            ErrorCode::MaxStepsExceeded,
            ErrorCode::InvalidResponse,
            ErrorCode::Cancelled,
        ];
        for code in codes {
            let json = serde_json::to_string(&code).unwrap();
            let back: ErrorCode = serde_json::from_str(&json).unwrap();
            assert_eq!(code, back);
        }
    }
}
