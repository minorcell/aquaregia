use serde::{Deserialize, Serialize};

/// Stable error categories exposed by the SDK.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    /// Request shape or arguments are invalid before/after provider call.
    InvalidRequest,
    /// Provider rejected authentication/authorization.
    AuthFailed,
    /// Provider rate-limited the request.
    RateLimited,
    /// Provider returned a 5xx class failure.
    ProviderServerError,
    /// Network/transport failure not mapped to timeout.
    Transport,
    /// Request timed out.
    Timeout,
    /// Streaming protocol payload was malformed or incomplete.
    StreamProtocol,
    /// Model requested a tool that is not registered.
    UnknownTool,
    /// Tool arguments failed schema validation or deserialization.
    InvalidToolArgs,
    /// Tool execution failed.
    ToolExecutionFailed,
    /// Agent exceeded configured max tool-loop steps.
    MaxStepsExceeded,
    /// Provider response shape could not be parsed/normalized.
    InvalidResponse,
    /// Request was cancelled via cancellation token.
    Cancelled,
}

/// Rich error payload returned by all fallible SDK operations.
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
}

impl Error {
    /// Creates a new error and derives `retryable` from [`ErrorCode`].
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
        }
    }

    /// Attaches the provider identifier to this error.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Attaches HTTP status metadata.
    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    /// Attaches provider request id metadata.
    pub fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = request_id;
        self
    }

    /// Attaches raw provider response body metadata.
    pub fn with_raw_body(mut self, raw_body: Option<String>) -> Self {
        self.raw_body = raw_body;
        self
    }
}

/// Maps HTTP status codes to stable SDK [`ErrorCode`] values.
pub fn classify_http_error(status: u16) -> ErrorCode {
    match status {
        401 | 403 => ErrorCode::AuthFailed,
        429 => ErrorCode::RateLimited,
        500..=599 => ErrorCode::ProviderServerError,
        400..=499 => ErrorCode::InvalidRequest,
        _ => ErrorCode::Transport,
    }
}

/// Returns whether errors in this category are generally safe to retry.
pub fn is_retryable(code: ErrorCode) -> bool {
    matches!(
        code,
        ErrorCode::RateLimited
            | ErrorCode::ProviderServerError
            | ErrorCode::Transport
            | ErrorCode::Timeout
    )
}

pub(crate) fn provider_http_error(
    provider_id: &str,
    status: u16,
    body: Option<String>,
    request_id: Option<String>,
) -> Error {
    let code = classify_http_error(status);
    let message = body
        .clone()
        .unwrap_or_else(|| format!("provider returned HTTP status {}", status));
    Error::new(code, message)
        .with_provider(provider_id)
        .with_status(status)
        .with_raw_body(body)
        .with_request_id(request_id)
}

pub(crate) fn transport_error(provider_id: &str, err: reqwest::Error) -> Error {
    let code = if err.is_timeout() {
        ErrorCode::Timeout
    } else {
        ErrorCode::Transport
    };
    Error::new(code, err.to_string()).with_provider(provider_id)
}
