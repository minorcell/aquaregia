use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    InvalidRequest,
    AuthFailed,
    RateLimited,
    ProviderServerError,
    Transport,
    Timeout,
    StreamProtocol,
    UnknownTool,
    InvalidToolArgs,
    ToolExecutionFailed,
    MaxStepsExceeded,
    InvalidResponse,
    Cancelled,
}

#[derive(Debug, thiserror::Error, Clone, Serialize, Deserialize)]
#[error("{code:?}: {message}")]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
    pub provider: Option<String>,
    pub status: Option<u16>,
    pub retryable: bool,
    pub request_id: Option<String>,
    pub raw_body: Option<String>,
}

impl Error {
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

    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = request_id;
        self
    }

    pub fn with_raw_body(mut self, raw_body: Option<String>) -> Self {
        self.raw_body = raw_body;
        self
    }
}

pub fn classify_http_error(status: u16) -> ErrorCode {
    match status {
        401 | 403 => ErrorCode::AuthFailed,
        429 => ErrorCode::RateLimited,
        500..=599 => ErrorCode::ProviderServerError,
        400..=499 => ErrorCode::InvalidRequest,
        _ => ErrorCode::Transport,
    }
}

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
