use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiErrorCode {
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
}

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
#[error("{code:?}: {message}")]
pub struct AiError {
    pub code: AiErrorCode,
    pub message: String,
    pub provider: Option<String>,
    pub status: Option<u16>,
    pub retryable: bool,
    pub request_id: Option<String>,
    pub raw_body: Option<String>,
}

impl AiError {
    pub fn new(code: AiErrorCode, message: impl Into<String>) -> Self {
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

pub fn classify_http_error(status: u16) -> AiErrorCode {
    match status {
        401 | 403 => AiErrorCode::AuthFailed,
        429 => AiErrorCode::RateLimited,
        500..=599 => AiErrorCode::ProviderServerError,
        400..=499 => AiErrorCode::InvalidRequest,
        _ => AiErrorCode::Transport,
    }
}

pub fn is_retryable(code: AiErrorCode) -> bool {
    matches!(
        code,
        AiErrorCode::RateLimited
            | AiErrorCode::ProviderServerError
            | AiErrorCode::Transport
            | AiErrorCode::Timeout
    )
}

pub(crate) fn provider_http_error(
    provider_id: &str,
    status: u16,
    body: Option<String>,
    request_id: Option<String>,
) -> AiError {
    let code = classify_http_error(status);
    let message = body
        .clone()
        .unwrap_or_else(|| format!("provider returned HTTP status {}", status));
    AiError::new(code, message)
        .with_provider(provider_id)
        .with_status(status)
        .with_raw_body(body)
        .with_request_id(request_id)
}

pub(crate) fn transport_error(provider_id: &str, err: reqwest::Error) -> AiError {
    let code = if err.is_timeout() {
        AiErrorCode::Timeout
    } else {
        AiErrorCode::Transport
    };
    AiError::new(code, err.to_string()).with_provider(provider_id)
}
