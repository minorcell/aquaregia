//! Anthropic API adapter for Aquaregia.
//!
//! This module provides the `AnthropicAdapter` implementation for communicating
//! with Anthropic's Messages API.
//!
//! ## Features
//!
//! - Non-streaming and streaming text generation
//! - Thinking/reasoning content support (`thinking` blocks)
//! - Redacted thinking support
//! - Tool use with incremental JSON parsing
//! - Cache token tracking (prompt caching)
//!
//! ## Supported Models
//!
//! - Claude 3.5 Sonnet, Claude 3.5 Haiku
//! - Claude 3 Opus, Sonnet, Haiku
//! - Claude Sonnet 4 and newer
//!
//! ## Example
//!
//! ```rust,no_run
//! use aquaregia::{LlmClient, GenerateTextRequest};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = LlmClient::anthropic("api-key").build()?;
//!
//! let response = client
//!     .generate(GenerateTextRequest::from_user_prompt("claude-sonnet-4-5", "Hello!"))
//!     .await?;
//!
//! println!("{}", response.output_text);
//! # Ok(())
//! # }
//! ```

#![allow(clippy::collapsible_if)]
use std::collections::HashMap;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use futures_util::StreamExt;
use reqwest::header::CONTENT_TYPE;
use serde_json::{Map, Value, json};

use crate::error::{Error, ErrorCode};
use crate::model_adapters::{ModelAdapter, check_response_status, map_send_error};
use crate::stream::drain_sse_frames;
use crate::types::{
    Anthropic, ContentPart, FinishReason, GenerateTextRequest, GenerateTextResponse, ImagePart,
    MediaData, Message, MessageRole, ReasoningPart, StreamEvent, TextStream, ToolCall, Usage,
};

/// Provider slug used in ids and error metadata.
pub const PROVIDER_SLUG: &str = "anthropic";
/// Default Anthropic API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Default Anthropic API version header value.
pub const DEFAULT_API_VERSION: &str = "2023-06-01";

/// Runtime settings for the Anthropic adapter.
pub struct AnthropicAdapterSettings {
    /// Base URL for API requests.
    pub base_url: String,
    /// API key sent via `x-api-key`.
    pub api_key: String,
    /// API version sent via `anthropic-version`.
    pub api_version: String,
}

impl AnthropicAdapterSettings {
    /// Creates settings with default base URL and API version.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            api_version: DEFAULT_API_VERSION.to_string(),
        }
    }
}

/// Anthropic adapter implementation.
pub struct AnthropicAdapter {
    base_url: String,
    api_key: String,
    api_version: String,
    http: Arc<reqwest::Client>,
}

impl AnthropicAdapter {
    /// Creates an adapter from validated settings and shared HTTP client.
    pub fn from_settings(settings: AnthropicAdapterSettings, http: Arc<reqwest::Client>) -> Self {
        Self {
            base_url: settings.base_url,
            api_key: settings.api_key,
            api_version: settings.api_version,
            http,
        }
    }
}

#[async_trait]
impl ModelAdapter<Anthropic> for AnthropicAdapter {
    async fn generate_text(
        &self,
        req: &GenerateTextRequest<Anthropic>,
    ) -> Result<GenerateTextResponse, Error> {
        let payload = build_anthropic_payload(req, false);
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let cancel_token = req.cancellation_token.clone();
        let send_fut = self
            .http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header(CONTENT_TYPE, "application/json")
            .json(&payload)
            .send();
        let response = tokio::select! {
            r = send_fut => r.map_err(|e| map_send_error(PROVIDER_SLUG, e))?,
            _ = async move {
                match cancel_token {
                    Some(t) => t.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
            } => return Err(Error::new(ErrorCode::Cancelled, "request cancelled")),
        };
        let response = check_response_status(PROVIDER_SLUG, response).await?;
        let body: Value = response
            .json()
            .await
            .map_err(|e| Error::new(ErrorCode::InvalidResponse, e.to_string()))?;
        normalize_anthropic_response(body)
    }

    async fn stream_text(&self, req: &GenerateTextRequest<Anthropic>) -> Result<TextStream, Error> {
        let payload = build_anthropic_payload(req, true);
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let cancel_token = req.cancellation_token.clone();
        let cancel_token_stream = cancel_token.clone();
        let send_fut = self
            .http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header(CONTENT_TYPE, "application/json")
            .json(&payload)
            .send();
        let response = tokio::select! {
            r = send_fut => r.map_err(|e| map_send_error(PROVIDER_SLUG, e))?,
            _ = async move {
                match cancel_token {
                    Some(t) => t.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
            } => return Err(Error::new(ErrorCode::Cancelled, "request cancelled")),
        };
        let response = check_response_status(PROVIDER_SLUG, response).await?;
        let mut byte_stream = response.bytes_stream();

        let stream = try_stream! {
            let mut buffer = String::new();
            let mut pending_calls: HashMap<u64, PendingToolUse> = HashMap::new();
            let mut reasoning_blocks: HashMap<u64, (String, Option<Value>)> = HashMap::new();
            let mut done = false;

            while let Some(chunk) = byte_stream.next().await {
                if cancel_token_stream.as_ref().map(|t| t.is_cancelled()).unwrap_or(false) {
                    Err(Error::new(ErrorCode::Cancelled, "stream cancelled"))?;
                }
                let chunk = chunk.map_err(|e| Error::new(ErrorCode::Transport, e.to_string()))?;
                let text = std::str::from_utf8(&chunk)
                    .map_err(|e| Error::new(ErrorCode::StreamProtocol, e.to_string()))?;
                buffer.push_str(text);

                let frames = drain_sse_frames(&mut buffer);
                for frame in frames {
                    let data = frame.data.trim();
                    if data.is_empty() {
                        continue;
                    }

                    let value: Value = serde_json::from_str(data)
                        .map_err(|e| Error::new(ErrorCode::StreamProtocol, e.to_string()))?;

                    if let Some(event_type) = value.get("type").and_then(Value::as_str) {
                        match event_type {
                            "content_block_delta" => {
                                if let Some(delta_type) = value
                                    .get("delta")
                                    .and_then(|d| d.get("type"))
                                    .and_then(Value::as_str)
                                {
                                    match delta_type {
                                        "text_delta" => {
                                            if let Some(text_delta) = value
                                                .get("delta")
                                                .and_then(|d| d.get("text"))
                                                .and_then(Value::as_str)
                                            {
                                                if !text_delta.is_empty() {
                                                    yield StreamEvent::TextDelta {
                                                        text: text_delta.to_string(),
                                                    };
                                                }
                                            }
                                        }
                                        "thinking_delta" => {
                                            if let (Some(index), Some(reasoning_delta)) = (
                                                value.get("index").and_then(Value::as_u64),
                                                value
                                                    .get("delta")
                                                    .and_then(|d| d.get("thinking"))
                                                    .and_then(Value::as_str),
                                            ) {
                                                if !reasoning_delta.is_empty() {
                                                    let (block_id, _) = reasoning_blocks
                                                        .entry(index)
                                                        .or_insert_with(|| (format!("reasoning-{index}"), None));
                                                    yield StreamEvent::ReasoningDelta {
                                                        block_id: block_id.clone(),
                                                        text: reasoning_delta.to_string(),
                                                        provider_metadata: None,
                                                    };
                                                }
                                            }
                                        }
                                        "signature_delta" => {
                                            if let (Some(index), Some(signature)) = (
                                                value.get("index").and_then(Value::as_u64),
                                                value
                                                    .get("delta")
                                                    .and_then(|d| d.get("signature"))
                                                    .and_then(Value::as_str),
                                            ) {
                                                let metadata = Some(json!({
                                                    "anthropic": {
                                                        "signature": signature,
                                                    }
                                                }));
                                                let (block_id, block_metadata) = reasoning_blocks
                                                    .entry(index)
                                                    .or_insert_with(|| (format!("reasoning-{index}"), None));
                                                *block_metadata = metadata.clone();
                                                yield StreamEvent::ReasoningDelta {
                                                    block_id: block_id.clone(),
                                                    text: String::new(),
                                                    provider_metadata: metadata,
                                                };
                                            }
                                        }
                                        "input_json_delta" => {
                                            if let Some(index) = value.get("index").and_then(Value::as_u64) {
                                                let entry = pending_calls.entry(index).or_default();
                                                if let Some(partial_json) = value
                                                    .get("delta")
                                                    .and_then(|d| d.get("partial_json"))
                                                    .and_then(Value::as_str)
                                                {
                                                    entry.args_buf.push_str(partial_json);
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_start" => {
                                if let Some(index) = value.get("index").and_then(Value::as_u64) {
                                    let block = value.get("content_block").cloned().unwrap_or(Value::Null);
                                    match block.get("type").and_then(Value::as_str) {
                                        Some("tool_use") => {
                                            let entry = pending_calls.entry(index).or_default();
                                            entry.call_id = block.get("id").and_then(Value::as_str).map(ToOwned::to_owned);
                                            entry.tool_name = block.get("name").and_then(Value::as_str).map(ToOwned::to_owned);
                                            if let Some(input) = block.get("input") {
                                                entry.args_json = Some(input.clone());
                                            }
                                        }
                                        Some("thinking") => {
                                            let block_id = format!("reasoning-{index}");
                                            reasoning_blocks.insert(index, (block_id.clone(), None));
                                            yield StreamEvent::ReasoningStarted {
                                                block_id,
                                                provider_metadata: None,
                                            };
                                        }
                                        Some("redacted_thinking") => {
                                            let block_id = format!("reasoning-{index}");
                                            let metadata = Some(json!({
                                                "anthropic": {
                                                    "redacted_data": block.get("data").cloned().unwrap_or(Value::Null),
                                                }
                                            }));
                                            reasoning_blocks.insert(index, (block_id.clone(), metadata.clone()));
                                            yield StreamEvent::ReasoningStarted {
                                                block_id,
                                                provider_metadata: metadata,
                                            };
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_stop" => {
                                if let Some(index) = value.get("index").and_then(Value::as_u64) {
                                    if let Some((block_id, metadata)) = reasoning_blocks.remove(&index) {
                                        yield StreamEvent::ReasoningDone {
                                            block_id,
                                            provider_metadata: metadata,
                                        };
                                    }
                                    if let Some(pending) = pending_calls.remove(&index) {
                                        if let Some(call) = pending.into_tool_call()? {
                                            yield StreamEvent::ToolCallReady { call };
                                        }
                                    }
                                }
                            }
                            "message_delta" => {
                                if let Some(usage) = value.get("usage").and_then(parse_anthropic_usage) {
                                    yield StreamEvent::Usage { usage };
                                }
                            }
                            "message_stop" => {
                                for (_, (block_id, metadata)) in reasoning_blocks.drain() {
                                    yield StreamEvent::ReasoningDone {
                                        block_id,
                                        provider_metadata: metadata,
                                    };
                                }
                                done = true;
                                yield StreamEvent::Done;
                            }
                            _ => {}
                        }
                    }
                }

                if done {
                    break;
                }
            }

            if !done {
                for (_, (block_id, metadata)) in reasoning_blocks.drain() {
                    yield StreamEvent::ReasoningDone {
                        block_id,
                        provider_metadata: metadata,
                    };
                }
                Err(Error::new(
                    ErrorCode::StreamProtocol,
                    "anthropic stream closed without message_stop",
                ))?;
            }
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Default)]
struct PendingToolUse {
    call_id: Option<String>,
    tool_name: Option<String>,
    args_json: Option<Value>,
    args_buf: String,
}

impl PendingToolUse {
    fn into_tool_call(self) -> Result<Option<ToolCall>, Error> {
        let Some(call_id) = self.call_id else {
            return Ok(None);
        };
        let Some(tool_name) = self.tool_name else {
            return Ok(None);
        };
        let args_json = if !self.args_buf.trim().is_empty() {
            serde_json::from_str(&self.args_buf).map_err(|e| {
                Error::new(
                    ErrorCode::InvalidToolArgs,
                    format!("invalid anthropic streamed tool arguments: {}", e),
                )
            })?
        } else {
            self.args_json.unwrap_or_else(|| json!({}))
        };

        Ok(Some(ToolCall {
            call_id,
            tool_name,
            args_json,
        }))
    }
}

fn build_anthropic_payload(req: &GenerateTextRequest<Anthropic>, stream: bool) -> Value {
    let mut payload = Map::new();
    payload.insert(
        "model".to_string(),
        Value::String(req.model.model().to_string()),
    );
    payload.insert(
        "messages".to_string(),
        Value::Array(
            req.messages
                .iter()
                .filter(|msg| msg.role != MessageRole::System)
                .map(to_anthropic_message)
                .collect(),
        ),
    );
    payload.insert("stream".to_string(), Value::Bool(stream));
    payload.insert(
        "max_tokens".to_string(),
        Value::from(req.max_output_tokens.unwrap_or(1024)),
    );

    let system = req
        .messages
        .iter()
        .filter(|msg| msg.role == MessageRole::System)
        .flat_map(|msg| {
            msg.parts.iter().filter_map(|part| match part {
                ContentPart::Text(text) => Some(text.clone()),
                _ => None,
            })
        })
        .collect::<Vec<_>>()
        .join("\n");
    if !system.is_empty() {
        payload.insert("system".to_string(), Value::String(system));
    }

    if let Some(temperature) = req.temperature {
        payload.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = req.top_p {
        payload.insert("top_p".to_string(), Value::from(top_p));
    }
    if !req.stop_sequences.is_empty() {
        payload.insert(
            "stop_sequences".to_string(),
            Value::Array(
                req.stop_sequences
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect(),
            ),
        );
    }
    if let Some(tools) = &req.tools {
        payload.insert(
            "tools".to_string(),
            Value::Array(
                tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        })
                    })
                    .collect(),
            ),
        );
    }

    Value::Object(payload)
}

fn to_anthropic_message(message: &Message) -> Value {
    match message.role {
        MessageRole::User | MessageRole::Assistant => {
            let role = if message.role == MessageRole::User {
                "user"
            } else {
                "assistant"
            };
            let mut content = Vec::new();
            for part in &message.parts {
                match part {
                    ContentPart::Text(text) => content.push(json!({
                        "type": "text",
                        "text": text,
                    })),
                    ContentPart::Image(image) => {
                        content.push(anthropic_image_block(image));
                    }
                    ContentPart::Reasoning(reasoning) => {
                        let signature = reasoning
                            .provider_metadata
                            .as_ref()
                            .and_then(|meta| meta.get("anthropic"))
                            .and_then(|meta| meta.get("signature"))
                            .and_then(Value::as_str);
                        let redacted_data = reasoning
                            .provider_metadata
                            .as_ref()
                            .and_then(|meta| meta.get("anthropic"))
                            .and_then(|meta| meta.get("redacted_data"))
                            .cloned();

                        if let Some(signature) = signature {
                            content.push(json!({
                                "type": "thinking",
                                "thinking": reasoning.text,
                                "signature": signature,
                            }));
                        } else if let Some(redacted_data) = redacted_data {
                            content.push(json!({
                                "type": "redacted_thinking",
                                "data": redacted_data,
                            }));
                        } else if !reasoning.text.is_empty() {
                            // Fall back to text when anthropic-specific reasoning metadata is missing.
                            content.push(json!({
                                "type": "text",
                                "text": reasoning.text,
                            }));
                        }
                    }
                    ContentPart::ToolCall(call) => content.push(json!({
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.tool_name,
                        "input": call.args_json,
                    })),
                    ContentPart::ToolResult(_) => {}
                }
            }
            json!({
                "role": role,
                "content": content,
            })
        }
        MessageRole::Tool => {
            let tool_result = message.parts.iter().find_map(|part| {
                if let ContentPart::ToolResult(result) = part {
                    Some(result)
                } else {
                    None
                }
            });
            if let Some(result) = tool_result {
                json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.output_json.to_string(),
                        "is_error": result.is_error,
                    }]
                })
            } else {
                json!({
                    "role": "user",
                    "content": [],
                })
            }
        }
        MessageRole::System => json!({
            "role": "user",
            "content": [],
        }),
    }
}

fn anthropic_image_block(image: &ImagePart) -> Value {
    match &image.data {
        MediaData::Url(url) => json!({
            "type": "image",
            "source": { "type": "url", "url": url }
        }),
        MediaData::Base64(b64) => {
            let media_type = image.media_type.as_deref().unwrap_or("image/jpeg");
            json!({
                "type": "image",
                "source": { "type": "base64", "media_type": media_type, "data": b64 }
            })
        }
        MediaData::Bytes(bytes) => {
            let data = STANDARD.encode(bytes);
            let media_type = image.media_type.as_deref().unwrap_or("image/jpeg");
            json!({
                "type": "image",
                "source": { "type": "base64", "media_type": media_type, "data": data }
            })
        }
    }
}

fn normalize_anthropic_response(body: Value) -> Result<GenerateTextResponse, Error> {
    let content = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidResponse,
                "anthropic response missing content",
            )
        })?;

    let mut output_text = String::new();
    let mut reasoning_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    output_text.push_str(text);
                }
            }
            Some("thinking") => {
                let thinking = block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let signature = block
                    .get("signature")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                reasoning_parts.push(ReasoningPart {
                    text: thinking,
                    provider_metadata: signature.map(|signature| {
                        json!({
                            "anthropic": {
                                "signature": signature,
                            }
                        })
                    }),
                });
            }
            Some("redacted_thinking") => {
                reasoning_parts.push(ReasoningPart {
                    text: String::new(),
                    provider_metadata: Some(json!({
                        "anthropic": {
                            "redacted_data": block.get("data").cloned().unwrap_or(Value::Null),
                        }
                    })),
                });
            }
            Some("tool_use") => {
                let call_id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::new(ErrorCode::InvalidResponse, "tool_use missing id"))?;
                let tool_name = block.get("name").and_then(Value::as_str).ok_or_else(|| {
                    Error::new(ErrorCode::InvalidResponse, "tool_use missing name")
                })?;
                let args_json = block.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(ToolCall {
                    call_id: call_id.to_string(),
                    tool_name: tool_name.to_string(),
                    args_json,
                });
            }
            _ => {}
        }
    }
    let reasoning_text = reasoning_parts
        .iter()
        .map(|part| part.text.as_str())
        .collect::<String>();

    let finish_reason = map_anthropic_finish_reason(
        body.get("stop_reason")
            .and_then(Value::as_str)
            .unwrap_or("end_turn"),
    );

    let usage = body
        .get("usage")
        .and_then(parse_anthropic_usage)
        .unwrap_or_default();

    Ok(GenerateTextResponse {
        output_text,
        reasoning_text,
        reasoning_parts,
        finish_reason,
        usage,
        tool_calls,
        raw_provider_response: Some(body),
    })
}

fn parse_anthropic_usage(value: &Value) -> Option<Usage> {
    let base_input_tokens = value.get("input_tokens")?.as_u64()? as u32;
    let base_output_tokens = value.get("output_tokens")?.as_u64()? as u32;

    let (no_cache_input_tokens, output_tokens) =
        parse_anthropic_iteration_totals(value).unwrap_or((base_input_tokens, base_output_tokens));

    let cache_read_tokens = value
        .get("cache_read_input_tokens")
        .and_then(Value::as_u64)
        .map(|v| v as u32)
        .unwrap_or(0);
    let cache_write_tokens = value
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .map(|v| v as u32)
        .unwrap_or(0);

    let input_tokens = no_cache_input_tokens
        .saturating_add(cache_read_tokens)
        .saturating_add(cache_write_tokens);

    Some(
        Usage::from_totals(input_tokens, output_tokens, 0, None)
            .with_input_cache_split(cache_read_tokens, cache_write_tokens)
            .with_output_split(output_tokens, 0)
            .with_raw_usage(value.clone()),
    )
}

fn parse_anthropic_iteration_totals(value: &Value) -> Option<(u32, u32)> {
    let iterations = value.get("iterations")?.as_array()?;
    if iterations.is_empty() {
        return None;
    }

    let mut input_total = 0u32;
    let mut output_total = 0u32;
    for item in iterations {
        let input = item.get("input_tokens")?.as_u64()? as u32;
        let output = item.get("output_tokens")?.as_u64()? as u32;
        input_total = input_total.saturating_add(input);
        output_total = output_total.saturating_add(output);
    }
    Some((input_total, output_total))
}

fn map_anthropic_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        _ => FinishReason::Unknown(reason.to_string()),
    }
}
