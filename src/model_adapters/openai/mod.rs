//! OpenAI API adapter for Aquaregia.
//!
//! This module provides the `OpenAiAdapter` implementation for communicating
//! with OpenAI's Chat Completions API.
//!
//! ## Features
//!
//! - Non-streaming and streaming text generation
//! - Reasoning content extraction (`reasoning_content` field)
//! - Tool/function calling support
//! - Usage token parsing with cache details
//!
//! ## Supported Models
//!
//! - GPT-4o, GPT-4o-mini, GPT-4 Turbo
//! - o1, o3 reasoning models
//! - Legacy GPT-3.5 Turbo
//!
//! ## Example
//!
//! ```rust,no_run
//! use aquaregia::{LlmClient, GenerateTextRequest};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = LlmClient::openai("api-key").build()?;
//!
//! let response = client
//!     .generate(GenerateTextRequest::from_user_prompt("gpt-4o", "Hello!"))
//!     .await?;
//!
//! println!("{}", response.output_text);
//! # Ok(())
//! # }
//! ```

#![allow(clippy::collapsible_if)]
use std::collections::BTreeMap;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::{Map, Value, json};

use crate::error::{Error, ErrorCode};
use crate::model_adapters::{ModelAdapter, check_response_status, map_send_error};
use crate::stream::drain_sse_frames;
use crate::types::{
    ContentPart, FinishReason, GenerateTextRequest, GenerateTextResponse, ImagePart, MediaData,
    Message, MessageRole, OpenAi, ReasoningPart, StreamEvent, TextStream, ToolCall, Usage,
};

/// Provider slug used in ids and error metadata.
pub const PROVIDER_SLUG: &str = "openai";

/// Default OpenAI API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Runtime settings for the OpenAI adapter.
///
/// Contains configuration for API endpoint and authentication.
pub struct OpenAiAdapterSettings {
    /// Base URL for API requests.
    pub base_url: String,
    /// API key sent as bearer token.
    pub api_key: String,
}

impl OpenAiAdapterSettings {
    /// Creates settings with default base URL.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
        }
    }
}

/// OpenAI adapter implementation.
pub struct OpenAiAdapter {
    base_url: String,
    api_key: String,
    http: Arc<reqwest::Client>,
}

impl OpenAiAdapter {
    /// Creates an adapter from validated settings and shared HTTP client.
    pub fn from_settings(settings: OpenAiAdapterSettings, http: Arc<reqwest::Client>) -> Self {
        Self {
            base_url: settings.base_url,
            api_key: settings.api_key,
            http,
        }
    }
}

#[async_trait]
impl ModelAdapter<OpenAi> for OpenAiAdapter {
    async fn generate_text(
        &self,
        req: &GenerateTextRequest<OpenAi>,
    ) -> Result<GenerateTextResponse, Error> {
        let payload = build_openai_payload(req, false);
        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let cancel_token = req.cancellation_token.clone();
        let send_fut = self
            .http
            .post(url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
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
        normalize_openai_response(body)
    }

    async fn stream_text(&self, req: &GenerateTextRequest<OpenAi>) -> Result<TextStream, Error> {
        let payload = build_openai_payload(req, true);
        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let cancel_token = req.cancellation_token.clone();
        let cancel_token_stream = cancel_token.clone();
        let send_fut = self
            .http
            .post(url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
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
            let mut tool_partial: BTreeMap<usize, PartialToolCall> = BTreeMap::new();
            let mut done = false;
            let mut saw_payload_frame = false;
            let mut reasoning_active = false;
            let reasoning_block_id = "reasoning-0".to_string();

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
                    saw_payload_frame = true;
                    let data = frame.data.trim();
                    if data == "[DONE]" {
                        if reasoning_active {
                            yield StreamEvent::ReasoningDone {
                                block_id: reasoning_block_id.clone(),
                                provider_metadata: None,
                            };
                            reasoning_active = false;
                        }
                        done = true;
                        yield StreamEvent::Done;
                        break;
                    }

                    let value: Value = serde_json::from_str(data)
                        .map_err(|e| Error::new(ErrorCode::StreamProtocol, e.to_string()))?;
                    if let Some(reasoning_delta) = value
                        .get("choices")
                        .and_then(Value::as_array)
                        .and_then(|arr| arr.first())
                        .and_then(|choice| choice.get("delta"))
                        .and_then(|delta| {
                            delta
                                .get("reasoning_content")
                                .or_else(|| delta.get("reasoning"))
                        })
                        .and_then(Value::as_str)
                    {
                        if !reasoning_delta.is_empty() {
                            if !reasoning_active {
                                yield StreamEvent::ReasoningStarted {
                                    block_id: reasoning_block_id.clone(),
                                    provider_metadata: None,
                                };
                                reasoning_active = true;
                            }
                            yield StreamEvent::ReasoningDelta {
                                block_id: reasoning_block_id.clone(),
                                text: reasoning_delta.to_string(),
                                provider_metadata: None,
                            };
                        }
                    }
                    if let Some(text_delta) = value
                        .get("choices")
                        .and_then(Value::as_array)
                        .and_then(|arr| arr.first())
                        .and_then(|choice| choice.get("delta"))
                        .and_then(|delta| delta.get("content"))
                        .and_then(Value::as_str)
                    {
                        if !text_delta.is_empty() {
                            if reasoning_active {
                                yield StreamEvent::ReasoningDone {
                                    block_id: reasoning_block_id.clone(),
                                    provider_metadata: None,
                                };
                                reasoning_active = false;
                            }
                            yield StreamEvent::TextDelta {
                                text: text_delta.to_string(),
                            };
                        }
                    }

                    if let Some(tool_calls) = value
                        .get("choices")
                        .and_then(Value::as_array)
                        .and_then(|arr| arr.first())
                        .and_then(|choice| choice.get("delta"))
                        .and_then(|delta| delta.get("tool_calls"))
                        .and_then(Value::as_array)
                    {
                        if reasoning_active {
                            yield StreamEvent::ReasoningDone {
                                block_id: reasoning_block_id.clone(),
                                provider_metadata: None,
                            };
                            reasoning_active = false;
                        }
                        for call in tool_calls {
                            if let Some(index) = call.get("index").and_then(Value::as_u64) {
                                let index = index as usize;
                                let entry = tool_partial.entry(index).or_default();

                                if let Some(id) = call.get("id").and_then(Value::as_str) {
                                    entry.call_id = Some(id.to_string());
                                }
                                if let Some(name) = call
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(Value::as_str)
                                {
                                    entry.tool_name = Some(name.to_string());
                                }
                                if let Some(args) = call
                                    .get("function")
                                    .and_then(|f| f.get("arguments"))
                                    .and_then(Value::as_str)
                                {
                                    entry.args_buf.push_str(args);
                                }
                            }
                        }
                    }

                    if let Some(usage) = value.get("usage").and_then(parse_openai_usage) {
                        yield StreamEvent::Usage { usage };
                    }

                    if let Some(finish_reason) = value
                        .get("choices")
                        .and_then(Value::as_array)
                        .and_then(|arr| arr.first())
                        .and_then(|choice| choice.get("finish_reason"))
                        .and_then(Value::as_str)
                    {
                        if finish_reason == "tool_calls" && !tool_partial.is_empty() {
                            for partial in tool_partial.values() {
                                if let Some(call) = partial.to_tool_call()? {
                                    yield StreamEvent::ToolCallReady { call };
                                }
                            }
                            tool_partial.clear();
                        }
                    }
                }

                if done {
                    break;
                }
            }

            if reasoning_active {
                yield StreamEvent::ReasoningDone {
                    block_id: reasoning_block_id,
                    provider_metadata: None,
                };
            }

            if !done {
                if saw_payload_frame {
                    // Some OpenAI-compatible servers close the stream after final chunk
                    // without emitting a terminal `[DONE]` marker.
                    yield StreamEvent::Done;
                } else {
                    Err(Error::new(
                        ErrorCode::StreamProtocol,
                        "openai stream closed without payload",
                    ))?;
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Default)]
struct PartialToolCall {
    call_id: Option<String>,
    tool_name: Option<String>,
    args_buf: String,
}

impl PartialToolCall {
    fn to_tool_call(&self) -> Result<Option<ToolCall>, Error> {
        let Some(call_id) = self.call_id.clone() else {
            return Ok(None);
        };
        let Some(tool_name) = self.tool_name.clone() else {
            return Ok(None);
        };
        let args_json = if self.args_buf.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&self.args_buf).map_err(|e| {
                Error::new(
                    ErrorCode::InvalidToolArgs,
                    format!("invalid streamed tool arguments: {}", e),
                )
            })?
        };
        Ok(Some(ToolCall {
            call_id,
            tool_name,
            args_json,
        }))
    }
}

fn build_openai_payload(req: &GenerateTextRequest<OpenAi>, stream: bool) -> Value {
    let mut payload = Map::new();
    payload.insert(
        "model".to_string(),
        Value::String(req.model.model().to_string()),
    );
    payload.insert(
        "messages".to_string(),
        Value::Array(req.messages.iter().map(to_openai_message).collect()),
    );
    payload.insert("stream".to_string(), Value::Bool(stream));

    if let Some(temperature) = req.temperature {
        payload.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = req.top_p {
        payload.insert("top_p".to_string(), Value::from(top_p));
    }
    if let Some(max_output_tokens) = req.max_output_tokens {
        payload.insert("max_tokens".to_string(), Value::from(max_output_tokens));
    }
    if !req.stop_sequences.is_empty() {
        payload.insert(
            "stop".to_string(),
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
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            }
                        })
                    })
                    .collect(),
            ),
        );
    }

    Value::Object(payload)
}

fn to_openai_message(message: &Message) -> Value {
    match message.role {
        MessageRole::System => json!({
            "role": "system",
            "content": text_content_from_parts(&message.parts),
        }),
        MessageRole::User => json!({
            "role": "user",
            "content": openai_user_content(&message.parts),
        }),
        MessageRole::Assistant => {
            let reasoning_content = reasoning_content_from_parts(&message.parts);
            let tool_calls: Vec<Value> = message
                .parts
                .iter()
                .filter_map(|part| {
                    if let ContentPart::ToolCall(call) = part {
                        Some(json!({
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.tool_name,
                                "arguments": call.args_json.to_string(),
                            }
                        }))
                    } else {
                        None
                    }
                })
                .collect();
            let mut payload = Map::new();
            payload.insert("role".to_string(), Value::String("assistant".to_string()));
            payload.insert(
                "content".to_string(),
                text_content_from_parts(&message.parts),
            );
            if !reasoning_content.is_empty() {
                payload.insert(
                    "reasoning_content".to_string(),
                    Value::String(reasoning_content),
                );
            }
            if !tool_calls.is_empty() {
                payload.insert("tool_calls".to_string(), Value::Array(tool_calls));
            }
            Value::Object(payload)
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
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.output_json.to_string(),
                })
            } else {
                json!({
                    "role": "tool",
                    "content": "",
                })
            }
        }
    }
}

fn text_content_from_parts(parts: &[ContentPart]) -> Value {
    let texts: Vec<String> = parts
        .iter()
        .filter_map(|part| {
            if let ContentPart::Text(text) = part {
                Some(text.clone())
            } else {
                None
            }
        })
        .collect();
    Value::String(texts.join(""))
}

fn reasoning_content_from_parts(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| {
            if let ContentPart::Reasoning(reasoning) = part {
                Some(reasoning.text.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn openai_user_content(parts: &[ContentPart]) -> Value {
    let has_images = parts.iter().any(|p| matches!(p, ContentPart::Image(_)));
    if has_images {
        Value::Array(
            parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text(text) => Some(json!({ "type": "text", "text": text })),
                    ContentPart::Image(img) => Some(openai_image_content_part(img)),
                    _ => None,
                })
                .collect(),
        )
    } else {
        text_content_from_parts(parts)
    }
}

fn openai_image_content_part(image: &ImagePart) -> Value {
    let url = match &image.data {
        MediaData::Url(url) => url.clone(),
        MediaData::Base64(b64) => {
            let mt = image.media_type.as_deref().unwrap_or("image/jpeg");
            format!("data:{};base64,{}", mt, b64)
        }
        MediaData::Bytes(bytes) => {
            let mt = image.media_type.as_deref().unwrap_or("image/jpeg");
            format!("data:{};base64,{}", mt, STANDARD.encode(bytes))
        }
    };
    json!({ "type": "image_url", "image_url": { "url": url } })
}

fn normalize_openai_response(body: Value) -> Result<GenerateTextResponse, Error> {
    let Some(choice) = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
    else {
        return Err(Error::new(
            ErrorCode::InvalidResponse,
            "openai response missing choices[0]",
        ));
    };

    let message = choice
        .get("message")
        .ok_or_else(|| Error::new(ErrorCode::InvalidResponse, "missing message"))?;

    let output_text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let reasoning_text = message
        .get("reasoning_content")
        .or_else(|| message.get("reasoning"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let reasoning_parts = if reasoning_text.is_empty() {
        Vec::new()
    } else {
        vec![ReasoningPart {
            text: reasoning_text.clone(),
            provider_metadata: None,
        }]
    };

    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(parse_openai_tool_call)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let finish_reason = map_openai_finish_reason(
        choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("stop"),
    );

    let usage = body
        .get("usage")
        .and_then(parse_openai_usage)
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

fn parse_openai_tool_call(value: &Value) -> Result<ToolCall, Error> {
    let call_id = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new(ErrorCode::InvalidResponse, "tool call missing id"))?;
    let tool_name = value
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidResponse,
                "tool call missing function name",
            )
        })?;
    let args_raw = value
        .get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let args_json = serde_json::from_str(args_raw).map_err(|e| {
        Error::new(
            ErrorCode::InvalidToolArgs,
            format!("failed to parse tool arguments JSON: {}", e),
        )
    })?;
    Ok(ToolCall {
        call_id: call_id.to_string(),
        tool_name: tool_name.to_string(),
        args_json,
    })
}

fn parse_openai_usage(value: &Value) -> Option<Usage> {
    let input_tokens = value.get("prompt_tokens")?.as_u64()? as u32;
    let output_tokens = value.get("completion_tokens")?.as_u64()? as u32;
    let cached_input_tokens = value
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .map(|n| n as u32)
        .unwrap_or(0);
    let reasoning_tokens = value
        .get("completion_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .map(|n| n as u32)
        .unwrap_or(0);
    let total_tokens = value
        .get("total_tokens")
        .and_then(Value::as_u64)
        .map(|n| n as u32)
        .unwrap_or_else(|| input_tokens.saturating_add(output_tokens));
    Some(
        Usage::from_totals(
            input_tokens,
            output_tokens,
            reasoning_tokens,
            Some(total_tokens),
        )
        .with_input_cache_split(cached_input_tokens, 0)
        .with_output_split(
            output_tokens.saturating_sub(reasoning_tokens),
            reasoning_tokens,
        )
        .with_raw_usage(value.clone()),
    )
}

fn map_openai_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Unknown(reason.to_string()),
    }
}
