use std::collections::HashMap;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::CONTENT_TYPE;
use serde_json::{Map, Value, json};

use crate::error::{AiError, AiErrorCode};
use crate::model_adapters::{ModelAdapter, check_response_status, map_send_error};
use crate::stream::drain_sse_frames;
use crate::types::{
    Anthropic, ContentPart, FinishReason, GenerateTextRequest, GenerateTextResponse, Message,
    MessageRole, StreamEvent, TextStream, ToolCall, Usage,
};

pub const PROVIDER_SLUG: &str = "anthropic";
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
pub const DEFAULT_API_VERSION: &str = "2023-06-01";

pub struct AnthropicAdapterSettings {
    pub base_url: String,
    pub api_key: String,
    pub api_version: String,
}

impl AnthropicAdapterSettings {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            api_version: DEFAULT_API_VERSION.to_string(),
        }
    }
}

pub struct AnthropicAdapter {
    base_url: String,
    api_key: String,
    api_version: String,
    http: Arc<reqwest::Client>,
}

impl AnthropicAdapter {
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
    ) -> Result<GenerateTextResponse, AiError> {
        let payload = build_anthropic_payload(req, false);
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let response = self
            .http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header(CONTENT_TYPE, "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| map_send_error(PROVIDER_SLUG, e))?;
        let response = check_response_status(PROVIDER_SLUG, response).await?;
        let body: Value = response
            .json()
            .await
            .map_err(|e| AiError::new(AiErrorCode::InvalidResponse, e.to_string()))?;
        normalize_anthropic_response(body)
    }

    async fn stream_text(
        &self,
        req: &GenerateTextRequest<Anthropic>,
    ) -> Result<TextStream, AiError> {
        let payload = build_anthropic_payload(req, true);
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let response = self
            .http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header(CONTENT_TYPE, "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| map_send_error(PROVIDER_SLUG, e))?;
        let response = check_response_status(PROVIDER_SLUG, response).await?;
        let mut byte_stream = response.bytes_stream();

        let stream = try_stream! {
            let mut buffer = String::new();
            let mut pending_calls: HashMap<u64, PendingToolUse> = HashMap::new();
            let mut done = false;

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| AiError::new(AiErrorCode::Transport, e.to_string()))?;
                let text = std::str::from_utf8(&chunk)
                    .map_err(|e| AiError::new(AiErrorCode::StreamProtocol, e.to_string()))?;
                buffer.push_str(text);

                let frames = drain_sse_frames(&mut buffer);
                for frame in frames {
                    let data = frame.data.trim();
                    if data.is_empty() {
                        continue;
                    }

                    let value: Value = serde_json::from_str(data)
                        .map_err(|e| AiError::new(AiErrorCode::StreamProtocol, e.to_string()))?;

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
                                    if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                                        let entry = pending_calls.entry(index).or_default();
                                        entry.call_id = block.get("id").and_then(Value::as_str).map(ToOwned::to_owned);
                                        entry.tool_name = block.get("name").and_then(Value::as_str).map(ToOwned::to_owned);
                                        if let Some(input) = block.get("input") {
                                            entry.args_json = Some(input.clone());
                                        }
                                    }
                                }
                            }
                            "content_block_stop" => {
                                if let Some(index) = value.get("index").and_then(Value::as_u64) {
                                    if let Some(pending) = pending_calls.remove(&index) {
                                        if let Some(call) = pending.to_tool_call()? {
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
                Err(AiError::new(
                    AiErrorCode::StreamProtocol,
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
    fn to_tool_call(self) -> Result<Option<ToolCall>, AiError> {
        let Some(call_id) = self.call_id else {
            return Ok(None);
        };
        let Some(tool_name) = self.tool_name else {
            return Ok(None);
        };
        let args_json = if !self.args_buf.trim().is_empty() {
            serde_json::from_str(&self.args_buf).map_err(|e| {
                AiError::new(
                    AiErrorCode::InvalidToolArgs,
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

fn normalize_anthropic_response(body: Value) -> Result<GenerateTextResponse, AiError> {
    let content = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            AiError::new(
                AiErrorCode::InvalidResponse,
                "anthropic response missing content",
            )
        })?;

    let mut output_text = String::new();
    let mut tool_calls = Vec::new();

    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    output_text.push_str(text);
                }
            }
            Some("tool_use") => {
                let call_id = block.get("id").and_then(Value::as_str).ok_or_else(|| {
                    AiError::new(AiErrorCode::InvalidResponse, "tool_use missing id")
                })?;
                let tool_name = block.get("name").and_then(Value::as_str).ok_or_else(|| {
                    AiError::new(AiErrorCode::InvalidResponse, "tool_use missing name")
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
        finish_reason,
        usage,
        tool_calls,
        raw_provider_response: Some(body),
    })
}

fn parse_anthropic_usage(value: &Value) -> Option<Usage> {
    let input_tokens = value.get("input_tokens")?.as_u64()? as u32;
    let output_tokens = value.get("output_tokens")?.as_u64()? as u32;
    Some(Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens.saturating_add(output_tokens),
    })
}

fn map_anthropic_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        _ => FinishReason::Unknown(reason.to_string()),
    }
}
