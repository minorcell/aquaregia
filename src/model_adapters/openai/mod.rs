//! OpenAI Responses API adapter for Aquaregia.
//!
//! Implements the OpenAI Responses API (`POST /v1/responses`).
//!
//! ## Example
//!
//! ```rust,no_run
//! use aquaregia::{LlmClient, GenerateTextRequest};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = LlmClient::openai().api_key("api-key").build()?;
//!
//! let response = client
//!     .generate(GenerateTextRequest::from_user_prompt("gpt-5.5", "Hello!"))
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
use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::{Map, Value, json};

use crate::error::{Error, ErrorCode};
use crate::model_adapters::{ModelAdapter, base64_encode, check_response_status, map_send_error};
use crate::stream::drain_sse_frames;
use crate::types::{
    ContentPart, FinishReason, GenerateTextRequest, GenerateTextResponse, ImagePart, MediaData,
    Message, MessageRole, ReasoningPart, StreamEvent, TextStream, ToolCall, Usage,
};

pub const PROVIDER_SLUG: &str = "openai";
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Runtime settings for the OpenAI adapter.
pub struct OpenAiAdapterSettings {
    pub base_url: String,
    pub api_key: String,
}

impl OpenAiAdapterSettings {
    pub fn new() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: String::new(),
        }
    }
}

impl Default for OpenAiAdapterSettings {
    fn default() -> Self {
        Self::new()
    }
}

/// OpenAI adapter — targets the Responses API (`/v1/responses`).
pub struct OpenAiAdapter {
    base_url: String,
    api_key: String,
    http: Arc<reqwest::Client>,
}

impl OpenAiAdapter {
    pub fn from_settings(settings: OpenAiAdapterSettings, http: Arc<reqwest::Client>) -> Self {
        Self {
            base_url: settings.base_url,
            api_key: settings.api_key,
            http,
        }
    }
}

#[async_trait]
impl ModelAdapter for OpenAiAdapter {
    async fn generate_text(
        &self,
        req: &GenerateTextRequest,
    ) -> Result<GenerateTextResponse, Error> {
        let payload = build_payload(req, false);
        let url = format!("{}/v1/responses", self.base_url.trim_end_matches('/'));
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
        normalize_response(body)
    }

    async fn stream_text(&self, req: &GenerateTextRequest) -> Result<TextStream, Error> {
        let payload = build_payload(req, true);
        let url = format!("{}/v1/responses", self.base_url.trim_end_matches('/'));
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
            let mut done = false;
            // Indexed by output_index from the streaming events.
            let mut fn_partials: BTreeMap<u64, PartialFnCall> = BTreeMap::new();
            // Reasoning blocks keyed by output_index. Multiple `reasoning` items can
            // appear in one response; the canonical id is `rs_*` from the SDK.
            let mut reasoning_blocks: BTreeMap<u64, String> = BTreeMap::new();

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
                    let data = frame.trim();
                    if data == "[DONE]" {
                        // Defensive: treat [DONE] as terminal if response.completed was missed.
                        if !done {
                            for (_, block_id) in std::mem::take(&mut reasoning_blocks) {
                                yield StreamEvent::ReasoningDone {
                                    block_id,
                                    provider_metadata: None,
                                };
                            }
                            done = true;
                            yield StreamEvent::Done;
                        }
                        break;
                    }

                    let value: Value = serde_json::from_str(data)
                        .map_err(|e| Error::new(ErrorCode::StreamProtocol, e.to_string()))?;
                    let event_type = value.get("type").and_then(Value::as_str).unwrap_or("");

                    match event_type {
                        "response.output_item.added" => {
                            // Track function_call and reasoning items so we can assemble them.
                            if let Some(item) = value.get("item") {
                                let output_index = value
                                    .get("output_index")
                                    .and_then(Value::as_u64)
                                    .unwrap_or(0);
                                match item.get("type").and_then(Value::as_str) {
                                    Some("function_call") => {
                                        let call_id = item
                                            .get("call_id")
                                            .and_then(Value::as_str)
                                            .unwrap_or("")
                                            .to_string();
                                        let name = item
                                            .get("name")
                                            .and_then(Value::as_str)
                                            .unwrap_or("")
                                            .to_string();
                                        fn_partials.insert(output_index, PartialFnCall {
                                            call_id,
                                            name,
                                            args_buf: String::new(),
                                        });
                                    }
                                    Some("reasoning") => {
                                        // Prefer the upstream item id for round-tripping; otherwise synthesize.
                                        let block_id = item
                                            .get("id")
                                            .and_then(Value::as_str)
                                            .map(ToOwned::to_owned)
                                            .unwrap_or_else(|| format!("reasoning-{output_index}"));
                                        reasoning_blocks.insert(output_index, block_id.clone());
                                        yield StreamEvent::ReasoningStarted {
                                            block_id,
                                            provider_metadata: None,
                                        };
                                    }
                                    _ => {}
                                }
                            }
                        }

                        "response.output_item.done" => {
                            let output_index = value
                                .get("output_index")
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            if let Some(block_id) = reasoning_blocks.remove(&output_index) {
                                yield StreamEvent::ReasoningDone {
                                    block_id,
                                    provider_metadata: None,
                                };
                            }
                        }

                        "response.output_text.delta" => {
                            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                                if !delta.is_empty() {
                                    yield StreamEvent::TextDelta { text: delta.to_string() };
                                }
                            }
                        }

                        // Refusal text is delivered through a separate channel; surface as text
                        // so callers see what the model produced. The raw provider response is
                        // still available via raw_provider_response on the non-streaming path.
                        "response.refusal.delta" => {
                            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                                if !delta.is_empty() {
                                    yield StreamEvent::TextDelta { text: delta.to_string() };
                                }
                            }
                        }

                        "response.reasoning_summary_text.delta"
                        | "response.reasoning_text.delta" => {
                            let output_index = value
                                .get("output_index")
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                                if !delta.is_empty() {
                                    // Lazily start a reasoning block if `output_item.added` was
                                    // never observed (some servers send deltas directly).
                                    let block_id = match reasoning_blocks.get(&output_index) {
                                        Some(id) => id.clone(),
                                        None => {
                                            let id = format!("reasoning-{output_index}");
                                            reasoning_blocks.insert(output_index, id.clone());
                                            yield StreamEvent::ReasoningStarted {
                                                block_id: id.clone(),
                                                provider_metadata: None,
                                            };
                                            id
                                        }
                                    };
                                    yield StreamEvent::ReasoningDelta {
                                        block_id,
                                        text: delta.to_string(),
                                        provider_metadata: None,
                                    };
                                }
                            }
                        }

                        "response.function_call_arguments.delta" => {
                            let output_index = value
                                .get("output_index")
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            if let Some(partial) = fn_partials.get_mut(&output_index) {
                                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                                    partial.args_buf.push_str(delta);
                                }
                            }
                        }

                        "response.function_call_arguments.done" => {
                            let output_index = value
                                .get("output_index")
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            if let Some(partial) = fn_partials.remove(&output_index) {
                                let args_str = value
                                    .get("arguments")
                                    .and_then(Value::as_str)
                                    .unwrap_or(&partial.args_buf);
                                let args_json = parse_args_json(args_str)?;
                                yield StreamEvent::ToolCallReady {
                                    call: ToolCall {
                                        call_id: partial.call_id,
                                        tool_name: partial.name,
                                        args_json,
                                    },
                                };
                            }
                        }

                        "response.completed" | "response.incomplete" => {
                            if let Some(resp) = value.get("response") {
                                if let Some(usage) = resp.get("usage").and_then(parse_usage) {
                                    yield StreamEvent::Usage { usage };
                                }
                            }
                            // Drain any still-open reasoning blocks so callers see balanced
                            // ReasoningStarted/Done pairs even when output_item.done was missed.
                            for (_, block_id) in std::mem::take(&mut reasoning_blocks) {
                                yield StreamEvent::ReasoningDone {
                                    block_id,
                                    provider_metadata: None,
                                };
                            }
                            done = true;
                            yield StreamEvent::Done;
                            break;
                        }

                        "response.failed" => {
                            let msg = value
                                .get("response")
                                .and_then(|r| r.get("error"))
                                .and_then(|e| e.get("message"))
                                .and_then(Value::as_str)
                                .unwrap_or("openai response failed");
                            Err(Error::new(ErrorCode::InvalidResponse, msg))?;
                        }

                        "response.error" => {
                            let msg = value
                                .get("message")
                                .and_then(Value::as_str)
                                .or_else(|| {
                                    value.get("error").and_then(|e| e.get("message")).and_then(Value::as_str)
                                })
                                .unwrap_or("openai stream error");
                            Err(Error::new(ErrorCode::InvalidResponse, msg))?;
                        }

                        _ => {}
                    }
                }

                if done {
                    break;
                }
            }

            if !done {
                for (_, block_id) in std::mem::take(&mut reasoning_blocks) {
                    yield StreamEvent::ReasoningDone {
                        block_id,
                        provider_metadata: None,
                    };
                }
                Err(Error::new(
                    ErrorCode::StreamProtocol,
                    "openai stream closed without response.completed",
                ))?;
            }
        };

        Ok(Box::pin(stream))
    }
}

struct PartialFnCall {
    call_id: String,
    name: String,
    args_buf: String,
}

/// Builds the Responses API request payload.
fn build_payload(req: &GenerateTextRequest, stream: bool) -> Value {
    let mut payload = Map::new();
    payload.insert("model".to_string(), Value::String(req.model.clone()));
    payload.insert("stream".to_string(), Value::Bool(stream));

    // System messages become the top-level `instructions` field.
    let instructions: String = req
        .messages
        .iter()
        .filter(|m| m.role == MessageRole::System)
        .flat_map(|m| m.parts.iter())
        .filter_map(|p| {
            if let ContentPart::Text(t) = p {
                Some(t.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if !instructions.is_empty() {
        payload.insert("instructions".to_string(), Value::String(instructions));
    }

    // Non-system messages become the `input` array.
    let input = build_input(&req.messages);
    payload.insert("input".to_string(), Value::Array(input));

    if let Some(temperature) = req.temperature {
        payload.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = req.top_p {
        payload.insert("top_p".to_string(), Value::from(top_p));
    }
    if let Some(max_output_tokens) = req.max_output_tokens {
        payload.insert(
            "max_output_tokens".to_string(),
            Value::from(max_output_tokens),
        );
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
                        // Responses API: name at top level, not nested in "function".
                        json!({
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                            "strict": true,
                        })
                    })
                    .collect(),
            ),
        );
    }

    if let Some(output_schema) = &req.output_schema {
        let mut text_format = Map::new();
        text_format.insert("type".to_string(), Value::String("json_schema".to_string()));
        text_format.insert(
            "name".to_string(),
            Value::String(output_schema.name.clone()),
        );
        text_format.insert("schema".to_string(), output_schema.json_schema.clone());
        text_format.insert("strict".to_string(), Value::Bool(true));
        payload.insert(
            "text".to_string(),
            json!({ "format": Value::Object(text_format) }),
        );
    }

    Value::Object(payload)
}

/// Converts the non-system messages to Responses API `input` items.
fn build_input(messages: &[Message]) -> Vec<Value> {
    let mut items: Vec<Value> = Vec::new();
    for message in messages {
        match message.role {
            MessageRole::System => {} // handled as `instructions`
            MessageRole::User => {
                let content = user_content_items(&message.parts);
                items.push(json!({ "type": "message", "role": "user", "content": content }));
            }
            MessageRole::Assistant => {
                // Text parts → message item.
                let text: String = message
                    .parts
                    .iter()
                    .filter_map(|p| {
                        if let ContentPart::Text(t) = p {
                            Some(t.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");
                if !text.is_empty() {
                    items.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": text }],
                    }));
                }
                // Each ToolCall part → a separate function_call item.
                for part in &message.parts {
                    if let ContentPart::ToolCall(call) = part {
                        items.push(json!({
                            "type": "function_call",
                            "call_id": call.call_id,
                            "name": call.tool_name,
                            "arguments": call.args_json.to_string(),
                        }));
                    }
                }
            }
            MessageRole::Tool => {
                // Each ToolResult part → a function_call_output item.
                for part in &message.parts {
                    if let ContentPart::ToolResult(result) = part {
                        items.push(json!({
                            "type": "function_call_output",
                            "call_id": result.call_id,
                            "output": result.output_json.to_string(),
                        }));
                    }
                }
            }
        }
    }
    items
}

/// Converts user message parts to Responses API content items.
fn user_content_items(parts: &[ContentPart]) -> Value {
    let has_images = parts.iter().any(|p| matches!(p, ContentPart::Image(_)));
    if has_images {
        Value::Array(
            parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text(text) => Some(json!({ "type": "input_text", "text": text })),
                    ContentPart::Image(img) => Some(image_content_item(img)),
                    _ => None,
                })
                .collect(),
        )
    } else {
        let text: String = parts
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");
        json!([{ "type": "input_text", "text": text }])
    }
}

fn image_content_item(image: &ImagePart) -> Value {
    let url = match &image.data {
        MediaData::Url(url) => url.clone(),
        MediaData::Base64(b64) => {
            let mt = image.media_type.as_deref().unwrap_or("image/jpeg");
            format!("data:{};base64,{}", mt, b64)
        }
        MediaData::Bytes(bytes) => {
            let mt = image.media_type.as_deref().unwrap_or("image/jpeg");
            format!("data:{};base64,{}", mt, base64_encode(bytes))
        }
    };
    json!({ "type": "input_image", "image_url": { "url": url } })
}

fn normalize_response(body: Value) -> Result<GenerateTextResponse, Error> {
    if body.get("status").and_then(Value::as_str) == Some("failed") {
        let msg = body
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("openai response failed");
        return Err(Error::new(ErrorCode::InvalidResponse, msg));
    }

    let output = body
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new(ErrorCode::InvalidResponse, "openai response missing output"))?;

    let mut output_text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut reasoning_parts: Vec<ReasoningPart> = Vec::new();

    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for c in content {
                        match c.get("type").and_then(Value::as_str) {
                            Some("output_text") => {
                                if let Some(text) = c.get("text").and_then(Value::as_str) {
                                    output_text.push_str(text);
                                }
                            }
                            // Refusal content is structurally distinct in the Responses API
                            // but functionally still text the caller cares about.
                            Some("refusal") => {
                                if let Some(text) = c.get("refusal").and_then(Value::as_str) {
                                    output_text.push_str(text);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Some("function_call") => {
                let call = parse_fn_call_item(item)?;
                tool_calls.push(call);
            }
            Some("reasoning") => {
                // The Responses API exposes reasoning items with a `summary` array of
                // `{ type: "summary_text", text: ... }` blocks. Encrypted reasoning text
                // (when requested) lives in `content[].text` on the same item.
                if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                    let text = summary
                        .iter()
                        .filter(|b| b.get("type").and_then(Value::as_str) == Some("summary_text"))
                        .filter_map(|b| b.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("");
                    if !text.is_empty() {
                        reasoning_parts.push(ReasoningPart {
                            text,
                            provider_metadata: None,
                        });
                    }
                }
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for c in content {
                        if let Some(text) = c.get("text").and_then(Value::as_str) {
                            if !text.is_empty() {
                                reasoning_parts.push(ReasoningPart {
                                    text: text.to_string(),
                                    provider_metadata: None,
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let reasoning_text = reasoning_parts
        .iter()
        .map(|p| p.text.as_str())
        .collect::<String>();

    let finish_reason = if tool_calls.is_empty() {
        FinishReason::Stop
    } else {
        FinishReason::ToolCalls
    };

    let usage = body.get("usage").and_then(parse_usage).unwrap_or_default();

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

fn parse_fn_call_item(item: &Value) -> Result<ToolCall, Error> {
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new(ErrorCode::InvalidResponse, "function_call missing call_id"))?;
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new(ErrorCode::InvalidResponse, "function_call missing name"))?;
    let args_raw = item
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let args_json = parse_args_json(args_raw)?;
    Ok(ToolCall {
        call_id: call_id.to_string(),
        tool_name: name.to_string(),
        args_json,
    })
}

fn parse_args_json(raw: &str) -> Result<Value, Error> {
    let s = raw.trim();
    if s.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(s).map_err(|e| {
        Error::new(
            ErrorCode::InvalidToolArgs,
            format!("invalid tool arguments JSON: {}", e),
        )
    })
}

fn parse_usage(value: &Value) -> Option<Usage> {
    let input_tokens = value.get("input_tokens")?.as_u64()? as u32;
    let output_tokens = value.get("output_tokens")?.as_u64()? as u32;
    let cached_tokens = value
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .map(|n| n as u32)
        .unwrap_or(0);
    let reasoning_tokens = value
        .get("output_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
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
        .with_input_cache_split(cached_tokens, 0)
        .with_output_split(
            output_tokens.saturating_sub(reasoning_tokens),
            reasoning_tokens,
        )
        .with_raw_usage(value.clone()),
    )
}
