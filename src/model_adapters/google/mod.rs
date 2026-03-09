#![allow(clippy::collapsible_if)]
use std::collections::HashMap;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::CONTENT_TYPE;
use serde_json::{Map, Value, json};

use crate::error::{Error, ErrorCode};
use crate::model_adapters::{ModelAdapter, check_response_status, map_send_error};
use crate::stream::drain_sse_frames;
use crate::types::{
    ContentPart, FinishReason, GenerateTextRequest, GenerateTextResponse, Google, Message,
    MessageRole, StreamEvent, TextStream, ToolCall, Usage,
};

pub const PROVIDER_SLUG: &str = "google";
pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

pub struct GoogleAdapterSettings {
    pub base_url: String,
    pub api_key: String,
}

impl GoogleAdapterSettings {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
        }
    }
}

pub struct GoogleAdapter {
    base_url: String,
    api_key: String,
    http: Arc<reqwest::Client>,
}

impl GoogleAdapter {
    pub fn from_settings(settings: GoogleAdapterSettings, http: Arc<reqwest::Client>) -> Self {
        Self {
            base_url: settings.base_url,
            api_key: settings.api_key,
            http,
        }
    }

    fn generate_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            model
        )
    }

    fn stream_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url.trim_end_matches('/'),
            model
        )
    }
}

#[async_trait]
impl ModelAdapter<Google> for GoogleAdapter {
    async fn generate_text(
        &self,
        req: &GenerateTextRequest<Google>,
    ) -> Result<GenerateTextResponse, Error> {
        let payload = build_google_payload(req);
        let cancel_token = req.cancellation_token.clone();
        let send_fut = self
            .http
            .post(self.generate_url(req.model.model()))
            .header("x-goog-api-key", &self.api_key)
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
        normalize_google_response(body)
    }

    async fn stream_text(&self, req: &GenerateTextRequest<Google>) -> Result<TextStream, Error> {
        let payload = build_google_payload(req);
        let cancel_token = req.cancellation_token.clone();
        let cancel_token_stream = cancel_token.clone();
        let send_fut = self
            .http
            .post(self.stream_url(req.model.model()))
            .header("x-goog-api-key", &self.api_key)
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
            let mut tool_counter = 0u32;

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

                    if let Some(candidate) = value
                        .get("candidates")
                        .and_then(Value::as_array)
                        .and_then(|arr| arr.first())
                    {
                        if let Some(parts) = candidate
                            .get("content")
                            .and_then(|content| content.get("parts"))
                            .and_then(Value::as_array)
                        {
                            for part in parts {
                                if let Some(text) = part.get("text").and_then(Value::as_str) {
                                    if !text.is_empty() {
                                        yield StreamEvent::TextDelta {
                                            text: text.to_string(),
                                        };
                                    }
                                }

                                if let Some(function_call) = part.get("functionCall") {
                                    if let Some(name) = function_call.get("name").and_then(Value::as_str) {
                                        let args_json = function_call
                                            .get("args")
                                            .cloned()
                                            .unwrap_or_else(|| json!({}));
                                        tool_counter = tool_counter.saturating_add(1);
                                        yield StreamEvent::ToolCallReady {
                                            call: ToolCall {
                                                call_id: format!("google_call_{}", tool_counter),
                                                tool_name: name.to_string(),
                                                args_json,
                                            },
                                        };
                                    }
                                }
                            }
                        }

                        if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
                            if reason != "FINISH_REASON_UNSPECIFIED" {
                                done = true;
                            }
                        }
                    }

                    if let Some(usage) = value.get("usageMetadata").and_then(parse_google_usage) {
                        yield StreamEvent::Usage { usage };
                    }
                }

                if done {
                    break;
                }
            }

            yield StreamEvent::Done;
        };

        Ok(Box::pin(stream))
    }
}

fn build_google_payload(req: &GenerateTextRequest<Google>) -> Value {
    let (contents, system_instruction) = to_google_messages(&req.messages);

    let mut payload = Map::new();
    payload.insert("contents".to_string(), Value::Array(contents));

    if let Some(system_instruction) = system_instruction {
        payload.insert(
            "systemInstruction".to_string(),
            json!({ "parts": [{ "text": system_instruction }] }),
        );
    }

    let mut generation_config = Map::new();
    if let Some(max_output_tokens) = req.max_output_tokens {
        generation_config.insert(
            "maxOutputTokens".to_string(),
            Value::from(max_output_tokens),
        );
    }
    if let Some(temperature) = req.temperature {
        generation_config.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = req.top_p {
        generation_config.insert("topP".to_string(), Value::from(top_p));
    }
    if !req.stop_sequences.is_empty() {
        generation_config.insert(
            "stopSequences".to_string(),
            Value::Array(
                req.stop_sequences
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect(),
            ),
        );
    }
    if !generation_config.is_empty() {
        payload.insert(
            "generationConfig".to_string(),
            Value::Object(generation_config),
        );
    }

    if let Some(tools) = &req.tools {
        let declarations = tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                })
            })
            .collect::<Vec<_>>();

        payload.insert(
            "tools".to_string(),
            Value::Array(vec![json!({ "functionDeclarations": declarations })]),
        );
    }

    Value::Object(payload)
}

fn to_google_messages(messages: &[Message]) -> (Vec<Value>, Option<String>) {
    let mut contents = Vec::new();
    let mut system_texts = Vec::new();
    let mut tool_name_by_call_id: HashMap<String, String> = HashMap::new();

    for message in messages {
        if message.role == MessageRole::Assistant {
            for part in &message.parts {
                if let ContentPart::ToolCall(call) = part {
                    tool_name_by_call_id.insert(call.call_id.clone(), call.tool_name.clone());
                }
            }
        }
    }

    for message in messages {
        match message.role {
            MessageRole::System => {
                let text = text_content_from_parts(&message.parts);
                if !text.is_empty() {
                    system_texts.push(text);
                }
            }
            MessageRole::User => {
                let parts = text_parts_from_message(message);
                if !parts.is_empty() {
                    contents.push(json!({
                        "role": "user",
                        "parts": parts,
                    }));
                }
            }
            MessageRole::Assistant => {
                let mut parts = Vec::new();
                for part in &message.parts {
                    match part {
                        ContentPart::Text(text) => {
                            if !text.is_empty() {
                                parts.push(json!({ "text": text }));
                            }
                        }
                        ContentPart::ToolCall(call) => {
                            parts.push(json!({
                                "functionCall": {
                                    "name": call.tool_name,
                                    "args": call.args_json,
                                }
                            }));
                        }
                        ContentPart::ToolResult(_) => {}
                    }
                }
                if !parts.is_empty() {
                    contents.push(json!({
                        "role": "model",
                        "parts": parts,
                    }));
                }
            }
            MessageRole::Tool => {
                let mut parts = Vec::new();
                for part in &message.parts {
                    if let ContentPart::ToolResult(result) = part {
                        let tool_name = tool_name_by_call_id
                            .get(&result.call_id)
                            .cloned()
                            .unwrap_or_else(|| "tool".to_string());

                        parts.push(json!({
                            "functionResponse": {
                                "name": tool_name,
                                "response": {
                                    "name": tool_name,
                                    "content": result.output_json,
                                }
                            }
                        }));
                    }
                }
                if !parts.is_empty() {
                    contents.push(json!({
                        "role": "user",
                        "parts": parts,
                    }));
                }
            }
        }
    }

    let system_instruction = if system_texts.is_empty() {
        None
    } else {
        Some(system_texts.join("\n"))
    };

    (contents, system_instruction)
}

fn text_parts_from_message(message: &Message) -> Vec<Value> {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text(text) if !text.is_empty() => Some(json!({ "text": text })),
            _ => None,
        })
        .collect()
}

fn text_content_from_parts(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| {
            if let ContentPart::Text(text) = part {
                Some(text.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn normalize_google_response(body: Value) -> Result<GenerateTextResponse, Error> {
    let Some(candidate) = body
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
    else {
        return Err(Error::new(
            ErrorCode::InvalidResponse,
            "google response missing candidates[0]",
        ));
    };

    let mut output_text = String::new();
    let mut tool_calls = Vec::new();

    if let Some(parts) = candidate
        .get("content")
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
    {
        for (index, part) in parts.iter().enumerate() {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                output_text.push_str(text);
            }

            if let Some(function_call) = part.get("functionCall") {
                if let Some(name) = function_call.get("name").and_then(Value::as_str) {
                    let args_json = function_call
                        .get("args")
                        .cloned()
                        .unwrap_or_else(|| json!({}));

                    tool_calls.push(ToolCall {
                        call_id: format!("google_call_{}", index + 1),
                        tool_name: name.to_string(),
                        args_json,
                    });
                }
            }
        }
    }

    let finish_reason = map_google_finish_reason(
        candidate
            .get("finishReason")
            .and_then(Value::as_str)
            .unwrap_or("STOP"),
        !tool_calls.is_empty(),
    );

    let usage = body
        .get("usageMetadata")
        .and_then(parse_google_usage)
        .unwrap_or_default();

    Ok(GenerateTextResponse {
        output_text,
        finish_reason,
        usage,
        tool_calls,
        raw_provider_response: Some(body),
    })
}

fn parse_google_usage(value: &Value) -> Option<Usage> {
    let input_tokens = value.get("promptTokenCount")?.as_u64()? as u32;
    let output_tokens = value.get("candidatesTokenCount")?.as_u64()? as u32;
    let total_tokens = value
        .get("totalTokenCount")
        .and_then(Value::as_u64)
        .map(|v| v as u32)
        .unwrap_or_else(|| input_tokens.saturating_add(output_tokens));
    Some(Usage {
        input_tokens,
        output_tokens,
        total_tokens,
    })
}

fn map_google_finish_reason(reason: &str, has_tool_calls: bool) -> FinishReason {
    match reason {
        "STOP" => {
            if has_tool_calls {
                FinishReason::ToolCalls
            } else {
                FinishReason::Stop
            }
        }
        "MAX_TOKENS" => FinishReason::Length,
        "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            FinishReason::ContentFilter
        }
        _ => FinishReason::Unknown(reason.to_string()),
    }
}
