use aquaregia::{
    ContentPart, ErrorCode, FinishReason, GenerateTextRequest, LlmClient, Message, MessageRole,
    StreamEvent, ToolResult,
};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

// ─── Google streaming ───────────────────────────────────────────────────

#[tokio::test]
async fn google_stream_emits_text_usage_done() {
    let server = MockServer::start().await;
    let sse_body = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"thoughtsTokenCount\":1,\"totalTokenCount\":6}}\n\n";

    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-google-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google()
        .api_key("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::from_user_prompt("gemini-3.5-flash", "hello");

    let mut stream = client
        .stream(req)
        .await
        .expect("stream_text should succeed");

    let mut saw_text = false;
    let mut saw_usage = false;
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::TextDelta { text } if text == "Hello" => saw_text = true,
            StreamEvent::Usage { usage } if usage.input_tokens == 3 && usage.total_tokens == 6 => {
                saw_usage = true;
            }
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert!(saw_text);
    assert!(saw_usage);
    assert!(saw_done);
}

#[tokio::test]
async fn google_stream_with_reasoning() {
    let server = MockServer::start().await;
    let sse_body = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"I think...\",\"thought\":true},{\"text\":\"Answer\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":3,\"totalTokenCount\":8}}\n\n";

    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-google-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google()
        .api_key("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::from_user_prompt("gemini-3.5-flash", "hello");

    let mut stream = client
        .stream(req)
        .await
        .expect("stream_text should succeed");

    let mut reasoning_text = String::new();
    let mut output_text = String::new();
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::TextDelta { text } => output_text.push_str(&text),
            StreamEvent::ReasoningDelta { text, .. } => reasoning_text.push_str(&text),
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert_eq!(reasoning_text, "I think...");
    assert_eq!(output_text, "Answer");
    assert!(saw_done);
}

// ─── Google generate non-streaming reasoning ────────────────────────────

#[tokio::test]
async fn google_generate_with_reasoning() {
    let server = MockServer::start().await;

    let body = json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        { "text": "internal reasoning", "thought": true },
                        { "text": "Final answer" }
                    ]
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8
        }
    });

    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:generateContent"))
        .and(header("x-goog-api-key", "test-google-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google()
        .api_key("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-3.5-flash",
            "hello",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.output_text, "Final answer");
    assert!(response.reasoning_text.contains("internal reasoning"));
    assert_eq!(response.reasoning_parts.len(), 1);
    assert!(response.raw_provider_response.is_some());
}

#[tokio::test]
async fn google_generate_with_tool_calls() {
    let server = MockServer::start().await;

    let body = json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "weather",
                                "args": { "city": "Tokyo" }
                            }
                        }
                    ]
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8
        }
    });

    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:generateContent"))
        .and(header("x-goog-api-key", "test-google-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google()
        .api_key("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-3.5-flash",
            "weather?",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].tool_name, "weather");
}

#[tokio::test]
async fn google_generate_with_thought_signature() {
    let server = MockServer::start().await;

    let body = json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "reasoning",
                            "thought": true,
                            "thoughtSignature": "sig123"
                        }
                    ]
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8
        }
    });

    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google()
        .api_key("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-3.5-flash",
            "think",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.reasoning_text, "reasoning");
    assert!(
        response
            .reasoning_parts
            .first()
            .and_then(|p| p.provider_metadata.as_ref())
            .is_some()
    );
}

// ─── Google error responses ─────────────────────────────────────────────

#[tokio::test]
async fn google_503_maps_to_provider_server_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(503).set_body_string("service unavailable"))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google()
        .api_key("test-google-key")
        .base_url(server.uri())
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-3.5-flash",
            "hello",
        ))
        .await
        .expect_err("request should fail");

    assert_eq!(err.code, ErrorCode::ProviderServerError);
    assert!(err.retryable);
}

#[tokio::test]
async fn google_invalid_response_missing_candidates() {
    let server = MockServer::start().await;

    let body = json!({"not_candidates": []});
    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google()
        .api_key("test-google-key")
        .base_url(server.uri())
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-3.5-flash",
            "hello",
        ))
        .await
        .expect_err("should fail");

    assert_eq!(err.code, ErrorCode::InvalidResponse);
}

// ─── Anthropic streaming with reasoning ─────────────────────────────────

#[tokio::test]
async fn anthropic_stream_with_thinking() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::anthropic()
        .api_key("test-anthropic-key")
        .base_url(server.uri())
        .api_version("2023-06-01")
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder("claude-3-5-haiku-latest")
        .message(Message::user_text("hello"))
        .temperature(0.2)
        .max_output_tokens(128)
        .build()
        .expect("request should build");

    let mut stream = client
        .stream(req)
        .await
        .expect("stream_text should succeed");

    let mut reasoning = String::new();
    let mut text = String::new();
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::TextDelta { text: t } => text.push_str(&t),
            StreamEvent::ReasoningDelta { text: r, .. } => reasoning.push_str(&r),
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert_eq!(reasoning, "plan");
    assert_eq!(text, "answer");
    assert!(saw_done);
}

// ─── OpenAI streaming with reasoning ────────────────────────────────────

#[tokio::test]
async fn openai_stream_with_reasoning_delta() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"thinking step 1\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"content_index\":0,\"delta\":\"answer\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":3,\"total_tokens\":8}}}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder("gpt-5.4-mini")
        .message(Message::user_text("hello"))
        .temperature(0.2)
        .max_output_tokens(128)
        .build()
        .expect("request should build");

    let mut stream = client
        .stream(req)
        .await
        .expect("stream_text should succeed");

    let mut reasoning = String::new();
    let mut text = String::new();
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::TextDelta { text: t } => text.push_str(&t),
            StreamEvent::ReasoningDelta { text: r, .. } => reasoning.push_str(&r),
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert_eq!(reasoning, "thinking step 1");
    assert_eq!(text, "answer");
    assert!(saw_done);
}

#[tokio::test]
async fn openai_stream_with_tool_calls_and_finish() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"weather\",\"status\":\"in_progress\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"item_id\":\"fc_1\",\"delta\":\"{\\\"city\\\":\\\"NYC\\\"}\"}\n\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"item_id\":\"fc_1\",\"arguments\":\"{\\\"city\\\":\\\"NYC\\\"}\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":3,\"total_tokens\":8}}}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder("gpt-5.4-mini")
        .message(Message::user_text("weather?"))
        .temperature(0.2)
        .max_output_tokens(128)
        .build()
        .expect("request should build");

    let mut stream = client
        .stream(req)
        .await
        .expect("stream_text should succeed");

    let mut saw_tool_call = false;
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::ToolCallReady { call } if call.tool_name == "weather" => {
                saw_tool_call = true;
            }
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert!(saw_tool_call);
    assert!(saw_done);
}

// ─── OpenAI generate with reasoning ─────────────────────────────────────

#[tokio::test]
async fn openai_generate_with_reasoning() {
    let server = MockServer::start().await;
    let body = json!({
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "answer" }]
            }
        ],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gpt-5.4-mini",
            "hello",
        ))
        .await
        .expect("request should succeed");

    // Responses API does not expose raw thinking in non-streaming responses.
    assert_eq!(response.output_text, "answer");
    assert_eq!(response.reasoning_text, "");
    assert!(response.reasoning_parts.is_empty());
}

#[tokio::test]
async fn openai_generate_with_tool_calls() {
    let server = MockServer::start().await;
    let body = json!({
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "{\"city\":\"Tokyo\"}"
            }
        ],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gpt-5.4-mini",
            "weather?",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].tool_name, "weather");
}

// ─── Anthropic generate with reasoning and tool use ─────────────────────

#[tokio::test]
async fn anthropic_generate_with_thinking_and_tool_use() {
    let server = MockServer::start().await;
    let body = json!({
        "content": [
            {
                "type": "thinking",
                "thinking": "Let me check the weather",
                "signature": "sig123"
            },
            {
                "type": "text",
                "text": "Checking..."
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "weather",
                "input": { "city": "Paris" }
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::anthropic()
        .api_key("test-anthropic-key")
        .base_url(server.uri())
        .api_version("2023-06-01")
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "claude-3-5-haiku-latest",
            "weather?",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.output_text, "Checking...");
    assert_eq!(
        response.reasoning_parts.first().unwrap().text,
        "Let me check the weather"
    );
    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    assert_eq!(response.tool_calls.len(), 1);
}

#[tokio::test]
async fn anthropic_generate_with_redacted_thinking() {
    let server = MockServer::start().await;
    let body = json!({
        "content": [
            {
                "type": "redacted_thinking",
                "data": "redacted"
            },
            {
                "type": "text",
                "text": "safe answer"
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::anthropic()
        .api_key("test-anthropic-key")
        .base_url(server.uri())
        .api_version("2023-06-01")
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "claude-3-5-haiku-latest",
            "hello",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.output_text, "safe answer");
    assert_eq!(response.reasoning_parts.len(), 1);
}

// ─── Anthropic error handling ───────────────────────────────────────────

#[tokio::test]
async fn anthropic_invalid_response_missing_content() {
    let server = MockServer::start().await;
    let body = json!({"not_content": []});

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::anthropic()
        .api_key("test-anthropic-key")
        .base_url(server.uri())
        .api_version("2023-06-01")
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "claude-3-5-haiku-latest",
            "hello",
        ))
        .await
        .expect_err("should fail");

    assert_eq!(err.code, ErrorCode::InvalidResponse);
}

#[tokio::test]
async fn anthropic_500_maps_to_provider_server_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(502).set_body_string("bad gateway"))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::anthropic()
        .api_key("test-anthropic-key")
        .base_url(server.uri())
        .api_version("2023-06-01")
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "claude-3-5-haiku-latest",
            "hello",
        ))
        .await
        .expect_err("request should fail");

    assert_eq!(err.code, ErrorCode::ProviderServerError);
}

// ─── OpenAI 403 error mapping ───────────────────────────────────────────

#[tokio::test]
async fn openai_403_maps_to_auth_failed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(403).set_body_string("forbidden"))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gpt-5.4-mini",
            "hello",
        ))
        .await
        .expect_err("request should fail");

    assert_eq!(err.code, ErrorCode::AuthFailed);
}

// ─── OpenAI invalid response ────────────────────────────────────────────

#[tokio::test]
async fn openai_invalid_response_missing_output() {
    let server = MockServer::start().await;
    let body = json!({"status": "completed", "no_output": []});

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gpt-5.4-mini",
            "hello",
        ))
        .await
        .expect_err("should fail");

    assert_eq!(err.code, ErrorCode::InvalidResponse);
}

// ─── Refresh tests added during the May 2026 SDK audit ──────────────────

/// Refusal text on the OpenAI Responses streaming channel must reach the caller.
/// Before the audit the adapter listened only to `response.output_text.delta`,
/// so a refused stream produced an empty TextDelta sequence.
#[tokio::test]
async fn openai_stream_refusal_surfaces_as_text_delta() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"response.refusal.delta\",\"output_index\":0,\"delta\":\"I cannot help with that.\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0,\"total_tokens\":1}}}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let mut stream = client
        .stream(GenerateTextRequest::from_user_prompt("gpt-5.4-mini", "hi"))
        .await
        .expect("stream_text should succeed");

    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event.expect("stream event should parse") {
            StreamEvent::TextDelta { text: t } => text.push_str(&t),
            StreamEvent::Done => break,
            _ => {}
        }
    }
    assert_eq!(text, "I cannot help with that.");
}

/// Reasoning summary blocks returned in the non-streaming `output` array must be
/// preserved. Before the audit, only `message` and `function_call` items were
/// scanned and reasoning was silently dropped.
#[tokio::test]
async fn openai_generate_extracts_reasoning_summary_from_output_array() {
    let server = MockServer::start().await;
    let body = json!({
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [
                    { "type": "summary_text", "text": "Step one." },
                    { "type": "summary_text", "text": " Step two." }
                ]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "answer" }]
            }
        ],
        "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gpt-5.4-mini",
            "hello",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.output_text, "answer");
    assert_eq!(response.reasoning_text, "Step one. Step two.");
    // Each reasoning item joins its summary segments into one ReasoningPart.
    assert_eq!(response.reasoning_parts.len(), 1);
    assert_eq!(response.reasoning_parts[0].text, "Step one. Step two.");
}

/// `response.error` is a top-level streaming event distinct from `response.failed`
/// in the SDK union. The adapter must propagate it as an error rather than
/// dropping it silently.
#[tokio::test]
async fn openai_stream_error_event_propagates_as_invalid_response() {
    let server = MockServer::start().await;
    let sse_body =
        "data: {\"type\":\"response.error\",\"message\":\"upstream blew up\"}\n\n".to_string();

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let mut stream = client
        .stream(GenerateTextRequest::from_user_prompt("gpt-5.4-mini", "hi"))
        .await
        .expect("stream_text should succeed");

    let mut saw_error = false;
    while let Some(event) = stream.next().await {
        if let Err(err) = event {
            assert_eq!(err.code, ErrorCode::InvalidResponse);
            assert!(err.to_string().contains("upstream blew up"));
            saw_error = true;
            break;
        }
    }
    assert!(
        saw_error,
        "expected response.error to surface as a stream error"
    );
}

/// The OpenAI Chat Completions contract uses `refusal` as a parallel channel to
/// `content`. The compatible adapter must surface it so a refused completion is
/// not silently empty.
#[tokio::test]
async fn openai_compatible_response_refusal_surfaces_as_output_text() {
    let server = MockServer::start().await;
    let body = json!({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": null,
                    "refusal": "I cannot help with that."
                },
                "finish_reason": "stop"
            }
        ]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible()
        .base_url(server.uri())
        .api_key("k")
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt("any-model", "hi"))
        .await
        .expect("request should succeed");

    assert_eq!(response.output_text, "I cannot help with that.");
}

/// When the upstream tool result is already a `Value::String`, the request body
/// must carry the plain string content. Previously `Value::to_string()` was
/// applied unconditionally, double-encoding the string and feeding the model
/// `"\"text\""` instead of `text`.
#[tokio::test]
async fn openai_compatible_tool_result_string_value_is_not_double_encoded() {
    use std::sync::{Arc, Mutex};

    let server = MockServer::start().await;

    #[derive(Clone)]
    struct CaptureBody {
        captured: Arc<Mutex<Option<serde_json::Value>>>,
    }
    impl Respond for CaptureBody {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("body should be JSON");
            *self.captured.lock().unwrap() = Some(body);
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "role": "assistant", "content": "ok" }, "finish_reason": "stop" }]
            }))
        }
    }

    let captured = Arc::new(Mutex::new(None));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(CaptureBody {
            captured: Arc::clone(&captured),
        })
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible()
        .base_url(server.uri())
        .api_key("k")
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder("any-model")
        .message(Message::user_text("ask"))
        .message(
            Message::new(
                MessageRole::Assistant,
                vec![ContentPart::ToolCall(aquaregia::ToolCall {
                    call_id: "c1".into(),
                    tool_name: "echo".into(),
                    args_json: json!({}),
                })],
            )
            .expect("assistant tool call message"),
        )
        .message(Message::tool_result(ToolResult {
            call_id: "c1".into(),
            output_json: serde_json::Value::String("plain text result".into()),
            is_error: false,
        }))
        .build()
        .expect("request should build");

    let _ = client.generate(req).await.expect("request should succeed");

    let body = captured
        .lock()
        .unwrap()
        .clone()
        .expect("request body captured");
    let messages = body
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .expect("messages array");

    let tool_msg = messages
        .iter()
        .find(|m| m.get("role").and_then(serde_json::Value::as_str) == Some("tool"))
        .expect("tool message in payload");
    assert_eq!(
        tool_msg.get("content").and_then(serde_json::Value::as_str),
        Some("plain text result"),
        "Value::String tool results must be sent as plain text, not JSON-encoded"
    );

    let assistant_msg = messages
        .iter()
        .find(|m| m.get("role").and_then(serde_json::Value::as_str) == Some("assistant"))
        .expect("assistant message in payload");
    // Per OpenAI Chat Completions: when only tool_calls are present, content is null.
    assert!(
        assistant_msg.get("content").is_some_and(|v| v.is_null()),
        "assistant tool-only message must serialize content as null, got {:?}",
        assistant_msg.get("content")
    );
}

/// When Gemini blocks a prompt entirely it returns no candidates and surfaces a
/// `promptFeedback.blockReason`. The adapter error must include that reason so
/// callers can distinguish blocked prompts from a malformed upstream payload.
#[tokio::test]
async fn google_response_block_reason_surfaces_in_error() {
    let server = MockServer::start().await;
    let body = json!({
        "promptFeedback": { "blockReason": "SAFETY" }
    });

    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google()
        .api_key("test-google-key")
        .base_url(server.uri())
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-3.5-flash",
            "hello",
        ))
        .await
        .expect_err("blocked prompts must error");

    assert_eq!(err.code, ErrorCode::InvalidResponse);
    assert!(
        err.to_string().contains("SAFETY"),
        "block reason must be carried in the error message, got: {err}"
    );
}

/// Newer Gemini models populate `functionCall.id`. The adapter must round-trip
/// it as `tool_call.call_id` rather than always synthesizing `google_call_*`.
#[tokio::test]
async fn google_uses_upstream_function_call_id_when_present() {
    let server = MockServer::start().await;
    let body = json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        { "functionCall": { "id": "fc_abc", "name": "weather", "args": { "city": "NYC" } } }
                    ]
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2 }
    });

    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google()
        .api_key("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-3.5-flash",
            "weather?",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].call_id, "fc_abc");
    assert_eq!(response.tool_calls[0].tool_name, "weather");
}

// ─── Structured output — Anthropic tool-use trick ──────────────────────────

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct WeatherOutput {
    city: String,
    temp_c: f64,
}

#[tokio::test]
async fn anthropic_generate_object_extracts_tool_call_args() {
    let server = MockServer::start().await;
    let body = json!({
        "id": "msg_001",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{
            "type": "tool_use",
            "id": "toolu_001",
            "name": "respond",
            "input": { "city": "NYC", "temp_c": 23.0 }
        }],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 42, "output_tokens": 15 }
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let req = GenerateTextRequest::builder("claude-sonnet-4-6")
        .message(Message::user_text("weather in NYC"))
        .build()
        .expect("request should build");

    let client = LlmClient::anthropic()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate_object::<WeatherOutput>(req)
        .await
        .expect("generate_object should succeed");

    assert_eq!(response.object.city, "NYC");
    assert_eq!(response.object.temp_c, 23.0);
    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
}

#[tokio::test]
async fn anthropic_generate_object_injects_respond_tool() {
    let server = MockServer::start().await;
    let body = json!({
        "id": "msg_001",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{
            "type": "tool_use",
            "id": "toolu_001",
            "name": "respond",
            "input": { "city": "LA", "temp_c": 28.0 }
        }],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 40, "output_tokens": 12 }
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let req = GenerateTextRequest::builder("claude-sonnet-4-6")
        .message(Message::user_text("weather in LA"))
        .build()
        .expect("request should build");

    let client = LlmClient::anthropic()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    client
        .generate_object::<WeatherOutput>(req)
        .await
        .expect("generate_object should succeed");

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should record requests");
    let body: serde_json::Value = requests[0]
        .body_json()
        .expect("request body should be valid json");

    // Verify the injected "respond" tool.
    let tools = body["tools"].as_array().expect("tools should be present");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "respond");
    assert_eq!(tools[0]["type"], "custom");
    assert!(tools[0]["input_schema"].is_object());

    // Verify tool_choice forces "respond".
    let tc = &body["tool_choice"];
    assert_eq!(tc["type"], "tool");
    assert_eq!(tc["name"], "respond");
}

// ─── Structured output — Google function-calling trick ─────────────────────

#[tokio::test]
async fn google_generate_object_extracts_function_call_args() {
    let server = MockServer::start().await;
    let body = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "respond",
                        "args": { "city": "NYC", "temp_c": 23.0 }
                    }
                }]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "totalTokenCount": 30
        }
    });
    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:generateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let req = GenerateTextRequest::builder("gemini-3.5-flash")
        .message(Message::user_text("weather in NYC"))
        .build()
        .expect("request should build");

    let client = LlmClient::google()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate_object::<WeatherOutput>(req)
        .await
        .expect("generate_object should succeed");

    assert_eq!(response.object.city, "NYC");
    assert_eq!(response.object.temp_c, 23.0);
}

#[tokio::test]
async fn google_generate_object_injects_respond_function() {
    let server = MockServer::start().await;
    let body = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "respond",
                        "args": { "city": "LA", "temp_c": 28.0 }
                    }
                }]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "totalTokenCount": 30
        }
    });
    Mock::given(method("POST"))
        .and(path("/models/gemini-3.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let req = GenerateTextRequest::builder("gemini-3.5-flash")
        .message(Message::user_text("weather in LA"))
        .build()
        .expect("request should build");

    let client = LlmClient::google()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    client
        .generate_object::<WeatherOutput>(req)
        .await
        .expect("generate_object should succeed");

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should record requests");
    let body: serde_json::Value = requests[0]
        .body_json()
        .expect("request body should be valid json");

    // Verify the injected "respond" function declaration.
    let tools = body["tools"].as_array().expect("tools should be present");
    let decls = tools[0]["functionDeclarations"]
        .as_array()
        .expect("functionDeclarations should be present");
    assert_eq!(decls.len(), 1);
    assert_eq!(decls[0]["name"], "respond");
    assert!(decls[0]["parameters"].is_object());

    // Verify toolConfig forces "respond".
    let tc = &body["toolConfig"]["functionCallingConfig"];
    assert_eq!(tc["mode"], "ANY");
    assert_eq!(tc["allowedFunctionNames"][0], "respond");
}
