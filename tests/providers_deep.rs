use aquaregia::{ErrorCode, FinishReason, GenerateTextRequest, LlmClient, Message, StreamEvent};
use futures_util::StreamExt;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ─── Google streaming ───────────────────────────────────────────────────

#[tokio::test]
async fn google_stream_emits_text_usage_done() {
    let server = MockServer::start().await;
    let sse_body = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"thoughtsTokenCount\":1,\"totalTokenCount\":6}}\n\n";

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-google-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::from_user_prompt("gemini-2.0-flash", "hello");

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
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-google-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::from_user_prompt("gemini-2.0-flash", "hello");

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
        .and(path("/models/gemini-2.0-flash:generateContent"))
        .and(header("x-goog-api-key", "test-google-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-2.0-flash",
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
        .and(path("/models/gemini-2.0-flash:generateContent"))
        .and(header("x-goog-api-key", "test-google-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-2.0-flash",
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
        .and(path("/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google("test-google-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-2.0-flash",
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
        .and(path("/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(503).set_body_string("service unavailable"))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google("test-google-key")
        .base_url(server.uri())
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-2.0-flash",
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
        .and(path("/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::google("test-google-key")
        .base_url(server.uri())
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gemini-2.0-flash",
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

    let client = LlmClient::anthropic("test-anthropic-key")
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

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder("gpt-4o-mini")
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

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder("gpt-4o-mini")
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

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gpt-4o-mini",
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

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gpt-4o-mini",
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

    let client = LlmClient::anthropic("test-anthropic-key")
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

    let client = LlmClient::anthropic("test-anthropic-key")
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

    let client = LlmClient::anthropic("test-anthropic-key")
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

    let client = LlmClient::anthropic("test-anthropic-key")
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

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gpt-4o-mini",
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

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .max_retries(0)
        .build()
        .expect("client should build");

    let err = client
        .generate(GenerateTextRequest::from_user_prompt(
            "gpt-4o-mini",
            "hello",
        ))
        .await
        .expect_err("should fail");

    assert_eq!(err.code, ErrorCode::InvalidResponse);
}
