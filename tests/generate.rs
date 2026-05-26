use aquaregia::{
    ErrorCode, GenerateTextRequest, LlmClient, Message, OutputSchema, StreamObjectEvent,
    ToolDescriptor,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn openai_request() -> GenerateTextRequest {
    GenerateTextRequest::builder("gpt-5.4-mini")
        .message(Message::user_text("hello"))
        .temperature(0.2)
        .max_output_tokens(64)
        .build()
        .expect("request should build")
}

#[tokio::test]
async fn openai_generate_text_success() {
    let server = MockServer::start().await;
    let body = json!({
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "Hello from OpenAI" }]
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15
        }
    });
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", "Bearer test-openai-key"))
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
        .generate(openai_request())
        .await
        .expect("generate_text should succeed");

    assert_eq!(response.output_text, "Hello from OpenAI");
    assert_eq!(response.usage.total_tokens, 15);
}

#[tokio::test]
async fn openai_401_maps_to_auth_failed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let err = client
        .generate(openai_request())
        .await
        .expect_err("request should fail");

    assert_eq!(err.code, ErrorCode::AuthFailed);
    assert_eq!(err.status, Some(401));
    assert!(!err.retryable);
}

#[tokio::test]
async fn openai_responses_api_request_shape() {
    // Verifies the outbound payload uses Responses API conventions, not Chat Completions.
    let server = MockServer::start().await;
    let body = json!({
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "ok" }]
            }
        ],
        "usage": { "input_tokens": 5, "output_tokens": 2, "total_tokens": 7 }
    });
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let tool = ToolDescriptor {
        name: "my_tool".to_string(),
        description: "a tool".to_string(),
        input_schema: json!({ "type": "object", "properties": {} }),
    };

    let req = GenerateTextRequest::builder("gpt-5.5")
        .message(Message::system_text("be helpful"))
        .message(Message::user_text("hello"))
        .tools([tool])
        .build()
        .expect("request should build");

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    client.generate(req).await.expect("request should succeed");

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should record requests");
    let body: serde_json::Value = requests[0]
        .body_json()
        .expect("request body should be valid json");

    // System message must become top-level `instructions`, not a message in `input`.
    assert_eq!(body["instructions"], "be helpful");
    assert!(
        body["input"]
            .as_array()
            .expect("input should be an array")
            .iter()
            .all(|item| item.get("role").and_then(|r| r.as_str()) != Some("system")),
        "system role must not appear in input array"
    );

    // Tool definition must have `name` at top level (Responses API format, not nested in `function`).
    let tools = body["tools"].as_array().expect("tools should be an array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "my_tool");
    assert!(
        tools[0].get("function").is_none(),
        "name must not be nested in `function`"
    );
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct WeatherResult {
    #[serde(default)]
    city: String,
    #[serde(default)]
    temp_c: f64,
}

#[tokio::test]
async fn openai_responses_api_request_shape_with_output_schema() {
    // Verifies the Responses API payload includes `text.format` when output_schema is set.
    let server = MockServer::start().await;
    let body = json!({
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": r#"{"city":"NYC","temp_c":23.0}"# }]
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

    let req = GenerateTextRequest::builder("gpt-5.5")
        .message(Message::user_text("weather in NYC"))
        .build()
        .expect("request should build");

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    client
        .generate_object::<WeatherResult>(req)
        .await
        .expect("generate_object should succeed");

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should record requests");
    let body: serde_json::Value = requests[0]
        .body_json()
        .expect("request body should be valid json");

    let text_format = body["text"]["format"].clone();
    assert_eq!(text_format["type"], "json_schema");
    assert_eq!(text_format["name"], "WeatherResult");
    assert_eq!(text_format["strict"], true);
    assert!(text_format["schema"].is_object());
}

#[tokio::test]
async fn generate_object_deserializes_response() {
    let server = MockServer::start().await;
    let body = json!({
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": r#"{"city":"NYC","temp_c":23.0}"# }]
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

    let req = GenerateTextRequest::builder("gpt-5.5")
        .message(Message::user_text("weather in NYC"))
        .build()
        .expect("request should build");

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate_object::<WeatherResult>(req)
        .await
        .expect("generate_object should succeed");

    assert_eq!(response.object.city, "NYC");
    assert_eq!(response.object.temp_c, 23.0);
    assert_eq!(response.usage.total_tokens, 8);
    assert_eq!(response.finish_reason, aquaregia::FinishReason::Stop);
}

#[tokio::test]
async fn generate_object_rejects_invalid_json() {
    let server = MockServer::start().await;
    let body = json!({
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "not json at all" }]
            }
        ],
        "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
    });
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let req = GenerateTextRequest::builder("gpt-5.5")
        .message(Message::user_text("test"))
        .build()
        .expect("request should build");

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let err = client
        .generate_object::<WeatherResult>(req)
        .await
        .expect_err("invalid JSON should fail");

    assert_eq!(err.code, ErrorCode::InvalidResponse);
    assert!(err.message.contains("failed to parse structured output"));
    assert_eq!(err.raw_body.as_deref(), Some("not json at all"));
}

#[tokio::test]
async fn generate_object_via_builder_output_schema() {
    let server = MockServer::start().await;
    let body = json!({
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": r#"{"city":"LA","temp_c":28.0}"# }]
            }
        ],
        "usage": { "input_tokens": 4, "output_tokens": 2, "total_tokens": 6 }
    });
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let schema = schemars::schema_for!(WeatherResult);
    let json_schema = serde_json::to_value(&schema).expect("serialize schema");
    let req = GenerateTextRequest::builder("gpt-5.5")
        .message(Message::user_text("weather in LA"))
        .output_schema(OutputSchema {
            name: "weather".into(),
            description: Some("weather forecast".into()),
            json_schema,
        })
        .build()
        .expect("request should build");

    let client = LlmClient::openai()
        .api_key("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let response = client
        .generate_object::<WeatherResult>(req)
        .await
        .expect("generate_object should succeed");

    assert_eq!(response.object.city, "LA");
}

#[tokio::test]
async fn openai_compatible_generate_object_request_shape() {
    // Verifies the Chat Completions payload includes `response_format.json_schema`.
    let server = MockServer::start().await;
    let body = json!({
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": r#"{"city":"NYC","temp_c":23.0}"# },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10 }
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let req = GenerateTextRequest::builder("deepseek-v4-pro")
        .message(Message::user_text("weather in NYC"))
        .build()
        .expect("request should build");

    let client = LlmClient::openai_compatible()
        .base_url(server.uri())
        .build()
        .expect("client should build");

    client
        .generate_object::<WeatherResult>(req)
        .await
        .expect("generate_object should succeed");

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should record requests");
    let body: serde_json::Value = requests[0]
        .body_json()
        .expect("request body should be valid json");

    let rf = &body["response_format"];
    assert_eq!(rf["type"], "json_schema");
    assert_eq!(rf["json_schema"]["name"], "WeatherResult");
    assert_eq!(rf["json_schema"]["strict"], true);
    assert!(rf["json_schema"]["schema"].is_object());
}

// ─── Streaming structured output ──────────────────────────────────────────

#[tokio::test]
async fn stream_object_emits_progressive_partials_and_final_object() {
    let server = MockServer::start().await;

    // First delta is truncated mid-value: second delta completes it.
    let sse_body = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"{\\\"city\\\":\\\"NYC\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"\\\",\\\"temp_c\\\":23.0}\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":4,\"total_tokens\":9}}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder("gpt-5.5")
        .message(Message::user_text("weather"))
        .build()
        .expect("request should build");

    let mut stream = client
        .stream_object::<WeatherResult>(req)
        .await
        .expect("stream_object should succeed");

    let mut saw_partial = false;
    let mut saw_object = false;

    use futures_util::StreamExt;
    while let Some(event) = stream.next().await {
        match event.expect("event should parse") {
            StreamObjectEvent::Partial { partial } => {
                assert_eq!(partial.city, "NYC");
                // temp_c may or may not have arrived
                saw_partial = true;
            }
            StreamObjectEvent::Object { object } => {
                assert_eq!(object.city, "NYC");
                assert_eq!(object.temp_c, 23.0);
                saw_object = true;
            }
        }
    }

    assert!(
        saw_partial,
        "should have emitted at least one Partial event"
    );
    assert!(saw_object, "should have emitted final Object event");
}
