use aquaregia::{ErrorCode, GenerateTextRequest, LlmClient, Message, ToolDescriptor};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn openai_request() -> GenerateTextRequest {
    GenerateTextRequest::builder("gpt-4o-mini")
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

    let client = LlmClient::openai("test-openai-key")
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

    let client = LlmClient::openai("test-openai-key")
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

    let req = GenerateTextRequest::builder("gpt-4o")
        .message(Message::system_text("be helpful"))
        .message(Message::user_text("hello"))
        .tools([tool])
        .build()
        .expect("request should build");

    let client = LlmClient::openai("test-openai-key")
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
    assert!(tools[0].get("function").is_none(), "name must not be nested in `function`");
}
