//! Integration tests for provider_options passthrough.

use aquaregia::{
    Agent, ContentPart, GenerateTextRequest, LlmClient, Message, MessageRole, TextPart, tool,
};
use serde_json::json;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn request_serializes_provider_options() {
    let req = GenerateTextRequest::builder("claude-sonnet-4-6")
        .message(Message::user_text("test"))
        .provider_options(json!({
            "anthropic": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 5000
                }
            },
            "google": {
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ]
            }
        }))
        .build()
        .unwrap();

    let serialized = serde_json::to_value(&req).unwrap();
    assert!(serialized.get("provider_options").is_some());
    assert!(serialized["provider_options"]["anthropic"]["thinking"]["budget_tokens"] == 5000);
}

#[test]
fn request_without_provider_options_omits_field() {
    let req = GenerateTextRequest::builder("gpt-4")
        .message(Message::user_text("test"))
        .build()
        .unwrap();

    let serialized = serde_json::to_value(&req).unwrap();
    assert!(serialized.get("provider_options").is_none());
}

#[test]
fn request_deserializes_provider_options() {
    let json_str = r#"{
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "User", "parts": [{"Text": {"text": "hello"}}], "name": null}],
        "temperature": null,
        "top_p": null,
        "max_output_tokens": null,
        "stop_sequences": [],
        "tools": null,
        "output_schema": null,
        "provider_options": {
            "anthropic": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            }
        }
    }"#;

    let req: GenerateTextRequest = serde_json::from_str(json_str).unwrap();
    assert!(req.provider_options().is_some());
    let options = req.provider_options().unwrap();
    assert_eq!(options["anthropic"]["thinking"]["budget_tokens"], 10000);
}

#[tokio::test]
async fn agent_run_passes_provider_options_to_every_step() {
    // The risk this test guards against: a multi-step agent loop drops
    // provider_options after the first step. We assert the marker field
    // appears in both the initial call and the post-tool-result call.
    let server = MockServer::start().await;

    let step1 = json!({
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });

    let step2 = json!({
        "choices": [{
            "message": { "content": "done" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });

    // Step 2: post-tool-result body must still carry the provider_options marker.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"role\":\"tool\""))
        .and(body_string_contains("\"agent_marker_field\""))
        .and(body_string_contains("\"agent-marker-value\""))
        .respond_with(ResponseTemplate::new(200).set_body_json(step2))
        .expect(1)
        .with_priority(1)
        .mount(&server)
        .await;

    // Step 1: initial body must carry it too.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"agent_marker_field\""))
        .and(body_string_contains("\"agent-marker-value\""))
        .respond_with(ResponseTemplate::new(200).set_body_json(step1))
        .expect(1)
        .with_priority(5)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible()
        .base_url(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let ping = tool("ping")
        .description("noop")
        .raw_schema(json!({ "type": "object", "properties": {} }))
        .execute_raw(|_| async move { Ok(json!({ "ok": true })) });

    let agent = Agent::builder(client, "gpt-5.4-mini")
        .tools([ping])
        .max_steps(3)
        .provider_options(json!({
            "openai-compatible": {
                "agent_marker_field": "agent-marker-value"
            }
        }))
        .build()
        .expect("agent should build");

    let response = agent
        .run("trigger the tool")
        .await
        .expect("agent run should succeed");

    assert_eq!(response.output_text, "done");
    assert_eq!(response.steps, 2);
}

#[tokio::test]
async fn agent_without_provider_options_sends_no_marker() {
    // Boundary: when the builder is not given provider_options, the request
    // body must not carry the agent-level marker. Guards against accidental
    // leakage from a default value.
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "hi" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible()
        .base_url(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, "gpt-5.4-mini")
        .build()
        .expect("agent should build");

    let response = agent.run("hello").await.expect("agent run should succeed");
    assert_eq!(response.output_text, "hi");
}

#[tokio::test]
async fn message_level_provider_options_ride_each_message() {
    // The risk: per-Message provider_options is silently dropped by the
    // adapter when serializing the request. We assert the marker key appears
    // on the user message in the outbound body.
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "ok" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"msg_marker_field\""))
        .and(body_string_contains("\"msg-marker-value\""))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible()
        .base_url(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let message = Message::user_text("hello").with_provider_options(json!({
        "openai-compatible": { "msg_marker_field": "msg-marker-value" }
    }));
    let req = GenerateTextRequest::builder("gpt-5.4-mini")
        .message(message)
        .build()
        .unwrap();

    let resp = client.generate(req).await.expect("generate should succeed");
    assert_eq!(resp.output_text, "ok");
}

#[tokio::test]
async fn text_block_provider_options_ride_each_text_block() {
    // The risk: per-TextPart provider_options is dropped on the wire, which
    // would silently break Anthropic-style cache_control breakpoints. We use
    // openai-compatible here for mockability; the same adapter rule is
    // exercised in all four implementations.
    //
    // Setting options on a text block also forces user/system content into
    // the typed-array form (a plain string cannot carry per-block fields),
    // so the wire must include the canonical `{"type":"text", ...}` shape.
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "ok" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"block_marker_field\""))
        .and(body_string_contains("\"block-marker-value\""))
        .and(body_string_contains("\"type\":\"text\""))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible()
        .base_url(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let text = TextPart::new("cache me").with_provider_options(json!({
        "openai-compatible": { "block_marker_field": "block-marker-value" }
    }));
    let message = Message::new(MessageRole::User, vec![ContentPart::Text(text)]).unwrap();
    let req = GenerateTextRequest::builder("gpt-5.4-mini")
        .message(message)
        .build()
        .unwrap();

    let resp = client.generate(req).await.expect("generate should succeed");
    assert_eq!(resp.output_text, "ok");
}

#[tokio::test]
async fn text_without_provider_options_stays_plain_string() {
    // Boundary: when no text part carries provider_options and there are no
    // images, the openai-compatible adapter must keep the canonical plain
    // string content shape. Switching to the typed-array form unconditionally
    // would inflate every request and break vendor endpoints that only accept
    // strings for `content`.
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{ "message": { "content": "ok" }, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });

    // Plain string content: the user content is the literal string, not an
    // object with "type":"text".
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"content\":\"hello\""))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible()
        .base_url(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder("gpt-5.4-mini")
        .user_prompt("hello")
        .build()
        .unwrap();

    let resp = client.generate(req).await.expect("generate should succeed");
    assert_eq!(resp.output_text, "ok");
}
