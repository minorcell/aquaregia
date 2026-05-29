//! Integration tests for provider_options passthrough.

use aquaregia::{Agent, GenerateTextRequest, LlmClient, Message, tool};
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
        "messages": [{"role": "User", "parts": [{"Text": "hello"}], "name": null}],
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
