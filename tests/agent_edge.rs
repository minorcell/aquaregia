use aquaregia::{Agent, AgentPreparedStep, ErrorCode, LlmClient, Message, ToolErrorPolicy};
use serde::Deserialize;
use serde_json::json;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ─── Max steps exceeded ─────────────────────────────────────────────────

#[tokio::test]
async fn agent_max_steps_exceeded() {
    let server = MockServer::start().await;

    // A response that keeps asking for a tool call, so the agent loops.
    let step_body = json!({
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "dummy",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(step_body))
        .expect(3) // max_steps=3, so 3 loops
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let dummy = aquaregia::tool("dummy")
        .description("dummy")
        .execute_raw(|_args| async move { Ok(json!({"ok": true})) });

    let agent = Agent::builder(client, "gpt-4o-mini")
        .tools([dummy])
        .max_steps(3)
        .build()
        .expect("agent should build");

    let err = agent
        .run("loop forever")
        .await
        .expect_err("should hit max steps");

    assert_eq!(err.code, ErrorCode::MaxStepsExceeded);
}

// ─── Stop when predicate ────────────────────────────────────────────────

#[tokio::test]
async fn agent_stop_when_predicate_stops_early() {
    let server = MockServer::start().await;

    let step1 = json!({
        "choices": [{
            "message": { "content": "First response" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(step1))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, "gpt-4o-mini")
        .max_steps(5)
        .stop_when(|step| step.output_text.contains("First"))
        .build()
        .expect("agent should build");

    let response = agent
        .run("say something")
        .await
        .expect("agent should succeed");

    assert_eq!(response.output_text, "First response");
    assert_eq!(response.steps, 1);
}

// ─── ToolErrorPolicy::FailFast with deserialization failure ─────────────

#[tokio::test]
async fn agent_fail_fast_on_invalid_tool_args() {
    let server = MockServer::start().await;

    let step1 = json!({
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": "{\"bad_key\": 123}" // typed executor requires "city"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(step1))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct WeatherArgs {
        city: String,
    }

    let weather = aquaregia::tool("weather")
        .description("Get weather by city")
        .execute(|args: WeatherArgs| async move { Ok(json!({"city": args.city, "temp": 22})) });

    let agent = Agent::builder(client, "gpt-4o-mini")
        .tools([weather])
        .max_steps(3)
        .tool_error_policy(ToolErrorPolicy::FailFast)
        .build()
        .expect("agent should build");

    let err = agent
        .run("weather?")
        .await
        .expect_err("should fail on invalid tool args");

    assert_eq!(err.code, ErrorCode::ToolExecutionFailed);
}

// ─── ToolErrorPolicy::ContinueAsToolResult (default) ────────────────────

#[tokio::test]
async fn agent_continue_on_invalid_tool_args() {
    let server = MockServer::start().await;

    let step1 = json!({
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": "{\"bad_key\": 123}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    });

    let step2 = json!({
        "choices": [{
            "message": { "content": "Recovered from error" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"role\":\"tool\""))
        .respond_with(ResponseTemplate::new(200).set_body_json(step2))
        .expect(1)
        .with_priority(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(step1))
        .expect(1)
        .with_priority(5)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let weather = aquaregia::tool("weather")
        .description("Get weather by city")
        .raw_schema(json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }))
        .execute_raw(|args| async move { Ok(json!({"city": args["city"], "temp": 22})) });

    let agent = Agent::builder(client, "gpt-4o-mini")
        .tools([weather])
        .max_steps(3)
        .tool_error_policy(ToolErrorPolicy::ContinueAsToolResult)
        .build()
        .expect("agent should build");

    let response = agent.run("weather?").await.expect("agent should succeed");

    assert_eq!(response.output_text, "Recovered from error");
    assert_eq!(response.steps, 2);
}

// ─── Agent with no tools returns immediately ────────────────────────────

#[tokio::test]
async fn agent_without_tools_returns_first_response() {
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "Direct answer" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, "gpt-4o-mini")
        .max_steps(5)
        .build()
        .expect("agent should build");

    let response = agent.run("hello").await.expect("agent should succeed");

    assert_eq!(response.output_text, "Direct answer");
    assert_eq!(response.steps, 1);
}

// ─── Agent run_messages ─────────────────────────────────────────────────

#[tokio::test]
async fn agent_run_messages_uses_explicit_message_list() {
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "Response" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("custom-message-content"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, "gpt-4o-mini")
        .build()
        .expect("agent should build");

    let response = agent
        .run_messages(vec![Message::user_text("custom-message-content")])
        .await
        .expect("agent should succeed");

    assert_eq!(response.output_text, "Response");
}

// ─── Agent instructions already has system message ──────────────────────

#[tokio::test]
async fn agent_respects_existing_system_message() {
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "OK" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("my-custom-instruction"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    // With agent instructions set AND an explicit system message,
    // the explicit system message should take precedence over agent instructions.
    let agent = Agent::builder(client, "gpt-4o-mini")
        .instructions("agent-level-instruction")
        .build()
        .expect("agent should build");

    let response = agent
        .run_messages(vec![Message::system_text("my-custom-instruction")])
        .await
        .expect("agent should succeed");

    assert_eq!(response.output_text, "OK");
}

// ─── Agent prepare_step changes tools ───────────────────────────────────

#[tokio::test]
async fn agent_prepare_step_changes_tools() {
    let server = MockServer::start().await;

    // The prepare_step hook removes all tools, so the agent finishes in 1 step.
    let body = json!({
        "choices": [{
            "message": { "content": "No tools, direct answer" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-key")
        .build()
        .expect("client should build");

    let dummy = aquaregia::tool("dummy")
        .description("dummy")
        .execute_raw(|_args| async move { Ok(json!({"ok": true})) });

    let agent = Agent::builder(client, "gpt-4o-mini")
        .tools([dummy])
        .max_steps(5)
        .prepare_step(|event| AgentPreparedStep {
            model: event.model.clone(),
            messages: event.messages.clone(),
            tools: vec![], // remove all tools
            temperature: event.temperature,
            max_output_tokens: event.max_output_tokens,
            stop_sequences: event.stop_sequences.clone(),
        })
        .build()
        .expect("agent should build");

    let response = agent.run("hello").await.expect("agent should succeed");

    assert_eq!(response.output_text, "No tools, direct answer");
    assert_eq!(response.steps, 1);
}
