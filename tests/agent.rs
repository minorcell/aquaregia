use std::sync::{Arc, Mutex};

use aquaregia::{Agent, AgentPreparedStep, LlmClient, Message, openai, tool};
use serde_json::json;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn agent_run_includes_instructions() {
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "hello" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"role\":\"system\""))
        .and(body_string_contains("You are concise."))
        .and(body_string_contains("Say hi"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, openai("gpt-4o-mini"))
        .instructions("You are concise.")
        .build()
        .expect("agent should build");

    let response = agent
        .run("Say hi")
        .await
        .expect("agent call should succeed");

    assert_eq!(response.output_text, "hello");
    assert_eq!(response.steps, 1);
}

#[tokio::test]
async fn agent_tool_loop_works() {
    let server = MockServer::start().await;

    let step1 = json!({
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Shanghai\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    });

    let step2 = json!({
        "choices": [{
            "message": { "content": "Shanghai is about 23C." },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12
        }
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

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let weather = tool("get_weather")
        .description("Get weather by city")
        .raw_schema(json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }))
        .execute_raw(|args| async move {
            let city = args
                .get("city")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            Ok(json!({ "city": city, "temp_c": 23 }))
        });

    let agent = Agent::builder(client, openai("gpt-4o-mini"))
        .tools([weather])
        .max_steps(3)
        .build()
        .expect("agent should build");

    let response = agent
        .run("What's the weather in Shanghai?")
        .await
        .expect("agent call should succeed");

    assert_eq!(response.output_text, "Shanghai is about 23C.");
    assert_eq!(response.steps, 2);
    assert_eq!(response.usage_total.total_tokens, 27);
}

#[tokio::test]
async fn agent_prepare_call_can_override_messages() {
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "prepared-call-ok" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("from-prepare-call"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, openai("gpt-4o-mini"))
        .prepare_call(|plan| {
            plan.messages = vec![
                Message::system_text("from-prepare-call"),
                Message::user_text("hello"),
            ];
        })
        .build()
        .expect("agent should build");

    let response = agent
        .run_messages(vec![Message::user_text("ignored")])
        .await
        .expect("agent call should succeed");

    assert_eq!(response.output_text, "prepared-call-ok");
    assert_eq!(response.steps, 1);
}

#[tokio::test]
async fn agent_prepare_step_can_override_messages() {
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "prepared-step-ok" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 2,
            "total_tokens": 8
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("from-agent-prepare-step"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, openai("gpt-4o-mini"))
        .max_steps(1)
        .prepare_step(|event| AgentPreparedStep {
            model: event.model.clone(),
            messages: vec![
                Message::system_text("from-agent-prepare-step"),
                Message::user_text("hello"),
            ],
            tools: event.tools.clone(),
            temperature: event.temperature,
            max_output_tokens: event.max_output_tokens,
            stop_sequences: event.stop_sequences.clone(),
        })
        .build()
        .expect("agent should build");

    let response = agent
        .run("Say hi")
        .await
        .expect("agent call should succeed");

    assert_eq!(response.output_text, "prepared-step-ok");
    assert_eq!(response.steps, 1);
}

#[tokio::test]
async fn agent_prepare_call_can_override_sampling_and_on_start_receives_model_id() {
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "prepared-plan-ok" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 2,
            "total_tokens": 8
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let start_event = Arc::new(Mutex::new(None::<(String, usize, usize, u8)>));
    let start_event_for_hook = Arc::clone(&start_event);
    let agent = Agent::builder(client, openai("gpt-4o-mini"))
        .max_steps(2)
        .top_p(0.3)
        .max_output_tokens(16)
        .stop_sequences(["STOP"])
        .prepare_call(|plan| {
            plan.model = openai("gpt-4.1-mini");
            plan.messages = vec![
                Message::system_text("from-prepare-call"),
                Message::user_text("hello"),
            ];
            plan.top_p = Some(0.7);
            plan.max_output_tokens = Some(42);
            plan.stop_sequences = vec!["END".to_string()];
        })
        .on_start(move |event| {
            *start_event_for_hook
                .lock()
                .expect("start_event mutex should not be poisoned") = Some((
                event.model_id.clone(),
                event.messages.len(),
                event.tool_count,
                event.max_steps,
            ));
        })
        .build()
        .expect("agent should build");

    let response = agent
        .run_messages(vec![Message::user_text("ignored")])
        .await
        .expect("agent call should succeed");
    let requests = server
        .received_requests()
        .await
        .expect("wiremock should record requests");
    let request_body: serde_json::Value = requests[0]
        .body_json()
        .expect("request body should be valid json");

    assert_eq!(response.output_text, "prepared-plan-ok");
    assert_eq!(response.steps, 1);
    assert_eq!(request_body["model"], "gpt-4.1-mini");
    assert!(
        (request_body["top_p"]
            .as_f64()
            .expect("top_p should be a number")
            - 0.7)
            .abs()
            < 1e-6
    );
    assert_eq!(request_body["max_tokens"], 42);
    assert_eq!(request_body["stop"], json!(["END"]));
    assert!(
        request_body["messages"]
            .as_array()
            .expect("messages should be an array")
            .iter()
            .any(|message| message["content"] == "from-prepare-call")
    );
    assert_eq!(
        start_event
            .lock()
            .expect("start_event mutex should not be poisoned")
            .clone(),
        Some(("openai/gpt-4.1-mini".to_string(), 2, 0, 2))
    );
}
