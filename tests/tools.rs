use std::sync::{Arc, Mutex};

use aquaregia::{
    Agent, AgentPreparedStep, AiErrorCode, LlmClient, Message, Tool, ToolDescriptor,
    ToolExecError, ToolExecutor, openai_model,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct DummyWeatherTool;

#[async_trait]
impl ToolExecutor for DummyWeatherTool {
    async fn execute(&self, args: Value) -> Result<Value, ToolExecError> {
        let city = args
            .get("city")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        Ok(json!({ "city": city, "temp_c": 22 }))
    }
}

fn make_weather_tool() -> Tool {
    Tool {
        descriptor: ToolDescriptor {
            name: "get_weather".to_string(),
            description: "Get weather by city".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        },
        executor: Arc::new(DummyWeatherTool),
    }
}

#[tokio::test]
async fn run_tools_two_step_success() {
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
            "message": { "content": "Shanghai is about 22C." },
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

    let agent = Agent::builder(client, openai_model("gpt-4o-mini"))
        .tools([make_weather_tool()])
        .max_steps(3)
        .temperature(0.2)
        .max_output_tokens(256)
        .build()
        .expect("agent should build");

    let response = agent
        .run("What's the weather in Shanghai?")
        .await
        .expect("agent should succeed");

    assert_eq!(response.output_text, "Shanghai is about 22C.");
    assert_eq!(response.steps, 2);
    assert_eq!(response.usage_total.total_tokens, 27);
    assert!(response.transcript.len() >= 4);
}

#[tokio::test]
async fn run_tools_unknown_tool_fails() {
    let server = MockServer::start().await;
    let step1 = json!({
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "missing_tool",
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

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(step1))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, openai_model("gpt-4o-mini"))
        .tools([make_weather_tool()])
        .max_steps(3)
        .build()
        .expect("agent should build");

    let err = agent
        .run("What's the weather in Shanghai?")
        .await
        .expect_err("agent should fail for unknown tool");

    assert_eq!(err.code, AiErrorCode::UnknownTool);
}

#[tokio::test]
async fn run_tools_lifecycle_hooks_fire() {
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
            "message": { "content": "Shanghai is about 22C." },
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

    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    fn push_event(events: &Arc<Mutex<Vec<String>>>, label: String) {
        events
            .lock()
            .expect("events mutex should not be poisoned")
            .push(label);
    }

    let agent = {
        let e = Arc::clone(&events);
        let agent = Agent::builder(client, openai_model("gpt-4o-mini"))
            .tools([make_weather_tool()])
            .max_steps(3)
            .temperature(0.2)
            .max_output_tokens(256)
            .on_start({
                let e = Arc::clone(&e);
                move |_| push_event(&e, "start".to_string())
            })
            .on_step_start({
                let e = Arc::clone(&e);
                move |event| push_event(&e, format!("step_start:{}", event.step))
            })
            .on_tool_call_start({
                let e = Arc::clone(&e);
                move |event| push_event(&e, format!("tool_call_start:{}", event.tool_call.tool_name))
            })
            .on_tool_call_finish({
                let e = Arc::clone(&e);
                move |event| push_event(
                    &e,
                    format!("tool_call_finish:{}:{}", event.tool_call.tool_name, event.tool_result.is_error),
                )
            })
            .on_step_finish({
                let e = Arc::clone(&e);
                move |event| push_event(&e, format!("step_finish:{}", event.step))
            })
            .on_finish({
                let e = Arc::clone(&e);
                move |event| push_event(&e, format!("finish:{}", event.step_count))
            })
            .build()
            .expect("agent should build");
        drop(e);
        agent
    };

    let response = agent
        .run("What's the weather in Shanghai?")
        .await
        .expect("agent should succeed");

    assert_eq!(response.steps, 2);
    let observed = events
        .lock()
        .expect("events mutex should not be poisoned")
        .clone();
    assert_eq!(
        observed,
        vec![
            "start",
            "step_start:1",
            "tool_call_start:get_weather",
            "tool_call_finish:get_weather:false",
            "step_finish:1",
            "step_start:2",
            "step_finish:2",
            "finish:2"
        ]
    );
}

#[tokio::test]
async fn run_tools_prepare_step_can_override_step_input() {
    let server = MockServer::start().await;

    let body = json!({
        "choices": [{
            "message": { "content": "prepared-step-ok" },
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
        .and(body_string_contains("from-prepare-step"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, openai_model("gpt-4o-mini"))
        .max_steps(1)
        .prepare_step(|event| AgentPreparedStep {
            model: event.model.clone(),
            messages: vec![
                Message::system_text("from-prepare-step"),
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
        .expect("agent should succeed");

    assert_eq!(response.output_text, "prepared-step-ok");
    assert_eq!(response.steps, 1);
}
