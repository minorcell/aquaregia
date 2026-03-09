use std::sync::Arc;

use aquaregia::{
    Agent, CancellationToken, ErrorCode, GenerateTextRequest, LlmClient, Tool, ToolDescriptor,
    ToolExecError, ToolExecutor, openai,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ─── cancel before request fires ─────────────────────────────────────────────

#[tokio::test]
async fn cancel_before_request_fires() {
    let server = MockServer::start().await;
    // No mock mounted — the request should never reach the server.

    let token = CancellationToken::new();
    token.cancel(); // pre-cancelled

    let client = LlmClient::openai("test-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder(openai("gpt-4o-mini"))
        .user_prompt("hello")
        .cancellation_token(token)
        .build()
        .expect("request should build");

    let err = client
        .generate(req)
        .await
        .expect_err("should fail with Cancelled");

    assert_eq!(err.code, ErrorCode::Cancelled);
}

// ─── cancel before agent step 1 ──────────────────────────────────────────────

#[tokio::test]
async fn cancel_before_agent_step() {
    let server = MockServer::start().await;
    // No mock — the agent step check catches the pre-cancelled token.

    let token = CancellationToken::new();
    token.cancel();

    let client = LlmClient::openai("test-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, openai("gpt-4o-mini"))
        .max_steps(3)
        .build()
        .expect("agent should build");

    let err = agent
        .run_cancellable("hello", token)
        .await
        .expect_err("should fail with Cancelled");

    assert_eq!(err.code, ErrorCode::Cancelled);
}

// ─── cancel between agent steps ──────────────────────────────────────────────

struct CancelOnExecTool(CancellationToken);

#[async_trait]
impl ToolExecutor for CancelOnExecTool {
    async fn execute(&self, _args: Value) -> Result<Value, ToolExecError> {
        self.0.cancel();
        Ok(json!({ "done": true }))
    }
}

#[tokio::test]
async fn cancel_between_agent_steps() {
    let server = MockServer::start().await;

    let step1 = json!({
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "cancel_tool",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
    });

    // Only mount step 1; step 2 should never be reached.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(step1))
        .expect(1)
        .mount(&server)
        .await;

    let token = CancellationToken::new();

    let cancel_tool = Tool {
        descriptor: ToolDescriptor {
            name: "cancel_tool".to_string(),
            description: "Cancels the run".to_string(),
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        executor: Arc::new(CancelOnExecTool(token.clone())),
    };

    let client = LlmClient::openai("test-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let agent = Agent::builder(client, openai("gpt-4o-mini"))
        .tools([cancel_tool])
        .max_steps(5)
        .build()
        .expect("agent should build");

    let err = agent
        .run_cancellable("go", token)
        .await
        .expect_err("should fail with Cancelled");

    assert_eq!(err.code, ErrorCode::Cancelled);
}
