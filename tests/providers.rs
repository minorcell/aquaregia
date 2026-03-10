use aquaregia::{GenerateTextRequest, LlmClient, anthropic, google, openai_compatible};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn google_generate_text_success() {
    let server = MockServer::start().await;

    let body = json!({
        "candidates": [
            {
                "content": {
                    "parts": [{ "text": "Hello from Gemini" }]
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 8,
            "candidatesTokenCount": 4,
            "cachedContentTokenCount": 2,
            "thoughtsTokenCount": 1,
            "totalTokenCount": 12
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
            google("gemini-2.0-flash"),
            "hello",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.output_text, "Hello from Gemini");
    assert_eq!(response.usage.total_tokens, 12);
    assert_eq!(response.usage.input_tokens, 8);
    assert_eq!(response.usage.input_cache_read_tokens, 2);
    assert_eq!(response.usage.input_no_cache_tokens, 6);
    assert_eq!(response.usage.output_tokens, 5);
    assert_eq!(response.usage.output_text_tokens, 4);
    assert_eq!(response.usage.reasoning_tokens, 1);
    assert!(response.usage.raw_usage.is_some());
}

#[tokio::test]
async fn openai_compatible_generate_text_success() {
    let server = MockServer::start().await;

    let body = json!({
        "choices": [
            {
                "message": { "content": "Hello from compatible endpoint" },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 3,
            "prompt_tokens_details": {
                "cached_tokens": 1
            },
            "completion_tokens_details": {
                "reasoning_tokens": 2
            },
            "total_tokens": 9
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-compatible-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-compatible-key")
        .build()
        .expect("client should build");

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            openai_compatible("deepseek-chat"),
            "hello",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.output_text, "Hello from compatible endpoint");
    assert_eq!(response.usage.total_tokens, 9);
    assert_eq!(response.usage.input_tokens, 6);
    assert_eq!(response.usage.input_cache_read_tokens, 1);
    assert_eq!(response.usage.input_no_cache_tokens, 5);
    assert_eq!(response.usage.output_tokens, 3);
    assert_eq!(response.usage.output_text_tokens, 1);
    assert_eq!(response.usage.reasoning_tokens, 2);
    assert!(response.usage.raw_usage.is_some());
}

#[tokio::test]
async fn anthropic_generate_text_usage_parses_cache_and_iterations() {
    let server = MockServer::start().await;

    let body = json!({
        "content": [
            {
                "type": "text",
                "text": "Hello from Anthropic"
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "cache_creation_input_tokens": 1,
            "cache_read_input_tokens": 2,
            "iterations": [
                { "type": "compaction", "input_tokens": 4, "output_tokens": 1 },
                { "type": "message", "input_tokens": 3, "output_tokens": 2 }
            ]
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
            anthropic("claude-3-5-haiku-latest"),
            "hello",
        ))
        .await
        .expect("request should succeed");

    assert_eq!(response.output_text, "Hello from Anthropic");
    assert_eq!(response.usage.input_tokens, 10);
    assert_eq!(response.usage.input_no_cache_tokens, 7);
    assert_eq!(response.usage.input_cache_read_tokens, 2);
    assert_eq!(response.usage.input_cache_write_tokens, 1);
    assert_eq!(response.usage.output_tokens, 3);
    assert_eq!(response.usage.output_text_tokens, 3);
    assert_eq!(response.usage.reasoning_tokens, 0);
    assert_eq!(response.usage.total_tokens, 13);
    assert!(response.usage.raw_usage.is_some());
}
