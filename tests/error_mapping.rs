use aquaregia::{AiErrorCode, GenerateTextRequest, LlmClient, Message, anthropic};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn anthropic_request() -> GenerateTextRequest<aquaregia::Anthropic> {
    GenerateTextRequest::builder(anthropic("claude-3-5-haiku-latest"))
        .message(Message::user_text("hi"))
        .temperature(0.2)
        .max_output_tokens(32)
        .build()
        .expect("request should build")
}

#[tokio::test]
async fn anthropic_429_maps_to_rate_limited() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
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
        .generate_request(anthropic_request())
        .await
        .expect_err("request should fail");

    assert_eq!(err.code, AiErrorCode::RateLimited);
    assert_eq!(err.status, Some(429));
    assert!(err.retryable);
}
