use aquaregia::{
    GenerateTextRequest, LlmClient, Message, StreamEvent, anthropic, openai, openai_compatible,
};
use futures_util::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn anthropic_stream_emits_text_usage_done() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::anthropic("test-anthropic-key")
        .base_url(server.uri())
        .api_version("2023-06-01")
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder(anthropic("claude-3-5-haiku-latest"))
        .message(Message::user_text("hello"))
        .temperature(0.2)
        .max_output_tokens(32)
        .build()
        .expect("request should build");

    let mut stream = client
        .stream(req)
        .await
        .expect("stream_text should succeed");

    let mut saw_text = false;
    let mut saw_usage = false;
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::TextDelta { text } => {
                if text == "Hello" {
                    saw_text = true;
                }
            }
            StreamEvent::Usage { usage } => {
                if usage.input_tokens == 3 && usage.output_tokens == 1 {
                    saw_usage = true;
                }
            }
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert!(saw_text);
    assert!(saw_usage);
    assert!(saw_done);
}

#[tokio::test]
async fn openai_stream_accepts_eof_without_done_marker() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"id\":\"chunk-1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}],\"usage\":null}\n\n",
        "data: {\"id\":\"chunk-2\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai("test-openai-key")
        .base_url(server.uri())
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder(openai("gpt-4o-mini"))
        .message(Message::user_text("hello"))
        .temperature(0.2)
        .max_output_tokens(32)
        .build()
        .expect("request should build");

    let mut stream = client
        .stream(req)
        .await
        .expect("stream_text should succeed");

    let mut saw_text = false;
    let mut saw_usage = false;
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::TextDelta { text } => {
                if text == "Hello" {
                    saw_text = true;
                }
            }
            StreamEvent::Usage { usage } => {
                if usage.total_tokens == 3 {
                    saw_usage = true;
                }
            }
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert!(saw_text);
    assert!(saw_usage);
    assert!(saw_done);
}

#[tokio::test]
async fn openai_compatible_stream_accepts_eof_without_done_or_finish_reason() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"id\":\"chunk-1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}],\"usage\":null}\n\n",
        "data: {\"id\":\"chunk-2\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body.to_string()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible(server.uri())
        .api_key("test-compatible-key")
        .build()
        .expect("client should build");

    let req = GenerateTextRequest::builder(openai_compatible("demo-model"))
        .message(Message::user_text("hello"))
        .temperature(0.2)
        .max_output_tokens(32)
        .build()
        .expect("request should build");

    let mut stream = client
        .stream(req)
        .await
        .expect("stream_text should succeed");

    let mut saw_text = false;
    let mut saw_usage = false;
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::TextDelta { text } => {
                if text == "Hi" {
                    saw_text = true;
                }
            }
            StreamEvent::Usage { usage } => {
                if usage.total_tokens == 3 {
                    saw_usage = true;
                }
            }
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert!(saw_text);
    assert!(saw_usage);
    assert!(saw_done);
}
