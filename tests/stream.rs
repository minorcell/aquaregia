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
        "data: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1,\"cache_creation_input_tokens\":1,\"cache_read_input_tokens\":2,\"iterations\":[{\"type\":\"compaction\",\"input_tokens\":4,\"output_tokens\":2},{\"type\":\"message\",\"input_tokens\":3,\"output_tokens\":1}]}}\n\n",
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
                if usage.input_tokens == 10
                    && usage.input_no_cache_tokens == 7
                    && usage.input_cache_read_tokens == 2
                    && usage.input_cache_write_tokens == 1
                    && usage.output_tokens == 3
                    && usage.output_text_tokens == 3
                    && usage.total_tokens == 13
                {
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
        "data: {\"id\":\"chunk-2\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":1},\"completion_tokens\":3,\"completion_tokens_details\":{\"reasoning_tokens\":1},\"total_tokens\":5}}\n\n"
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
                if usage.total_tokens == 5
                    && usage.input_cache_read_tokens == 1
                    && usage.input_no_cache_tokens == 1
                    && usage.output_tokens == 3
                    && usage.output_text_tokens == 2
                    && usage.reasoning_tokens == 1
                {
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
        "data: {\"id\":\"chunk-2\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":1},\"completion_tokens\":3,\"completion_tokens_details\":{\"reasoning_tokens\":1},\"total_tokens\":5}}\n\n"
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
                if usage.total_tokens == 5
                    && usage.input_cache_read_tokens == 1
                    && usage.input_no_cache_tokens == 1
                    && usage.output_tokens == 3
                    && usage.output_text_tokens == 2
                    && usage.reasoning_tokens == 1
                {
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
async fn openai_compatible_stream_splits_think_tags_when_enabled() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"id\":\"chunk-1\",\"choices\":[{\"delta\":{\"content\":\"<thinking>alpha\"},\"finish_reason\":null}],\"usage\":null}\n\n",
        "data: {\"id\":\"chunk-2\",\"choices\":[{\"delta\":{\"content\":\" beta</thinking>Hi\"},\"finish_reason\":null}],\"usage\":null}\n\n",
        "data: [DONE]\n\n"
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
        .think_tag_parsing(true)
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

    let mut output_text = String::new();
    let mut reasoning_text = String::new();
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::TextDelta { text } => output_text.push_str(&text),
            StreamEvent::ReasoningDelta { text, .. } => reasoning_text.push_str(&text),
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert_eq!(output_text, "Hi");
    assert_eq!(reasoning_text, "alpha beta");
    assert!(saw_done);
}

#[tokio::test]
async fn openai_compatible_stream_prefers_standard_reasoning_field() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"id\":\"chunk-1\",\"choices\":[{\"delta\":{\"reasoning_content\":\"plan\",\"content\":\"<thinking>private</thinking>answer\"},\"finish_reason\":null}],\"usage\":null}\n\n",
        "data: [DONE]\n\n"
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
        .think_tag_parsing(true)
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

    let mut output_text = String::new();
    let mut reasoning_text = String::new();
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        let event = event.expect("stream event should parse");
        match event {
            StreamEvent::TextDelta { text } => output_text.push_str(&text),
            StreamEvent::ReasoningDelta { text, .. } => reasoning_text.push_str(&text),
            StreamEvent::Done => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }

    assert_eq!(reasoning_text, "plan");
    assert_eq!(output_text, "<thinking>private</thinking>answer");
    assert!(saw_done);
}
