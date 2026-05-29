//! Integration tests for FilePart media_type dispatch across adapters.

use aquaregia::{
    ContentPart, ErrorCode, FilePart, GenerateTextRequest, LlmClient, MediaData, Message,
    MessageRole, TextPart,
};
use serde_json::json;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const PDF_B64_FIXTURE: &str = "JVBERi0xLjQKJ...";
const PNG_B64_FIXTURE: &str = "iVBORw0KGgo=";

fn pdf_message() -> Message {
    let pdf = FilePart::new(MediaData::Base64(PDF_B64_FIXTURE.into()), "application/pdf")
        .with_filename("doc.pdf");
    Message::new(
        MessageRole::User,
        vec![
            ContentPart::Text(TextPart::new("Summarise this document.")),
            ContentPart::File(pdf),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn anthropic_pdf_uses_document_block() {
    let server = MockServer::start().await;

    let body = json!({
        "content": [{ "type": "text", "text": "ok" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_string_contains("\"type\":\"document\""))
        .and(body_string_contains("\"media_type\":\"application/pdf\""))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::anthropic()
        .api_key("test")
        .base_url(server.uri())
        .build()
        .unwrap();

    let req = GenerateTextRequest::builder("claude-sonnet-4-6")
        .message(pdf_message())
        .build()
        .unwrap();
    client.generate(req).await.expect("generate should succeed");
}

#[tokio::test]
async fn anthropic_image_still_uses_image_block() {
    // Regression guard: image media types must keep emitting the legacy
    // image block, not the new document block.
    let server = MockServer::start().await;

    let body = json!({
        "content": [{ "type": "text", "text": "ok" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_string_contains("\"type\":\"image\""))
        .and(body_string_contains("\"media_type\":\"image/png\""))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::anthropic()
        .api_key("test")
        .base_url(server.uri())
        .build()
        .unwrap();

    let img = FilePart::new(MediaData::Base64(PNG_B64_FIXTURE.into()), "image/png");
    let message = Message::new(
        MessageRole::User,
        vec![
            ContentPart::Text(TextPart::new("What is this?")),
            ContentPart::File(img),
        ],
    )
    .unwrap();
    let req = GenerateTextRequest::builder("claude-sonnet-4-6")
        .message(message)
        .build()
        .unwrap();
    client.generate(req).await.expect("generate should succeed");
}

#[tokio::test]
async fn openai_pdf_uses_input_file_block_with_filename() {
    let server = MockServer::start().await;

    let body = json!({
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "ok" }]
        }],
        "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_string_contains("\"type\":\"input_file\""))
        .and(body_string_contains("\"filename\":\"doc.pdf\""))
        .and(body_string_contains("data:application/pdf;base64,"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = LlmClient::openai()
        .api_key("test")
        .base_url(server.uri())
        .build()
        .unwrap();

    let req = GenerateTextRequest::builder("gpt-5.5")
        .message(pdf_message())
        .build()
        .unwrap();
    client.generate(req).await.expect("generate should succeed");
}

#[tokio::test]
async fn openai_compatible_pdf_is_rejected_locally() {
    // Boundary: openai-compatible chat completions has no standard PDF
    // representation; the adapter must reject the request locally rather
    // than send something the upstream cannot interpret. No HTTP call is
    // expected.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    let client = LlmClient::openai_compatible()
        .base_url(server.uri())
        .api_key("test")
        .build()
        .unwrap();

    let req = GenerateTextRequest::builder("gpt-4o-mini")
        .message(pdf_message())
        .build()
        .unwrap();
    let err = client
        .generate(req)
        .await
        .expect_err("PDF on openai-compatible must error");
    assert_eq!(err.code, ErrorCode::InvalidRequest);
    assert!(
        err.message.contains("application/pdf"),
        "error should mention the unsupported media type: {}",
        err.message
    );
}
