// These tests require the `axum` feature.
#![cfg(feature = "axum")]

use aquaregia::axum_sse::stream_to_sse;
use aquaregia::types::{StreamEvent, ToolCall, Usage};
use axum::response::IntoResponse;
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::json;

async fn collect_sse_body(events: Vec<Result<StreamEvent, aquaregia::Error>>) -> String {
    let text_stream = Box::pin(stream::iter(events));
    let sse_response = stream_to_sse(text_stream);
    let response = sse_response.into_response();
    let (_, body) = response.into_parts();
    let collected = body.collect().await.expect("should collect body");
    let buf = collected.to_bytes();
    String::from_utf8_lossy(&buf).to_string()
}

#[tokio::test]
async fn stream_to_sse_text_delta_and_done() {
    let body = collect_sse_body(vec![
        Ok(StreamEvent::TextDelta {
            text: "hello".into(),
        }),
        Ok(StreamEvent::Done),
    ])
    .await;
    assert!(body.contains("\"text\":\"hello\""));
    assert!(body.contains("event: done"));
    assert!(body.contains("event: token"));
}

#[tokio::test]
async fn stream_to_sse_reasoning_events() {
    let body = collect_sse_body(vec![
        Ok(StreamEvent::ReasoningStarted {
            block_id: "r1".into(),
            provider_metadata: Some(json!({"sig": "abc"})),
        }),
        Ok(StreamEvent::ReasoningDelta {
            block_id: "r1".into(),
            text: "think".into(),
            provider_metadata: None,
        }),
        Ok(StreamEvent::ReasoningDone {
            block_id: "r1".into(),
            provider_metadata: None,
        }),
        Ok(StreamEvent::Done),
    ])
    .await;
    assert!(body.contains("event: reasoning_start"));
    assert!(body.contains("event: reasoning_token"));
    assert!(body.contains("event: reasoning_end"));
}

#[tokio::test]
async fn stream_to_sse_tool_call() {
    let body = collect_sse_body(vec![
        Ok(StreamEvent::ToolCallReady {
            call: ToolCall {
                call_id: "c1".into(),
                tool_name: "weather".into(),
                args_json: json!({"city": "NYC"}),
            },
        }),
        Ok(StreamEvent::Done),
    ])
    .await;
    assert!(body.contains("event: tool_call"));
    assert!(body.contains("weather"));
}

#[tokio::test]
async fn stream_to_sse_usage_event() {
    let body = collect_sse_body(vec![
        Ok(StreamEvent::Usage {
            usage: Usage::from_totals(10, 5, 0, Some(15)),
        }),
        Ok(StreamEvent::Done),
    ])
    .await;
    assert!(body.contains("event: usage"));
}

#[tokio::test]
async fn stream_to_sse_error_event() {
    let body = collect_sse_body(vec![Err(aquaregia::Error::new(
        aquaregia::ErrorCode::RateLimited,
        "rate limited",
    ))])
    .await;
    assert!(body.contains("event: error"));
    assert!(body.contains("RateLimited"));
}

#[tokio::test]
async fn stream_to_sse_empty_stream() {
    let body = collect_sse_body(vec![Ok(StreamEvent::Done)]).await;
    assert!(body.contains("event: done"));
}
