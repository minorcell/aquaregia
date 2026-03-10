use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_util::StreamExt;
use serde_json::json;

use crate::types::{StreamEvent, TextStream};

/// Converts a [`TextStream`] into an Axum [`Sse`] response stream.
///
/// Stream errors are converted into `event: error` payloads, so the resulting
/// stream item type is `Result<Event, Infallible>`.
pub fn stream_to_sse(
    stream: TextStream,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let mapped = stream.map(|item| {
        let event = match item {
            Ok(StreamEvent::ReasoningStarted {
                block_id,
                provider_metadata,
            }) => Event::default().event("reasoning_start").data(
                json!({
                    "block_id": block_id,
                    "provider_metadata": provider_metadata,
                })
                .to_string(),
            ),
            Ok(StreamEvent::ReasoningDelta {
                block_id,
                text,
                provider_metadata,
            }) => Event::default().event("reasoning_token").data(
                json!({
                    "block_id": block_id,
                    "text": text,
                    "provider_metadata": provider_metadata,
                })
                .to_string(),
            ),
            Ok(StreamEvent::ReasoningDone {
                block_id,
                provider_metadata,
            }) => Event::default().event("reasoning_end").data(
                json!({
                    "block_id": block_id,
                    "provider_metadata": provider_metadata,
                })
                .to_string(),
            ),
            Ok(StreamEvent::TextDelta { text }) => Event::default()
                .event("token")
                .data(json!({ "text": text }).to_string()),
            Ok(StreamEvent::ToolCallReady { call }) => Event::default()
                .event("tool_call")
                .data(json!({ "call": call }).to_string()),
            Ok(StreamEvent::Usage { usage }) => Event::default()
                .event("usage")
                .data(json!({ "usage": usage }).to_string()),
            Ok(StreamEvent::Done) => Event::default().event("done").data("{}"),
            Err(err) => Event::default().event("error").data(
                json!({ "code": format!("{:?}", err.code), "message": err.message }).to_string(),
            ),
        };
        Ok::<Event, Infallible>(event)
    });

    Sse::new(mapped)
}
