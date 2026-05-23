//! SSE (Server-Sent Events) frame parsing helpers for Aquaregia streaming.
//!
//! This module provides utilities for parsing SSE streams from LLM providers:
//!
//! - `SseFrame`: Parsed SSE frame with optional event name and data payload
//! - `parse_sse_lines`: Parse complete SSE data into frames
//!
//! ## SSE Format
//!
//! Server-Sent Events use a simple text-based format:
//! ```text
//! event: message_type
//! data: {"json": "payload"}
//!
//! data: [DONE]
//!
//! ```
//!
//! Each frame is separated by a blank line (`\n\n`), and fields are prefixed
//! with `event:`, `data:`, `id:`, or `retry:`.

/// Parsed SSE frame.
///
/// This struct represents a single parsed SSE (Server-Sent Events) frame,
/// containing an optional event name and the data payload.
///
/// # Fields
///
/// - `event`: Optional event type name (e.g., "message", "done")
/// - `data`: Data payload, typically JSON for LLM streaming responses
#[derive(Debug, Clone)]
pub struct SseFrame {
    /// Optional SSE event name.
    pub event: Option<String>,
    /// Frame payload from one or more `data:` lines.
    pub data: String,
}

/// Parses one or more SSE frames from a complete string payload.
///
/// This helper normalizes line endings (`\r\n` to `\n`) and ensures a trailing
/// frame separator so callers can pass chunked test fixtures conveniently.
///
/// # Arguments
///
/// * `input` - Raw SSE data string containing one or more frames
///
/// # Returns
///
/// A vector of parsed [`SseFrame`] items. Frames without data are skipped.
///
/// # Example
///
/// ```
/// use aquaregia::stream::parse_sse_lines;
///
/// let input = "event: message\ndata: {\"text\": \"hello\"}\n\ndata: [DONE]\n\n";
/// let frames = parse_sse_lines(input);
///
/// assert_eq!(frames.len(), 2);
/// assert_eq!(frames[0].event, Some("message".to_string()));
/// assert_eq!(frames[1].data, "[DONE]");
/// ```
pub fn parse_sse_lines(input: &str) -> Vec<SseFrame> {
    let mut buffer = input.replace("\r\n", "\n");
    if !buffer.ends_with("\n\n") {
        buffer.push_str("\n\n");
    }
    drain_sse_frames(&mut buffer)
}

/// Internal helper to drain parsed frames from a buffer.
pub(crate) fn drain_sse_frames(buffer: &mut String) -> Vec<SseFrame> {
    let mut out = Vec::new();
    while let Some(idx) = buffer.find("\n\n") {
        let raw = buffer[..idx].to_string();
        buffer.drain(..idx + 2);
        if let Some(frame) = parse_frame(&raw) {
            out.push(frame);
        }
    }
    out
}

/// Parse a single SSE frame from raw text.
fn parse_frame(raw: &str) -> Option<SseFrame> {
    let mut event = None;
    let mut data_lines = Vec::new();

    for line in raw.lines() {
        // Skip comment lines
        if line.starts_with(':') {
            continue;
        }

        // Parse event name
        if let Some(value) = line.strip_prefix("event:") {
            event = Some(value.trim().to_string());
            continue;
        }

        // Parse data line
        if let Some(value) = line.strip_prefix("data:") {
            data_lines.push(value.trim_start().to_string());
        }
    }

    // Frames without data are invalid
    if data_lines.is_empty() {
        return None;
    }

    Some(SseFrame {
        event,
        data: data_lines.join("\n"),
    })
}

#[cfg(test)]
mod tests {
    use super::parse_sse_lines;

    #[test]
    fn parses_multiple_frames() {
        let input = "event: one\ndata: {\"a\":1}\n\ndata: [DONE]\n\n";
        let frames = parse_sse_lines(input);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].event.as_deref(), Some("one"));
        assert_eq!(frames[1].data, "[DONE]");
    }
}
