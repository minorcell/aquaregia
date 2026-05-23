//! Internal SSE (Server-Sent Events) frame parsing helpers.
//!
//! Used by streaming adapters to drain frames from incrementally received
//! response bodies. Not exposed to library users.

#[derive(Debug, Clone)]
pub(crate) struct SseFrame {
    pub data: String,
}

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

fn parse_frame(raw: &str) -> Option<SseFrame> {
    let mut data_lines = Vec::new();

    for line in raw.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(value) = line.strip_prefix("data:") {
            data_lines.push(value.trim_start().to_string());
        }
    }

    if data_lines.is_empty() {
        return None;
    }

    Some(SseFrame {
        data: data_lines.join("\n"),
    })
}

#[cfg(test)]
mod tests {
    use super::drain_sse_frames;

    #[test]
    fn drains_multiple_frames() {
        let mut buffer =
            String::from("event: one\ndata: {\"a\":1}\n\ndata: [DONE]\n\nincomplete: tail");
        let frames = drain_sse_frames(&mut buffer);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].data, "{\"a\":1}");
        assert_eq!(frames[1].data, "[DONE]");
        assert_eq!(buffer, "incomplete: tail");
    }
}
