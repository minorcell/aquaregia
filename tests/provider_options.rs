//! Integration tests for provider_options passthrough.

use aquaregia::{GenerateTextRequest, Message};
use serde_json::json;

#[test]
fn request_serializes_provider_options() {
    let req = GenerateTextRequest::builder("claude-sonnet-4-6")
        .message(Message::user_text("test"))
        .provider_options(json!({
            "anthropic": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 5000
                }
            },
            "google": {
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ]
            }
        }))
        .build()
        .unwrap();

    let serialized = serde_json::to_value(&req).unwrap();
    assert!(serialized.get("provider_options").is_some());
    assert!(serialized["provider_options"]["anthropic"]["thinking"]["budget_tokens"] == 5000);
}

#[test]
fn request_without_provider_options_omits_field() {
    let req = GenerateTextRequest::builder("gpt-4")
        .message(Message::user_text("test"))
        .build()
        .unwrap();

    let serialized = serde_json::to_value(&req).unwrap();
    assert!(serialized.get("provider_options").is_none());
}

#[test]
fn request_deserializes_provider_options() {
    let json_str = r#"{
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "User", "parts": [{"Text": "hello"}], "name": null}],
        "temperature": null,
        "top_p": null,
        "max_output_tokens": null,
        "stop_sequences": [],
        "tools": null,
        "output_schema": null,
        "provider_options": {
            "anthropic": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            }
        }
    }"#;

    let req: GenerateTextRequest = serde_json::from_str(json_str).unwrap();
    assert!(req.provider_options().is_some());
    let options = req.provider_options().unwrap();
    assert_eq!(options["anthropic"]["thinking"]["budget_tokens"], 10000);
}
