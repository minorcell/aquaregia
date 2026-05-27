use aquaregia::{ErrorCode, LlmClient};
use std::time::Duration;

// ─── OpenAI client builder ──────────────────────────────────────────────

#[test]
fn openai_client_builds_with_all_settings() {
    let client = LlmClient::openai()
        .api_key("sk-test")
        .base_url("https://custom.openai.com")
        .timeout(Duration::from_secs(120))
        .max_retries(5)
        .default_max_steps(16)
        .user_agent("my-agent/1.0")
        .build()
        .expect("client should build");
    let _ = client;
}

#[test]
fn openai_client_rejects_empty_api_key() {
    match LlmClient::openai().api_key("  ").build() {
        Err(err) => assert_eq!(err.code, ErrorCode::AuthFailed),
        Ok(_) => panic!("empty api key should fail"),
    }
}

// ─── Anthropic client builder ───────────────────────────────────────────

#[test]
fn anthropic_client_builds_with_all_settings() {
    let client = LlmClient::anthropic()
        .api_key("sk-ant-test")
        .base_url("https://custom.anthropic.com")
        .api_version("2024-02-15")
        .timeout(Duration::from_secs(90))
        .max_retries(2)
        .default_max_steps(10)
        .build()
        .expect("client should build");
    let _ = client;
}

#[test]
fn anthropic_client_rejects_empty_api_key() {
    match LlmClient::anthropic().api_key("").build() {
        Err(err) => assert_eq!(err.code, ErrorCode::AuthFailed),
        Ok(_) => panic!("empty api key should fail"),
    }
}

// ─── Google client builder ──────────────────────────────────────────────

#[test]
fn google_client_builds_with_all_settings() {
    let client = LlmClient::google()
        .api_key("g-test-key")
        .base_url("https://custom.google.com")
        .timeout(Duration::from_secs(60))
        .max_retries(1)
        .default_max_steps(5)
        .build()
        .expect("client should build");
    let _ = client;
}

#[test]
fn google_client_rejects_empty_api_key() {
    match LlmClient::google().api_key("  ").build() {
        Err(err) => assert_eq!(err.code, ErrorCode::AuthFailed),
        Ok(_) => panic!("empty api key should fail"),
    }
}

// ─── OpenAI-compatible client builder ───────────────────────────────────

#[test]
fn openai_compatible_builds_without_api_key() {
    let client = LlmClient::openai_compatible()
        .base_url("https://api.example.com")
        .no_api_key()
        .build()
        .expect("client should build without api key");
    let _ = client;
}

#[test]
fn openai_compatible_builds_with_custom_headers_and_query_params() {
    let client = LlmClient::openai_compatible()
        .base_url("https://api.example.com")
        .api_key("sk-custom")
        .header("X-Custom", "value")
        .query_param("version", "2")
        .chat_completions_path("/custom/chat")
        .build()
        .expect("client should build");
    let _ = client;
}

#[test]
fn openai_compatible_rejects_empty_base_url() {
    match LlmClient::openai_compatible().base_url("  ").build() {
        Err(err) => assert_eq!(err.code, ErrorCode::InvalidRequest),
        Ok(_) => panic!("empty base url should fail"),
    }
}

// ─── ClientBuilder default_max_steps: 0 means unlimited, no upper cap ───

#[test]
fn client_accepts_zero_default_max_steps_as_unlimited() {
    LlmClient::openai()
        .api_key("sk-test")
        .default_max_steps(0)
        .build()
        .expect("0 max_steps is unlimited and must build successfully");
}

#[test]
fn client_accepts_large_default_max_steps() {
    LlmClient::openai()
        .api_key("sk-test")
        .default_max_steps(10_000)
        .build()
        .expect("no upper bound on max_steps");
}
