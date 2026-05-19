use aquaregia::{
    AnthropicAdapterSettings, ErrorCode, GoogleAdapterSettings, LlmClient,
    OpenAiAdapterSettings, OpenAiCompatibleAdapterSettings,
};
use std::time::Duration;

// ─── OpenAI client builder ──────────────────────────────────────────────

#[test]
fn openai_client_builds_with_all_settings() {
    let client = LlmClient::openai("sk-test")
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
    match LlmClient::openai("  ").build() {
        Err(err) => assert_eq!(err.code, ErrorCode::AuthFailed),
        Ok(_) => panic!("empty api key should fail"),
    }
}

// ─── Anthropic client builder ───────────────────────────────────────────

#[test]
fn anthropic_client_builds_with_all_settings() {
    let client = LlmClient::anthropic("sk-ant-test")
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
    match LlmClient::anthropic("").build() {
        Err(err) => assert_eq!(err.code, ErrorCode::AuthFailed),
        Ok(_) => panic!("empty api key should fail"),
    }
}

// ─── Google client builder ──────────────────────────────────────────────

#[test]
fn google_client_builds_with_all_settings() {
    let client = LlmClient::google("g-test-key")
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
    match LlmClient::google("  ").build() {
        Err(err) => assert_eq!(err.code, ErrorCode::AuthFailed),
        Ok(_) => panic!("empty api key should fail"),
    }
}

// ─── OpenAI-compatible client builder ───────────────────────────────────

#[test]
fn openai_compatible_builds_without_api_key() {
    let client = LlmClient::openai_compatible("https://api.example.com")
        .no_api_key()
        .build()
        .expect("client should build without api key");
    let _ = client;
}

#[test]
fn openai_compatible_builds_with_custom_headers_and_query_params() {
    let client = LlmClient::openai_compatible("https://api.example.com")
        .api_key("sk-custom")
        .header("X-Custom", "value")
        .query_param("version", "2")
        .chat_completions_path("/custom/chat")
        .think_tag_parsing(true)
        .think_tag_case_insensitive(false)
        .build()
        .expect("client should build");
    let _ = client;
}

#[test]
fn openai_compatible_rejects_empty_base_url() {
    match LlmClient::openai_compatible("  ").build() {
        Err(err) => assert_eq!(err.code, ErrorCode::InvalidRequest),
        Ok(_) => panic!("empty base url should fail"),
    }
}

// ─── OpenAiCompatibleAdapterSettings ────────────────────────────────────

#[test]
fn compatible_settings_new_has_defaults() {
    let s = OpenAiCompatibleAdapterSettings::new("https://api.example.com");
    assert_eq!(s.base_url, "https://api.example.com");
}

// ─── Provider adapter settings constructors ─────────────────────────────

#[test]
fn openai_settings_default_base_url() {
    let settings = OpenAiAdapterSettings::new("sk-key");
    assert!(settings.base_url.contains("api.openai.com"));
    assert_eq!(settings.api_key, "sk-key");
}

#[test]
fn anthropic_settings_default_values() {
    let settings = AnthropicAdapterSettings::new("sk-ant-key");
    assert!(settings.base_url.contains("api.anthropic.com"));
    assert_eq!(settings.api_key, "sk-ant-key");
    assert!(settings.api_version.contains("2023-06-01"));
}

#[test]
fn google_settings_default_base_url() {
    let settings = GoogleAdapterSettings::new("g-key");
    assert!(settings.base_url.contains("generativelanguage.googleapis.com"));
    assert_eq!(settings.api_key, "g-key");
}

// ─── ClientBuilder default_max_steps validation ─────────────────────────

#[test]
fn client_rejects_invalid_default_max_steps() {
    match LlmClient::openai("sk-test").default_max_steps(0).build() {
        Err(err) => assert!(err.message.contains("1..=32")),
        Ok(_) => panic!("0 max_steps should fail"),
    }
}

// ─── ClientBuilder max_steps 33 fails ───────────────────────────────────

#[test]
fn client_rejects_default_max_steps_33() {
    match LlmClient::openai("sk-test").default_max_steps(33).build() {
        Err(err) => assert!(err.message.contains("1..=32")),
        Ok(_) => panic!("33 max_steps should fail"),
    }
}
