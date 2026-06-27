use aquaregia::ErrorCode;
use std::sync::Mutex;
use std::time::Duration;

static ENV_LOCK: Mutex<()> = Mutex::new(());

// ─── OpenAI client builder ──────────────────────────────────────────────

#[test]
fn openai_client_builds_with_all_settings() {
    let client = aquaregia::providers::openai::Client::builder()
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
    match aquaregia::providers::openai::Client::builder()
        .api_key("  ")
        .build()
    {
        Err(err) => assert_eq!(err.code, ErrorCode::AuthFailed),
        Ok(_) => panic!("empty api key should fail"),
    }
}

#[test]
fn openai_client_reports_missing_api_key_env_var() {
    let env_var = "AQUAREGIA_TEST_OPENAI_API_KEY_SHOULD_NOT_EXIST_9DFB2C22";
    match aquaregia::providers::openai::Client::builder()
        .api_key_from_env(env_var)
        .build()
    {
        Err(err) => {
            assert_eq!(err.code, ErrorCode::AuthFailed);
            assert!(err.message.contains(env_var));
        }
        Ok(_) => panic!("missing api key env var should fail"),
    }
}

// ─── Anthropic client builder ───────────────────────────────────────────

#[test]
fn anthropic_client_builds_with_all_settings() {
    let client = aquaregia::providers::anthropic::Client::builder()
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
    match aquaregia::providers::anthropic::Client::builder()
        .api_key("")
        .build()
    {
        Err(err) => assert_eq!(err.code, ErrorCode::AuthFailed),
        Ok(_) => panic!("empty api key should fail"),
    }
}

// ─── Google client builder ──────────────────────────────────────────────

#[test]
fn google_client_builds_with_all_settings() {
    let client = aquaregia::providers::google::Client::builder()
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
    match aquaregia::providers::google::Client::builder()
        .api_key("  ")
        .build()
    {
        Err(err) => assert_eq!(err.code, ErrorCode::AuthFailed),
        Ok(_) => panic!("empty api key should fail"),
    }
}

// ─── OpenAI-compatible client builder ───────────────────────────────────

#[test]
fn openai_compatible_builds_without_api_key() {
    let client = aquaregia::providers::openai_compatible::Client::builder()
        .base_url("https://api.example.com")
        .no_api_key()
        .build()
        .expect("client should build without api key");
    let _ = client;
}

#[test]
fn openai_compatible_builds_with_custom_headers_and_query_params() {
    let client = aquaregia::providers::openai_compatible::Client::builder()
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
    match aquaregia::providers::openai_compatible::Client::builder()
        .base_url("  ")
        .build()
    {
        Err(err) => assert_eq!(err.code, ErrorCode::InvalidRequest),
        Ok(_) => panic!("empty base url should fail"),
    }
}

#[test]
fn openai_compatible_from_env_uses_base_url_and_optional_api_key() {
    let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
    unsafe {
        std::env::set_var("OPENAI_COMPATIBLE_BASE_URL", "https://api.example.com");
        std::env::set_var("OPENAI_COMPATIBLE_API_KEY", "sk-compatible");
    }

    let client = aquaregia::providers::openai_compatible::Client::from_env()
        .expect("client should build from env");
    let _ = client;

    unsafe {
        std::env::remove_var("OPENAI_COMPATIBLE_BASE_URL");
        std::env::remove_var("OPENAI_COMPATIBLE_API_KEY");
    }
}

#[test]
fn openai_compatible_from_env_requires_base_url() {
    let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
    unsafe {
        std::env::remove_var("OPENAI_COMPATIBLE_BASE_URL");
        std::env::remove_var("OPENAI_COMPATIBLE_API_KEY");
    }

    match aquaregia::providers::openai_compatible::Client::from_env() {
        Err(err) => assert_eq!(err.code, ErrorCode::InvalidRequest),
        Ok(_) => panic!("missing base url should fail"),
    }
}

// ─── ClientBuilder default_max_steps: 0 means unlimited, no upper cap ───

#[test]
fn client_accepts_zero_default_max_steps_as_unlimited() {
    aquaregia::providers::openai::Client::builder()
        .api_key("sk-test")
        .default_max_steps(0)
        .build()
        .expect("0 max_steps is unlimited and must build successfully");
}

#[test]
fn client_accepts_large_default_max_steps() {
    aquaregia::providers::openai::Client::builder()
        .api_key("sk-test")
        .default_max_steps(10_000)
        .build()
        .expect("no upper bound on max_steps");
}
