use aquaregia::{LlmClient, OpenAiCompatibleAdapterSettings};

const DEFAULT_DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-chat";

/// 场景：演示 DeepSeek(OpenAI-compatible) 的两种接入方式。
///
/// 运行：
/// DEEPSEEK_API_KEY=... cargo run --example provider_selection_demo
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let base_url = std::env::var("DEEPSEEK_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_DEEPSEEK_BASE_URL.to_string());
    let model =
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| DEFAULT_DEEPSEEK_MODEL.to_string());

    // 1) 直接方式：openai_compatible(base_url).api_key(...)
    let simple_client = LlmClient::openai_compatible(base_url.clone())
        .api_key(api_key.clone())
        .build()?;

    let response = simple_client
        .generate(model.clone(), "Reply with exactly: deepseek-ok")
        .await?;
    println!("simple client result: {}", response.output_text);

    // 2) 高级方式：openai_compatible_with_settings(...)
    let settings = OpenAiCompatibleAdapterSettings::new(base_url)
        .api_key(api_key)
        .header("x-demo", "provider-selection");

    let settings_client = LlmClient::openai_compatible_with_settings(settings).build()?;

    let second = settings_client
        .generate(model, "Reply with exactly: deepseek-settings-ok")
        .await?;
    println!("settings client result: {}", second.output_text);
    Ok(())
}
