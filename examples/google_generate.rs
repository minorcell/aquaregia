use aquaregia::LlmClient;

const DEFAULT_DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-chat";

/// 场景：一次性调用 DeepSeek（保留文件名用于兼容旧示例索引）。
///
/// 运行：
/// DEEPSEEK_API_KEY=... cargo run --example google_generate
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let base_url = std::env::var("DEEPSEEK_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_DEEPSEEK_BASE_URL.to_string());
    let model =
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| DEFAULT_DEEPSEEK_MODEL.to_string());

    let client = LlmClient::openai_compatible(base_url)
        .api_key(api_key)
        .build()?;

    let response = client
        .generate(
            model,
            "Summarize what ownership and borrowing means in Rust in 3 bullet points.",
        )
        .await?;

    println!("{}", response.output_text);
    Ok(())
}
