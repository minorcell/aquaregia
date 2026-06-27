use aquaregia::ChatRequest;

const DEFAULT_OPENAI_COMPATIBLE_BASE_URL: &str = "https://api.example.com";
const DEFAULT_OPENAI_COMPATIBLE_MODEL: &str = "gpt-5.5";

/// 场景：接入 OpenAI-Compatible 服务，并配置自定义 headers/query/path。
///
/// 运行示例：
/// OPENAI_COMPATIBLE_API_KEY=... OPENAI_COMPATIBLE_BASE_URL=... cargo run --example openai_compatible_custom
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = std::env::var("OPENAI_COMPATIBLE_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_OPENAI_COMPATIBLE_BASE_URL.to_string());
    let model = std::env::var("OPENAI_COMPATIBLE_MODEL")
        .unwrap_or_else(|_| DEFAULT_OPENAI_COMPATIBLE_MODEL.to_string());

    let client = aquaregia::providers::openai_compatible::Client::builder()
        .base_url(base_url)
        .api_key(std::env::var("OPENAI_COMPATIBLE_API_KEY")?)
        // 可选：部分兼容服务需要额外 header 或 query 参数。
        .header("x-trace-source", "aquaregia-example")
        .query_param("source", "aquaregia")
        // 默认是 /v1/chat/completions，这里保持默认也可。
        .chat_completions_path("/v1/chat/completions")
        .build()?;

    let response = client
        .generate(ChatRequest::from_prompt(model, "Say hello in Chinese."))
        .await?;

    println!("{}", response.output_text);
    Ok(())
}
