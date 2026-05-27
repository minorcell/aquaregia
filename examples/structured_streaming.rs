use aquaregia::{GenerateTextRequest, LlmClient, Message, StreamObjectEvent};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

const DEFAULT_DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-v4-pro";

/// 场景：流式获取结构化输出，每个字段到位即可读，无需等整段 JSON 拼完。
///
/// `stream_object` 内部带 partial JSON 修复：截断的字符串/数组/嵌套对象
/// 会被补齐成合法 JSON，再尝试反序列化为 `T`。未到达的字段保持 `Default`，
/// 所以 `T` 推荐整体加 `#[serde(default)]`。
///
/// 运行：
/// DEEPSEEK_API_KEY=... cargo run --example structured_streaming
#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default)]
struct ProductBrief {
    name: String,
    tagline: String,
    key_features: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let base_url = std::env::var("DEEPSEEK_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_DEEPSEEK_BASE_URL.to_string());
    let model =
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| DEFAULT_DEEPSEEK_MODEL.to_string());

    let client = LlmClient::openai_compatible()
        .base_url(base_url)
        .api_key(api_key)
        .build()?;

    let req = GenerateTextRequest::builder(model)
        .message(Message::user_text(
            "Generate a short product brief for a Rust LLM SDK. \
             Include a punchy tagline and 4 key features.",
        ))
        .temperature(0.3)
        .build()?;

    let mut stream = client.stream_object::<ProductBrief>(req).await?;

    let mut partial_count = 0u32;
    println!("=== progressive partials ===");
    while let Some(event) = stream.next().await {
        match event? {
            StreamObjectEvent::Partial { partial } => {
                partial_count += 1;
                println!(
                    "[partial #{}] name={:?} features={}",
                    partial_count,
                    partial.name,
                    partial.key_features.len()
                );
            }
            StreamObjectEvent::Object { object } => {
                println!("\n=== final object ===");
                println!("name:     {}", object.name);
                println!("tagline:  {}", object.tagline);
                println!("features: {}", object.key_features.len());
                for (i, f) in object.key_features.iter().enumerate() {
                    println!("  {}. {}", i + 1, f);
                }
            }
        }
    }
    println!(
        "\n--- emitted {} partial(s) before final ---",
        partial_count
    );
    Ok(())
}
