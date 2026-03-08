use aquaregia::{Agent, LlmClient, tool};
use serde_json::{Value, json};

const DEFAULT_DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-chat";

#[tool(description = "Get weather by city")]
async fn get_weather(city: String) -> Result<Value, String> {
    Ok(json!({ "city": city, "temp_c": 23, "condition": "sunny" }))
}

/// 场景：20~30 行级别的最小 Agent（带 1 个工具）。
///
/// 运行：
/// DEEPSEEK_API_KEY=... cargo run --example agent_minimal
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

    let agent = Agent::builder(client, model)
        .instructions("You are a concise assistant.")
        .tools([get_weather])
        .max_steps(4)
        .build()?;

    let result = agent
        .run("上海天气怎么样？请在调用工具后给出简洁结论。")
        .await?;

    println!("{}", result.output_text);
    Ok(())
}
