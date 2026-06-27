use aquaregia::{Tool, tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

const DEFAULT_OPENAI_MODEL: &str = "gpt-5.5";

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

fn get_weather() -> Tool {
    tool("get_weather")
        .description("Get weather by city")
        .execute(|args: WeatherArgs| async move {
            json!({ "city": args.city, "temp_c": 23, "condition": "sunny" })
        })
}

/// 场景：20~30 行级别的最小 Agent（带 1 个工具）。
///
/// 运行：
/// OPENAI_API_KEY=... cargo run --example agent_minimal
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string());
    let client = aquaregia::providers::openai::Client::from_env()?;

    let agent = client
        .agent(model)
        .instructions("You are a concise assistant.")
        .tool(get_weather())
        .max_steps(4)
        .build()?;

    let result = agent
        .run("上海天气怎么样？请在调用工具后给出简洁结论。")
        .await?;

    println!("{}", result.output_text);
    Ok(())
}
