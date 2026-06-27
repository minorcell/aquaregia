use aquaregia::tool;
use serde_json::json;

const DEFAULT_OPENAI_MODEL: &str = "gpt-5.5";

/// 场景：工具循环 + 最大步数保护。
///
/// 运行：
/// OPENAI_API_KEY=... cargo run --example tools_max_steps
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string());
    let client = aquaregia::providers::openai::Client::from_env()?;

    // 工具 1：天气查询（mock 数据）
    let weather_tool = tool("get_weather")
        .description("Get current weather for a city")
        .raw_schema(json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }))
        .execute_raw(|args| async move {
            let city = args
                .get("city")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Ok(json!({ "city": city, "temp_c": 23, "condition": "sunny" }))
        });

    // 工具 2：汇率查询（mock 数据）
    let fx_tool = tool("get_fx_rate")
        .description("Get FX rate by currency pair, e.g. USD/CNY")
        .raw_schema(json!({
            "type": "object",
            "properties": {
                "pair": { "type": "string" }
            },
            "required": ["pair"]
        }))
        .execute_raw(|args| async move {
            let pair = args
                .get("pair")
                .and_then(|v| v.as_str())
                .unwrap_or("USD/CNY");
            Ok(json!({ "pair": pair, "rate": 7.18 }))
        });

    let agent = client
        .agent(model)
        .instructions(
            "You can call tools. If a tool is useful, call it first, then answer concisely.",
        )
        .tools([weather_tool, fx_tool])
        // 避免异常循环；生产中建议根据任务复杂度设置。
        .max_steps(4)
        .temperature(0.2)
        .max_output_tokens(300)
        .build()?;

    let response = agent
        .run("What is the weather in Shanghai and what is USD/CNY now? Use tools if needed.")
        .await?;

    println!("=== agent answer ===");
    println!("{}", response.output_text);
    println!("steps: {}", response.steps);
    println!(
        "usage_total: input={} output={} total={}",
        response.usage_total.input_tokens,
        response.usage_total.output_tokens,
        response.usage_total.total_tokens
    );

    Ok(())
}
