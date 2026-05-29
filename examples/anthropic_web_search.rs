use aquaregia::{GenerateTextRequest, LlmClient};
use serde_json::json;

/// 场景：调用 Anthropic 的 native `web_search` 工具。
///
/// 这是一类 provider 在服务端执行的 native 工具——你不提供 executor，
/// Aquaregia 也不需要为它造一个新类型。直接通过 `provider_options` 把工具
/// 描述塞进请求体的 `tools` 数组即可。
///
/// 注意：顶层 merge 是覆盖语义。如果你同时调了 `.tools([...])`，那 array
/// 会被这里的 `provider_options.anthropic.tools` 覆盖；想混着用就一并放在
/// `provider_options.anthropic.tools` 里、不要再调 `.tools(...)`。
///
/// 运行：
/// ANTHROPIC_API_KEY=... cargo run --example anthropic_web_search
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let client = LlmClient::anthropic().api_key(api_key).build()?;

    let req = GenerateTextRequest::builder("claude-sonnet-4-6")
        .user_prompt("What did Rust 1.85 ship? Cite sources.")
        .provider_options(json!({
            "anthropic": {
                "tools": [{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 3
                }]
            }
        }))
        .max_output_tokens(1024)
        .build()?;

    let resp = client.generate(req).await?;
    println!("=== answer ===");
    println!("{}", resp.output_text);
    println!(
        "\ninput={} output={} total={}",
        resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.total_tokens
    );

    Ok(())
}
