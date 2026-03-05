use aquaregia::LlmClient;

const DEFAULT_DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-chat";

/// 场景：一次性非流式调用（最常见的“问答/改写/总结”请求）。
///
/// 运行：
/// DEEPSEEK_API_KEY=... cargo run --example basic_generate
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

    // 这里用一个贴近日常开发的提示词：让模型产出可执行结论。
    let prompt = r#"
You are a senior Rust reviewer.
Summarize the key ownership/lifetime pitfalls in 5 bullet points,
and give one quick fix tip for each point.
"#;

    let response = client.generate(model, prompt).await?;

    println!("=== one-shot result ===");
    println!("{}", response.output_text);
    println!("\nfinish_reason: {:?}", response.finish_reason);
    println!(
        "usage: input={} output={} total={}",
        response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens
    );

    Ok(())
}
