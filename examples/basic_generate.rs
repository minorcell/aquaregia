use aquaregia::ChatRequest;

const DEFAULT_OPENAI_MODEL: &str = "gpt-5.5";

/// 场景：一次性非流式调用（最常见的“问答/改写/总结”请求）。
///
/// 运行：
/// OPENAI_API_KEY=... cargo run --example basic_generate
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string());
    let client = aquaregia::providers::openai::Client::from_env()?;

    // 这里用一个贴近日常开发的提示词：让模型产出可执行结论。
    let prompt = r#"
You are a senior Rust reviewer.
Summarize the key ownership/lifetime pitfalls in 5 bullet points,
and give one quick fix tip for each point.
"#;

    let response = client
        .generate(ChatRequest::from_prompt(model, prompt))
        .await?;

    println!("=== one-shot result ===");
    println!("{}", response.output_text);
    println!("\nfinish_reason: {:?}", response.finish_reason);
    println!(
        "usage: input={} output={} total={}",
        response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens
    );

    Ok(())
}
