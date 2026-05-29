use aquaregia::{ContentPart, GenerateTextRequest, LlmClient, Message, MessageRole, TextPart};
use serde_json::json;

/// 场景：用 Anthropic prompt caching 把一段长 system 上下文设为缓存断点。
///
/// 重点：`TextPart::with_provider_options(...)` 把 `cache_control` 直接挂在
/// 内容块上——这种 per-content-block 字段没法通过请求顶层选项表达。
///
/// 第一次调用建 cache（计为 cache_creation_input_tokens），后续命中相同前缀
/// 的调用算作 cache_read_input_tokens，单价更低、延迟更短。
///
/// 运行：
/// ANTHROPIC_API_KEY=... cargo run --example anthropic_prompt_caching
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let client = LlmClient::anthropic().api_key(api_key).build()?;

    // 假装这是一份长 system 上下文（实测要 ≥ 1024 tokens 才会真正进 cache）。
    let long_context = "You are reviewing the following codebase.\n\n".to_string()
        + &"// pretend this is a 4000-token excerpt of source files\n".repeat(200);

    let cached_system = TextPart::new(long_context).with_provider_options(json!({
        "anthropic": { "cache_control": { "type": "ephemeral" } }
    }));

    let system = Message::new(MessageRole::System, vec![ContentPart::Text(cached_system)])?;
    let question = Message::user_text("List the top 3 readability issues.");

    let req = GenerateTextRequest::builder("claude-sonnet-4-6")
        .message(system)
        .message(question)
        .max_output_tokens(512)
        .build()?;

    let resp = client.generate(req).await?;
    println!("=== answer ===");
    println!("{}", resp.output_text);
    println!(
        "\ninput={} cache_write={} cache_read={} output={}",
        resp.usage.input_tokens,
        resp.usage.input_cache_write_tokens,
        resp.usage.input_cache_read_tokens,
        resp.usage.output_tokens,
    );

    Ok(())
}
