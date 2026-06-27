use aquaregia::{ChatRequest, StreamEvent};
use futures_util::StreamExt;

const DEFAULT_OPENAI_MODEL: &str = "gpt-5.5";

/// 场景：流式输出，适合 CLI/Chat UI 一边生成一边展示。
///
/// 运行：
/// OPENAI_API_KEY=... cargo run --example basic_stream
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string());
    let client = aquaregia::providers::openai::Client::from_env()?;

    let mut stream = client
        .stream(ChatRequest::from_prompt(
            model,
            "Write a short release note for a Rust SDK refactor (Chinese).",
        ))
        .await?;

    let mut full_text = String::new();

    println!("=== streaming output ===");
    while let Some(chunk) = stream.next().await {
        match chunk? {
            StreamEvent::TextDelta { text } => {
                full_text.push_str(&text);
                print!("{text}");
            }
            StreamEvent::Done { .. } => break,
            _ => {}
        }
    }
    println!("\n=== stream done ===");

    println!(
        "\n--- final text length: {} chars ---",
        full_text.chars().count()
    );
    Ok(())
}
