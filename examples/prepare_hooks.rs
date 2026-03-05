use aquaregia::{Agent, ContentPart, LlmClient, Message, MessageRole, tool};
use serde_json::json;

const DEFAULT_DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-chat";

/// 场景：动态控制一次调用与每一步执行（AI SDK 风格）。
///
/// 运行：
/// DEEPSEEK_API_KEY=... cargo run --example prepare_hooks
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

    let time_tool = tool("get_time")
        .description("Return current time in mock format")
        .raw_schema(json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }))
        .execute_raw(|_| async move { Ok(json!({ "time": "2026-03-05T12:00:00+08:00" })) });

    let agent = Agent::builder(client, model)
        .instructions("You may call tools, then answer concisely.")
        .tool(time_tool)
        .max_steps(4)
        // 对应 AI SDK prepareCall：在一次调用开始前可动态改调用计划。
        .prepare_call(|plan| {
            if user_mentions_keyword(&plan.messages, "json") {
                plan.temperature = Some(0.0);
                plan.max_output_tokens = Some(400);
            }
        })
        // 对应 AI SDK prepareStep：每一步前可动态改模型/消息/工具等。
        .prepare_step(|event| {
            let mut next = event.to_prepared();

            next.messages.push(Message::system_text(format!(
                "Current step: {}. Use minimal tools.",
                event.step
            )));

            // 示意：第一步允许工具，后续步骤禁用工具，避免无限循环。
            if event.step >= 2 {
                next.tools.clear();
            }

            next
        })
        .on_step_start(|event| {
            println!(
                "[step_start] step={} messages={}",
                event.step,
                event.messages.len()
            );
        })
        .on_step_finish(|step| {
            println!(
                "[step_finish] step={} tool_calls={} finish={:?}",
                step.step,
                step.tool_calls.len(),
                step.finish_reason
            );
        })
        .build()?;

    let out = agent
        .run("Use tool once, then answer in JSON with one short sentence.")
        .await?;

    println!("\n=== answer ===");
    println!("{}", out.output_text);
    println!("steps={}", out.steps);
    println!(
        "usage total={}/{}",
        out.usage_total.input_tokens, out.usage_total.output_tokens
    );

    Ok(())
}

fn user_mentions_keyword(messages: &[Message], keyword: &str) -> bool {
    let keyword = keyword.to_ascii_lowercase();
    messages.iter().any(|message| {
        if message.role() != MessageRole::User {
            return false;
        }
        message.parts().iter().any(|part| match part {
            ContentPart::Text(text) => text.to_ascii_lowercase().contains(&keyword),
            _ => false,
        })
    })
}
