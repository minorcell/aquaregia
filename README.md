# Aquaregia

> Note: Aquaregia is in rapid iteration before v0.2.0, and the API may have breaking changes. use with caution.

Aquaregia is a provider-agnostic Rust toolkit for building AI applications and tool-using agents.

It provides a unified API across OpenAI, Anthropic, Google, and OpenAI-compatible services, with first-class support for streaming output and multi-step tool execution.

Read the [API docs](https://docs.rs/aquaregia), browse [examples](./examples/README.md), or switch to [中文文档](./README_CN.md).

## Installation

You need Rust and a Tokio async runtime in your project.

```bash
cargo add aquaregia
```

Default features enable `openai` and `anthropic`. You can also validate minimal/provider-specific builds:

```bash
cargo check --no-default-features
cargo check --no-default-features --features openai
cargo check --no-default-features --features anthropic
```

## Unified Provider Architecture

One `LlmClient` binds to one provider configuration.
Each call passes a `GenerateTextRequest` that carries both the model and the messages.

| Provider          | Register API                                                                                                                                            | Model argument              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| OpenAI            | `LlmClient::openai(api_key)` (+ optional `.base_url(...)`)                                                                                              | `"gpt-4o"`                  |
| Anthropic         | `LlmClient::anthropic(api_key)` (+ optional `.base_url(...)`, `.api_version(...)`)                                                                      | `"claude-sonnet-4-5"`       |
| Google            | `LlmClient::google(api_key)` (+ optional `.base_url(...)`)                                                                                              | `"gemini-2.0-flash"`        |
| OpenAI-compatible | `LlmClient::openai_compatible(base_url).api_key(...)`                                                                                                   | `"deepseek-chat"`           |

## Usage

### Generating Text

```rust
use aquaregia::{GenerateTextRequest, LlmClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;

    let client = LlmClient::openai_compatible("https://api.deepseek.com")
        .api_key(api_key)
        .build()?;

    let out = client
        .generate(GenerateTextRequest::from_user_prompt(
            "deepseek-chat",
            "Explain Rust ownership in 3 bullet points.",
        ))
        .await?;

    println!("{}", out.output_text);
    Ok(())
}
```

### Streaming Text

```rust
use aquaregia::{GenerateTextRequest, LlmClient, StreamEvent};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = LlmClient::openai_compatible("https://api.deepseek.com")
        .api_key(api_key)
        .build()?;

    let mut stream = client
        .stream(GenerateTextRequest::from_user_prompt(
            "deepseek-chat",
            "Write a short release note.",
        ))
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text } => print!("{text}"),
            StreamEvent::Done => break,
            _ => {}
        }
    }
    Ok(())
}
```

### Error Handling

```rust
use aquaregia::{ErrorCode, GenerateTextRequest, LlmClient};

match client
    .generate(GenerateTextRequest::from_user_prompt("deepseek-chat", "hello"))
    .await
{
    Ok(out) => println!("{}", out.output_text),
    Err(err) => match err.code {
        ErrorCode::RateLimited => eprintln!("rate limited; retry later"),
        ErrorCode::AuthFailed => eprintln!("check API key"),
        _ => eprintln!("request failed: {}", err),
    },
}
```

### Agent + Tool Loop

```rust
use aquaregia::{Agent, LlmClient, tool};
use serde_json::{Value, json};

#[tool(description = "Get weather by city")]
async fn get_weather(city: String) -> Result<Value, String> {
    Ok(json!({ "city": city, "temp_c": 23, "condition": "sunny" }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = LlmClient::openai_compatible("https://api.deepseek.com")
        .api_key(api_key)
        .build()?;

    let agent = Agent::builder(client, "deepseek-chat")
        .instructions("You can call tools before answering.")
        .tools([get_weather])
        .max_steps(4)
        .build()?;

    let out = agent.run("What is the weather in Shanghai?").await?;
    println!("{}", out.output_text);
    Ok(())
}
```

### Dynamic Planning (`prepare_call` / `prepare_step`)

```rust
use aquaregia::{Agent, LlmClient};

let agent = Agent::builder(client, "deepseek-chat")
    .max_steps(4)
    .prepare_call(|plan| {
        plan.temperature = Some(0.2);
    })
    .prepare_step(|event| {
        let mut next = event.to_prepared();
        if event.step >= 2 {
            next.tools.clear();
        }
        next
    })
    .build()?;
```

### OpenAI-Compatible Advanced Settings

```rust
use aquaregia::LlmClient;

let client = LlmClient::openai_compatible("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .header("x-trace-source", "aquaregia")
    .query_param("source", "sdk")
    .chat_completions_path("/v1/chat/completions")
    .build()?;
```

## Examples

| Example                             | Command                                        | Focus                                      |
| ----------------------------------- | ---------------------------------------------- | ------------------------------------------ |
| Basic generation                    | `cargo run --example basic_generate`           | one-shot `generate`                        |
| Basic stream                        | `cargo run --example basic_stream`             | `stream` + `StreamEvent` handling          |
| Minimal agent                       | `cargo run --example agent_minimal`            | `Agent::builder` + one tool                |
| Tool loop guardrails                | `cargo run --example tools_max_steps`          | multi-step tools + `max_steps`             |
| Dynamic hooks                       | `cargo run --example prepare_hooks`            | `prepare_call` / `prepare_step`            |
| Compatible custom path/query/header | `cargo run --example openai_compatible_custom` | custom headers / query params / path       |
| Mini terminal code agent            | `cargo run --example mini_claude_code`         | `Agent::builder` + `#[tool]` + local tools |

## Development

```bash
cargo fmt
cargo test
cargo check --examples
cargo check --no-default-features
cargo check --no-default-features --features openai
cargo check --no-default-features --features anthropic
cargo check --features axum
```

For `ai-sdk/` workspace only:

```bash
cd ai-sdk && pnpm build && pnpm lint && pnpm type-check
```

## Contributing

Contributions are welcome. For behavior changes, include integration tests (happy path + error mapping + tool/stream flows where relevant).

- [Contributing Guide](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Security Policy](./SECURITY.md)
- [MIT License](./LICENSE)
