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
Each call can pass a model id string directly (for example, `"deepseek-chat"`).

| Provider          | Register API                                                                                                                                            | Model argument              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| OpenAI            | `LlmClient::openai(api_key)` (+ optional `.base_url(...)`)                                                                                              | `"gpt-4o-mini"`             |
| Anthropic         | `LlmClient::anthropic(api_key)` (+ optional `.base_url(...)`, `.api_version(...)`)                                                                      | `"claude-3-5-haiku-latest"` |
| Google            | `LlmClient::google(api_key)` (+ optional `.base_url(...)`)                                                                                              | `"gemini-2.0-flash"`        |
| OpenAI-compatible | `LlmClient::openai_compatible(base_url).api_key(...)` / `LlmClient::openai_compatible_no_auth(base_url)` / `LlmClient::openai_compatible_with_settings` | `"deepseek-chat"`           |

## Usage

### Generating Text

```rust
use aquaregia::LlmClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;

    let client = LlmClient::openai_compatible("https://api.deepseek.com")
        .api_key(api_key)
        .build()?;

    let out = client
        .generate("deepseek-chat", "Explain Rust ownership in 3 bullet points.")
        .await?;

    println!("{}", out.output_text);
    Ok(())
}
```

### Streaming Text (Simple)

```rust
use aquaregia::LlmClient;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = LlmClient::openai_compatible("https://api.deepseek.com")
        .api_key(api_key)
        .build()?;

    let mut stream = client.stream_text("deepseek-chat", "Write a short release note.").await?;

    while let Some(chunk) = stream.next().await {
        print!("{}", chunk?);
    }
    Ok(())
}
```

For full events (`TextDelta / Usage / ToolCallReady / Done`), use `client.stream_prompt(...)`.

### Error Handling

```rust
use aquaregia::{AiErrorCode, LlmClient};

match client.generate("deepseek-chat", "hello").await {
    Ok(out) => println!("{}", out.output_text),
    Err(err) => match err.code {
        AiErrorCode::RateLimited => eprintln!("rate limited; retry later"),
        AiErrorCode::AuthFailed => eprintln!("check API key"),
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
        .tool(get_weather)
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
use aquaregia::{LlmClient, OpenAiCompatibleAdapterSettings};

let settings = OpenAiCompatibleAdapterSettings::new("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .header("x-trace-source", "aquaregia")
    .query_param("source", "sdk");

let client = LlmClient::openai_compatible_with_settings(settings)
    .build()?;
```

## Breaking Migration Notes

- Model helpers are now infallible:
  - `openai("gpt-4o-mini")` now returns `ModelRef<_>` directly (no `?`).
- Prompt APIs now accept model strings directly:
  - `client.generate("deepseek-chat", "...")`.
- OpenAI-compatible quick constructor is now base-url-first:
  - `LlmClient::openai_compatible(base_url).api_key(api_key)`.
  - no-auth endpoints can use `LlmClient::openai_compatible_no_auth(base_url)`.
- `AgentBuilder::build()` now returns `Result<Agent<_>, AiError>`:
  - call `.build()?`.
- Agent prompt execution API was renamed:
  - `agent.run("...")` for prompt input.
  - `agent.run_messages(messages)` for explicit message lists.
- `MaxSteps` / `Temperature` / `TopP` newtypes were removed:
  - use primitive types (`u8`, `f32`) and rely on validation in build/run paths.
- Tool builder split into typed and raw modes:
  - typed path: `.execute(|args: MyArgs| async move { ... })` with `Deserialize + JsonSchema`.
  - raw escape hatch: `.raw_schema(json!(...)).execute_raw(|args| async move { ... })`.
- `prepare_step` now has `event.to_prepared()` to clone the full default plan safely.
- `OpenAiCompatibleAdapterSettings` fields are private:
  - configure via chain methods (`api_key`, `header`, `query_param`, `chat_completions_path`).

## Examples

| Example                             | Command                                        | Focus                               |
| ----------------------------------- | ---------------------------------------------- | ----------------------------------- |
| Basic generation                    | `cargo run --example basic_generate`           | one-shot `generate`                 |
| Basic stream                        | `cargo run --example basic_stream`             | `StreamEvent` handling              |
| Minimal agent                       | `cargo run --example agent_minimal`            | `Agent::builder` + one tool         |
| Tool loop guardrails                | `cargo run --example tools_max_steps`          | multi-step tools + `max_steps`      |
| Dynamic hooks                       | `cargo run --example prepare_hooks`            | `prepare_call` / `prepare_step`     |
| Provider settings                   | `cargo run --example provider_selection_demo`  | quick vs. advanced compatible setup |
| Compatible custom path/query/header | `cargo run --example openai_compatible_custom` | `OpenAiCompatibleAdapterSettings`   |
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
