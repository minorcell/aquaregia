# Aquaregia

> Note: Aquaregia is in rapid iteration before v0.2.0, and the API may have breaking changes. use with caution.

Aquaregia is a provider-agnostic Rust toolkit for building AI applications and tool-using agents.

It provides a unified API across OpenAI, Anthropic, Google, and OpenAI-compatible services, with first-class support for reasoning-aware output, streaming events, and multi-step tool execution.

Read the [API docs](https://docs.rs/aquaregia), browse [examples](./examples/README.md), or switch to [中文文档](./README_CN.md).

## Installation

You need Rust and a Tokio async runtime in your project.

```bash
cargo add aquaregia
```

Default features enable `openai` and `anthropic`. Optional features:

| Feature     | Description                                                          |
| ----------- | -------------------------------------------------------------------- |
| `openai`    | OpenAI adapter (default)                                             |
| `anthropic` | Anthropic adapter (default)                                          |
| `telemetry` | `tracing` spans for `generate`, `stream`, agent steps and tool calls |

```bash
cargo check --no-default-features
cargo check --no-default-features --features openai
cargo check --no-default-features --features anthropic
cargo check --features telemetry
```

## Unified Provider Architecture

One `LlmClient` binds to one provider configuration.
Each call passes a `GenerateTextRequest` that carries both the model and the messages.

| Provider          | Register API                                                                       | Model argument        |
| ----------------- | ---------------------------------------------------------------------------------- | --------------------- |
| OpenAI            | `LlmClient::openai(api_key)` (+ optional `.base_url(...)`)                         | `"gpt-4o"`            |
| Anthropic         | `LlmClient::anthropic(api_key)` (+ optional `.base_url(...)`, `.api_version(...)`) | `"claude-sonnet-4-5"` |
| Google            | `LlmClient::google(api_key)` (+ optional `.base_url(...)`)                         | `"gemini-2.0-flash"`  |
| OpenAI-compatible | `LlmClient::openai_compatible(base_url).api_key(...)`                              | `"deepseek-chat"`     |

## Usage

### Generating Text

```rust
use aquaregia::{GenerateTextRequest, LlmClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::openai_compatible("https://api.deepseek.com")
        .api_key(std::env::var("DEEPSEEK_API_KEY"))
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
    let client = LlmClient::openai_compatible("https://api.deepseek.com")
        .api_key(std::env::var("DEEPSEEK_API_KEY"))
        .build()?;

    let mut stream = client
        .stream(GenerateTextRequest::from_user_prompt(
            "deepseek-chat",
            "Write a short release note.",
        ))
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ReasoningStarted { block_id, .. } => {
                eprintln!("\n[reasoning:{block_id}]");
            }
            StreamEvent::ReasoningDelta { text, .. } => {
                eprint!("{text}");
            }
            StreamEvent::ReasoningDone { .. } => {
                eprintln!();
            }
            StreamEvent::TextDelta { text } => print!("{text}"),
            StreamEvent::Usage { usage } => {
                eprintln!(
                    "\nusage: in={} (no_cache={} cache_read={} cache_write={}) out={} (text={} reasoning={}) total={}",
                    usage.input_tokens,
                    usage.input_no_cache_tokens,
                    usage.input_cache_read_tokens,
                    usage.input_cache_write_tokens,
                    usage.output_tokens,
                    usage.output_text_tokens,
                    usage.reasoning_tokens,
                    usage.total_tokens
                );
            }
            StreamEvent::Done => break,
            _ => {}
        }
    }
    Ok(())
}
```

`StreamEvent` covers all variants:
`ReasoningStarted`, `ReasoningDelta`, `ReasoningDone`, `TextDelta`, `ToolCallReady`, `Usage`, and `Done`.

### Reasoning

Reasoning is exposed in both non-streaming and streaming APIs.

```rust
let out = client
    .generate(GenerateTextRequest::from_user_prompt(
        "deepseek-chat",
        "Solve this step by step.",
    ))
    .await?;

println!("answer: {}", out.output_text);
println!("reasoning text: {}", out.reasoning_text);
println!("reasoning tokens: {}", out.usage.reasoning_tokens);

for part in &out.reasoning_parts {
    println!("reasoning block: {}", part.text);
}
```

Unified output fields:

- `GenerateTextResponse.reasoning_text`: flattened reasoning text (convenience field).
- `GenerateTextResponse.reasoning_parts`: structured reasoning blocks with optional provider metadata.
- `Usage.input_tokens`: total input tokens reported by provider.
- `Usage.input_no_cache_tokens`: non-cached input tokens (best effort).
- `Usage.input_cache_read_tokens` / `Usage.input_cache_write_tokens`: cache read/write split when available.
- `Usage.output_tokens`: total output tokens.
- `Usage.output_text_tokens`: output text token split when available.
- `Usage.reasoning_tokens`: provider-reported reasoning tokens when available.
- `Usage.raw_usage`: raw provider usage payload for debugging/future extension.
- `Message.parts`: assistant messages can include `ContentPart::Reasoning(...)` for transcript replay.

Provider mapping:

| Provider | Reasoning Content | Usage Mapping |
| --- | --- | --- |
| OpenAI / OpenAI-compatible | `reasoning_content` (or `reasoning`) in sync + stream | parses `prompt_tokens_details.cached_tokens` + `completion_tokens_details.reasoning_tokens` |
| Anthropic | `thinking` / `redacted_thinking`, stream `thinking_delta` + `signature_delta` | parses `cache_read_input_tokens` / `cache_creation_input_tokens`; reasoning token split unavailable |
| Google | parts with `thought: true`, optional `thoughtSignature` metadata | parses `cachedContentTokenCount` + `thoughtsTokenCount` |

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
        ErrorCode::AuthFailed  => eprintln!("check API key"),
        ErrorCode::Cancelled   => eprintln!("request was cancelled"),
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
    let client = LlmClient::openai_compatible("https://api.deepseek.com")
        .api_key(std::env::var("DEEPSEEK_API_KEY"))
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

### Cancellation

Every request and agent run can be cancelled via a `CancellationToken`.

```rust
use aquaregia::{Agent, CancellationToken, GenerateTextRequest, LlmClient};
use tokio::time::{Duration, sleep};

// Cancel a single generate call
let token = CancellationToken::new();
let token_clone = token.clone();
tokio::spawn(async move {
    sleep(Duration::from_millis(200)).await;
    token_clone.cancel();
});

let req = GenerateTextRequest::builder("deepseek-chat")
    .user_prompt("Write a 10000-word essay.")
    .cancellation_token(token)
    .build()?;

match client.generate(req).await {
    Err(e) if e.code == ErrorCode::Cancelled => println!("cancelled as expected"),
    other => println!("{other:?}"),
}
```

Agents expose dedicated helpers:

```rust
let token = CancellationToken::new();
token.cancel(); // or cancel later from another task

// Returns Err with ErrorCode::Cancelled
agent.run_cancellable("hello", token).await?;

// Pass your own message list
agent.run_messages_cancellable(messages, token).await?;
```

Cancellation is checked:

- **Before every HTTP send** (via `tokio::select!` — zero overhead when not cancelled)
- **After every SSE chunk** in streaming responses
- **At the top of every agent step** in the tool loop

### Telemetry

Enable the `telemetry` feature to get `tracing` spans automatically:

```toml
aquaregia = { version = "*", features = ["telemetry"] }
```

Spans emitted:

| Span                  | Fields              |
| --------------------- | ------------------- |
| `aquaregia::generate` | `model`, `provider` |
| `aquaregia::stream`   | `model`             |
| `agent_step`          | `step`              |
| `tool_call`           | `tool.name`         |

Wire your own subscriber (e.g. `tracing-subscriber`, `tracing-opentelemetry`) — Aquaregia does not configure one for you.

```rust
tracing_subscriber::fmt::init(); // or any other subscriber

let out = client.generate(req).await?; // emits a span
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
cargo test --features telemetry
cargo clippy -- -D warnings
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
