# Oxide

Use Oxide to quickly build your Rust AI application, with a unified interface to multiple providers and powerful tool execution capabilities.

[中文文档 (README_CN.md)](./README_CN.md)

Oxide is a provider-agnostic Rust toolkit for AI apps and agents.
Recommended path: run first, understand layers second, then unlock advanced control.

## Start Here

- [1) Get Running First](#1-get-running-first)
- [2) Understand the Layers](#2-understand-the-layers)
- [3) Explore Advanced Capabilities](#3-explore-advanced-capabilities)
- [Open Source](#open-source)

## 1) Get Running First

### Installation

```toml
[dependencies]
oxide = { path = "." }
# or after publishing to crates.io:
# oxide = "x.y.z"
```

### Environment Variables (DeepSeek example)

```bash
export DEEPSEEK_API_KEY="your_api_key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"   # optional
export DEEPSEEK_MODEL="deepseek-chat"                  # optional
```

### First Successful Call (2 minutes)

```rust
use oxide::{AiClient, openai_compatible};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = AiClient::builder()
        .with_openai_compatible(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".to_string()),
            Some(std::env::var("DEEPSEEK_API_KEY")?),
        )
        .build()?;

    let out = client
        .generate_prompt(
            openai_compatible(
                std::env::var("DEEPSEEK_MODEL")
                    .unwrap_or_else(|_| "deepseek-chat".to_string()),
            )?,
            "Explain Rust ownership in 3 bullets.",
        )
        .await?;

    println!("{}", out.output_text);
    Ok(())
}
```

### Examples Learning Path

| Stage | Command | What you learn |
| --- | --- | --- |
| 1 | `cargo run --example basic_generate` | Minimal one-shot generation |
| 2 | `cargo run --example basic_stream` | Streaming output handling |
| 3 | `cargo run --example agent_minimal` | First agent + one tool |
| 4 | `cargo run --example tools_max_steps` | Multi-step tool loop and guardrails |
| 5 | `cargo run --example prepare_hooks` | Dynamic per-call and per-step control |
| 6 | `cargo run --example openai_compatible_custom` | Custom headers/query/path for compatible providers |

More examples: [examples/README.md](./examples/README.md)

## 2) Understand the Layers

### Unified Provider Architecture

- One `AiClient` binds to one provider configuration.
- One `ModelRef` (`openai(...)`, `anthropic(...)`, `google(...)`, `openai_compatible(...)`) selects a model per call.
- If your app needs multiple providers, create multiple clients.

### Provider Registration + Model Selection

| Provider kind | Register API | Model selector |
| --- | --- | --- |
| OpenAI GPT | `.with_openai(api_key, base_url)` | `openai("gpt-4o-mini")` |
| Anthropic Claude | `.with_anthropic(api_key, base_url, api_version)` | `anthropic("claude-3-5-haiku-latest")` |
| Google Gemini | `.with_google(api_key, base_url)` | `google("gemini-2.0-flash")` |
| OpenAI-compatible | `.with_openai_compatible(base_url, api_key)` / `.with_openai_compatible_settings(...)` | `openai_compatible("deepseek-chat")` |

### Layer Map

| Layer | Responsibility | Core APIs |
| --- | --- | --- |
| Provider binding | Configure transport, retries, and provider adapter | `AiClient::builder()` |
| Model selection | Keep provider/model identity explicit | `openai(...)`, `anthropic(...)`, `google(...)`, `openai_compatible(...)` |
| Generation | One-shot + streaming text | `generate_prompt`, `stream_prompt`, `generate_text`, `stream_text` |
| Tool runtime | Multi-step reasoning + tool execution loop | `run_tools`, `tool(...)`, `max_steps`, `stop_when` |
| Agent facade | Reusable workflow (model + instructions + tools + hooks) | `Agent::builder(...).generate_prompt(...)` |

## 3) Explore Advanced Capabilities

### Agent + Tool Loop

```rust
use oxide::{Agent, openai_compatible, tool};
use serde_json::json;

let weather = tool("get_weather")
    .description("Get weather by city")
    .input_schema(json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
    }))
    .execute(|args| async move {
        Ok(json!({ "city": args["city"], "temp_c": 23, "condition": "sunny" }))
    });

let agent = Agent::builder(client)
    .model(openai_compatible("deepseek-chat")?)
    .instructions("You can call tools before answering.")
    .tool(weather)
    .max_steps(4)
    .build()?;
```

### Dynamic Planning Hooks

- `prepare_call`: adjust the call plan once before a run starts.
- `prepare_step`: adjust model/messages/tools before each step.

```rust
let agent = Agent::builder(client)
    .prepare_call(|plan| {
        let mut next = plan.clone();
        next.temperature = Some(0.2);
        next
    })
    .prepare_step(|event| {
        let mut next = oxide::RunToolsPreparedStep {
            model: event.model.clone(),
            messages: event.messages.clone(),
            tools: event.tools.clone(),
            temperature: event.temperature,
            max_output_tokens: event.max_output_tokens,
            stop_sequences: event.stop_sequences.clone(),
        };
        next.messages
            .push(oxide::Message::system_text(format!("step={}", event.step)));
        next
    })
    .build()?;
```

### Full Lifecycle Callbacks

`on_start`, `on_step_start`, `on_tool_call_start`, `on_tool_call_finish`, `on_step_finish`, `on_finish`, `stop_when`

### OpenAI-Compatible Advanced Settings

Need custom headers/query/path? Use `OpenAiCompatibleAdapterSettings`.

```bash
cargo run --example openai_compatible_custom
```

### Axum SSE Integration

```bash
cargo check --features axum
```

## Open Source

### Contributing

Issues and PRs are welcome. For behavior changes, include tests.

### Local Development Checks

```bash
cargo fmt
cargo test
cargo check --examples
cargo check --no-default-features
cargo check --no-default-features --features openai
cargo check --no-default-features --features anthropic
cargo check --features axum
```

### Governance

- [MIT License](./LICENSE)
- [Contributing Guide](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Security Policy](./SECURITY.md)
