<div align="center">

# Aquaregia

**The universal AI layer for Rust.**

[![Crates.io](https://img.shields.io/crates/v/aquaregia.svg)](https://crates.io/crates/aquaregia)
[![Docs.rs](https://docs.rs/aquaregia/badge.svg)](https://docs.rs/aquaregia)
[![License: MIT](https://img.shields.io/crates/l/aquaregia.svg)](./LICENSE)
[![Downloads](https://img.shields.io/crates/d/aquaregia.svg)](https://crates.io/crates/aquaregia)

[API Docs](https://docs.rs/aquaregia) · [Examples](./examples/README.md) · [中文文档](./README_CN.md)

</div>

One crate to build LLM applications and agents on a single, provider-agnostic foundation — OpenAI, Anthropic, Google, or any OpenAI-compatible endpoint (DeepSeek, vLLM, Ollama, …).

---

## Highlights

- **Typed provider markers** — `LlmClient::openai(...)` returns a `BoundClient<OpenAi>`; a mismatched `ModelRef<P>` is a compile-time error, not a 400.
- **Unified text · stream · reasoning · vision · tools** — same `GenerateTextRequest`, same `StreamEvent`, same `Usage`, across all four provider families.
- **Agents with AI SDK-style hooks** — `prepare_call`, `prepare_step`, plus a full event chain (`on_start` → `on_step_start` → `on_tool_call_*` → `on_step_finish` → `on_finish`).
- **Production reliability** — built-in retries with exponential backoff and `Retry-After` parsing, `CancellationToken` checkpoints, optional `tracing` spans.
- **Multi-turn out of the box** — `AgentResponse.transcript` round-trips back into `agent.run_messages(...)` for free.
- **Multimodal vision** — URL / base64 / raw bytes, mapped to each provider's native image format.

---

## Quick start

```rust
use aquaregia::{GenerateTextRequest, LlmClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::openai_compatible("https://api.deepseek.com")
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .build()?;

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "deepseek-chat",
            "Explain Rust ownership in 3 bullet points.",
        ))
        .await?;

    println!("{}", response.output_text);
    println!(
        "usage: {} → {} tokens",
        response.usage.input_tokens, response.usage.output_tokens
    );
    Ok(())
}
```

Swap the constructor and the same call works against Anthropic, OpenAI, or Google.

---

## Installation

```bash
cargo add aquaregia
```

You also need a Tokio runtime in your project.

| Feature     | Description                                                          |
| ----------- | -------------------------------------------------------------------- |
| `openai`    | OpenAI adapter — default                                             |
| `anthropic` | Anthropic adapter — default                                          |
| `telemetry` | `tracing` spans for `generate`, `stream`, agent steps, and tools     |

Google and OpenAI-compatible adapters are always available.

---

## Providers

Pick a constructor; the resulting `BoundClient<P>` is parameterized over the provider marker `P`.

| Provider          | Constructor                                              | Model argument        |
| ----------------- | -------------------------------------------------------- | --------------------- |
| OpenAI            | `LlmClient::openai(api_key)`                             | `"gpt-4o"`            |
| Anthropic         | `LlmClient::anthropic(api_key)`                          | `"claude-sonnet-4-5"` |
| Google            | `LlmClient::google(api_key)`                             | `"gemini-2.0-flash"`  |
| OpenAI-compatible | `LlmClient::openai_compatible(base_url).api_key(...)`    | `"deepseek-chat"`     |

### Client configuration

```rust
use std::time::Duration;

let client = LlmClient::openai(std::env::var("OPENAI_API_KEY")?)
    .base_url("https://api.openai.com")          // custom upstream
    .timeout(Duration::from_secs(60))            // per-request timeout
    .max_retries(3)                              // transient-failure retries
    .default_max_steps(8)                        // default for Agents built from this client
    .user_agent("my-app/1.0")
    .build()?;
```

### Typed `ModelRef<P>`

To prevent passing an OpenAI model name to an Anthropic client at runtime, use the typed factory helpers:

```rust
use aquaregia::{anthropic, openai, Anthropic, ModelRef, OpenAi};

let gpt:    ModelRef<OpenAi>    = openai("gpt-4o");
let claude: ModelRef<Anthropic> = anthropic("claude-sonnet-4-5");

// `client_openai.generate(GenerateTextRequest::from_user_prompt(claude, "..."))`
// is a compile-time error against a `BoundClient<OpenAi>`.
```

`GenerateTextRequest::from_user_prompt` accepts anything implementing `IntoModelRef<P>` — a bare `&str` for ergonomic inline calls, or the typed factories above for stronger guarantees.

### OpenAI-compatible deep configuration

```rust
let client = LlmClient::openai_compatible("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .header("x-trace-source", "aquaregia")
    .query_param("source", "sdk")
    .chat_completions_path("/v1/chat/completions") // override the endpoint path
    .think_tag_parsing(true)                       // parse <think>...</think> as reasoning
    .think_tag_case_insensitive(true)
    .build()?;
```

`think_tag_parsing` extracts `<think>` / `<thinking>` blocks from the assistant message and routes them into `reasoning_parts`, matching the unified surface used by native reasoning providers.

### Provider differences at a glance

| Capability                       | OpenAI | Anthropic | Google | OpenAI-Compatible |
| -------------------------------- | :----: | :-------: | :----: | :---------------: |
| Custom `base_url`                |   ✓    |     ✓     |   ✓    |         ✓         |
| Custom headers / query / path    |        |           |        |         ✓         |
| `api_version` (header)           |        |     ✓     |        |                   |
| Native reasoning content         |   ✓    |     ✓     |   ✓    |  via think tags   |
| Tool-call streaming              |   ✓    |     ✓     |   ✓    |         ✓         |
| Cache-token split in `Usage`     |   ✓    |     ✓     |   ✓    |   if reported     |

---

## Generating text

### One-shot `generate`

```rust
let response = client
    .generate(GenerateTextRequest::from_user_prompt(
        "deepseek-chat",
        "Summarize Rust's borrow checker for a Go developer.",
    ))
    .await?;

println!("{}", response.output_text);
println!("finish: {:?}", response.finish_reason);
```

Full builder when you need messages, sampling, or tools:

```rust
use aquaregia::{GenerateTextRequest, Message};

let req = GenerateTextRequest::builder("deepseek-chat")
    .message(Message::system_text("You are concise."))
    .message(Message::user_text("Write a release note."))
    .temperature(0.2)
    .max_output_tokens(300)
    .build()?;
```

### Streaming

```rust
use aquaregia::StreamEvent;
use futures_util::StreamExt;

let mut stream = client.stream(request).await?;
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta { text }          => print!("{text}"),
        StreamEvent::ReasoningDelta { text, .. } => eprint!("{text}"),
        StreamEvent::ToolCallReady { call }      => eprintln!("\n[tool] {}", call.tool_name),
        StreamEvent::Usage { usage }             => eprintln!(
            "\nin={} out={} total={}",
            usage.input_tokens, usage.output_tokens, usage.total_tokens,
        ),
        StreamEvent::Done                        => break,
        _ => {}
    }
}
```

All variants:

| Event                | Fields                                  |
| -------------------- | --------------------------------------- |
| `ReasoningStarted`   | `block_id`, `provider_metadata`         |
| `ReasoningDelta`     | `block_id`, `text`, `provider_metadata` |
| `ReasoningDone`      | `block_id`, `provider_metadata`         |
| `TextDelta`          | `text`                                  |
| `ToolCallReady`      | `call: ToolCall`                        |
| `Usage`              | `usage: Usage`                          |
| `Done`               | —                                       |

### Reasoning

Reasoning is exposed in both sync and streaming output:

```rust
let out = client.generate(req).await?;

println!("answer:     {}", out.output_text);
println!("thinking:   {}", out.reasoning_text);
println!("rsn-tokens: {}", out.usage.reasoning_tokens);

for part in &out.reasoning_parts {
    println!("[block] {}", part.text);
    // part.provider_metadata carries signature blocks (Anthropic), thought
    // signatures (Google), and provider-specific extras.
}
```

| Provider                   | Reasoning content                                                              | Usage mapping                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| OpenAI / OpenAI-compatible | `reasoning_content` (or `reasoning`); `<think>` tags if enabled                | parses `prompt_tokens_details.cached_tokens` + `completion_tokens_details.reasoning_tokens`         |
| Anthropic                  | `thinking` / `redacted_thinking`; stream `thinking_delta` + `signature_delta`  | parses `cache_read_input_tokens` / `cache_creation_input_tokens`; reasoning split unavailable       |
| Google                     | parts with `thought: true`, optional `thoughtSignature` metadata               | parses `cachedContentTokenCount` + `thoughtsTokenCount`                                             |

### `Usage` and aggregation

```rust
pub struct Usage {
    pub input_tokens:             u32, // total
    pub input_no_cache_tokens:    u32,
    pub input_cache_read_tokens:  u32,
    pub input_cache_write_tokens: u32,
    pub output_tokens:            u32, // total
    pub output_text_tokens:       u32,
    pub reasoning_tokens:         u32,
    pub total_tokens:             u32,
    pub raw_usage:                Option<serde_json::Value>,
}
```

`Usage` implements `Add` and `AddAssign`, so totaling tokens across agent steps is a one-liner. `AgentResponse.usage_total` is already aggregated for you.

---

## Tools & Agents

### Defining tools

Tools are built with the `tool(name)` function (there is no `#[tool]` proc-macro). Two execution styles are supported.

**Typed args** — `schemars` derives the JSON Schema from your struct:

```rust
use aquaregia::{Tool, tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherArgs { city: String }

fn get_weather() -> Tool {
    tool("get_weather")
        .description("Get weather by city")
        .execute(|args: WeatherArgs| async move {
            Ok(json!({ "city": args.city, "temp_c": 23, "condition": "sunny" }))
        })
}
```

**Raw schema** — write the JSON Schema by hand, receive a `serde_json::Value`:

```rust
let fx_tool = tool("get_fx_rate")
    .description("Get FX rate by currency pair, e.g. USD/CNY")
    .raw_schema(json!({
        "type": "object",
        "properties": { "pair": { "type": "string" } },
        "required": ["pair"]
    }))
    .execute_raw(|args| async move {
        let pair = args.get("pair").and_then(|v| v.as_str()).unwrap_or("USD/CNY");
        Ok(json!({ "pair": pair, "rate": 7.18 }))
    });
```

Tool names must match `^[a-zA-Z0-9_-]{1,64}$` and be unique within an agent.

### Minimal Agent

```rust
use aquaregia::{Agent, LlmClient};

let client = LlmClient::openai_compatible("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .build()?;

let agent = Agent::builder(client, "deepseek-chat")
    .instructions("You can call tools before answering.")
    .tools([get_weather])
    .max_steps(4)
    .build()?;

let response = agent.run("Weather in Shanghai?").await?;
println!("{}", response.output_text);
println!("steps={} total={}", response.steps, response.usage_total.total_tokens);
```

### Event hooks

The agent loop emits an event at every meaningful boundary. All hooks are `Fn + Send + Sync`, so you can attach them as closures.

```rust
let agent = Agent::builder(client, "deepseek-chat")
    .tools([get_weather])
    .on_start(|e|            println!("[start] tools={} max_steps={}", e.tool_count, e.max_steps))
    .on_step_start(|e|       println!("[step:{}] msgs={}", e.step, e.messages.len()))
    .on_tool_call_start(|e|  println!("[tool:{}] {}", e.step, e.tool_call.tool_name))
    .on_tool_call_finish(|e| println!("[tool:{}] {} in {}ms", e.step, e.tool_call.tool_name, e.duration_ms))
    .on_step_finish(|s|      println!("[step:{}] finish={:?}", s.step, s.finish_reason))
    .on_finish(|f|           println!("[done] {} steps, {} total tokens", f.step_count, f.usage_total.total_tokens))
    .build()?;
```

### Dynamic planning — `prepare_call` / `prepare_step`

`prepare_call` runs once before the agent loop starts and mutates the run plan in place:

```rust
let agent = Agent::builder(client, "deepseek-chat")
    .prepare_call(|plan| {
        if plan.messages.len() > 6 {
            plan.temperature = Some(0.0);
            plan.max_output_tokens = Some(400);
        }
    })
    .build()?;
```

`prepare_step` runs before every step and returns a fresh prepared plan — useful for shrinking the tool set, switching models, or injecting per-step instructions:

```rust
use aquaregia::Message;

let agent = Agent::builder(client, "deepseek-chat")
    .tools([get_weather, get_fx_rate])
    .prepare_step(|event| {
        let mut next = event.to_prepared();
        next.messages.push(Message::system_text(format!(
            "Step {}: be concise.", event.step,
        )));
        if event.step >= 2 {
            next.tools.clear(); // disallow tools after step 2
        }
        next
    })
    .build()?;
```

### Stopping policies

```rust
use aquaregia::ToolErrorPolicy;

let agent = Agent::builder(client, "deepseek-chat")
    .max_steps(8)                                                                 // hard cap
    .stop_when(|step| step.tool_calls.is_empty() && !step.output_text.is_empty()) // predicate
    .tool_error_policy(ToolErrorPolicy::ContinueAsToolResult)                     // default
    .build()?;
```

- `max_steps` — exceeding it returns `ErrorCode::MaxStepsExceeded`.
- `stop_when` — predicate evaluated after every step; truthy = stop early.
- `tool_error_policy` —
  - `ContinueAsToolResult` (default) — schema-validation failures, timeouts, and panics become `{ "error": "..." }` tool results so the model can recover.
  - `FailFast` — surface as `ErrorCode::ToolExecutionFailed` / `InvalidToolArgs` immediately.

### Multi-turn conversations

`AgentResponse.transcript` is a complete `Vec<Message>` (system + user + assistant + tool results) that you can feed straight back into the next turn:

```rust
let mut history = vec![Message::system_text("You are a careful assistant.")];

loop {
    let user_input = read_line()?;
    history.push(Message::user_text(user_input));

    let result = agent.run_messages(history.clone()).await?;
    println!("{}", result.output_text);

    history = result.transcript; // round-trip the full conversation
}
```

See `examples/mini_claude_code.rs` for a working terminal agent that uses this pattern with `bash` / `read` / `write` / `edit` tools.

---

## Multimodal vision

```rust
use aquaregia::{GenerateTextRequest, LlmClient, Message};

let client = LlmClient::anthropic(std::env::var("ANTHROPIC_API_KEY")?).build()?;

let out = client
    .generate(
        GenerateTextRequest::builder("claude-sonnet-4-5")
            .message(Message::user_text_and_image_url(
                "What's in this image?",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
            ))
            .build()?,
    )
    .await?;
```

Three convenience constructors plus a full-control form:

| Constructor                                                              | Use case                                  |
| ------------------------------------------------------------------------ | ----------------------------------------- |
| `Message::user_image_url(url)`                                           | Single image from a URL                   |
| `Message::user_image_bytes(bytes, mime)`                                 | Single image from raw bytes (auto base64) |
| `Message::user_text_and_image_url(text, url)`                            | Text + image in one user message          |
| `ContentPart::Image(ImagePart { data, media_type, provider_metadata })`  | Full control + provider-specific hints    |

Each provider sees its own native format:

| Provider            | URL                          | Base64 / Bytes                          |
| ------------------- | ---------------------------- | --------------------------------------- |
| Anthropic           | `source.type: url`           | `source.type: base64`                   |
| OpenAI / Compatible | `image_url` with remote URL  | `image_url` with `data:<mime>;base64,…` |
| Google              | `fileData.fileUri`           | `inlineData.data`                       |

---

## Cancellation

Every request and agent run is cancellable through a `CancellationToken`.

```rust
use aquaregia::{CancellationToken, ErrorCode, GenerateTextRequest};
use std::time::Duration;

let token = CancellationToken::new();
let bg = token.clone();
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_millis(200)).await;
    bg.cancel();
});

let req = GenerateTextRequest::builder("deepseek-chat")
    .user_prompt("Write a 10,000-word essay.")
    .cancellation_token(token)
    .build()?;

match client.generate(req).await {
    Err(e) if e.code == ErrorCode::Cancelled => println!("cancelled"),
    other => println!("{other:?}"),
}
```

Agents have dedicated helpers:

```rust
agent.run_cancellable("hello", token.clone()).await?;
agent.run_messages_cancellable(messages, token).await?;
```

Cancellation is checked **before every HTTP send** (via `tokio::select!`, zero overhead on the happy path), **after every SSE chunk** in streaming responses, and **at the top of every agent step** in the tool loop.

---

## Reliability

### Retries

```rust
let client = LlmClient::openai(api_key)
    .max_retries(3)                       // default: 0
    .timeout(Duration::from_secs(45))
    .build()?;
```

Aquaregia retries automatically on transient classes (`RateLimited`, `ProviderServerError`, `Transport`, `Timeout`) using exponential backoff with jitter. The `Retry-After` header is parsed and honored when present.

Every `Error` carries a `retryable: bool` flag matching the same classification, so you can layer your own retry/circuit-breaker on top if you need finer control.

### Telemetry

Enable the `telemetry` feature:

```toml
aquaregia = { version = "*", features = ["telemetry"] }
```

Aquaregia emits — but does not configure — `tracing` spans. Bring your own subscriber (e.g. `tracing-subscriber`, `tracing-opentelemetry`):

| Span                  | Fields              |
| --------------------- | ------------------- |
| `aquaregia::generate` | `model`, `provider` |
| `aquaregia::stream`   | `model`             |
| `agent_step`          | `step`              |
| `tool_call`           | `tool.name`         |

---

## Framework integration example (Axum)

Aquaregia intentionally keeps web framework adapters out of the crate. If you're building on Axum, adapt `TextStream` in your application layer:

```rust
use aquaregia::{BoundClient, GenerateTextRequest, OpenAiCompatible, StreamEvent, TextStream};
use axum::{
    extract::State,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::get,
    Router,
};
use futures_util::StreamExt;
use std::{convert::Infallible, sync::Arc};

fn to_axum_sse(
    stream: TextStream,
) -> impl IntoResponse {
    Sse::new(stream.map(|item| {
        let event = match item {
            Ok(StreamEvent::ReasoningStarted { .. }) => {
                Event::default().event("reasoning_start").data("{}")
            }
            Ok(StreamEvent::ReasoningDelta { text, .. }) => {
                Event::default().event("reasoning_token").data(text)
            }
            Ok(StreamEvent::ReasoningDone { .. }) => {
                Event::default().event("reasoning_end").data("{}")
            }
            Ok(StreamEvent::TextDelta { text }) => Event::default().event("token").data(text),
            Ok(StreamEvent::ToolCallReady { .. }) => Event::default().event("tool_call").data("{}"),
            Ok(StreamEvent::Usage { .. }) => Event::default().event("usage").data("{}"),
            Ok(StreamEvent::Done) => Event::default().event("done").data("{}"),
            Err(err) => Event::default().event("error").data(err.message),
        };
        Ok::<Event, Infallible>(event)
    }))
}

async fn chat(State(client): State<Arc<BoundClient<OpenAiCompatible>>>) -> impl IntoResponse {
    let stream = client
        .stream(GenerateTextRequest::from_user_prompt("deepseek-chat", "Hello."))
        .await
        .unwrap();
    to_axum_sse(stream)
}

let app: Router = Router::new()
    .route("/chat", get(chat))
    .with_state(Arc::new(client));
```

The example keeps non-text payloads minimal; in a real app, serialize tool calls, usage, and reasoning metadata into whatever wire format your frontend expects.

Map the `StreamEvent` variants you care about to named SSE events, websocket messages, or any other transport format your app uses.

---

## Error handling

```rust
use aquaregia::ErrorCode;

match client.generate(req).await {
    Ok(out) => println!("{}", out.output_text),
    Err(e) => match e.code {
        ErrorCode::RateLimited        => eprintln!("retry after {:?}s", e.retry_after_secs),
        ErrorCode::AuthFailed         => eprintln!("bad API key"),
        ErrorCode::Cancelled          => eprintln!("cancelled"),
        ErrorCode::MaxStepsExceeded   => eprintln!("agent loop too long"),
        ErrorCode::InvalidToolArgs    => eprintln!("schema mismatch: {}", e.message),
        ErrorCode::Timeout            => eprintln!("upstream timed out"),
        _                              => eprintln!("error: {e}"),
    },
}
```

Every `Error` carries:

- `code: ErrorCode` — one of 13 normalized variants
- `provider`, `status`, `request_id`, `raw_body`, `retry_after_secs` — for logging and triage
- `retryable: bool` — `true` iff Aquaregia's built-in retry will engage

---

## Examples

```bash
DEEPSEEK_API_KEY=... cargo run --example basic_generate
```

| Example                       | Focus                                                       |
| ----------------------------- | ----------------------------------------------------------- |
| `basic_generate`              | One-shot `generate` + usage reading                         |
| `basic_stream`                | `stream` + `StreamEvent` handling                           |
| `agent_minimal`               | `Agent::builder` with one typed tool                        |
| `tools_max_steps`             | Multi-tool loop with `max_steps` and sampling caps          |
| `prepare_hooks`               | `prepare_call`, `prepare_step`, `on_step_finish`            |
| `openai_compatible_custom`    | Custom headers / query params / chat path                   |
| `mini_claude_code`            | TUI code agent — `bash` / `read` / `write` / `edit` tools   |
| `multimodal_image`            | `Message::user_text_and_image_url` + Anthropic vision       |

Set `DEEPSEEK_API_KEY` for most examples; `ANTHROPIC_API_KEY` for `multimodal_image`. See [`examples/README.md`](./examples/README.md) for full descriptions.

---

## Development

```bash
cargo fmt
cargo test
cargo check --examples
cargo check --no-default-features
cargo check --no-default-features --features openai
cargo check --no-default-features --features anthropic
cargo test --features telemetry
cargo clippy -- -D warnings
```

---

## Contributing & License

Contributions are welcome. For behavior changes, include integration tests (happy path + error mapping + tool/stream flows where relevant).

- [Contributing Guide](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Security Policy](./SECURITY.md)
- Licensed under the [MIT License](./LICENSE)
