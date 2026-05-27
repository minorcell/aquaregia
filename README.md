<div align="center">

# Aquaregia

**The universal AI layer for Rust.**

[![Crates.io](https://img.shields.io/crates/v/aquaregia.svg)](https://crates.io/crates/aquaregia)
[![Docs.rs](https://docs.rs/aquaregia/badge.svg)](https://docs.rs/aquaregia)
[![License: MIT](https://img.shields.io/crates/l/aquaregia.svg)](./LICENSE)
[![Downloads](https://img.shields.io/crates/d/aquaregia.svg)](https://crates.io/crates/aquaregia)

[API Docs](https://docs.rs/aquaregia) · [Examples](./examples/README.md) · [中文文档](./README_CN.md)

</div>

---

## Introduction

### Why Aquaregia

You want to call an LLM from Rust. You probably also want to call _another_ LLM from Rust six months from now, without rewriting your prompts, your tool definitions, or your streaming code.

Aquaregia is a single crate that gives you one API across OpenAI, Anthropic, Google, and any OpenAI-compatible endpoint. You swap a constructor and the same `generate`, `stream`, `generate_object`, and `Agent` calls keep working. Streaming, structured output, reasoning tokens, multimodal input, agent loops, retries, cancellation — all of it normalised behind one type.

It's not a gateway, not a proxy, not a microservice. It's a Rust library you `cargo add` and call directly.

### At a glance

| Capability                        | What you get                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------- |
| **One API, four providers**       | OpenAI · Anthropic · Google · OpenAI-compatible (DeepSeek, Together, Groq, …)            |
| **Streaming & non-streaming**     | Same builder feeds `generate` or `stream`, with consistent `StreamEvent`s                |
| **Structured output**             | `generate_object::<T>()` and `stream_object::<T>()` with `schemars`-derived schemas      |
| **Reasoning content**             | First-class reasoning extraction, streaming reasoning deltas, reasoning-token usage      |
| **Tool-using agents**             | Multi-step loop with `prepare_step` hooks, `max_steps`, `stop_when`, error policies      |
| **Multimodal vision**             | Send images by URL, base64, or raw bytes — one `ImagePart` API across providers          |
| **Cancellation & retries**        | `CancellationToken` checked everywhere; exponential backoff with `Retry-After` honored   |

### When not to use it

If you only ever target one provider and don't care about being portable, that provider's native SDK probably gives you a more idiomatic surface. Aquaregia earns its keep when you want to **A/B providers**, **let users pick a model at runtime**, or **future-proof against the LLM landscape moving under you**. If none of those apply, you're not the target audience — and that's fine.

---

## Quick Start

### Install

```bash
cargo add aquaregia
```

You'll also need a Tokio runtime in your application — Aquaregia is async end-to-end.

### Hello world

The shortest path to a real model response:

```rust
use aquaregia::{GenerateTextRequest, LlmClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::openai_compatible()
        .base_url("https://api.deepseek.com")
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .build()?;

    let response = client
        .generate(GenerateTextRequest::from_user_prompt(
            "deepseek-v4-pro",
            "Explain Rust ownership in 3 bullet points.",
        ))
        .await?;

    println!("{}", response.output_text);
    Ok(())
}
```

Three moving parts: a **client** (provider + auth + transport), a **request** (model + messages), and a `.generate()` call. Everything in this guide is a refinement of one of those three.

### First streaming call

Same client, swap `.generate` for `.stream`:

```rust
use aquaregia::StreamEvent;
use futures_util::StreamExt;

let mut stream = client
    .stream(GenerateTextRequest::from_user_prompt(
        "deepseek-v4-pro",
        "Write a haiku about the borrow checker.",
    ))
    .await?;

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta { text } => print!("{text}"),
        StreamEvent::Done               => break,
        _                                => {}
    }
}
```

You've now seen the shape — a builder, a call, a result or an event loop. The rest of this guide unpacks each piece.

---

## Essentials

### Client

A `LlmClient` is just a namespace of constructors. Each constructor returns a typed `ClientBuilder<S>` parameterised on the provider's settings type, which builds into a reusable `BoundClient`. Think **constructor → builder → bound client** — that's the whole shape, and it's the same regardless of which provider you pick.

```rust
use std::time::Duration;

let client = LlmClient::openai()
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .base_url("https://api.openai.com")          // optional: custom upstream
    .timeout(Duration::from_secs(60))            // per-request HTTP timeout
    .max_retries(3)                              // transient-failure retries
    .default_max_steps(8)                        // default agent step cap
    .user_agent("my-app/1.0")
    .build()?;
```

Switching providers is just a different constructor — every method you'll see below works the same way on whichever `BoundClient` you end up with.

| Provider          | Constructor                                                         |
| ----------------- | ------------------------------------------------------------------- |
| OpenAI            | `LlmClient::openai().api_key(api_key)`                              |
| Anthropic         | `LlmClient::anthropic().api_key(api_key).api_version("2023-06-01")` |
| Google            | `LlmClient::google().api_key(api_key)`                              |
| OpenAI-compatible | `LlmClient::openai_compatible().base_url(url).api_key(token)`       |

#### Going deeper: OpenAI-compatible endpoints

If your provider speaks the OpenAI chat-completions wire format but lives at a different URL — DeepSeek, Together, Groq, your own gateway — `openai_compatible()` lets you bolt on custom headers, query params, and even a different chat path:

```rust
let client = LlmClient::openai_compatible()
    .base_url("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .header("x-trace-source", "aquaregia")
    .query_param("source", "sdk")
    .chat_completions_path("/v1/chat/completions")
    .build()?;
```

If the endpoint doesn't want any `Authorization` header at all, call `.no_api_key()` instead of `.api_key(...)`.

---

### Generating text

`generate` is the workhorse: one request, one response, all content in `output_text`.

```rust
let response = client
    .generate(GenerateTextRequest::from_user_prompt(
        "deepseek-v4-pro",
        "Summarize Rust's borrow checker for a Go developer.",
    ))
    .await?;

println!("{}", response.output_text);
println!("finish: {:?}", response.finish_reason);
```

`from_user_prompt(model, prompt)` is the one-line form. When you need more — system prompts, sampling controls, tools — reach for the builder:

```rust
use aquaregia::{GenerateTextRequest, Message};

let req = GenerateTextRequest::builder("deepseek-v4-pro")
    .message(Message::system_text("You are concise."))
    .message(Message::user_text("Write a release note."))
    .temperature(0.2)
    .max_output_tokens(300)
    .build()?;
```

The model argument is just a string — Aquaregia doesn't try to enumerate model IDs because every provider ships new ones every few weeks. Pass whatever your provider's docs say is current; if the provider rejects it, you'll get a clean `InvalidRequest` error back.

---

### Streaming responses

When the consumer is a UI or a terminal, you want tokens as they arrive, not after the whole reply lands. Swap `generate` for `stream` and consume `StreamEvent`s:

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

The event stream is the union of everything a model might emit. You'll typically only match a few variants; the rest you can `_ => {}`. Here's the full menu:

| Event                | Fields                                  | When it fires                                  |
| -------------------- | --------------------------------------- | ---------------------------------------------- |
| `ReasoningStarted`   | `block_id`, `provider_metadata`         | A reasoning block begins (Anthropic / Google)  |
| `ReasoningDelta`     | `block_id`, `text`, `provider_metadata` | Each token of thought                          |
| `ReasoningDone`      | `block_id`, `provider_metadata`         | A reasoning block closed                       |
| `TextDelta`          | `text`                                  | Each token of the visible answer               |
| `ToolCallReady`      | `call: ToolCall`                        | Model finished assembling a tool call          |
| `Usage`              | `usage: Usage`                          | Provider reports token counts                  |
| `Done`               | —                                       | Final event of every stream                    |

Once `Done` fires the stream is finished — don't poll after.

---

### Reasoning

Reasoning is the model's "thinking out loud" — separate from the visible answer, exposed as its own content block by Anthropic Extended Thinking, Google's `thoughts`, and OpenAI's reasoning-token accounting. Aquaregia surfaces it uniformly:

```rust
let out = client.generate(req).await?;

println!("answer:     {}", out.output_text);
println!("thinking:   {}", out.reasoning_text);
println!("rsn-tokens: {}", out.usage.reasoning_tokens);

for part in &out.reasoning_parts {
    println!("[block] {}", part.text);
    // part.provider_metadata carries signature blocks (Anthropic),
    // thought signatures (Google), and any other provider extras.
}
```

In streaming you'll see `ReasoningStarted` → `ReasoningDelta`(s) → `ReasoningDone` for each reasoning block before the visible text starts. The token-usage split comes from whichever fields the provider populates:

| Provider                   | Reasoning-token mapping                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| OpenAI / OpenAI-compatible | `prompt_tokens_details.cached_tokens` + `completion_tokens_details.reasoning_tokens`        |
| Anthropic                  | `cache_read_input_tokens` / `cache_creation_input_tokens`; reasoning-token split unavailable |
| Google                     | `cachedContentTokenCount` + `thoughtsTokenCount`                                            |

If a provider doesn't report a number, the field stays at `0` — Aquaregia never makes up data.

---

### Structured output

When you want a typed Rust value back instead of a blob of text, derive `JsonSchema` and call `generate_object::<T>()`. The schema is generated automatically and passed to the provider; the JSON response is parsed straight into `T`.

```rust
use aquaregia::{GenerateTextRequest, LlmClient, Message};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherResult {
    city: String,
    temp_c: f64,
}

let req = GenerateTextRequest::builder("gpt-5.5")
    .message(Message::user_text("What is the weather in Tokyo?"))
    .temperature(0.2)
    .build()?;

let result = client.generate_object::<WeatherResult>(req).await?;

println!("{} → {}°C", result.object.city, result.object.temp_c);
```

Providers without native structured-output mode (Anthropic, Google) fall back transparently to forced tool calls. From the caller's perspective there's no difference — you get a `T` either way.

#### Streaming partial objects

For UIs that should render fields as they arrive, `stream_object::<T>()` emits progressively-populated values. Each chunk is repaired and re-deserialised into a partial `T`. Fields not yet emitted by the model stay at their `Default`, so derive `Default` and add `#[serde(default)]`:

```rust
use aquaregia::{GenerateTextRequest, LlmClient, Message, StreamObjectEvent};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default)]
struct WeatherResult {
    city:   String,
    temp_c: f64,
}

let mut stream = client.stream_object::<WeatherResult>(req).await?;
while let Some(event) = stream.next().await {
    match event? {
        StreamObjectEvent::Partial { partial } => {
            println!("partial: {} ({}°C)", partial.city, partial.temp_c);
        }
        StreamObjectEvent::Object { object } => {
            println!("final:   {:?}", object);
        }
    }
}
```

Under the hood is a stack-based JSON repairer that handles truncated strings, unclosed arrays and objects, and escape sequences mid-token. A chunk that splits a field name or value still produces a valid partial — you never have to handle invalid JSON in your code. See `examples/structured_streaming.rs` for a runnable version.

---

### Tools

A tool lets the model call your Rust code mid-conversation. Think of one as a **typed `async fn` the LLM can invoke by name** — Aquaregia builds the JSON Schema from your argument type, marshals the call args back into Rust, and runs your function.

The typed style is the one you'll reach for most:

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

`tool(name)` starts a builder; `.execute(closure)` consumes a `JsonSchema`-derived arg type and finishes the build. The closure returns `Result<serde_json::Value, ToolExecError>` — the model gets back whatever JSON you produce.

If you'd rather hand-write the schema (unusual constraints, no derive available), use `.raw_schema(...)` + `.execute_raw(...)`:

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

Tool names must match `^[a-zA-Z0-9_-]{1,64}$` and be unique within an agent. That's validated at agent build time so an invalid registry never reaches the model.

---

### Agents

An `Agent` is a `generate`-plus-tools `while` loop with hooks: the model thinks → maybe calls tools → you run the tools → results go back to the model → repeat until it stops calling tools (or you cap it). You don't write the loop; you describe its inputs and outputs.

The minimum agent is one tool and a step cap:

```rust
use aquaregia::{Agent, LlmClient};

let client = LlmClient::openai_compatible()
    .base_url("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .build()?;

let agent = Agent::builder(client, "deepseek-v4-pro")
    .instructions("You can call tools before answering.")
    .tools([get_weather])
    .max_steps(4)
    .build()?;

let response = agent.run("Weather in Shanghai?").await?;
println!("{}", response.output_text);
println!("steps={} total={}", response.steps, response.usage_total.total_tokens);
```

`response.output_text` is the final user-visible answer. `response.steps` tells you how many round-trips it took; `response.usage_total` aggregates token counts across every call.

#### Event hooks

Every interesting boundary in the loop emits an event. All hooks are `Fn + Send + Sync`, so you can attach closures directly — useful for logging, metrics, debug UIs:

```rust
let agent = Agent::builder(client, "deepseek-v4-pro")
    .tools([get_weather])
    .on_start(|e|            println!("[start] tools={} max_steps={}", e.tool_count, e.max_steps))
    .on_step_start(|e|       println!("[step:{}] msgs={}", e.step, e.messages.len()))
    .on_tool_call_start(|e|  println!("[tool:{}] {}", e.step, e.tool_call.tool_name))
    .on_tool_call_finish(|e| println!("[tool:{}] {} in {}ms", e.step, e.tool_call.tool_name, e.duration_ms))
    .on_step_finish(|s|      println!("[step:{}] finish={:?}", s.step, s.finish_reason))
    .on_finish(|f|           println!("[done] {} steps, {} total tokens", f.step_count, f.usage_total.total_tokens))
    .build()?;
```

#### Dynamic planning with `prepare_step`

When you need to mutate the next step before it runs — narrow the tool list, swap models, inject a system prompt for that step only — give the agent a `prepare_step` closure:

```rust
use aquaregia::Message;

let agent = Agent::builder(client, "deepseek-v4-pro")
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

The returned `AgentPreparedStep` is what the agent actually sends to the model for this step. Anything you don't change carries over from the previous state.

#### Stopping policies

Three knobs control how the loop ends:

```rust
use aquaregia::ToolErrorPolicy;

let agent = Agent::builder(client, "deepseek-v4-pro")
    .max_steps(8)
    .stop_when(|step| step.tool_calls.is_empty() && !step.output_text.is_empty())
    .tool_error_policy(ToolErrorPolicy::ContinueAsToolResult)
    .build()?;
```

- **`max_steps`** — hard cap. Hitting it returns `ErrorCode::MaxStepsExceeded` instead of silently truncating.
- **`stop_when`** — predicate checked after every step. Useful when you have a sharper definition of "done" than "no more tool calls".
- **`tool_error_policy`** — what happens when a tool throws.
  - `ContinueAsToolResult` (default) — schema-validation failures, timeouts, and panics become `{ "error": "..." }` results so the model can recover on the next step.
  - `FailFast` — surface as `ErrorCode::ToolExecutionFailed` / `InvalidToolArgs` immediately.

#### Multi-turn conversations

`AgentResponse.transcript` is a complete `Vec<Message>` (system + user + assistant + tool results) you can feed straight back in for the next turn — no manual reconstruction:

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

### Multimodal

Vision models take images alongside text. Aquaregia hides the per-provider differences behind one `ImagePart` type — you say "URL" or "bytes" once, and the right thing happens on the wire.

The shortest path is `Message::user_image_url` or `Message::user_image_bytes`. For mixed content or multiple images per message, build a `Message` with explicit content parts:

```rust
use aquaregia::{
    ContentPart, GenerateTextRequest, ImagePart, LlmClient, MediaData, Message, MessageRole,
};

let client = LlmClient::anthropic().api_key(std::env::var("ANTHROPIC_API_KEY")?).build()?;

let out = client
    .generate(
        GenerateTextRequest::builder("claude-sonnet-4-6")
            .message(Message::new(
                MessageRole::User,
                vec![
                    ContentPart::Text("What's in this image?".into()),
                    ContentPart::Image(ImagePart {
                        data: MediaData::Url(
                            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg".into(),
                        ),
                        media_type: None,
                        provider_metadata: None,
                    }),
                ],
            )?)
            .build()?,
    )
    .await?;
```

The convenience constructors:

| Constructor                                                              | Use case                                  |
| ------------------------------------------------------------------------ | ----------------------------------------- |
| `Message::user_image_url(url)`                                           | One image from a URL                      |
| `Message::user_image_bytes(bytes, mime)`                                 | One image from raw bytes (auto base64)    |
| `Message::new(MessageRole::User, vec![Text, Image, …])`                  | Mixed content / multiple images           |
| `ContentPart::Image(ImagePart { data, media_type, provider_metadata })`  | Full control + provider-specific hints    |

Each provider receives whatever native format it wants:

| Provider            | URL                          | Base64 / Bytes                          |
| ------------------- | ---------------------------- | --------------------------------------- |
| Anthropic           | `source.type: url`           | `source.type: base64`                   |
| OpenAI / Compatible | `image_url` with remote URL  | `image_url` with `data:<mime>;base64,…` |
| Google              | `fileData.fileUri`           | `inlineData.data`                       |

---

## Production

Code that calls an LLM in production has to deal with three boring-but-essential concerns: stopping work that's no longer wanted, surviving transient failures, and reporting errors usefully.

### Cancellation

Every request and agent run accepts a `CancellationToken`. Cancel the token and the operation stops at the next safe boundary — Aquaregia checks before every HTTP send (via `tokio::select!`, zero overhead on the happy path), after every SSE chunk in streaming responses, and at the top of every agent step.

```rust
use aquaregia::{CancellationToken, ErrorCode, GenerateTextRequest};
use std::time::Duration;

let token = CancellationToken::new();
let bg = token.clone();
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_millis(200)).await;
    bg.cancel();
});

let req = GenerateTextRequest::builder("deepseek-v4-pro")
    .user_prompt("Write a 10,000-word essay.")
    .cancellation_token(token)
    .build()?;

match client.generate(req).await {
    Err(e) if e.code == ErrorCode::Cancelled => println!("cancelled"),
    other                                    => println!("{other:?}"),
}
```

Agents can take the token at builder time so every `agent.run(...)` call uses the same one:

```rust
let agent = Agent::builder(client, "deepseek-v4-pro")
    .cancellation_token(token.clone())
    .build()?;
```

### Retries & timeouts

Two knobs, set on the client:

```rust
let client = LlmClient::openai()
    .api_key(api_key)
    .max_retries(3)                          // default: 0
    .timeout(Duration::from_secs(45))
    .build()?;
```

Aquaregia retries on the classes that are actually transient: `RateLimited`, `ProviderServerError`, `Transport`, `Timeout`. Backoff is exponential with jitter. If the provider returns a `Retry-After` header, Aquaregia honours it instead of using its own delay.

Every `Error` also carries `retryable: bool` flagging the same classification, so if you need a custom retry / circuit-breaker layer, you don't have to redo the taxonomy.

### Error handling

`Error` is a structured payload, not just a string. Match on `code: ErrorCode` for control flow; read the rest for diagnostics:

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

- `code: ErrorCode` — a normalised variant (the discriminator you actually want to switch on)
- `provider`, `status`, `request_id`, `raw_body`, `retry_after_secs` — diagnostic fields for logs and support tickets
- `retryable: bool` — `true` iff Aquaregia's built-in retry would engage

---

## Integration

Aquaregia keeps web framework adapters out of the crate on purpose — `TextStream` is a `Stream` of `StreamEvent` and adapts cleanly to whatever transport your app already uses.

Here's the Axum SSE pattern. Every `StreamEvent` becomes a named SSE event your frontend can switch on:

```rust
use aquaregia::{BoundClient, GenerateTextRequest, StreamEvent, TextStream};
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

fn to_axum_sse(stream: TextStream) -> impl IntoResponse {
    Sse::new(stream.map(|item| {
        let event = match item {
            Ok(StreamEvent::ReasoningStarted { .. })       => Event::default().event("reasoning_start").data("{}"),
            Ok(StreamEvent::ReasoningDelta { text, .. })   => Event::default().event("reasoning_token").data(text),
            Ok(StreamEvent::ReasoningDone { .. })          => Event::default().event("reasoning_end").data("{}"),
            Ok(StreamEvent::TextDelta { text })            => Event::default().event("token").data(text),
            Ok(StreamEvent::ToolCallReady { .. })          => Event::default().event("tool_call").data("{}"),
            Ok(StreamEvent::Usage { .. })                  => Event::default().event("usage").data("{}"),
            Ok(StreamEvent::Done)                          => Event::default().event("done").data("{}"),
            Err(err)                                       => Event::default().event("error").data(err.message),
        };
        Ok::<Event, Infallible>(event)
    }))
}

async fn chat(State(client): State<Arc<BoundClient>>) -> impl IntoResponse {
    let stream = client
        .stream(GenerateTextRequest::from_user_prompt("deepseek-v4-pro", "Hello."))
        .await
        .unwrap();
    to_axum_sse(stream)
}

let app: Router = Router::new()
    .route("/chat", get(chat))
    .with_state(Arc::new(client));
```

For Actix, Warp, or your own gRPC layer the recipe is identical — map each `StreamEvent` variant to your wire format. The example keeps non-text payloads minimal; in a real app you'd serialise tool calls, usage, and reasoning metadata into whatever shape your frontend already speaks.

---

## Reference

Lookup tables for when you know roughly what you want and need the exact name.

### Capability matrix

| Capability                       | OpenAI | Anthropic | Google | OpenAI-Compatible |
| -------------------------------- | :----: | :-------: | :----: | :---------------: |
| Custom `base_url`                |   ✓    |     ✓     |   ✓    |         ✓         |
| Custom headers / query / path    |        |           |        |         ✓         |
| `api_version` (header)           |        |     ✓     |        |                   |
| Structured output                |   ✓    |     ✓     |   ✓    |         ✓         |
| Tool-call streaming              |   ✓    |     ✓     |   ✓    |         ✓         |
| Cache-token split in `Usage`     |   ✓    |     ✓     |   ✓    |   if reported     |

### `Usage` fields

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

`Usage` implements `Add` and `AddAssign`, so totalling tokens across agent steps is a one-liner. `AgentResponse.usage_total` is already aggregated for you.

### Examples

```bash
DEEPSEEK_API_KEY=... cargo run --example basic_generate
```

| Example                       | Focus                                                       |
| ----------------------------- | ----------------------------------------------------------- |
| `basic_generate`              | One-shot `generate` + usage reading                         |
| `basic_stream`                | `stream` + `StreamEvent` handling                           |
| `structured_streaming`        | `stream_object::<T>()` + progressive `Partial` events       |
| `agent_minimal`               | `Agent::builder` with one typed tool                        |
| `tools_max_steps`             | Multi-tool loop with `max_steps` and sampling caps          |
| `prepare_hooks`               | `prepare_step`, `on_step_finish`                            |
| `openai_compatible_custom`    | Custom headers / query params / chat path                   |
| `mini_claude_code`            | TUI code agent — `bash` / `read` / `write` / `edit` tools   |
| `multimodal_image`            | `Message::new` with mixed text + image parts + Anthropic vision |

Set `DEEPSEEK_API_KEY` for most examples; `ANTHROPIC_API_KEY` for `multimodal_image`. See [`examples/README.md`](./examples/README.md) for full descriptions.

### API reference

Full type signatures, every public item, every variant — on [docs.rs/aquaregia](https://docs.rs/aquaregia).

---

## Development

```bash
cargo fmt
cargo test
cargo check --examples
cargo clippy -- -D warnings
```

---

## AI-Assisted Development

AI-assisted development is welcome in this project, but the contributor remains responsible for the final result. If code, tests, docs, or API changes are proposed with AI help, the person submitting them is still expected to understand, review, and validate them.

This repository also keeps agent-facing guidance principle-based on purpose. Files such as `AGENTS.md` and `CLAUDE.md` should describe durable constraints and decision rules, not long checklists of internal APIs that drift away from the code.

---

## Contributing & License

Contributions are welcome. For behaviour changes, include integration tests (happy path + error mapping + tool/stream flows where relevant).

- [Contributing Guide](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Security Policy](./SECURITY.md)
- Licensed under the [MIT License](./LICENSE)
