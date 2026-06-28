<div align="center">

<img src="https://raw.githubusercontent.com/minorcell/aquaregia/dev/docs/public/brand/aquaregia-logo.svg" alt="Aquaregia logo" width="96" height="96">

# Aquaregia

**A lightweight agent SDK for Rust.**

[![Crates.io](https://img.shields.io/crates/v/aquaregia.svg)](https://crates.io/crates/aquaregia)
[![Docs.rs](https://docs.rs/aquaregia/badge.svg)](https://docs.rs/aquaregia)
[![License: MIT](https://img.shields.io/crates/l/aquaregia.svg)](./LICENSE)
[![Downloads](https://img.shields.io/crates/d/aquaregia.svg)](https://crates.io/crates/aquaregia)

[API Docs](https://docs.rs/aquaregia) · [Examples](./examples/README.md) 

</div>

---

## Introduction

### Why Aquaregia

You want an agent: a model that calls your tools, loops until the job is done, and streams its progress as it goes. Writing that loop by hand — dispatching tool calls, feeding results back, tracking steps, handling the model that won't stop — is the part nobody enjoys.

Aquaregia gives you that loop in a few lines. You describe the tools and the stopping rule; the `Agent` runs the think → call → observe → repeat cycle for you, with hooks at every boundary. And because the agent is built on one unified API across OpenAI, Anthropic, Google, and any OpenAI-compatible endpoint, the same agent runs on whichever provider you point it at — swap a constructor, keep the code.

It's not a framework, not a gateway, not a microservice. It's a Rust library you `cargo add` and call directly.

### At a glance

| Capability                        | What you get                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------- |
| **Tool-using agents**             | Multi-step loop with `prepare_step` hooks, `max_steps`, `stop_when`, error policies      |
| **Typed tools**                   | A tool is a typed `async fn` — `schemars` derives the schema, args marshal back to Rust  |
| **Runs on any provider**          | One agent, same code, over OpenAI · Anthropic · Google · OpenAI-compatible gateways      |
| **Streaming & non-streaming**     | Same builder feeds `generate` or `stream`, with consistent `StreamEvent`s                |
| **Structured output**             | `generate_object::<T>()` and `stream_object::<T>()` with `schemars`-derived schemas      |
| **Reasoning content**             | First-class reasoning extraction, streaming reasoning deltas, reasoning-token usage      |
| **Multimodal input**              | Send images, PDFs, and other files by URL, base64, or raw bytes — one `FilePart` API     |
| **Text embeddings**               | Generate vector embeddings with `embed()` — OpenAI, Google, OpenAI-compatible support    |
| **Cancellation & retries**        | `CancellationToken` checked everywhere; exponential backoff with `Retry-After` honored   |

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
use aquaregia::providers::openai;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = openai::Client::from_env()?
        .agent("gpt-5.5")
        .build()?;

    let response = agent
        .prompt("Explain Rust ownership in 3 bullet points.")
        .await?;

    println!("{response}");
    Ok(())
}
```

Two moving parts: a **provider client** (auth + transport) and a **model-bound agent**. `.prompt()` is the common string-in/string-out path; reach for `.run()` or `.generate()` when you need richer metadata or explicit requests.

### First streaming call

For a single streaming model call, use the provider client directly:

```rust
use aquaregia::StreamEvent;
use futures_util::StreamExt;

let mut stream = client
    .stream(ChatRequest::from_prompt(
        "gpt-5.5",
        "Write a haiku about the borrow checker.",
    ))
    .await?;

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta { text } => print!("{text}"),
        StreamEvent::Done { .. }        => break,
        _                               => {}
    }
}
```

Use `.prompt()` for ordinary text, and explicit `ChatRequest` values when you need lower-level control. The rest of this guide unpacks each piece.

---

## Essentials

### Provider clients

Provider modules are the constructor entry-point for all LLM operations. Use `from_env()` for providers with a standard endpoint, or `.builder()` when you need explicit settings.

```rust
use aquaregia::providers::openai;
use std::time::Duration;

let client = openai::Client::builder()
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .base_url("https://api.openai.com")          // optional: custom upstream
    .timeout(Duration::from_secs(60))            // per-request HTTP timeout
    .max_retries(3)                              // transient-failure retries
    .default_max_steps(8)                        // default agent step cap
    .user_agent("my-app/1.0")
    .build()?;
```

Switching providers is just a different module — every method you'll see below works the same way on whichever provider client you end up with.

| Provider          | Constructor                                                         |
| ----------------- | ------------------------------------------------------------------- |
| OpenAI            | `openai::Client::from_env()` or `openai::Client::builder().api_key(api_key)` |
| Anthropic         | `anthropic::Client::from_env()` or `anthropic::Client::builder().api_key(api_key)` |
| Google            | `google::Client::from_env()` or `google::Client::builder().api_key(api_key)` |
| OpenAI-compatible | `openai_compatible::Client::from_env()` or `openai_compatible::Client::builder().base_url(url).api_key(token)` |

#### Going deeper: OpenAI-compatible endpoints

If your provider speaks the OpenAI chat-completions wire format but lives at a different URL — Together, Groq, your own gateway — `openai_compatible::Client::builder()` lets you bolt on custom headers, query params, and even a different chat path:

```rust
use aquaregia::providers::openai_compatible;

let client = openai_compatible::Client::builder()
    .base_url("https://api.example.com")
    .api_key_from_env("OPENAI_COMPATIBLE_API_KEY")
    .header("x-trace-source", "aquaregia")
    .query_param("source", "sdk")
    .chat_completions_path("/v1/chat/completions")
    .build()?;
```

If the endpoint doesn't want any `Authorization` header at all, call `.no_api_key()` instead of `.api_key(...)`.

---

### Generating text

`agent.prompt(...)` is the common path when you only need visible text:

```rust
let response = client
    .agent("gpt-5.5")
    .build()?
    .prompt("Summarize Rust's borrow checker for a Go developer.")
    .await?;

println!("{response}");
```

`generate` is the explicit request path: one request, one response, all content in `output_text`.

```rust
let response = client
    .generate(ChatRequest::from_prompt(
        "gpt-5.5",
        "Summarize Rust's borrow checker for a Go developer.",
    ))
    .await?;

println!("{}", response.output_text);
println!("finish: {:?}", response.finish_reason);
```

`from_prompt(model, prompt)` is the one-line form. When you need more — system prompts, sampling controls, tools — reach for the builder:

```rust
use aquaregia::{ChatRequest, Message};

let req = ChatRequest::builder("gpt-5.5")
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
        StreamEvent::Done { .. }                 => break,
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
| `Done`               | `finish_reason: FinishReason`          | Final event of every stream                    |

Once `Done` fires the stream is finished — don't poll after. Use `finish_reason` to distinguish natural stops from token limits, tool calls, or provider-specific endings.

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
use aquaregia::{ChatRequest, Message};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherResult {
    city: String,
    temp_c: f64,
}

let req = ChatRequest::builder("gpt-5.5")
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
use aquaregia::types::StreamObjectEvent;
use aquaregia::{ChatRequest, Message};
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
            json!({ "city": args.city, "temp_c": 23, "condition": "sunny" })
        })
}
```

`tool(name)` starts a builder; `.execute(closure)` consumes a `JsonSchema`-derived arg type and returns any `Serialize` value. Use `.try_execute(...)` when the tool can fail with `ToolExecError`.

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
use aquaregia::providers::openai;

let client = openai::Client::from_env()?;

let agent = client
    .agent("gpt-5.5")
    .instructions("You can call tools before answering.")
    .tool(get_weather)
    .max_steps(4)
    .build()?;

let response = agent.prompt("Weather in Shanghai?").await?;
println!("{response}");
```

`prompt` returns the final user-visible answer directly. Use `run` when you need the full `AgentOutput`: `steps` tells you how many round-trips it took, and `usage_total` aggregates token counts across every call.

#### Streaming agent runs

Use `agent.stream(...)` when the consumer needs the whole execution as it happens: model deltas, tool calls, tool results, step snapshots, and the final output.

```rust
use aquaregia::{AgentStreamEvent, StreamEvent};
use futures_util::StreamExt;

let mut stream = agent.stream("Weather in Shanghai?").await?;

while let Some(event) = stream.next().await {
    match event? {
        AgentStreamEvent::Model {
            event: StreamEvent::TextDelta { text },
            ..
        } => print!("{text}"),
        AgentStreamEvent::ToolCallStart { event } => {
            eprintln!("[tool] {}", event.tool_call.tool_name);
        }
        AgentStreamEvent::Done { output } => {
            eprintln!("[done] steps={}", output.steps);
            break;
        }
        _ => {}
    }
}
```

#### Event hooks

Every interesting boundary in the loop emits an event. All hooks are `Fn + Send + Sync`, so you can attach closures directly — useful for logging, metrics, debug UIs:

```rust
let agent = client.agent("gpt-5.5")
    .tool(get_weather)
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

let agent = client.agent("gpt-5.5")
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

let agent = client.agent("gpt-5.5")
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

`AgentOutput.transcript` is a complete `Vec<Message>` (system + user + assistant + tool results) you can feed straight back in for the next turn — no manual reconstruction:

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

Images, PDFs, and other binary inputs all ride the same `FilePart` type, distinguished by an IANA `media_type`. The shortest path is `Message::user_file_url` / `Message::user_file_bytes`; for mixed content or multiple files per message, build a `Message` with explicit content parts:

```rust
use aquaregia::{
    ChatRequest, ContentPart, FilePart, MediaData, Message, MessageRole, TextPart,
};
use aquaregia::providers::openai;

let client = openai::Client::from_env()?;

let out = client
    .generate(
        ChatRequest::builder("gpt-5.5")
            .message(Message::new(
                MessageRole::User,
                vec![
                    ContentPart::Text(TextPart::new("What's in this image?")),
                    ContentPart::File(FilePart::new(
                        MediaData::Url(
                            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg".into(),
                        ),
                        "image/jpeg",
                    )),
                ],
            ))
            .build()?,
    )
    .await?;
```

The convenience constructors:

| Constructor                                                                                | Use case                                                |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| `Message::user_file_url(url, media_type)`                                                  | One file from a URL                                     |
| `Message::user_file_bytes(bytes, media_type)`                                              | One file from raw bytes (auto base64)                   |
| `Message::new(MessageRole::User, vec![Text, File, …])`                                     | Mixed content / multiple files                          |
| `FilePart::new(data, media_type).with_filename(...).with_provider_options(...)`            | Full control: filename hint, per-block provider options |

`media_type` is mandatory — IANA-style strings such as `image/jpeg`, `image/png`, `application/pdf`. Adapters dispatch on it:

| `media_type`         | Anthropic                | OpenAI Responses                          | OpenAI-compatible | Google                       |
| -------------------- | ------------------------ | ----------------------------------------- | ----------------- | ---------------------------- |
| `image/*`            | `image` block            | `input_image` (`image_url` data URL)      | `image_url`       | `inlineData` / `fileData`    |
| `application/pdf`    | `document` block         | `input_file` (with optional `filename`)   | ✗ rejected locally | `inlineData` / `fileData`    |
| anything else        | ✗ rejected locally       | ✗ rejected locally                        | ✗ rejected locally | passed through (mimeType)    |

Unsupported `media_type` surfaces as a local `ErrorCode::InvalidRequest` before any network round-trip — no silent fallback. Google receives the `media_type` verbatim and supports the broadest set (image / PDF / audio / video subtypes) at the API level; the others reject what they cannot represent.

---

### Embeddings

Generate vector embeddings for text using provider embedding models. Embeddings convert text into dense vector representations useful for semantic search, clustering, and similarity comparisons.

```rust
use aquaregia::embed::EmbedRequest;
use aquaregia::providers::openai;

let client = openai::Client::from_env()?;

let response = client.embed(
    EmbedRequest::new("text-embedding-3-small", vec!["Your text here"])
).await?;

println!("Dimension: {}", response.embeddings[0].len());
println!("Tokens: {}", response.usage.tokens);
```

**Provider support:**

| Provider          | Embedding API | Models                                    |
| ----------------- | :-----------: | ----------------------------------------- |
| OpenAI            | ✅            | `text-embedding-3-small`, `text-embedding-3-large` |
| Anthropic         | ❌            | —                                         |
| Google            | ✅            | `text-embedding-004`                      |
| OpenAI-compatible | ✅            | Depends on provider                            |

Anthropic does not provide an embedding API. For Anthropic-based applications, use a third-party embedding provider through the `openai_compatible` adapter (e.g., Voyage AI, Cohere).

**Batch processing:**

```rust
let texts = vec![
    "First document",
    "Second document",
    "Third document",
];

let response = client.embed(
    EmbedRequest::new("text-embedding-3-small", texts)
).await?;

// response.embeddings[i] corresponds to texts[i]
for (i, embedding) in response.embeddings.iter().enumerate() {
    println!("Text {}: {} dimensions", i, embedding.len());
}
```

**Provider-specific options:**

Use `provider_options` to access provider-specific features like dimension reduction:

```rust
use serde_json::json;

let response = client.embed(
    EmbedRequest::builder("text-embedding-3-large")
        .values(vec!["Some text"])
        .provider_options(json!({
            "openai": { "dimensions": 256 }  // Reduce from 3072 to 256
        }))
        .build()?
).await?;
```

See `examples/basic_embed.rs` and `examples/openai_embed.rs` for complete examples including similarity calculations.

---

The core request type sticks to the lowest common denominator — model, messages, sampling, tools. But every provider ships knobs that don't generalise: Anthropic's thinking budget, Google's safety thresholds, parameters that land on one provider and nowhere else. Rather than bloat the core type with fields that mean nothing to three out of four providers, Aquaregia gives you an escape hatch: `provider_options`.

You pass a JSON object keyed by provider slug. The core never parses it — each adapter picks out its own key and merges those fields into the request body it sends. Anything the provider's API accepts, you can set:

```rust
use aquaregia::ChatRequest;
use serde_json::json;

let req = ChatRequest::builder("gpt-5.5")
    .user("Prove the infinitude of primes.")
    .provider_options(json!({
        "openai": {
            "reasoning": { "effort": "medium" }
        }
    }))
    .build()?;
```

The slug (`"openai"`) is what routes the options, so a single request can carry settings for several providers — only the matching adapter reads its own block, the rest is ignored. That means you can keep one `provider_options` value and reuse it as you A/B across providers:

```rust
.provider_options(json!({
    "anthropic": { "thinking": { "type": "enabled", "budget_tokens": 10000 } },
    "google":    { "safetySettings": [
        { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH" }
    ]}
}))
```

| Provider slug       | Merged into                          |
| ------------------- | ------------------------------------ |
| `anthropic`         | Messages API request body (top level) |
| `openai`            | Responses API request body (top level) |
| `google`            | `generateContent` request body (top level) |
| `openai-compatible` | Chat Completions request body (top level) |

Because the merge is opaque, the responsibility is yours: keys you set override what the adapter computed, and a malformed value surfaces as a provider-side `InvalidRequest` rather than a compile error. This is the deliberate trade — full reach into provider features, no waiting on the core type to add a field.

The same setter is on the agent builder returned by `client.agent(...)` — once configured, the options ride every step of the tool loop:

```rust
let agent = client.agent("gpt-5.5")
    .tools([weather])
    .provider_options(json!({
        "openai": { "reasoning": { "effort": "medium" } }
    }))
    .build()?;
```

#### Per-message and per-content-block options

Some provider features attach to a single message or even a single content block — Anthropic's `cache_control` breakpoint is the canonical example: it tells the API where to start a prompt cache, and *where* you put it is the whole point. The same `provider_options` shape works at message and block level:

```rust
use aquaregia::{ContentPart, Message, MessageRole, TextPart};

let cached_system = TextPart::new(LONG_SYSTEM_PROMPT).with_provider_options(json!({
    "anthropic": { "cache_control": { "type": "ephemeral" } }
}));

let system = Message::new(
    MessageRole::System,
    vec![ContentPart::Text(cached_system)],
);
```

`Message::with_provider_options(...)` is the message-level analogue, merged into the message object itself rather than a block within it.

The same opaqueness contract holds at every level — adapters read their slug, merge what they find, and never invent semantics. For OpenAI's Responses API and the openai-compatible Chat Completions adapter, attaching `provider_options` to a text block also forces the `content` field into the typed-array form (a bare string can't carry per-block fields).

| Setter                          | Merged into                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `ChatRequest::builder().provider_options(…)` | Request body, top level                                      |
| `client.agent(...).provider_options(…)`              | Every per-step request body, top level                       |
| `Message::with_provider_options(…)`                  | The corresponding message object inside `messages` / `input` |
| `TextPart::with_provider_options(…)`                 | The corresponding text content block                         |

#### Provider-native capabilities

Anthropic's `web_search_20250305`, OpenAI's `web_search` / `file_search` / `code_interpreter`, Google's `googleSearch` and friends are server-side provider capabilities. They go into the request body's provider-specific fields through `provider_options`:

```rust
let req = ChatRequest::builder("gpt-5.5")
    .user("What did Rust 1.85 ship?")
    .provider_options(json!({
        "openai": {
            "tools": [{
                "type": "web_search"
            }]
        }
    }))
    .build()?;
```

One thing to know about the merge: top-level keys **overwrite** what the adapter computed for that key. If a provider uses a field named `tools` for server-side capabilities, put the complete provider-native array in `provider_options.<slug>.tools`. See `examples/anthropic_web_search.rs` for a runnable provider-native call.

---

## Production

Code that calls an LLM in production has to deal with three boring-but-essential concerns: stopping work that's no longer wanted, surviving transient failures, and reporting errors usefully.

### Cancellation

Every request and agent run accepts a `CancellationToken`. Cancel the token and the operation stops at the next safe boundary — Aquaregia checks before every HTTP send (via `tokio::select!`, zero overhead on the happy path), after every SSE chunk in streaming responses, and at the top of every agent step.

```rust
use aquaregia::{ChatRequest, ErrorCode};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

let token = CancellationToken::new();
let bg = token.clone();
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_millis(200)).await;
    bg.cancel();
});

let req = ChatRequest::builder("gpt-5.5")
    .user("Write a 10,000-word essay.")
    .cancellation_token(token)
    .build()?;

match client.generate(req).await {
    Err(e) if e.code == ErrorCode::Cancelled => println!("cancelled"),
    other                                    => println!("{other:?}"),
}
```

Agents can take the token at builder time so every `agent.run(...)` call uses the same one:

```rust
let agent = client.agent("gpt-5.5")
    .cancellation_token(token.clone())
    .build()?;
```

### Retries & timeouts

Two knobs, set on the client:

```rust
let client = aquaregia::providers::openai::Client::builder()
    .api_key(api_key)
    .max_retries(3)                          // default: 3
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
use aquaregia::{providers::openai, ChatRequest, StreamEvent, TextStream};
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
            Ok(StreamEvent::Done { finish_reason })        => Event::default()
                .event("done")
                .data(format!("{finish_reason:?}")),
            Err(err)                                       => Event::default().event("error").data(err.message),
        };
        Ok::<Event, Infallible>(event)
    }))
}

async fn chat(State(client): State<Arc<openai::Client>>) -> impl IntoResponse {
    let stream = client
        .stream(ChatRequest::from_prompt("gpt-5.5", "Hello."))
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
| `provider_options` passthrough   |   ✓    |     ✓     |   ✓    |         ✓         |
| Embeddings                       |   ✓    |           |   ✓    |         ✓         |

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

`Usage` implements `Add` and `AddAssign`, so totalling tokens across agent steps is a one-liner. `AgentOutput.usage_total` is already aggregated for you.

### Examples

```bash
OPENAI_API_KEY=... cargo run --example basic_generate
```

| Example                       | Focus                                                       |
| ----------------------------- | ----------------------------------------------------------- |
| `basic_generate`              | One-shot `generate` + usage reading                         |
| `basic_stream`                | `stream` + `StreamEvent` handling                           |
| `basic_embed`                 | Text embeddings with batch processing and similarity        |
| `openai_embed`                | OpenAI embeddings with dimension reduction                  |
| `structured_streaming`        | `stream_object::<T>()` + progressive `Partial` events       |
| `agent_minimal`               | `client.agent(model)` with one typed tool                    |
| `tools_max_steps`             | Multi-tool loop with `max_steps` and sampling caps          |
| `prepare_hooks`               | `prepare_step`, `on_step_finish`                            |
| `openai_compatible_custom`    | Custom headers / query params / chat path                   |
| `mini_claude_code`            | TUI code agent — `bash` / `read` / `write` / `edit` tools   |
| `multimodal_image`            | `Message::new` with mixed text + image parts + OpenAI vision    |
| `multimodal_pdf`              | Send a local PDF to OpenAI (`FilePart` + `application/pdf`)      |

Set `OPENAI_API_KEY` for most examples; `ANTHROPIC_API_KEY` for `anthropic_*` examples. The PDF demo also needs `PDF_PATH`. See [`examples/README.md`](./examples/README.md) for full descriptions.

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
