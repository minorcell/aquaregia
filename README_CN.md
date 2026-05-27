<div align="center">

# Aquaregia

**Rust 的通用 AI 层。**

[![Crates.io](https://img.shields.io/crates/v/aquaregia.svg)](https://crates.io/crates/aquaregia)
[![Docs.rs](https://docs.rs/aquaregia/badge.svg)](https://docs.rs/aquaregia)
[![License: MIT](https://img.shields.io/crates/l/aquaregia.svg)](./LICENSE)
[![Downloads](https://img.shields.io/crates/d/aquaregia.svg)](https://crates.io/crates/aquaregia)

[API 文档](https://docs.rs/aquaregia) · [示例指南](./examples/README.md) · [English](./README.md)

</div>

---

## Introduction

### 为什么是 Aquaregia

你想从 Rust 调用 LLM。半年后你大概率还会想调用_另一家_ LLM——而且不希望为此重写你的 prompt、tool 定义和流式代码。

Aquaregia 是一个 crate，让你用同一套 API 调用 OpenAI、Anthropic、Google 和任意 OpenAI-compatible 端点。换一个构造器，相同的 `generate`、`stream`、`generate_object`、`Agent` 继续工作。流式、结构化输出、推理内容、多模态输入、agent 循环、重试、取消——一切都归一在同一个类型背后。

它不是网关、不是代理、不是微服务。它就是一个 `cargo add` 进来直接调用的 Rust 库。

### 一览

| 能力                              | 你能拿到什么                                                                              |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| **一套 API，四个 provider**       | OpenAI · Anthropic · Google · OpenAI-compatible（DeepSeek、Together、Groq 等）             |
| **流式与非流式**                  | 同一个 builder 喂给 `generate` 或 `stream`，事件类型统一                                  |
| **结构化输出**                    | `generate_object::<T>()` 与 `stream_object::<T>()`，schema 由 `schemars` 自动推导          |
| **推理内容**                      | 一等公民式的 reasoning 提取、流式 reasoning delta、reasoning-token usage                   |
| **工具型 Agent**                  | 多步循环 + `prepare_step` 钩子 + `max_steps` + `stop_when` + 错误策略                      |
| **多模态视觉**                    | URL / base64 / 原始字节都支持——一套 `ImagePart` API 通吃所有 provider                       |
| **取消与重试**                    | `CancellationToken` 全程检查；transient 错误指数退避，遵循 `Retry-After`                   |

### 什么时候不该用

如果你的项目只会接一家 provider，也不在乎可移植性，那家 vendor 的原生 SDK 会给你更地道的接口。Aquaregia 真正发挥价值的场景：**A/B 多个 provider**、**让用户在运行时选模型**、**为未来 LLM 生态变化留余地**。如果上面这些都不沾——你不是目标受众，没关系。

---

## 快速上手

### 安装

```bash
cargo add aquaregia
```

你的项目还需要一个 Tokio runtime——Aquaregia 全程 async。

### Hello world

调到一个真实模型响应的最短路径：

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

三个动件：一个 **client**（provider + 认证 + 传输），一个 **request**（model + messages），一次 `.generate()` 调用。本文档剩下的所有内容都是这三件事的细化。

### 第一次流式调用

同一个 client，把 `.generate` 换成 `.stream`：

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

你已经看到了整个 shape——一个 builder、一次调用、要么是返回值要么是事件循环。剩下的章节把每一块拆开。

---

## 核心

### Client

`LlmClient` 只是构造器的命名空间。每个构造器返回一个用 provider 设置类型参数化的 `ClientBuilder<S>`，最终 `.build()` 出一个可复用的 `BoundClient`。记住这个 shape：**构造器 → builder → bound client**——无论选哪家 provider 都是这个套路。

```rust
use std::time::Duration;

let client = LlmClient::openai()
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .base_url("https://api.openai.com")          // 可选：自定义上游
    .timeout(Duration::from_secs(60))            // 单次请求的 HTTP 超时
    .max_retries(3)                              // transient 失败重试次数
    .default_max_steps(8)                        // Agent 默认步数上限
    .user_agent("my-app/1.0")
    .build()?;
```

切换 provider 只是换个构造器——下面所有方法在任何 `BoundClient` 上的用法都一样。

| Provider          | 构造器                                                                |
| ----------------- | -------------------------------------------------------------------- |
| OpenAI            | `LlmClient::openai().api_key(api_key)`                               |
| Anthropic         | `LlmClient::anthropic().api_key(api_key).api_version("2023-06-01")`  |
| Google            | `LlmClient::google().api_key(api_key)`                               |
| OpenAI-compatible | `LlmClient::openai_compatible().base_url(url).api_key(token)`        |

#### 深一层：OpenAI-compatible 端点

如果你的 provider 说 OpenAI chat-completions 协议但住在另一个 URL——DeepSeek、Together、Groq、你自家网关——`openai_compatible()` 让你叠加自定义 header、query 参数、甚至换 chat path：

```rust
let client = LlmClient::openai_compatible()
    .base_url("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .header("x-trace-source", "aquaregia")
    .query_param("source", "sdk")
    .chat_completions_path("/v1/chat/completions")
    .build()?;
```

如果端点根本不要 `Authorization` 头，用 `.no_api_key()` 替换 `.api_key(...)`。

---

### 文本生成

`generate` 是主力：一次请求一次响应，所有内容塞在 `output_text` 里。

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

`from_user_prompt(model, prompt)` 是一行式的。当你需要更多——系统提示词、采样控制、工具——切到 builder：

```rust
use aquaregia::{GenerateTextRequest, Message};

let req = GenerateTextRequest::builder("deepseek-v4-pro")
    .message(Message::system_text("You are concise."))
    .message(Message::user_text("Write a release note."))
    .temperature(0.2)
    .max_output_tokens(300)
    .build()?;
```

model 参数就是一个字符串——Aquaregia 不试图枚举模型 ID，因为每家 provider 几乎每隔几周都会出新型号。直接传你的 provider 文档说当前可用的 ID；如果 provider 拒收，你会拿到一个干净的 `InvalidRequest` 错误。

---

### 流式响应

当消费方是 UI 或终端时，你希望 token 边到边显示，而不是等整段回复落地。把 `generate` 换成 `stream`，然后消费 `StreamEvent`：

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

事件流是模型能发出的一切事件的全集。你通常只匹配几种，剩下 `_ => {}` 即可。全部菜单如下：

| 事件                 | 字段                                    | 何时触发                                       |
| -------------------- | --------------------------------------- | --------------------------------------------- |
| `ReasoningStarted`   | `block_id`、`provider_metadata`         | 一个 reasoning 块开始（Anthropic / Google）   |
| `ReasoningDelta`     | `block_id`、`text`、`provider_metadata` | 每个 thought token                            |
| `ReasoningDone`      | `block_id`、`provider_metadata`         | 一个 reasoning 块结束                         |
| `TextDelta`          | `text`                                  | 每个可见回答 token                            |
| `ToolCallReady`      | `call: ToolCall`                        | 模型完成一次 tool call 的组装                  |
| `Usage`              | `usage: Usage`                          | provider 回报 token 计数                       |
| `Done`               | —                                       | 每个 stream 的最后一个事件                     |

`Done` 一旦触发，stream 就结束了——别再 poll。

---

### Reasoning

Reasoning 是模型的"出声思考"——独立于可见回答，被 Anthropic Extended Thinking、Google `thoughts`、OpenAI 的 reasoning-token 账目作为单独内容块暴露。Aquaregia 用统一形式呈现：

```rust
let out = client.generate(req).await?;

println!("answer:     {}", out.output_text);
println!("thinking:   {}", out.reasoning_text);
println!("rsn-tokens: {}", out.usage.reasoning_tokens);

for part in &out.reasoning_parts {
    println!("[block] {}", part.text);
    // part.provider_metadata 载入 signature 块（Anthropic）、
    // thought signature（Google）以及其他 provider 私有附加。
}
```

在 streaming 中，可见文本前会先看到每个 reasoning 块的 `ReasoningStarted` → `ReasoningDelta`(s) → `ReasoningDone`。token usage 的拆分来自各 provider 填写的字段：

| Provider                   | Reasoning-token 字段映射                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| OpenAI / OpenAI-compatible | `prompt_tokens_details.cached_tokens` + `completion_tokens_details.reasoning_tokens`        |
| Anthropic                  | `cache_read_input_tokens` / `cache_creation_input_tokens`；reasoning-token 拆分不可用        |
| Google                     | `cachedContentTokenCount` + `thoughtsTokenCount`                                            |

如果 provider 没报某个数，对应字段保持 `0`——Aquaregia 从不编造数据。

---

### 结构化输出

当你想拿回一个类型化的 Rust 值而不是一段文本，给类型 derive `JsonSchema`，然后调 `generate_object::<T>()`。schema 自动生成并传给 provider；返回的 JSON 直接反序列化进 `T`。

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

不支持原生结构化输出的 provider（Anthropic、Google）会透明地降级到强制 tool call。从调用方视角看完全一样——你拿到的都是一个 `T`。

#### 流式 partial 对象

当 UI 需要字段一到位就渲染时，`stream_object::<T>()` 会持续输出渐进填充的对象。每个 chunk 都会被修复后重新反序列化成 partial `T`。尚未到达的字段保持 `Default`，所以请给类型 derive `Default` 并加 `#[serde(default)]`：

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

底层是一个基于栈的 JSON 修复器，处理截断字符串、未闭合的数组/对象、token 中间的转义序列。即使 chunk 把字段名或值切成两半也能拼出合法 partial——你在自己代码里永远不用处理非法 JSON。完整可运行示例见 `examples/structured_streaming.rs`。

---

### 工具

Tool 让模型在对话中段调用你的 Rust 代码。把一个 tool 想成**一个带 schema 的 `async fn`，LLM 按名字调用它**——Aquaregia 从你的参数类型推导 JSON Schema，把调用参数 marshal 回 Rust，再运行你的函数。

类型化风格是最常用的：

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

`tool(name)` 启动一个 builder；`.execute(closure)` 接收一个 `JsonSchema`-derived 的参数类型并完成构建。closure 返回 `Result<serde_json::Value, ToolExecError>`——模型拿到的是你产出的任何 JSON。

如果你想手写 schema（约束特殊、不方便用 derive），改用 `.raw_schema(...)` + `.execute_raw(...)`：

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

Tool 名必须匹配 `^[a-zA-Z0-9_-]{1,64}$` 且在一个 agent 内唯一。这条规则在 agent 构建时就校验——非法的工具表永远到不了模型那边。

---

### Agent

`Agent` 是一个 `generate` 加工具的 `while` 循环外加钩子：模型思考 → 也许调工具 → 你跑工具 → 结果回到模型 → 重复，直到模型不再调工具（或你设了上限）。你不写这个循环，你只描述它的输入与输出。

最小 agent 是一个工具加一个步数上限：

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

`response.output_text` 是最终给用户看的回答。`response.steps` 告诉你跑了几个 round-trip；`response.usage_total` 跨所有调用聚合 token 数。

#### 事件钩子

循环里每个有意思的边界都会发事件。所有钩子都是 `Fn + Send + Sync`，可以直接挂闭包——做日志、metrics、debug UI 都好使：

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

#### 动态规划 —— `prepare_step`

当你需要在下一步开跑前改它——收窄工具集、换模型、临时塞个系统提示——给 agent 一个 `prepare_step` 闭包：

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
            next.tools.clear(); // 第 2 步后禁用工具
        }
        next
    })
    .build()?;
```

返回的 `AgentPreparedStep` 就是 agent 实际发给模型的这一步内容。你没改的部分会从上一步状态继承下来。

#### 停止策略

控制循环何时结束的三个旋钮：

```rust
use aquaregia::ToolErrorPolicy;

let agent = Agent::builder(client, "deepseek-v4-pro")
    .max_steps(8)
    .stop_when(|step| step.tool_calls.is_empty() && !step.output_text.is_empty())
    .tool_error_policy(ToolErrorPolicy::ContinueAsToolResult)
    .build()?;
```

- **`max_steps`** —— 硬上限。超出返回 `ErrorCode::MaxStepsExceeded`，不会静默截断。
- **`stop_when`** —— 每步结束后判断的谓词。当你对"完成"有更精细的定义（而不仅仅是"没有 tool call"）时用它。
- **`tool_error_policy`** —— 工具 throw 时怎么办。
  - `ContinueAsToolResult`（默认） —— schema 校验失败、超时、panic 都会被包成 `{ "error": "..." }` 结果，让模型下一步自行恢复。
  - `FailFast` —— 直接抛出 `ErrorCode::ToolExecutionFailed` / `InvalidToolArgs`。

#### 多轮对话

`AgentResponse.transcript` 是一个完整的 `Vec<Message>`（system + user + assistant + tool results），可以直接喂给下一轮——不用手动拼装：

```rust
let mut history = vec![Message::system_text("You are a careful assistant.")];

loop {
    let user_input = read_line()?;
    history.push(Message::user_text(user_input));

    let result = agent.run_messages(history.clone()).await?;
    println!("{}", result.output_text);

    history = result.transcript; // 整段对话原地回流
}
```

`examples/mini_claude_code.rs` 用这个模式 + `bash` / `read` / `write` / `edit` 工具组合给出了一个可跑的终端 agent。

---

### 多模态

视觉模型既吃文本也吃图像。Aquaregia 用一个 `ImagePart` 把各 provider 之间的差异屏蔽掉——你说一次"URL"或"bytes"，正确的事情就发生在线路上。

最短路径是 `Message::user_image_url` 或 `Message::user_image_bytes`。当一条消息要混合内容或多张图，用 explicit content parts 构造 `Message`：

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

便捷构造器：

| 构造器                                                                   | 用途                                       |
| ------------------------------------------------------------------------ | ------------------------------------------ |
| `Message::user_image_url(url)`                                           | 单张图像，来自 URL                          |
| `Message::user_image_bytes(bytes, mime)`                                 | 单张图像，来自原始字节（自动 base64）       |
| `Message::new(MessageRole::User, vec![Text, Image, …])`                  | 混合内容 / 多图                             |
| `ContentPart::Image(ImagePart { data, media_type, provider_metadata })`  | 完整控制 + provider 私有附加                |

每个 provider 拿到的都是自己的原生格式：

| Provider            | URL                          | Base64 / Bytes                          |
| ------------------- | ---------------------------- | --------------------------------------- |
| Anthropic           | `source.type: url`           | `source.type: base64`                   |
| OpenAI / Compatible | `image_url` with remote URL  | `image_url` with `data:<mime>;base64,…` |
| Google              | `fileData.fileUri`           | `inlineData.data`                       |

---

## 生产化

生产环境调 LLM 要面对三个枯燥但绕不开的事：停止不再需要的工作、扛住瞬时失败、把错误报得有用。

### 取消

每个请求和 agent run 都接受一个 `CancellationToken`。取消 token，下一个安全 checkpoint 就停——Aquaregia 在每次 HTTP 发送前（通过 `tokio::select!`，happy path 零开销）、每个流式 SSE chunk 之后、每个 agent step 顶部都会检查。

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

Agent 可以在 builder 阶段就绑 token，这样每次 `agent.run(...)` 都用同一个：

```rust
let agent = Agent::builder(client, "deepseek-v4-pro")
    .cancellation_token(token.clone())
    .build()?;
```

### 重试与超时

两个旋钮，挂在 client 上：

```rust
let client = LlmClient::openai()
    .api_key(api_key)
    .max_retries(3)                          // 默认：0
    .timeout(Duration::from_secs(45))
    .build()?;
```

Aquaregia 只在确实是瞬时的类目上重试：`RateLimited`、`ProviderServerError`、`Transport`、`Timeout`。退避策略是指数加 jitter。若 provider 返回 `Retry-After` 头，Aquaregia 会遵守它而不是用自己的延时。

每个 `Error` 还带 `retryable: bool`，分类口径相同——如果你想加自己的重试 / 断路器，不必重做一遍分类法。

### 错误处理

`Error` 是结构化 payload 而不是一串字符串。`code: ErrorCode` 用来做控制流分支，其余字段用来诊断：

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

每个 `Error` 都带：

- `code: ErrorCode` —— 归一化变体（这才是你真正想 `match` 的分支器）
- `provider`、`status`、`request_id`、`raw_body`、`retry_after_secs` —— 日志和工单用的诊断字段
- `retryable: bool` —— `true` 当且仅当 Aquaregia 内置重试会处理

---

## 集成

Aquaregia 故意把 Web 框架适配器留在 crate 之外——`TextStream` 是一个 `StreamEvent` 的 `Stream`，干净地接到你 app 已经在用的任意传输层。

下面是 Axum SSE 的形态。每个 `StreamEvent` 变成一个命名 SSE 事件，前端可以 switch：

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

Actix、Warp、或你自家的 gRPC 层用同样的食谱——把每个 `StreamEvent` 变体映射到你自己的线协议。这里的示例为了简洁把非文本载荷都最小化了；实际项目你通常要把 tool call、usage、reasoning metadata 序列化成前端约定的 shape。

---

## 参考

当你知道想要什么、只差一个准确名字时查这里。

### Provider 能力矩阵

| 能力                              | OpenAI | Anthropic | Google | OpenAI-Compatible |
| --------------------------------- | :----: | :-------: | :----: | :---------------: |
| 自定义 `base_url`                 |   ✓    |     ✓     |   ✓    |         ✓         |
| 自定义 headers / query / path     |        |           |        |         ✓         |
| `api_version`（header）           |        |     ✓     |        |                   |
| Structured output                 |   ✓    |     ✓     |   ✓    |         ✓         |
| Tool-call streaming               |   ✓    |     ✓     |   ✓    |         ✓         |
| `Usage` 中 cache-token 拆分       |   ✓    |     ✓     |   ✓    |   有就报          |

### `Usage` 字段

```rust
pub struct Usage {
    pub input_tokens:             u32, // 总和
    pub input_no_cache_tokens:    u32,
    pub input_cache_read_tokens:  u32,
    pub input_cache_write_tokens: u32,
    pub output_tokens:            u32, // 总和
    pub output_text_tokens:       u32,
    pub reasoning_tokens:         u32,
    pub total_tokens:             u32,
    pub raw_usage:                Option<serde_json::Value>,
}
```

`Usage` 实现了 `Add` 和 `AddAssign`，跨 agent 步骤累加是一行的事。`AgentResponse.usage_total` 已经为你聚合好。

### 示例

```bash
DEEPSEEK_API_KEY=... cargo run --example basic_generate
```

| 示例                          | 重点                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `basic_generate`              | 一次性 `generate` + 读取 usage                                |
| `basic_stream`                | `stream` + `StreamEvent` 处理                                 |
| `structured_streaming`        | `stream_object::<T>()` + 渐进式 `Partial` 事件                |
| `agent_minimal`               | `Agent::builder` + 单个类型化工具                             |
| `tools_max_steps`             | 多工具循环 + `max_steps` + 采样参数                           |
| `prepare_hooks`               | `prepare_step`、`on_step_finish`                              |
| `openai_compatible_custom`    | 自定义 headers / query params / chat path                     |
| `mini_claude_code`            | TUI Code Agent —— `bash` / `read` / `write` / `edit` 工具     |
| `multimodal_image`            | `Message::new` 组合文字 + 图像 part + Anthropic 视觉           |

大多数示例需要 `DEEPSEEK_API_KEY`；`multimodal_image` 需要 `ANTHROPIC_API_KEY`。完整说明见 [`examples/README.md`](./examples/README.md)。

### API 参考

完整类型签名、所有公开项、所有变体——见 [docs.rs/aquaregia](https://docs.rs/aquaregia)。

---

## 本地开发

```bash
cargo fmt
cargo test
cargo check --examples
cargo clippy -- -D warnings
```

---

## AI 辅助开发

本项目欢迎 AI 辅助开发，但提交人对最终结果负责。如果代码、测试、文档或 API 变更是借助 AI 提出的，提交者依然要能理解、审查、并验证它们。

仓库里也刻意让 agent 面向的文档保持原则导向。`AGENTS.md`、`CLAUDE.md` 这类文件应该描述长期生效的约束和判断规则，而不是一堆很快就会与代码漂移开的内部 API checklist。

---

## 贡献与许可

欢迎贡献。涉及行为变化时，请附上集成测试（happy path + 错误映射 + tool/stream 流程，按相关性选择）。

- [贡献指南](./CONTRIBUTING.md)
- [行为准则](./CODE_OF_CONDUCT.md)
- [安全策略](./SECURITY.md)
- 基于 [MIT License](./LICENSE) 发布
