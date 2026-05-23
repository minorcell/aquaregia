<div align="center">

# Aquaregia

**Rust 生态的统一 AI 抽象层。**

[![Crates.io](https://img.shields.io/crates/v/aquaregia.svg)](https://crates.io/crates/aquaregia)
[![Docs.rs](https://docs.rs/aquaregia/badge.svg)](https://docs.rs/aquaregia)
[![License: MIT](https://img.shields.io/crates/l/aquaregia.svg)](./LICENSE)
[![Downloads](https://img.shields.io/crates/d/aquaregia.svg)](https://crates.io/crates/aquaregia)

[API 文档](https://docs.rs/aquaregia) · [示例指南](./examples/README.md) · [English](./README.md)

</div>

一个 crate 就能在任一 provider 上构建 LLM 应用与 Agent —— OpenAI、Anthropic、Google、或任何 OpenAI-compatible 端点（DeepSeek、vLLM、Ollama……）。

---

## 核心特性

- **类型化的 provider markers** —— `LlmClient::openai(...)` 返回 `BoundClient<OpenAi>`；把 `ModelRef<P>` 传错 provider 是编译错误，而不是运行时 400。
- **文本 · 流 · 推理 · 视觉 · 工具 统一接口** —— 跨四类 provider 共用同一份 `GenerateTextRequest`、`StreamEvent` 与 `Usage`。
- **AI SDK 风格的 Agent 钩子** —— `prepare_call`、`prepare_step`，以及完整事件链（`on_start` → `on_step_start` → `on_tool_call_*` → `on_step_finish` → `on_finish`)。
- **生产级可靠性** —— 内置指数退避重试与 `Retry-After` 解析、`CancellationToken` 检查点、可选 `tracing` span。
- **多轮对话开箱即用** —— `AgentResponse.transcript` 直接回灌到 `agent.run_messages(...)`，无需手工组装。
- **多模态视觉** —— URL / base64 / 原始字节，均映射到各 provider 的原生图像格式。

---

## 快速上手

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
            "用 3 个要点解释 Rust 所有权。",
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

换一个构造器，同样的调用就能跑在 Anthropic、OpenAI 或 Google 上。

---

## 安装

```bash
cargo add aquaregia
```

项目还需要 Tokio 异步运行时。

| Feature     | 说明                                                              |
| ----------- | ----------------------------------------------------------------- |
| `telemetry` | 为 `generate`、`stream`、Agent 步骤与工具调用注入 `tracing` span  |

---

## Providers

选择一个构造器，得到的 `BoundClient<P>` 以 provider marker `P` 为类型参数。

| Provider          | 构造器                                                | 模型参数              |
| ----------------- | ----------------------------------------------------- | --------------------- |
| OpenAI            | `LlmClient::openai(api_key)`                          | `"gpt-4o"`            |
| Anthropic         | `LlmClient::anthropic(api_key)`                       | `"claude-sonnet-4-5"` |
| Google            | `LlmClient::google(api_key)`                          | `"gemini-2.0-flash"`  |
| OpenAI-compatible | `LlmClient::openai_compatible(base_url).api_key(...)` | `"deepseek-chat"`     |

### 客户端配置

```rust
use std::time::Duration;

let client = LlmClient::openai(std::env::var("OPENAI_API_KEY")?)
    .base_url("https://api.openai.com")          // 自定义上游
    .timeout(Duration::from_secs(60))            // 单次请求超时
    .max_retries(3)                              // 瞬时失败重试次数
    .default_max_steps(8)                        // 基于此 client 构造的 Agent 默认 max_steps
    .user_agent("my-app/1.0")
    .build()?;
```

### 类型安全的 `ModelRef<P>`

如果你担心把 OpenAI 的模型名错传给 Anthropic 客户端，使用类型化工厂函数：

```rust
use aquaregia::{anthropic, openai, Anthropic, ModelRef, OpenAi};

let gpt:    ModelRef<OpenAi>    = openai("gpt-4o");
let claude: ModelRef<Anthropic> = anthropic("claude-sonnet-4-5");

// `client_openai.generate(GenerateTextRequest::from_user_prompt(claude, "..."))`
// 对 `BoundClient<OpenAi>` 是编译错误。
```

`GenerateTextRequest::from_user_prompt` 接受任何实现 `IntoModelRef<P>` 的值 —— 包括裸 `&str`（便捷写法），或上面这些类型化工厂函数（更强保证）。

### OpenAI-compatible 深度配置

```rust
let client = LlmClient::openai_compatible("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .header("x-trace-source", "aquaregia")
    .query_param("source", "sdk")
    .chat_completions_path("/v1/chat/completions") // 覆盖默认 endpoint 路径
    .think_tag_parsing(true)                       // 把 <think>...</think> 当作 reasoning 解析
    .think_tag_case_insensitive(true)
    .build()?;
```

`think_tag_parsing` 会从 assistant 内容里抽出 `<think>` / `<thinking>` 块并路由到 `reasoning_parts`，与原生支持 reasoning 的 provider 共用统一接口。

### Provider 能力差异速查

| 能力                          | OpenAI | Anthropic | Google | OpenAI-Compatible |
| ----------------------------- | :----: | :-------: | :----: | :---------------: |
| 自定义 `base_url`             |   ✓    |     ✓     |   ✓    |         ✓         |
| 自定义 headers / query / path |        |           |        |         ✓         |
| `api_version`（header）       |        |     ✓     |        |                   |
| 原生 reasoning 内容           |   ✓    |     ✓     |   ✓    |  依赖 think tag   |
| Tool-call 流式输出            |   ✓    |     ✓     |   ✓    |         ✓         |
| `Usage` 中的缓存 token 拆分   |   ✓    |     ✓     |   ✓    |  provider 上报时  |

---

## 文本生成

### 一次性 `generate`

```rust
let response = client
    .generate(GenerateTextRequest::from_user_prompt(
        "deepseek-chat",
        "用 Go 开发者能听懂的话总结 Rust 借用检查器。",
    ))
    .await?;

println!("{}", response.output_text);
println!("finish: {:?}", response.finish_reason);
```

需要多条消息、采样参数或工具时使用 builder：

```rust
use aquaregia::{GenerateTextRequest, Message};

let req = GenerateTextRequest::builder("deepseek-chat")
    .message(Message::system_text("You are concise."))
    .message(Message::user_text("Write a release note."))
    .temperature(0.2)
    .max_output_tokens(300)
    .build()?;
```

### 流式输出

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

完整事件：

| 事件                 | 字段                                    |
| -------------------- | --------------------------------------- |
| `ReasoningStarted`   | `block_id`、`provider_metadata`         |
| `ReasoningDelta`     | `block_id`、`text`、`provider_metadata` |
| `ReasoningDone`      | `block_id`、`provider_metadata`         |
| `TextDelta`          | `text`                                  |
| `ToolCallReady`      | `call: ToolCall`                        |
| `Usage`              | `usage: Usage`                          |
| `Done`               | —                                       |

### Reasoning

reasoning 在同步与流式输出中均可用：

```rust
let out = client.generate(req).await?;

println!("answer:     {}", out.output_text);
println!("thinking:   {}", out.reasoning_text);
println!("rsn-tokens: {}", out.usage.reasoning_tokens);

for part in &out.reasoning_parts {
    println!("[block] {}", part.text);
    // part.provider_metadata 承载 Anthropic 的 signature、Google 的 thoughtSignature 等元数据
}
```

| Provider                   | Reasoning 内容来源                                                          | Usage 映射                                                                                  |
| -------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| OpenAI / OpenAI-compatible | `reasoning_content`（或 `reasoning`）；启用后含 `<think>` 标签内容             | 解析 `prompt_tokens_details.cached_tokens` + `completion_tokens_details.reasoning_tokens`   |
| Anthropic                  | `thinking` / `redacted_thinking`，流式 `thinking_delta` + `signature_delta`  | 解析 `cache_read_input_tokens` / `cache_creation_input_tokens`;reasoning token 细分暂不可用 |
| Google                     | `thought: true` 的 part，可选 `thoughtSignature` 元数据                      | 解析 `cachedContentTokenCount` + `thoughtsTokenCount`                                       |

### `Usage` 与聚合

```rust
pub struct Usage {
    pub input_tokens:             u32, // 总输入
    pub input_no_cache_tokens:    u32,
    pub input_cache_read_tokens:  u32,
    pub input_cache_write_tokens: u32,
    pub output_tokens:            u32, // 总输出
    pub output_text_tokens:       u32,
    pub reasoning_tokens:         u32,
    pub total_tokens:             u32,
    pub raw_usage:                Option<serde_json::Value>,
}
```

`Usage` 实现了 `Add` 与 `AddAssign`，跨 Agent 步骤累加是一行的事。`AgentResponse.usage_total` 已经为你聚合好了。

---

## Tools 与 Agents

### 定义工具

工具通过 `tool(name)` 函数构造（没有 `#[tool]` 属性宏）。支持两种执行模式。

**类型化参数** —— `schemars` 自动从 struct 推导 JSON Schema：

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

**原始 schema** —— 手写 JSON Schema，回调收到 `serde_json::Value`：

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

工具名必须匹配 `^[a-zA-Z0-9_-]{1,64}$`，且在同一个 Agent 内唯一。

### 最小 Agent

```rust
use aquaregia::{Agent, LlmClient};

let client = LlmClient::openai_compatible("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .build()?;

let agent = Agent::builder(client, "deepseek-chat")
    .instructions("回答前可以先调用工具。")
    .tools([get_weather])
    .max_steps(4)
    .build()?;

let response = agent.run("上海天气怎么样？").await?;
println!("{}", response.output_text);
println!("steps={} total={}", response.steps, response.usage_total.total_tokens);
```

### 事件钩子

Agent 循环在每个关键边界都会触发事件。所有钩子都是 `Fn + Send + Sync`，可以直接传闭包。

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

### 动态规划 —— `prepare_call` / `prepare_step`

`prepare_call` 在整个 Agent 循环开始前运行一次，直接修改运行计划：

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

`prepare_step` 在每一步前运行，返回一份新的 prepared plan —— 适合裁剪工具集、切换模型、注入分步指令：

```rust
use aquaregia::Message;

let agent = Agent::builder(client, "deepseek-chat")
    .tools([get_weather, get_fx_rate])
    .prepare_step(|event| {
        let mut next = event.to_prepared();
        next.messages.push(Message::system_text(format!(
            "Step {}: 简洁回复。", event.step,
        )));
        if event.step >= 2 {
            next.tools.clear(); // 第 2 步起禁用工具
        }
        next
    })
    .build()?;
```

### 停止策略

```rust
use aquaregia::ToolErrorPolicy;

let agent = Agent::builder(client, "deepseek-chat")
    .max_steps(8)                                                                 // 硬上限
    .stop_when(|step| step.tool_calls.is_empty() && !step.output_text.is_empty()) // 提前停止谓词
    .tool_error_policy(ToolErrorPolicy::ContinueAsToolResult)                     // 默认
    .build()?;
```

- `max_steps` —— 超出返回 `ErrorCode::MaxStepsExceeded`。
- `stop_when` —— 每步结束后评估的谓词，为真则提前停止。
- `tool_error_policy` ——
  - `ContinueAsToolResult`（默认）—— schema 校验失败、超时、panic 都会变成 `{ "error": "..." }` 的 tool result，让模型自行恢复。
  - `FailFast` —— 直接抛出 `ErrorCode::ToolExecutionFailed` / `InvalidToolArgs`。

### 多轮对话

`AgentResponse.transcript` 是一份完整的 `Vec<Message>`（system + user + assistant + tool 结果），可以直接喂回下一轮：

```rust
let mut history = vec![Message::system_text("你是一名严谨的助手。")];

loop {
    let user_input = read_line()?;
    history.push(Message::user_text(user_input));

    let result = agent.run_messages(history.clone()).await?;
    println!("{}", result.output_text);

    history = result.transcript; // 把完整对话回滚到下一轮
}
```

`examples/mini_claude_code.rs` 给出了一个使用此模式 + `bash` / `read` / `write` / `edit` 工具的终端 Agent 示例。

---

## 多模态视觉

```rust
use aquaregia::{GenerateTextRequest, LlmClient, Message};

let client = LlmClient::anthropic(std::env::var("ANTHROPIC_API_KEY")?).build()?;

let out = client
    .generate(
        GenerateTextRequest::builder("claude-sonnet-4-5")
            .message(Message::user_text_and_image_url(
                "这张图里有什么？",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
            ))
            .build()?,
    )
    .await?;
```

三个便捷构造器 + 一个完全自定义形式：

| 构造器                                                                  | 场景                                  |
| ----------------------------------------------------------------------- | ------------------------------------- |
| `Message::user_image_url(url)`                                          | 单张图像，来自 URL                    |
| `Message::user_image_bytes(bytes, mime)`                                | 单张图像，来自原始字节（自动 base64） |
| `Message::user_text_and_image_url(text, url)`                           | 文字 + 图像 URL 合为一条用户消息      |
| `ContentPart::Image(ImagePart { data, media_type, provider_metadata })` | 完全自定义，可附带 provider 专用提示  |

各 provider 接收各自的原生格式：

| Provider            | URL                          | Base64 / 字节                           |
| ------------------- | ---------------------------- | --------------------------------------- |
| Anthropic           | `source.type: url`           | `source.type: base64`                   |
| OpenAI / Compatible | `image_url`（远程 URL）      | `image_url`（`data:<mime>;base64,…`）   |
| Google              | `fileData.fileUri`           | `inlineData.data`                       |

---

## 取消

每次请求与 Agent 运行都可以通过 `CancellationToken` 取消。

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
    .user_prompt("写一篇一万字的文章。")
    .cancellation_token(token)
    .build()?;

match client.generate(req).await {
    Err(e) if e.code == ErrorCode::Cancelled => println!("已取消"),
    other => println!("{other:?}"),
}
```

Agent 提供专用的可取消方法：

```rust
agent.run_cancellable("你好", token.clone()).await?;
agent.run_messages_cancellable(messages, token).await?;
```

取消检查点：**每次 HTTP 发送前**（通过 `tokio::select!`，未取消时零开销）、**每个 SSE chunk 之后**、**工具循环中每个 Agent 步骤开始时**。

---

## 可靠性

### 重试

```rust
let client = LlmClient::openai(api_key)
    .max_retries(3)                       // 默认 0
    .timeout(Duration::from_secs(45))
    .build()?;
```

Aquaregia 会在瞬时错误（`RateLimited`、`ProviderServerError`、`Transport`、`Timeout`）上自动重试，采用带 jitter 的指数退避。响应中如带 `Retry-After`，会解析并遵守。

每个 `Error` 都带 `retryable: bool` 字段，标识与内置重试相同的判定，你可以在外层自行叠加自己的重试 / 熔断策略。

### Telemetry

启用 `telemetry` feature：

```toml
aquaregia = { version = "*", features = ["telemetry"] }
```

Aquaregia 只发出 `tracing` span，不替你配置 subscriber。请自行接入（`tracing-subscriber`、`tracing-opentelemetry` 等）：

| Span                  | 字段                |
| --------------------- | ------------------- |
| `aquaregia::generate` | `model`、`provider` |
| `aquaregia::stream`   | `model`             |
| `agent_step`          | `step`              |
| `tool_call`           | `tool.name`         |

---

## 框架集成示例（Axum）

Aquaregia 有意不在主 crate 中内置 Web 框架适配层。如果你在用 Axum，可以在应用代码里自行把 `TextStream` 转成 SSE：

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
        .stream(GenerateTextRequest::from_user_prompt("deepseek-chat", "你好。"))
        .await
        .unwrap();
    to_axum_sse(stream)
}

let app: Router = Router::new()
    .route("/chat", get(chat))
    .with_state(Arc::new(client));
```

示例里把非文本事件的 payload 简化了；实际项目中，你通常会把 tool call、usage、reasoning metadata 序列化成前端约定的格式。

你可以按需把 `StreamEvent` 映射成命名 SSE 事件、WebSocket 消息，或任何适合你应用的传输格式。

---

## 错误处理

```rust
use aquaregia::ErrorCode;

match client.generate(req).await {
    Ok(out) => println!("{}", out.output_text),
    Err(e) => match e.code {
        ErrorCode::RateLimited        => eprintln!("retry after {:?}s", e.retry_after_secs),
        ErrorCode::AuthFailed         => eprintln!("API key 无效"),
        ErrorCode::Cancelled          => eprintln!("已取消"),
        ErrorCode::MaxStepsExceeded   => eprintln!("Agent 循环超出步数上限"),
        ErrorCode::InvalidToolArgs    => eprintln!("schema 校验失败: {}", e.message),
        ErrorCode::Timeout            => eprintln!("上游超时"),
        _                              => eprintln!("error: {e}"),
    },
}
```

每个 `Error` 都包含：

- `code: ErrorCode` —— 13 种归一化错误码之一
- `provider`、`status`、`request_id`、`raw_body`、`retry_after_secs` —— 用于日志和定位
- `retryable: bool` —— 为 `true` 当且仅当 Aquaregia 内置重试会处理它

---

## 示例

```bash
DEEPSEEK_API_KEY=... cargo run --example basic_generate
```

| 示例                          | 重点                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `basic_generate`              | 一次性 `generate` + 读取 usage                                |
| `basic_stream`                | `stream` + `StreamEvent` 处理                                 |
| `agent_minimal`               | `Agent::builder` + 单个类型化工具                             |
| `tools_max_steps`             | 多工具循环 + `max_steps` + 采样参数                           |
| `prepare_hooks`               | `prepare_call`、`prepare_step`、`on_step_finish`              |
| `openai_compatible_custom`    | 自定义 headers / query params / chat path                     |
| `mini_claude_code`            | TUI Code Agent —— `bash` / `read` / `write` / `edit` 工具     |
| `multimodal_image`            | `Message::user_text_and_image_url` + Anthropic 视觉           |

大多数示例需要 `DEEPSEEK_API_KEY`；`multimodal_image` 需要 `ANTHROPIC_API_KEY`。完整说明见 [`examples/README.md`](./examples/README.md)。

---

## 本地开发

```bash
cargo fmt
cargo test
cargo check --examples
cargo test --features telemetry
cargo clippy -- -D warnings
```

---

## 贡献与许可

欢迎贡献。涉及行为变更时，请补充集成测试（happy path + 错误映射 + tool/stream 流程）。

- [贡献指南](./CONTRIBUTING.md)
- [行为准则](./CODE_OF_CONDUCT.md)
- [安全策略](./SECURITY.md)
- [MIT 许可证](./LICENSE)
