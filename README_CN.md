# Aquaregia

> 注：在 v0.2.0 版本前，Aquaregia 处于快速迭代阶段，API 可能会有破坏性变更，谨慎使用。

Aquaregia 是一个 provider-agnostic 的 Rust AI 工具包，用于构建 AI 应用与可调用工具的 Agent。

它提供统一的多供应商 API（OpenAI、Anthropic、Google、OpenAI-compatible），并内置 reasoning 感知输出、流式事件和多步工具执行循环能力。

查看 [API 文档](https://docs.rs/aquaregia)、[示例指南](./examples/README.md)，或切换到 [English README](./README.md)。

## 安装

项目需要 Rust 和 Tokio 异步运行时。

```bash
cargo add aquaregia
```

默认 feature 包含 `openai` 与 `anthropic`。可选 feature 说明：

| Feature     | 说明                                                                 |
| ----------- | -------------------------------------------------------------------- |
| `openai`    | OpenAI 适配器（默认启用）                                            |
| `anthropic` | Anthropic 适配器（默认启用）                                         |
| `telemetry` | 为 `generate`、`stream`、Agent 步骤及工具调用自动注入 `tracing` span |

```bash
cargo check --no-default-features
cargo check --no-default-features --features openai
cargo check --no-default-features --features anthropic
cargo check --features telemetry
```

## 统一 Provider 架构

一个 `LlmClient` 绑定一个 provider 配置。
每次调用传入一个 `GenerateTextRequest`，其中包含模型和消息。

| Provider          | 注册 API                                                                      | 模型参数              |
| ----------------- | ----------------------------------------------------------------------------- | --------------------- |
| OpenAI            | `LlmClient::openai(api_key)`（可选 `.base_url(...)`）                         | `"gpt-4o"`            |
| Anthropic         | `LlmClient::anthropic(api_key)`（可选 `.base_url(...)`、`.api_version(...)`） | `"claude-sonnet-4-5"` |
| Google            | `LlmClient::google(api_key)`（可选 `.base_url(...)`）                         | `"gemini-2.0-flash"`  |
| OpenAI-compatible | `LlmClient::openai_compatible(base_url).api_key(...)`                         | `"deepseek-chat"`     |

## Usage

### 文本生成

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
            "用 3 个要点解释 Rust 所有权。",
        ))
        .await?;

    println!("{}", out.output_text);
    Ok(())
}
```

### 流式输出

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
            "写一段简短版本发布说明。",
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

如果你需要完整事件，`StreamEvent` 已覆盖：
`ReasoningStarted / ReasoningDelta / ReasoningDone / TextDelta / ToolCallReady / Usage / Done`。

### Reasoning

reasoning 在非流式与流式 API 中都可用。

```rust
let out = client
    .generate(GenerateTextRequest::from_user_prompt(
        "deepseek-chat",
        "请分步骤推理后给出答案。",
    ))
    .await?;

println!("answer: {}", out.output_text);
println!("reasoning text: {}", out.reasoning_text);
println!("reasoning tokens: {}", out.usage.reasoning_tokens);

for part in &out.reasoning_parts {
    println!("reasoning block: {}", part.text);
}
```

统一字段说明：

- `GenerateTextResponse.reasoning_text`：扁平化 reasoning 文本（便捷字段）。
- `GenerateTextResponse.reasoning_parts`：结构化 reasoning 分块，包含可选 provider 元数据。
- `Usage.input_tokens`：provider 返回的输入 token 总量。
- `Usage.input_no_cache_tokens`：非缓存输入 token（尽力推导）。
- `Usage.input_cache_read_tokens` / `Usage.input_cache_write_tokens`：缓存读取/写入 token（可用时）。
- `Usage.output_tokens`：输出 token 总量。
- `Usage.output_text_tokens`：输出文本 token（可用时）。
- `Usage.reasoning_tokens`：provider 若返回推理 token，将映射到该字段。
- `Usage.raw_usage`：provider 原始 usage 负载，便于调试与后续扩展。
- `Message.parts`：assistant 消息可包含 `ContentPart::Reasoning(...)`，可用于 transcript 回放。

Provider 映射：

| Provider | Reasoning 内容来源 | Usage 映射 |
| --- | --- | --- |
| OpenAI / OpenAI-compatible | 同步/流式中的 `reasoning_content`（或 `reasoning`） | 解析 `prompt_tokens_details.cached_tokens` + `completion_tokens_details.reasoning_tokens` |
| Anthropic | `thinking` / `redacted_thinking`，流式 `thinking_delta` + `signature_delta` | 解析 `cache_read_input_tokens` / `cache_creation_input_tokens`；reasoning token 细分暂不可用 |
| Google | `thought: true` 的 part，支持 `thoughtSignature` 元数据 | 解析 `cachedContentTokenCount` + `thoughtsTokenCount` |

### 错误处理

```rust
use aquaregia::{ErrorCode, GenerateTextRequest, LlmClient};

match client
    .generate(GenerateTextRequest::from_user_prompt("deepseek-chat", "hello"))
    .await
{
    Ok(out) => println!("{}", out.output_text),
    Err(err) => match err.code {
        ErrorCode::RateLimited => eprintln!("触发限流，请稍后重试"),
        ErrorCode::AuthFailed  => eprintln!("请检查 API Key"),
        ErrorCode::Cancelled   => eprintln!("请求已取消"),
        _ => eprintln!("请求失败: {}", err),
    },
}
```

### Agent + 工具循环

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
        .instructions("回答前可以先调用工具。")
        .tools([get_weather])
        .max_steps(4)
        .build()?;

    let out = agent.run("上海天气如何？").await?;
    println!("{}", out.output_text);
    Ok(())
}
```

### 动态规划（`prepare_call` / `prepare_step`）

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

### 取消请求

每次请求和 Agent 运行都可以通过 `CancellationToken` 取消。

```rust
use aquaregia::{Agent, CancellationToken, GenerateTextRequest, LlmClient};
use tokio::time::{Duration, sleep};

// 取消单次 generate 调用
let token = CancellationToken::new();
let token_clone = token.clone();
tokio::spawn(async move {
    sleep(Duration::from_millis(200)).await;
    token_clone.cancel();
});

let req = GenerateTextRequest::builder("deepseek-chat")
    .user_prompt("写一篇一万字的文章。")
    .cancellation_token(token)
    .build()?;

match client.generate(req).await {
    Err(e) if e.code == ErrorCode::Cancelled => println!("已按预期取消"),
    other => println!("{other:?}"),
}
```

Agent 提供专用的可取消方法：

```rust
let token = CancellationToken::new();
token.cancel(); // 或在另一个任务中稍后调用

// 返回 Err，错误码为 ErrorCode::Cancelled
agent.run_cancellable("你好", token).await?;

// 传入自定义消息列表
agent.run_messages_cancellable(messages, token).await?;
```

取消检查点：

- **每次 HTTP 发送前**（通过 `tokio::select!`，未取消时无额外开销）
- **流式响应的每个 SSE chunk 之后**
- **工具循环中每个 Agent 步骤开始时**

### 可观测性（Telemetry）

启用 `telemetry` feature 可自动注入 `tracing` span：

```toml
aquaregia = { version = "*", features = ["telemetry"] }
```

自动注入的 span：

| Span                  | 字段                |
| --------------------- | ------------------- |
| `aquaregia::generate` | `model`、`provider` |
| `aquaregia::stream`   | `model`             |
| `agent_step`          | `step`              |
| `tool_call`           | `tool.name`         |

Aquaregia 不会自动配置 subscriber，请自行接入（如 `tracing-subscriber`、`tracing-opentelemetry`）：

```rust
tracing_subscriber::fmt::init(); // 或其他 subscriber

let out = client.generate(req).await?; // 自动产生 span
```

### OpenAI-Compatible 高级配置

```rust
use aquaregia::LlmClient;

let client = LlmClient::openai_compatible("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .header("x-trace-source", "aquaregia")
    .query_param("source", "sdk")
    .chat_completions_path("/v1/chat/completions")
    .build()?;
```

## 示例

| 示例             | 命令                                           | 重点                                    |
| ---------------- | ---------------------------------------------- | --------------------------------------- |
| 基础文本生成     | `cargo run --example basic_generate`           | 一次性 `generate`                       |
| 基础流式输出     | `cargo run --example basic_stream`             | `stream` + `StreamEvent` 处理           |
| 最小 Agent       | `cargo run --example agent_minimal`            | `Agent::builder` + 单工具               |
| 工具循环保护     | `cargo run --example tools_max_steps`          | 多步工具调用 + `max_steps`              |
| 动态 hooks       | `cargo run --example prepare_hooks`            | `prepare_call` / `prepare_step`         |
| 兼容接口深度定制 | `cargo run --example openai_compatible_custom` | 自定义 headers / query params / path    |
| 终端代码 Agent   | `cargo run --example mini_claude_code`         | `Agent::builder` + `#[tool]` + 本地工具 |

## 本地开发检查

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

仅在 `ai-sdk/` 子工程开发时运行：

```bash
cd ai-sdk && pnpm build && pnpm lint && pnpm type-check
```

## 贡献与许可

欢迎提交 Issue 和 PR。涉及行为变更时，建议补充集成测试（happy path + 错误映射 + 工具/流式流程）。

- [贡献指南](./CONTRIBUTING.md)
- [行为准则](./CODE_OF_CONDUCT.md)
- [安全策略](./SECURITY.md)
- [MIT 许可证](./LICENSE)
