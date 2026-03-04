# Oxide

使用 Oxide 快速构建您的 Rust AI 应用，具有统一的多供应商接口和强大的工具执行能力。

[English README](./README.md)

Oxide 是一个面向 Rust 的 provider-agnostic AI 工具包。
推荐上手路径：先跑起来，再理解分层，最后使用高级控制能力。

## 快速入口

- [1）先跑起来](#cn-get-running-first)
- [2）再理解分层](#cn-understand-layers)
- [3）再看高级能力](#cn-advanced-capabilities)
- [开源协作](#cn-open-source)

<a id="cn-get-running-first"></a>
## 1）先跑起来

### 安装

```toml
[dependencies]
oxide = { path = "." }
# 发布到 crates.io 后可替换为版本号：
# oxide = "x.y.z"
```

### 环境变量（以 DeepSeek 为例）

```bash
export DEEPSEEK_API_KEY="your_api_key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"   # 可选
export DEEPSEEK_MODEL="deepseek-chat"                  # 可选
```

### 首次成功调用（2 分钟）

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
            "用 3 个要点解释 Rust 所有权。",
        )
        .await?;

    println!("{}", out.output_text);
    Ok(())
}
```

### 示例学习路径

| 阶段 | 命令 | 你会学到什么 |
| --- | --- | --- |
| 1 | `cargo run --example basic_generate` | 最小一次性生成 |
| 2 | `cargo run --example basic_stream` | 流式输出处理 |
| 3 | `cargo run --example agent_minimal` | 第一个 Agent + 单工具 |
| 4 | `cargo run --example tools_max_steps` | 多步工具循环与收敛保护 |
| 5 | `cargo run --example prepare_hooks` | 按调用/按步骤动态控制 |
| 6 | `cargo run --example openai_compatible_custom` | OpenAI-compatible 自定义 headers/query/path |

更多示例： [examples/README.md](./examples/README.md)

<a id="cn-understand-layers"></a>
## 2）再理解分层

### 统一 Provider 架构

- 一个 `AiClient` 绑定一个 provider 配置。
- 每次调用通过 `ModelRef`（`openai(...)`、`anthropic(...)`、`google(...)`、`openai_compatible(...)`）选择模型。
- 同一应用需接入多个 provider 时，创建多个 client。

### Provider 注册 + 模型选择

| 适配类型 | 注册方法 | 模型选择 |
| --- | --- | --- |
| OpenAI GPT | `.with_openai(api_key, base_url)` | `openai("gpt-4o-mini")` |
| Anthropic Claude | `.with_anthropic(api_key, base_url, api_version)` | `anthropic("claude-3-5-haiku-latest")` |
| Google Gemini | `.with_google(api_key, base_url)` | `google("gemini-2.0-flash")` |
| OpenAI 兼容接口 | `.with_openai_compatible(base_url, api_key)` / `.with_openai_compatible_settings(...)` | `openai_compatible("deepseek-chat")` |

### 分层地图

| 分层 | 职责 | 核心 API |
| --- | --- | --- |
| Provider 绑定层 | 配置传输、重试与 provider adapter | `AiClient::builder()` |
| 模型选择层 | 显式区分 provider/model | `openai(...)`、`anthropic(...)`、`google(...)`、`openai_compatible(...)` |
| 生成层 | 一次性 + 流式文本生成 | `generate_prompt`、`stream_prompt`、`generate_text`、`stream_text` |
| 工具运行时层 | 多步推理 + 工具执行循环 | `run_tools`、`tool(...)`、`max_steps`、`stop_when` |
| Agent 封装层 | 模型 + 指令 + 工具 + hooks 的复用工作流 | `Agent::builder(...).generate_prompt(...)` |

<a id="cn-advanced-capabilities"></a>
## 3）再看高级能力

### Agent + 工具循环

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
    .instructions("回答前可以先调用工具。")
    .tool(weather)
    .max_steps(4)
    .build()?;
```

### 动态规划 Hooks

- `prepare_call`：运行开始前，调整调用计划。
- `prepare_step`：每一步开始前，调整 model/messages/tools。

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

### 完整生命周期回调

`on_start`、`on_step_start`、`on_tool_call_start`、`on_tool_call_finish`、`on_step_finish`、`on_finish`、`stop_when`

### OpenAI-Compatible 深度配置

需要自定义 headers/query/path 时，使用 `OpenAiCompatibleAdapterSettings`。

```bash
cargo run --example openai_compatible_custom
```

### Axum SSE 集成

```bash
cargo check --features axum
```

<a id="cn-open-source"></a>
## 开源协作

### 如何贡献

欢迎提 Issue 和 PR。涉及行为变更请附带测试。

### 本地开发检查

```bash
cargo fmt
cargo test
cargo check --examples
cargo check --no-default-features
cargo check --no-default-features --features openai
cargo check --no-default-features --features anthropic
cargo check --features axum
```

### 治理文件

- [MIT 许可证](./LICENSE)
- [贡献指南](./CONTRIBUTING.md)
- [行为准则](./CODE_OF_CONDUCT.md)
- [安全策略](./SECURITY.md)
