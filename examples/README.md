# Examples Guide

本目录展示 Aquaregia 的推荐用法（对齐 AI SDK 风格）。

## 运行前准备

按示例设置对应环境变量（只需配置你要运行的那个示例）：

- `DEEPSEEK_API_KEY`（大多数示例必需）
- `ANTHROPIC_API_KEY`（`multimodal_image` 示例必需）
- `DEEPSEEK_BASE_URL`（可选，默认 `https://api.deepseek.com`）
- `DEEPSEEK_MODEL`（可选，默认 `deepseek-v4-pro`）

## 示例清单

1. `basic_generate.rs`

- 场景：一次性非流式调用。
- 重点：`generate`、usage 读取、最小上手路径。

2. `basic_stream.rs`

- 场景：流式输出到 CLI/UI。
- 重点：`stream` 消费 `StreamEvent`；可同时处理 `ReasoningStarted/ReasoningDelta/ReasoningDone`、`TextDelta`、`Usage`、`Done`。

3. `structured_streaming.rs`

- 场景：流式拿结构化输出，字段逐个到达即可消费。
- 重点：`stream_object::<T>()`、`StreamObjectEvent::Partial`/`Object`、`#[serde(default)]` 默认值约定、内置 partial JSON 修复。

4. `agent_minimal.rs`

- 场景：最小 Agent + 单工具。
- 重点：`Agent::builder(client, model)`、`tool()` builder、`tools([...])` 批量注册、`max_steps(...)`。

5. `tools_max_steps.rs`

- 场景：循环 + 多工具 + 步数保护。
- 重点：多个 tool 定义、`max_steps`、多步推理收敛。

6. `openai_compatible_custom.rs`

- 场景：接入 OpenAI-Compatible 服务。
- 重点：自定义 `header` / `query_param` / `chat_completions_path`。

7. `mini_claude_code.rs`

- 场景：最小终端 Code Agent（TUI + 系统提示词 + 工具循环）。
- 重点：`Agent::builder`、`tool()` builder、`on_step_finish`、`bash/read/write/edit` 工具组合。

8. `prepare_hooks.rs`

- 场景：动态控制每一步执行（对齐 AI SDK 的 `prepareStep`）。
- 重点：`prepare_step`、在 step 前动态改消息/工具/采样参数。

9. `multimodal_image.rs`

- 场景：向视觉模型发送图像（URL / base64 / 原始字节）。
- 重点：`Message::new` 组合文字 + 图像 part、`Message::user_image_bytes`、`ImagePart`、`MediaData`。
- 运行：`ANTHROPIC_API_KEY=<key> cargo run --example multimodal_image`

10. `anthropic_prompt_caching.rs`

- 场景：用 Anthropic prompt caching 把长 system 上下文设为缓存断点，重复调用时复用。
- 重点：`TextPart::with_provider_options(...)` 把 `cache_control` 直接挂在内容块上、`Usage` 里的 `input_cache_write_tokens` / `input_cache_read_tokens` 读取。
- 运行：`ANTHROPIC_API_KEY=<key> cargo run --example anthropic_prompt_caching`

11. `anthropic_web_search.rs`

- 场景：调用 Anthropic 的 native `web_search` 工具——provider 在服务端执行，不需要 executor。
- 重点：通过 `provider_options` 把 native 工具直接注入到请求体的 `tools` 数组、`provider_options` 顶层覆盖语义。
- 运行：`ANTHROPIC_API_KEY=<key> cargo run --example anthropic_web_search`

## 建议阅读顺序

1. `basic_generate.rs`
2. `basic_stream.rs`
3. `structured_streaming.rs`
4. `agent_minimal.rs`
5. `tools_max_steps.rs`
6. `openai_compatible_custom.rs`
7. `mini_claude_code.rs`
8. `prepare_hooks.rs`
9. `multimodal_image.rs`
10. `anthropic_prompt_caching.rs`
11. `anthropic_web_search.rs`

## Web 框架集成说明

Aquaregia 不在主 crate 中内置 Axum、Actix、Warp 等 Web 框架适配层。
如果你需要把流式输出接到 SSE / WebSocket，请在应用层基于 `TextStream` 与 `StreamEvent` 做一层转换。
Axum 的示例写法可参考顶层 [`README.md`](../README.md)。
