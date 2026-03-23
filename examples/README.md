# Examples Guide

本目录展示 Aquaregia 的推荐用法（对齐 AI SDK 风格）。

## 运行前准备

按示例设置对应环境变量（只需配置你要运行的那个示例）：

- `DEEPSEEK_API_KEY`（大多数示例必需）
- `ANTHROPIC_API_KEY`（`multimodal_image` 示例必需）
- `DEEPSEEK_BASE_URL`（可选，默认 `https://api.deepseek.com`）
- `DEEPSEEK_MODEL`（可选，默认 `deepseek-chat`）

## 示例清单

1. `basic_generate.rs`

- 场景：一次性非流式调用。
- 重点：`generate`、usage 读取、最小上手路径。

2. `basic_stream.rs`

- 场景：流式输出到 CLI/UI。
- 重点：`stream` 消费 `StreamEvent`；可同时处理 `ReasoningStarted/ReasoningDelta/ReasoningDone`、`TextDelta`、`Usage`、`Done`。

3. `agent_minimal.rs`

- 场景：最小 Agent + 单工具。
- 重点：`Agent::builder(client, model)`、`#[tool]` 宏、`tools([...])` 批量注册、`max_steps(...)`。

4. `tools_max_steps.rs`

- 场景：循环 + 多工具 + 步数保护。
- 重点：多个 tool 定义、`max_steps`、多步推理收敛。

5. `google_generate.rs`

- 场景：一次性调用（OpenAI-compatible 接口）。
- 重点：`generate` 的最小调用路径。

6. `openai_compatible_custom.rs`

- 场景：接入 OpenAI-Compatible 服务。
- 重点：自定义 `header` / `query_param` / `chat_completions_path`。

7. `mini_claude_code.rs`

- 场景：最小终端 Code Agent（TUI + 系统提示词 + 工具循环）。
- 重点：`Agent::builder`、`#[tool]`、`on_step_finish`、`bash/read/write/edit` 工具组合。

8. `prepare_hooks.rs`

- 场景：动态控制每次调用和每一步执行（对齐 AI SDK 的 `prepareCall/prepareStep`）。
- 重点：`prepare_call`、`prepare_step`、在 step 前动态改消息/工具/采样参数。

9. `multimodal_image.rs`

- 场景：向视觉模型发送图像（URL / base64 / 原始字节）。
- 重点：`Message::user_text_and_image_url`、`Message::user_image_bytes`、`ImagePart`、`MediaData`。
- 运行：`ANTHROPIC_API_KEY=<key> cargo run --example multimodal_image`

## 建议阅读顺序

1. `basic_generate.rs`
2. `basic_stream.rs`
3. `agent_minimal.rs`
4. `tools_max_steps.rs`
5. `openai_compatible_custom.rs`
6. `mini_claude_code.rs`
7. `prepare_hooks.rs`
8. `multimodal_image.rs`
