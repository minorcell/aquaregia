# Remote Agent 设计文档

## 1. 概要

这个 example 将 Aquaregia 组装成一个小型远程 agent 服务。客户端访问 HTTP
Gateway，Gateway 为进程内 session 运行 Aquaregia agent loop，所有工具调用都通过
MCP 转发到 Docker 沙箱执行。

设计目标是在不修改根 `aquaregia` crate 的前提下完成集成。远程 MCP 工具通过
`raw_schema` 和 `execute_raw` 适配成现有的 Aquaregia 工具。

## 2. 目标

- 展示服务形态的 agent example，而不是单个 API 调用。
- 通过 SSE 保留完整的 `AgentStreamEvent` 生命周期。
- 用进程内状态保存 session transcript、run 状态、事件回放数据和工具遥测。
- 在容器边界内执行工具，避免工具直接运行在宿主机上。
- 将 app-only 依赖全部限制在 `examples/remote_agent` 内。

## 3. 非目标

- 生产级认证、授权、计费或租户隔离。
- OpenAI 兼容的 chat completions endpoint。
- 在根 Aquaregia SDK 中内置 MCP 支持。
- 超出 Docker 容器边界的高级沙箱加固。
- 上下文压缩、transcript 裁剪或长期记忆。

## 4. 架构

```text
client
  |
  | REST + SSE
  v
HTTP Gateway
  |
  | create session / start run / stream events
  v
Session Manager + Aquaregia Agent loop
  |
  | rmcp tools/list + tools/call
  v
Docker sandbox
  |
  | sandbox-mcp process executes tools inside the container
  v
bash / git / node / file system
```

### 组件职责

| 组件 | 职责 |
| --- | --- |
| Gateway | HTTP 路由、请求校验、SSE 响应、健康检查 |
| Web App | 管理 session/run、提交任务、查看 live events、回放事件和工具 schema |
| Session Manager | session 生命周期、run 并发控制、transcript 交给 Aquaregia |
| MCP Bridge | 将远程 MCP 工具转换成 `aquaregia::Tool` |
| Telemetry Recorder | 将 stream event、usage 和工具调用耗时写入进程内状态 |
| Sandbox MCP Server | 通过 Streamable HTTP 暴露容器内工具 |
| In-memory Store | 保存 session、message、run 和 event，进程重启后清空 |

## 5. 运行流程

1. 客户端创建或选择一个 session。
2. 客户端向 `POST /v1/sessions/{sid}/runs` 提交一次用户输入。
3. Gateway 从内存 store 加载该 session 的 transcript，并追加新的 user message。
4. Session Manager 使用 `agent.stream_messages(...)` 启动一次 Aquaregia loop。
5. 单个 stream consumer 消费 `AgentStream`。
6. consumer 将每个事件广播给 live SSE subscriber。
7. consumer 同时把每个事件记录到内存 store，用于当前进程内回放。
8. 工具调用进入 `ToolExecutor::execute`，穿过 MCP bridge，在沙箱容器内执行。
9. loop 完成后，最终 `AgentOutput.transcript` 和 run summary 写回内存 store。
10. Gateway 发送 `run_done` 并关闭 SSE 响应。

## 6. HTTP API

| Method | Path | 用途 |
| --- | --- | --- |
| `POST` | `/v1/sessions` | 创建 session |
| `GET` | `/v1/sessions` | 列出 session |
| `GET` | `/v1/sessions/{sid}` | 获取 session 元数据和 usage |
| `DELETE` | `/v1/sessions/{sid}` | 归档 session |
| `GET` | `/v1/sessions/{sid}/messages` | 读取 transcript |
| `GET` | `/v1/sessions/{sid}/runs` | 列出 run history |
| `POST` | `/v1/sessions/{sid}/runs` | 启动一次 agent loop |
| `GET` | `/v1/sessions/{sid}/runs/{rid}` | 读取 run 状态或最终输出 |
| `GET` | `/v1/sessions/{sid}/runs/{rid}/events` | 回放当前进程内 run events |
| `GET` | `/v1/tools` | 列出桥接后的 MCP 工具 |
| `GET` | `/healthz` | 检查 Gateway 和沙箱连通性 |

### 创建 Session

```json
{
  "model": "deepseek/deepseek-v4-pro",
  "instructions": "You are a careful coding agent.",
  "metadata": {}
}
```

`model`、`instructions` 和 `metadata` 都是可选字段。未传时使用 example 默认值。

### 启动 Run

```json
{
  "input": "Clone this repository and count Rust source lines.",
  "stream": true
}
```

`stream: true` 返回 `text/event-stream`。第一帧是 `run_accepted`，包含
`run_id` 和 `session_id`，便于前端在流结束后读取 run status 和 replay events。
`stream: false` 阻塞到 run 完成，并返回最终 run object。

## 7. SSE 事件

SSE `event:` 名称直接映射 Aquaregia stream event。

| SSE event | 来源 | 必要数据 |
| --- | --- | --- |
| `run_accepted` | Gateway | run id、session id |
| `run_start` | `AgentStreamEvent::Start` | model、tool count、max steps |
| `step_start` | `AgentStreamEvent::StepStart` | step index |
| `model_delta` | `AgentStreamEvent::Model` | streamed model delta |
| `tool_call_start` | `AgentStreamEvent::ToolCallStart` | call id、tool name、arguments |
| `tool_call_finish` | `AgentStreamEvent::ToolCallFinish` | result、error flag、duration |
| `step_finish` | `AgentStreamEvent::StepFinish` | finish reason、usage、step result |
| `run_done` | `AgentStreamEvent::Done` | final output 和 usage total |
| `error` | stream error | code 和 message |

除 `run_accepted` 这类 transport-level 事件外，Gateway 不引入平行的 agent 事件
DTO。Gateway 直接保存并流式返回序列化后的 agent event payload，只根据事件变体
派生 SSE event 名称。

## 8. Session 与 Run 规则

- 一个 session 拥有一份累积 transcript。
- 一个 run 表示针对一次新用户输入执行一次 agent loop。
- 内存 store 是进程内事实源，进程重启后 session 和 run history 会丢失。
- Session Manager 可以维护 active run handle 的内存注册表。
- MVP 中，同一个 session 同时只允许一个 active run。
- 当同一 session 已有 active run 时，新的 run 请求返回 conflict。
- 多个不同 session 可以并发运行。
- live SSE 使用 `tokio::sync::broadcast`。
- 断线客户端可以在同一 Gateway 进程内回放 run events。

## 9. MCP Bridge

Gateway 启动时执行：

1. 使用 `SANDBOX_MCP_URL` 构造 rmcp Streamable HTTP client。
2. 使用 `SANDBOX_MCP_TOKEN` 配置 Bearer token。
3. 调用 `list_all_tools()`。
4. 将每个远程 MCP 工具转换成一个 `aquaregia::Tool`。
5. 在每个 `AgentBuilder` 上注册转换后的工具列表。

桥接契约：

| MCP 字段或行为 | Aquaregia 映射 |
| --- | --- |
| tool name | `tool(name)` |
| description | `.description(...)` |
| `inputSchema` | `.raw_schema(...)` |
| `tools/call` | `.execute_raw(...)` |
| `structuredContent` | 返回的 `serde_json::Value` |
| 第一个 text content block | fallback `{ "text": ... }` |
| rmcp service error | `ToolExecError::Execution(...)` |

实现骨架：

```rust
async fn bridge_all(peer: Arc<Peer<RoleClient>>) -> Result<Vec<aquaregia::Tool>, ToolExecError> {
    let remote_tools = peer
        .list_all_tools()
        .await
        .map_err(|err| ToolExecError::Execution(err.to_string()))?;

    remote_tools
        .into_iter()
        .map(|remote_tool| bridge_one(peer.clone(), remote_tool))
        .collect()
}
```

`CallToolRequestParams` 要求 arguments 是 JSON object。MVP 中，非 object arguments
按无参数处理，因为 Aquaregia 已经通过工具 schema 要求模型生成合法参数。

## 10. 遥测

`on_step_finish`、`on_tool_call_finish` 等 builder hook 是同步回调。streaming path
不能在这些 hook 中直接做慢 I/O。

流式 run 的处理规则：

- 只消费一次 `AgentStream`。
- 将每个事件广播给 live SSE subscriber。
- 将每个事件同步记录到内存 store。

非流式 run 的处理规则：

- 复用同一条 streaming pipeline。
- HTTP handler 等待 loop 完成后返回最终 run object。

遥测来源：

| 数据 | 来源 |
| --- | --- |
| 工具耗时 | `AgentToolCallFinish.duration_ms` |
| 单 step token usage | `AgentStep.usage` |
| 累计 token usage | `AgentOutput.usage_total` |
| transcript | `AgentOutput.transcript` |
| 工具参数和结果 | step tool calls 与 tool results |

## 11. 沙箱

沙箱是一个 Docker image，包含：

- `sandbox-mcp`，使用 rmcp server API 编写的 Rust binary。
- Node.js，用于 JavaScript 执行实验。
- Git，用于仓库操作。
- 作用域限制在容器内的可写工作目录。

当前工具：

| 工具 | 用途 |
| --- | --- |
| `bash` | 在容器工作目录中执行 shell 命令 |
| `read_file` | 读取容器内文件 |
| `write_file` | 写入容器内文件 |
| `git` | 执行常见 git 操作 |
| `node_eval` | 执行 Node.js 片段 |

这些工具都只在容器工作目录内执行。路径型工具拒绝绝对路径和 `..`。

## 12. 沙箱认证

Gateway 和沙箱共享 `SANDBOX_MCP_TOKEN`。

客户端行为：

- 使用 sandbox URL 配置 rmcp Streamable HTTP client。
- 通过 client transport 的 auth-header 支持传入 token。

服务端行为：

- rmcp 不提供适合本 example 的简单 Bearer token 校验。
- 在 axum 或 tower middleware 中包裹 `StreamableHttpService`。
- 当 `Authorization` header 不是 `Bearer <token>` 时拒绝请求。
- 为容器网络配置 allowed hosts；rmcp 默认 host allowlist 偏向 loopback 场景。

这是 example 级认证，不是生产身份体系。

## 13. 存储模型

```text
StoreState
  sessions: session_id -> SessionRecord
  messages: session_id -> Vec<Message>
  runs: run_id -> RunRecord
  run_events: run_id -> Vec<StoredEvent>
```

这是有意的 0 数据库设计。`run_events` 用于当前进程内的回放和调试；
重启 Gateway 后，session、transcript、run history 和 replay events 都会清空。

## 14. Package Layout

```text
examples/remote_agent/
  Cargo.toml
  README.md
  design.md
  index.html
  src/
    main.rs
    api.rs
    session.rs
    mcp_bridge.rs
    telemetry.rs
    store.rs
    bin/
      sandbox_mcp.rs
  sandbox/
    Dockerfile
  static/
    app.js
    style.css
```

example package 通过下面的方式依赖本地 SDK：

```toml
aquaregia = { path = "../.." }
```

`axum`、`rmcp` 和 Docker 相关文件只放在 `examples/remote_agent` 内。

## 15. 配置

| 变量 | 使用方 | 用途 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | Gateway | provider API key |
| `DEEPSEEK_MODEL` | Gateway | 可选 model override |
| `DEEPSEEK_BASE_URL` | Gateway | 可选 OpenAI-compatible base URL |
| `SANDBOX_MCP_URL` | Gateway | Streamable HTTP MCP endpoint |
| `SANDBOX_MCP_TOKEN` | Gateway 和 sandbox | 共享 Bearer token |

## 16. 落地切片

### Slice 1: 端到端 loop

- 新增独立 Cargo package。
- 新增只包含 `bash` 工具的 sandbox MCP server。
- 新增 Gateway session 创建和 streamed run 执行。
- 在内存中保留 session transcript。
- 通过 Docker、Gateway 启动和一次 curl streamed run 验证。

### Slice 2: 遥测与回放

- 在内存中保留 `run_events`。
- 新增 run status 和 event replay endpoints。
- 验证 streamed output、replayed output 和工具 duration record 一致。

### Slice 3: 完整 example 表面

- 新增 `read_file`、`write_file`、`git` 和 `node_eval`。
- 新增 `/v1/tools`。
- 新增包含沙箱连通性检查的 `/healthz`。
- 允许不同 session 并发，同时保持同一 session 只有一个 active run。
- 用两个 session 和每个 session 至少一次工具调用 run 做验证。

当前实现已覆盖三个切片的主要表面；后续改动应继续保持每个切片可独立验证。

## 17. 待实现时确认

- 确认所用 rmcp 版本中 `Tool.input_schema` 的准确 clone 写法。
- 确认共享 rmcp client connection 是否应跨 run 复用，或在沙箱连接失败后重建。
- 确认 stateful Streamable HTTP 行为不会和本 example 自己的 session ID 冲突。
