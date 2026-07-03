# Remote Agent Example

## 是什么

这是一个 Aquaregia 远程 agent 服务示例。

它启动一个 HTTP Gateway，把 Aquaregia agent 通过 REST + SSE 暴露给客户端；
agent 的工具调用会通过 MCP 转发到 Docker 沙箱中执行。Gateway 内置一个简单 Web App，
可以创建 session、发起 run、查看实时事件和当前进程内的 run 回放。

数据保存在进程内内存中，不使用数据库。Gateway 重启后，session、transcript、run history
和 replay events 都会清空。

## 如何使用

先启动沙箱 MCP server：

```bash
docker build -f examples/remote_agent/sandbox/Dockerfile -t remote-agent-sandbox .
docker run -e SANDBOX_MCP_TOKEN=dev-token -p 8931:8931 remote-agent-sandbox
```

再启动 Gateway。Gateway 会直接从当前环境变量读取 `DEEPSEEK_API_KEY`：

```bash
SANDBOX_MCP_URL=http://127.0.0.1:8931/mcp \
SANDBOX_MCP_TOKEN=dev-token \
cargo run --manifest-path examples/remote_agent/Cargo.toml
```

默认 Gateway 地址是 `http://127.0.0.1:3000`，打开后即可使用 Web App。

也可以直接用 curl：

```bash
SESSION_ID=$(curl -sS -X POST http://127.0.0.1:3000/v1/sessions \
  -H 'content-type: application/json' \
  -d '{}' | jq -r '.session_id')

curl -N -X POST "http://127.0.0.1:3000/v1/sessions/${SESSION_ID}/runs" \
  -H 'content-type: application/json' \
  -H 'accept: text/event-stream' \
  -d '{"input":"在沙箱里运行 pwd 和 ls -la","stream":true}'
```

常用环境变量：

- `DEEPSEEK_API_KEY`：DeepSeek API key，必填。
- `REMOTE_AGENT_ADDR`：Gateway 监听地址，默认 `127.0.0.1:3000`。
- `DEEPSEEK_MODEL`：模型名，默认 `deepseek-v4-pro`。
- `DEEPSEEK_BASE_URL`：OpenAI-compatible base URL，默认 `https://api.deepseek.com`。
- `SANDBOX_MCP_URL`：沙箱 MCP endpoint，默认 `http://127.0.0.1:8931/mcp`。
- `SANDBOX_MCP_TOKEN`：Gateway 和 sandbox 共享的 Bearer token，默认 `dev-token`。
