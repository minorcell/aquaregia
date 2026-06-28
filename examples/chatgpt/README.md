# ChatGPT Example

A small ChatGPT-style web app.

The browser owns the visible conversation state. The Rust server receives the
message list for each turn, streams Aquaregia text deltas back over server-sent
events, and lets the browser append the assistant response as it arrives.

## Run

```bash
DEEPSEEK_API_KEY=... cargo run --manifest-path examples/chatgpt/Cargo.toml
```

Optional:

```bash
DEEPSEEK_MODEL=deepseek-v4-pro cargo run --manifest-path examples/chatgpt/Cargo.toml
DEEPSEEK_BASE_URL=https://api.deepseek.com cargo run --manifest-path examples/chatgpt/Cargo.toml
```

Open `http://127.0.0.1:3000`.
