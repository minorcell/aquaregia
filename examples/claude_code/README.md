# Claude Code Example

A small terminal code-agent app inspired by Claude Code.

The UI is built with `ratatui`. The agent is built with Aquaregia and has local
tools for shell commands, file reads, full-file writes, and targeted string
edits.

## Run

```bash
DEEPSEEK_API_KEY=... cargo run --manifest-path examples/claude_code/Cargo.toml
```

Optional:

```bash
DEEPSEEK_MODEL=deepseek-v4-pro cargo run --manifest-path examples/claude_code/Cargo.toml
DEEPSEEK_BASE_URL=https://api.deepseek.com cargo run --manifest-path examples/claude_code/Cargo.toml
```

## Controls

- Type a message and press `Enter` to send it.
- Press `Esc` to exit.
- Press `Ctrl+C` to exit.

The app restricts file paths to the current working directory and blocks a small
set of obviously destructive shell commands. It is still an example, not a
sandbox.
