# Examples Guide

This directory contains runnable example applications built with Aquaregia.

Examples are app-driven: each one should show how to assemble Aquaregia into a
small working product shape, not how to call a single API method in isolation.
API-level snippets belong in the main README, docs, or tests.

## Structure

Use an independent Cargo package for each app example:

```text
examples/<app-name>/
  Cargo.toml
  README.md
  src/
    main.rs
```

Only include files the example needs to run. Web examples should keep their
frontend files beside the Rust server:

```text
examples/<web-app>/
  Cargo.toml
  README.md
  src/
    main.rs
  index.html
  static/
    app.js
    style.css
```

Each app package should depend on the local crate with:

```toml
aquaregia = { path = "../.." }
```

Keep app-only dependencies in the app package's `Cargo.toml`, not in the root
crate manifest.

## Running

Run an example through its package manifest:

```bash
cargo run --manifest-path examples/<app-name>/Cargo.toml
```

Set only the environment variables required by that example. Common variables
are:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_BASE_URL`

Each example's own `README.md` should list the exact variables it needs.

## What Belongs Here

Good examples:

- Build a small complete app.
- Keep the first run path obvious.
- Use plain local setup steps.
- Show one primary workflow end to end.
- Keep provider-specific code inside provider-specific examples.

Avoid examples that only demonstrate one method name such as `generate`,
`stream`, `embed`, or `tool`. Those belong in docs or tests unless they are part
of a complete app workflow.

## Naming

Name examples after the app or workflow, not the API surface.

Good names:

- `chatgpt`
- `claude_code`
- `semantic_search`
- `document_intake`
- `support_triage`

Avoid names that describe only an API method or provider option.
