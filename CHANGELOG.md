# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the crate is `0.y.z`, minor version bumps may introduce breaking changes.

## [0.2.0] — 2026-05-28

### Added

- **Structured streaming output**: `BoundClient::stream_object::<T>()` emits `StreamObjectEvent::Partial { partial }` as new fields arrive and a final `StreamObjectEvent::Object { object }` when the stream completes. Backed by a stack-based partial JSON repairer that handles truncated strings, unclosed arrays/objects, and escape sequences mid-token.
- **Structured (non-streaming) output**: `BoundClient::generate_object::<T>()` returns a typed `GenerateObjectResponse<T>`. Schema is derived from `T: schemars::JsonSchema`. Providers without native structured-output mode (Anthropic, Google) fall back transparently to forced tool calls.
- New example: `examples/structured_streaming.rs`.

### Changed (breaking)

- **Model reference is now `String`** (was `ModelRef` newtype). Pass any `impl Into<String>` to `GenerateTextRequest::from_user_prompt` / `::builder` / `Agent::builder`.
- **`BuildProvider` is now sealed** via `pub trait BuildProvider: private::Sealed`. External adapters belong in the `model_adapters` module rather than as downstream `impl`s.
- **`telemetry` feature flag removed**, and with it the optional `tracing` dependency. The flag previously gated no code.
- **`ProviderKind` enum removed** (was unused).
- **`ModelRef` newtype removed** (use `String` directly).

### Changed (non-breaking)

- `Agent` / `AgentBuilder` / `RunTools` field mirror collapsed — internal restructuring; user-facing builder API unchanged.
- Base64 encoding switched from a hand-rolled implementation to the `base64` crate.
- `#[doc(hidden)]` annotations on `BuildProvider` methods removed (the trait is sealed; the hidden marker was redundant).
- Default Claude model identifier in tests and examples bumped to `claude-haiku-4-5`.

### Fixed

- `partial_json` repair: rewritten as a stack-based state machine; hardened against unicode escape truncation; root arrays supported.
- `stream_object` schema injection deduplicated; `type_name` sanitised; final buffer flushed on EOF.
- `cargo doc` no longer warns about broken intra-doc links to the (private) `ToolRegistry`.

### Docs

- README.md and README_CN.md fully rewritten with an onboarding-flow narrative (Introduction → Quick Start → Essentials → Production → Integration → Reference), retaining all reference tables.
- `examples/README.md`: removed phantom `google_generate.rs` reference; added `structured_streaming.rs` to the list.
