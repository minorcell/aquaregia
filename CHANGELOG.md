# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the crate is `0.y.z`, minor version bumps may introduce breaking changes.

## [0.3.0] — 2026-05-30

### Added

- **`provider_options` escape hatch** at every layer of a request — `GenerateTextRequest::builder()`, `Agent::builder()`, `Message::with_provider_options()`, `TextPart::with_provider_options()`, `FilePart::with_provider_options()`. A JSON object keyed by provider slug; each adapter extracts its own block and merges it into the corresponding outbound JSON (request body, message object, or content block). Enables Anthropic extended thinking and prompt-caching breakpoints, Google safety settings, OpenAI reasoning effort, and any future provider-specific knob without waiting for a typed surface in the core.
- **PDF input** via the new `FilePart` type with `media_type: "application/pdf"`. Anthropic dispatches to `document` blocks; OpenAI Responses to `input_file` blocks (with optional `filename`); Google passes the `mimeType` through. The openai-compatible Chat Completions adapter rejects PDFs locally as `InvalidRequest`, since the wire contract has no standard representation.
- **Provider-native tools** (Anthropic `web_search_*`, OpenAI `web_search` / `file_search` / `code_interpreter`, Google `googleSearch`) work via `provider_options.<slug>.tools` — no dedicated `ProviderTool` type. README documents the overwrite semantics on the `tools` field.
- New examples: `anthropic_prompt_caching.rs`, `anthropic_web_search.rs`, `multimodal_pdf.rs`.

### Changed (breaking)

- **`ContentPart::Image(ImagePart)` → `ContentPart::File(FilePart)`**, aligned with the AI SDK v3 `LanguageModelV3FilePart` shape. `ImagePart` is removed.
- **`FilePart` requires `media_type: String`** (IANA media type, e.g. `image/jpeg`, `application/pdf`). The previous `Option<String>` field and the `unwrap_or("image/jpeg")` fallback inside every adapter are gone — unsupported media types surface as `ErrorCode::InvalidRequest` locally, before any HTTP round-trip.
- **`Message::user_image_url(url)` → `Message::user_file_url(url, media_type)`** — `media_type` is now a required argument.
- **`Message::user_image_bytes` → `Message::user_file_bytes`** — rename only; signature is unchanged.
- **`ContentPart::Text(String)` → `ContentPart::Text(TextPart)`**. Wire shape changes from `{"Text":"hi"}` to `{"Text":{"text":"hi"}}`. Convenience constructors (`Message::user_text`, etc.) still accept `impl Into<String>`; only callers that pattern-match the variant payload or hand-write the Message JSON need updates.
- **`ImagePart.provider_metadata` removed** — the field was never read by any adapter. The replacement on `FilePart` is `provider_options`, matching the outbound escape-hatch convention used by `TextPart` and `Message`.
- **`IntoTool` no longer implemented for `FnOnce() -> Tool`**. Call the constructor directly or use `ToolBuilder`. No production code path used the implicit form.
- **Top-level `provider_options` overwrite, not union**. Keys you set replace what the adapter computed for the same key — most relevant for `tools` arrays. To combine user tools with provider-native tools, put both in `provider_options.<slug>.tools` and skip `.tools(...)`.

### Changed (non-breaking)

- New `pub(crate) merge_provider_options` helper in `model_adapters/mod.rs` centralises the slug-to-target merge used by all four adapters. The four hand-rolled top-level merge blocks introduced in 0.2.x are gone.
- New `pub(crate) unsupported_media_type` helper produces a uniform `InvalidRequest` error across adapters for unsupported `FilePart` media types.
- OpenAI Responses and openai-compatible Chat Completions content shape stays a plain string by default; the typed content-array form is only emitted when a `TextPart` carries `provider_options` or a `FilePart` is present (since per-block fields can only ride a block object).

### Fixed

- Adapters no longer silently mislabel non-JPEG image payloads. Previously every base64/bytes image without `media_type` was sent with `image/jpeg`, which Anthropic and Google reject for PNG/WebP/HEIC sources.
- The `provider_options` field, once merged, properly rides every step of an `Agent` run — not only the first call. (Regression: 0.2.x hard-coded `None` inside `BoundClient::run_tools`.)

### Docs

- README and README_CN: new sections for `provider_options` across all four layers, a dispatch matrix for `FilePart` media types, and an explanation of the overwrite semantics on the `tools` field.
- Multimodal walkthrough rewritten around `FilePart` and IANA media types.

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
