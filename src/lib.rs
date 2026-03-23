//! # Aquaregia
//!
//! A provider-agnostic Rust toolkit for building AI applications and tool-using agents.
//!
//! Aquaregia provides a unified API across OpenAI, Anthropic, Google, and OpenAI-compatible services,
//! with first-class support for reasoning-aware output, streaming events, multi-step tool execution,
//! and vision/image inputs.
//!
//! ## Features
//!
//! - **Unified Provider API**: One `LlmClient` binds to one provider configuration with support for
//!   OpenAI, Anthropic, Google, and OpenAI-compatible endpoints.
//! - **Streaming & Non-Streaming**: Both `generate` and `stream` APIs with consistent event handling.
//! - **Reasoning Support**: First-class reasoning content extraction and streaming events.
//! - **Tool-Using Agents**: Multi-step agent loops with configurable tool execution and error handling.
//! - **Multimodal Vision**: Send images to vision-capable models via URL, base64, or raw bytes.
//! - **Cancellation**: All requests and agent runs support cancellation via `CancellationToken`.
//! - **Telemetry**: Optional `tracing` spans for generate, stream, and agent operations.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use aquaregia::{GenerateTextRequest, LlmClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmClient::openai_compatible("https://api.deepseek.com")
//!         .api_key(std::env::var("DEEPSEEK_API_KEY")?)
//!         .build()?;
//!
//!     let out = client
//!         .generate(GenerateTextRequest::from_user_prompt(
//!             "deepseek-chat",
//!             "Explain Rust ownership in 3 bullet points.",
//!         ))
//!         .await?;
//!
//!     println!("{}", out.output_text);
//!     Ok(())
//! }
//! ```
//!
//! ## Crate Features
//!
//! | Feature     | Description                                                          |
//! | ----------- | -------------------------------------------------------------------- |
//! | `openai`    | OpenAI adapter (default)                                             |
//! | `anthropic` | Anthropic adapter (default)                                          |
//! | `telemetry` | `tracing` spans for `generate`, `stream`, agent steps and tool calls |
//! | `axum`      | Axum SSE bridge for converting streams into SSE responses            |
//!
//! ## Architecture
//!
//! - [`LlmClient`]: Entry point for creating provider-bound clients.
//! - [`BoundClient`]: Reusable client for `generate`, `stream`, and agent loops.
//! - [`Agent`]: Multi-step tool-using agent with configurable hooks.
//! - [`ModelAdapter`]: Trait for provider-specific request/response handling.
//! - [`Tool`]: Executable tool definitions with JSON Schema validation.

/// Agent runtime and builder APIs.
pub mod agent;
/// Provider-bound client types and retry behavior.
pub mod client;
/// Unified error types and HTTP-to-error mapping helpers.
pub mod error;
/// Provider adapter traits and concrete provider implementations.
pub mod model_adapters;
/// SSE frame parsing helpers used by streaming adapters.
pub mod stream;
/// Tool definition, execution, and registry types.
pub mod tool;
/// Shared request/response and event types.
pub mod types;

#[cfg(feature = "axum")]
/// Axum SSE bridge for converting [`TextStream`] into SSE responses.
///
/// This module provides integration with the Axum web framework, allowing
/// streaming responses to be converted into Server-Sent Events (SSE) for HTTP streaming.
pub mod axum_sse;

#[doc(hidden)]
/// Re-export of `schemars` for use in procedural macros.
///
/// This is an internal re-export used by the `#[tool]` macro to generate
/// JSON Schema implementations without requiring users to add `schemars`
/// as a direct dependency.
pub use schemars as __aquaregia_schemars;

#[doc(hidden)]
/// Re-export of `serde` for use in procedural macros.
///
/// This is an internal re-export used by the `#[tool]` macro to generate
/// Deserialize implementations without requiring users to add `serde`
/// as a direct dependency.
pub use serde as __aquaregia_serde;

pub use agent::{Agent, AgentBuilder, AgentRunPlan};
pub use aquaregia_macros::tool;
pub use client::{BoundClient, ClientBuilder, LlmClient};
pub use error::{Error, ErrorCode};
pub use model_adapters::ModelAdapter;
pub use model_adapters::anthropic::AnthropicAdapterSettings;
pub use model_adapters::google::GoogleAdapterSettings;
pub use model_adapters::openai::OpenAiAdapterSettings;
pub use model_adapters::openai_compatible::OpenAiCompatibleAdapterSettings;
pub use tokio_util::sync::CancellationToken;

pub use tool::{
    IntoTool, Tool, ToolBuilder, ToolDescriptor, ToolExecError, ToolExecutor, ToolRegistry, tool,
};
pub use types::{
    AgentFinish,
    AgentPrepareStep,
    AgentPreparedStep,
    AgentResponse,
    // Agent event types (used in hook callbacks)
    AgentStart,
    AgentStep,
    AgentStepStart,
    AgentToolCallFinish,
    AgentToolCallStart,
    // Provider markers
    Anthropic,
    // Messages
    ContentPart,
    // Streaming
    FinishReason,
    // Requests & responses
    GenerateTextRequest,
    GenerateTextResponse,
    Google,
    ImagePart,
    IntoModelRef,
    MediaData,
    Message,
    MessageRole,
    ModelRef,
    OpenAi,
    OpenAiCompatible,
    ProviderKind,
    ProviderMarker,
    ReasoningPart,
    StreamEvent,
    TextDeltaStream,
    TextStream,
    // Tool types
    ToolCall,
    ToolErrorPolicy,
    ToolResult,
    Usage,
};

/// Creates a typed OpenAI model reference (`openai/<model>`).
///
/// This function provides a convenient way to create a [`ModelRef`] for OpenAI models
/// with compile-time provider type safety.
///
/// # Arguments
///
/// * `model` - The OpenAI model identifier (e.g., `"gpt-4o"`, `"gpt-4o-mini"`)
///
/// # Example
///
/// ```
/// use aquaregia::openai;
///
/// let model = openai("gpt-4o");
/// assert_eq!(model.id(), "openai/gpt-4o");
/// ```
pub fn openai(model: impl Into<String>) -> ModelRef<OpenAi> {
    ModelRef::<OpenAi>::new(model)
}

/// Creates a typed Anthropic model reference (`anthropic/<model>`).
///
/// This function provides a convenient way to create a [`ModelRef`] for Anthropic models
/// with compile-time provider type safety.
///
/// # Arguments
///
/// * `model` - The Anthropic model identifier (e.g., `"claude-sonnet-4-5"`, `"claude-3-5-sonnet"`)
///
/// # Example
///
/// ```
/// use aquaregia::anthropic;
///
/// let model = anthropic("claude-sonnet-4-5");
/// assert_eq!(model.id(), "anthropic/claude-sonnet-4-5");
/// ```
pub fn anthropic(model: impl Into<String>) -> ModelRef<Anthropic> {
    ModelRef::<Anthropic>::new(model)
}

/// Creates a typed Google model reference (`google/<model>`).
///
/// This function provides a convenient way to create a [`ModelRef`] for Google Generative AI models
/// with compile-time provider type safety.
///
/// # Arguments
///
/// * `model` - The Google model identifier (e.g., `"gemini-2.0-flash"`, `"gemini-1.5-pro"`)
///
/// # Example
///
/// ```
/// use aquaregia::google;
///
/// let model = google("gemini-2.0-flash");
/// assert_eq!(model.id(), "google/gemini-2.0-flash");
/// ```
pub fn google(model: impl Into<String>) -> ModelRef<Google> {
    ModelRef::<Google>::new(model)
}

/// Creates a typed OpenAI-compatible model reference (`openai-compatible/<model>`).
///
/// This function provides a convenient way to create a [`ModelRef`] for OpenAI-compatible
/// endpoints (e.g., DeepSeek, local LLM servers) with compile-time provider type safety.
///
/// # Arguments
///
/// * `model` - The model identifier for the compatible endpoint (e.g., `"deepseek-chat"`)
///
/// # Example
///
/// ```
/// use aquaregia::openai_compatible;
///
/// let model = openai_compatible("deepseek-chat");
/// assert_eq!(model.id(), "openai-compatible/deepseek-chat");
/// ```
pub fn openai_compatible(model: impl Into<String>) -> ModelRef<OpenAiCompatible> {
    ModelRef::<OpenAiCompatible>::new(model)
}
