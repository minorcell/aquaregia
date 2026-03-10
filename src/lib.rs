//! Provider-agnostic Rust toolkit for AI text generation, streaming, and tool-using agents.
//!
//! This crate exposes:
//! - [`LlmClient`] for provider-bound generate/stream calls.
//! - [`Agent`] for multi-step tool loops.
//! - [`tool()`], [`macro@tool`], and related types for defining executable tools.
//! - Shared request/response and stream types in [`types`].

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
pub mod axum_sse;

#[doc(hidden)]
pub use schemars as __aquaregia_schemars;
#[doc(hidden)]
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
    IntoModelRef,
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
pub fn openai(model: impl Into<String>) -> ModelRef<OpenAi> {
    ModelRef::<OpenAi>::new(model)
}

/// Creates a typed Anthropic model reference (`anthropic/<model>`).
pub fn anthropic(model: impl Into<String>) -> ModelRef<Anthropic> {
    ModelRef::<Anthropic>::new(model)
}

/// Creates a typed Google model reference (`google/<model>`).
pub fn google(model: impl Into<String>) -> ModelRef<Google> {
    ModelRef::<Google>::new(model)
}

/// Creates a typed OpenAI-compatible model reference (`openai-compatible/<model>`).
pub fn openai_compatible(model: impl Into<String>) -> ModelRef<OpenAiCompatible> {
    ModelRef::<OpenAiCompatible>::new(model)
}
