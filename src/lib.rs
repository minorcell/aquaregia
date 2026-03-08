pub mod agent;
pub mod client;
pub mod error;
pub mod model_adapters;
pub mod stream;
pub mod tool;
pub mod types;

#[cfg(feature = "axum")]
pub mod axum_sse;

#[doc(hidden)]
pub use schemars as __aquaregia_schemars;
#[doc(hidden)]
pub use serde as __aquaregia_serde;

pub use agent::{Agent, AgentBuilder, AgentCallPlan};
pub use aquaregia_macros::tool;
pub use client::{BoundClient, ClientBuilder, LlmClient};
pub use error::{AiError, AiErrorCode};
pub use model_adapters::ModelAdapter;
pub use model_adapters::anthropic::AnthropicAdapterSettings;
pub use model_adapters::google::GoogleAdapterSettings;
pub use model_adapters::openai::OpenAiAdapterSettings;

pub use tool::{IntoTool, Tool, ToolBuilder, ToolDescriptor, ToolExecError, ToolExecutor, ToolRegistry, tool};
pub use types::{
    // Provider markers
    Anthropic, Google, OpenAi, OpenAiCompatible,
    IntoModelRef, ModelRef, ProviderKind, ProviderMarker,
    // Messages
    ContentPart, Message, MessageRole,
    // Requests & responses
    GenerateTextRequest, GenerateTextResponse, AgentResponse,
    // Agent event types (used in hook callbacks)
    AgentStart, AgentStep, AgentStepStart, AgentFinish,
    AgentToolCallStart, AgentToolCallFinish,
    AgentPrepareStep, AgentPreparedStep,
    // Streaming
    FinishReason, StreamEvent, TextDeltaStream, TextStream,
    // Tool types
    ToolCall, ToolErrorPolicy, ToolResult, Usage,
};

pub fn openai_model(model: impl Into<String>) -> ModelRef<OpenAi> {
    ModelRef::<OpenAi>::new(model)
}

pub fn anthropic_model(model: impl Into<String>) -> ModelRef<Anthropic> {
    ModelRef::<Anthropic>::new(model)
}

pub fn google_model(model: impl Into<String>) -> ModelRef<Google> {
    ModelRef::<Google>::new(model)
}

pub fn openai_compatible_model(model: impl Into<String>) -> ModelRef<OpenAiCompatible> {
    ModelRef::<OpenAiCompatible>::new(model)
}
