pub mod agent;
pub mod client;
pub mod error;
pub mod model_adapters;
pub mod provider;
pub mod stream;
pub mod tool;
pub mod types;

#[cfg(feature = "axum")]
pub mod axum_sse;

#[doc(hidden)]
pub use schemars as __aquaregia_schemars;
#[doc(hidden)]
pub use serde as __aquaregia_serde;

pub use agent::{Agent, AgentBuilder, AgentCallPlan, PrepareCallCallback};
pub use aquaregia_macros::tool;
pub use client::{BoundClient, ClientBuilder, LlmClient};
pub use error::{AiError, AiErrorCode};
pub use model_adapters::ModelAdapter;
pub use model_adapters::anthropic::AnthropicAdapterSettings;
pub use model_adapters::google::GoogleAdapterSettings;
pub use model_adapters::openai::OpenAiAdapterSettings;
pub use model_adapters::openai_compatible::OpenAiCompatibleAdapterSettings;
pub use tool::{
    IntoTool, Tool, ToolBuilder, ToolDescriptor, ToolExecError, ToolExecutor, ToolRegistry, tool,
};
pub use types::{
    Anthropic, ContentPart, FinishCallback, FinishReason, GenerateTextRequest,
    GenerateTextResponse, Google, IntoModelRef, Message, MessageRole, ModelRef, OpenAi,
    OpenAiCompatible, PrepareStepCallback, ProviderKind, ProviderMarker, RunTools, RunToolsFinish,
    RunToolsPrepareStep, RunToolsPreparedStep, RunToolsResponse, RunToolsStart, RunToolsStep,
    RunToolsStepStart, RunToolsToolCallFinish, RunToolsToolCallStart, StartCallback, StepCallback,
    StepStartCallback, StopWhen, StreamEvent, TextDeltaStream, TextStream, ToolCall,
    ToolCallFinishCallback, ToolCallStartCallback, ToolErrorPolicy, ToolResult, Usage,
};

pub fn openai(model: impl Into<String>) -> ModelRef<OpenAi> {
    ModelRef::<OpenAi>::new(model)
}

pub fn anthropic(model: impl Into<String>) -> ModelRef<Anthropic> {
    ModelRef::<Anthropic>::new(model)
}

pub fn google(model: impl Into<String>) -> ModelRef<Google> {
    ModelRef::<Google>::new(model)
}

pub fn openai_compatible(model: impl Into<String>) -> ModelRef<OpenAiCompatible> {
    ModelRef::<OpenAiCompatible>::new(model)
}
