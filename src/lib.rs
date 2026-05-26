//! # Aquaregia
//!
//! A single crate for building LLM-powered applications and agents across any provider.
//!
//! ## Features
//!
//! - **Unified Provider API**: One `LlmClient` binds to one provider configuration with support for
//!   OpenAI, Anthropic, Google, and OpenAI-compatible endpoints.
//! - **Streaming & Non-Streaming**: Both `generate` and `stream` APIs with consistent event handling.
//! - **Structured Output**: `generate_object::<T>()` deserialises responses directly into Rust types
//!   using `schemars`-derived JSON Schema, with provider-native support (OpenAI) and tool-use fallback
//!   (Anthropic, Google).
//! - **Reasoning Support**: First-class reasoning content extraction and streaming events.
//! - **Tool-Using Agents**: Multi-step agent loops with configurable tool execution and error handling.
//! - **Multimodal Vision**: Send images to vision-capable models via URL, base64, or raw bytes.
//! - **Cancellation**: All requests and agent runs support cancellation via `CancellationToken`.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use aquaregia::{GenerateTextRequest, LlmClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmClient::openai_compatible().base_url("https://api.deepseek.com")
//!         .api_key(std::env::var("DEEPSEEK_API_KEY")?)
//!         .build()?;
//!
//!     let out = client
//!         .generate(GenerateTextRequest::from_user_prompt(
//!             "deepseek-v4-pro",
//!             "Explain Rust ownership in 3 bullet points.",
//!         ))
//!         .await?;
//!
//!     println!("{}", out.output_text);
//!     Ok(())
//! }
//! ```
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
pub(crate) mod partial_json;
pub(crate) mod stream;
/// Tool definition, execution, and registry types.
pub mod tool;
/// Shared request/response and event types.
pub mod types;

pub use agent::{Agent, AgentBuilder};
pub use client::{BoundClient, ClientBuilder, LlmClient};
pub use error::{Error, ErrorCode};
pub use model_adapters::ModelAdapter;
pub use model_adapters::anthropic::AnthropicAdapterSettings;
pub use model_adapters::google::GoogleAdapterSettings;
pub use model_adapters::openai::OpenAiAdapterSettings;
pub use model_adapters::openai_compatible::OpenAiCompatibleAdapterSettings;
pub use tokio_util::sync::CancellationToken;

pub use tool::{IntoTool, Tool, ToolBuilder, ToolDescriptor, ToolExecError, ToolExecutor, tool};
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
    // Messages
    ContentPart,
    // Streaming
    FinishReason,
    // Structured output
    GenerateObjectResponse,
    // Requests & responses
    GenerateTextRequest,
    GenerateTextResponse,
    ImagePart,
    MediaData,
    Message,
    MessageRole,
    ModelRef,
    ObjectStream,
    OutputSchema,
    ProviderKind,
    ReasoningPart,
    StreamEvent,
    StreamObjectEvent,
    TextDeltaStream,
    TextStream,
    // Tool types
    ToolCall,
    ToolErrorPolicy,
    ToolResult,
    Usage,
};
