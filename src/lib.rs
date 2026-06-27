//! # Aquaregia
//!
//! A lightweight agent SDK for Rust: build tool-using LLM agents that run on any provider.
//!
//! ## Features
//!
//! - **Tool-Using Agents**: Multi-step agent loops with `prepare_step` hooks, `stop_when`, and
//!   configurable tool execution and error handling — you describe the tools, the agent runs the loop.
//! - **Provider-Portable**: The same agent (and every `generate`/`stream` call) runs on OpenAI,
//!   Anthropic, Google, and OpenAI-compatible endpoints — swap a constructor to change provider.
//! - **Streaming & Non-Streaming**: Both `generate` and `stream` APIs with consistent event handling.
//! - **Structured Output**: `generate_object::<T>()` deserialises responses directly into Rust types
//!   using `schemars`-derived JSON Schema, with provider-native support (OpenAI) and tool-use fallback
//!   (Anthropic, Google).
//! - **Reasoning Support**: First-class reasoning content extraction and streaming events.
//! - **Multimodal Vision**: Send images to vision-capable models via URL, base64, or raw bytes.
//! - **Cancellation**: All requests and agent runs support cancellation via `CancellationToken`.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use aquaregia::providers::openai;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let agent = openai::Client::from_env()?
//!         .agent("gpt-5.5")
//!         .build()?;
//!
//!     let out = agent.prompt("Explain Rust ownership in 3 bullet points.").await?;
//!
//!     println!("{out}");
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! - [`providers`]: Provider-specific client entry points.
//! - [`Agent`]: Multi-step tool-using agent with configurable hooks.
//! - [`Tool`]: Executable tool definitions with JSON Schema validation.

pub(crate) mod adapters;
/// Agent runtime and builder APIs.
pub mod agent;
pub(crate) mod client;
/// Embedding generation types and APIs.
pub mod embed;
/// Unified error types and HTTP-to-error mapping helpers.
pub mod error;
pub(crate) mod partial_json;
/// Provider-specific client entry points.
pub mod providers;
pub(crate) mod stream;
/// Tool definition, execution, and registry types.
pub mod tool;
/// Shared request/response and event types.
pub mod types;

pub use agent::Agent;
pub use error::{Error, ErrorCode};

pub use tool::{Tool, tool};
pub use types::{
    AgentOutput, AgentStream, AgentStreamEvent, ChatRequest, ChatResponse, ContentPart, FilePart,
    FinishReason, MediaData, Message, MessageRole, ObjectResponse, OutputSchema, ReasoningPart,
    StreamEvent, TextPart, TextStream, ToolCall, ToolErrorPolicy, ToolResult, Usage,
};
