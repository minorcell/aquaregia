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
//! use aquaregia::Client;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Client::openai_compatible().base_url("https://api.deepseek.com")
//!         .api_key(std::env::var("DEEPSEEK_API_KEY")?)
//!         .build()?;
//!
//!     let out = client
//!         .generate(aquaregia::ChatRequest::from_prompt(
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
//! - [`Client`]: Provider-bound client for `generate`, `stream`, and agent loops.
//! - [`Agent`]: Multi-step tool-using agent with configurable hooks.
//! - [`ModelAdapter`](adapters::ModelAdapter): Trait for provider-specific request/response handling.
//! - [`Tool`]: Executable tool definitions with JSON Schema validation.

/// Provider adapter traits and concrete provider implementations.
pub mod adapters;
/// Agent runtime and builder APIs.
pub mod agent;
/// Provider-bound client types and retry behavior.
pub mod client;
/// Embedding generation types and APIs.
pub mod embed;
/// Unified error types and HTTP-to-error mapping helpers.
pub mod error;
pub(crate) mod partial_json;
pub(crate) mod stream;
/// Tool definition, execution, and registry types.
pub mod tool;
/// Shared request/response and event types.
pub mod types;

pub use agent::Agent;
pub use client::Client;
pub use error::{Error, ErrorCode};

pub use tool::{Tool, tool};
pub use types::{
    AgentOutput, ChatRequest, ChatResponse, ContentPart, FilePart, FinishReason, MediaData,
    Message, MessageRole, ObjectResponse, OutputSchema, ReasoningPart, StreamEvent, TextPart,
    TextStream, ToolCall, ToolErrorPolicy, ToolResult, Usage,
};
