//! Shared request/response types and streaming events for Aquaregia.
//!
//! This module defines the core data structures used throughout the Aquaregia SDK:
//!
//! - **Messages**: Provider-agnostic chat message types with support for text, images, reasoning, and tool content
//! - **Requests/Responses**: Structured generation request and response types
//! - **Streaming**: Event types emitted during streaming generation
//! - **Agent Types**: Event types and plan structures for multi-step agent loops
//! - **Usage**: Token usage counters with cache and reasoning token support

use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, ErrorCode};
use crate::tool::{IntoTool, Tool, ToolDescriptor};

/// Chat message role used across providers.
///
/// This enum represents the standard roles in a multi-turn conversation,
/// following the convention used by major LLM providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    /// System/instruction message providing behavioral guidelines.
    System,
    /// User message containing a prompt or question.
    User,
    /// Assistant/model message containing a response.
    Assistant,
    /// Tool result message containing execution output.
    Tool,
}

/// Provider-agnostic chat message.
///
/// This struct represents a single message in a conversation, supporting
/// multiple content types through [`ContentPart`] enumeration. Messages
/// are the fundamental building blocks of LLM conversations.
///
/// # Structure
///
/// A message consists of:
/// - A [`MessageRole`] indicating the sender
/// - A list of [`ContentPart`] items (text, reasoning, tool calls, etc.)
/// - An optional name for authorship attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message role indicating the sender type.
    pub(crate) role: MessageRole,
    /// Content parts making up the message body.
    pub(crate) parts: Vec<ContentPart>,
    /// Optional author/tool name for attribution.
    pub(crate) name: Option<String>,
}

impl Message {
    /// Creates a message with custom role and parts.
    ///
    /// Prefer the named constructors ([`Message::system_text`], [`Message::user_text`],
    /// [`Message::assistant_text`], [`Message::tool_result`]) for common cases. Use `new` only
    /// when you need to build a message with custom [`ContentPart`] combinations.
    ///
    /// # Arguments
    ///
    /// * `role` - The message role
    /// * `parts` - Vector of content parts
    ///
    /// # Errors
    ///
    /// Returns an error if the message parts are invalid for the given role.
    pub fn new(role: MessageRole, parts: Vec<ContentPart>) -> Result<Self, Error> {
        validate_message_parts(role.clone(), &parts)?;
        Ok(Self {
            role,
            parts,
            name: None,
        })
    }

    /// Creates a system message containing one text part.
    ///
    /// System messages provide behavioral instructions to the model,
    /// setting the context and tone for the conversation.
    ///
    /// # Arguments
    ///
    /// * `text` - The system instruction text
    pub fn system_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            parts: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    /// Creates a user message containing one text part.
    ///
    /// User messages represent prompts or questions from the end user.
    ///
    /// # Arguments
    ///
    /// * `text` - The user prompt text
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            parts: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    /// Creates an assistant message containing one text part.
    ///
    /// Assistant messages represent model responses in a conversation.
    ///
    /// # Arguments
    ///
    /// * `text` - The assistant response text
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            parts: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    /// Creates a tool-role message containing one tool result part.
    ///
    /// Tool messages carry the execution results back to the model
    /// after a tool call has been processed.
    ///
    /// # Arguments
    ///
    /// * `result` - The tool execution result
    pub fn tool_result(result: ToolResult) -> Self {
        Self {
            role: MessageRole::Tool,
            parts: vec![ContentPart::ToolResult(result)],
            name: None,
        }
    }

    /// Returns the message role.
    pub fn role(&self) -> MessageRole {
        self.role.clone()
    }

    /// Returns message content parts.
    pub fn parts(&self) -> &[ContentPart] {
        &self.parts
    }

    /// Internal constructor for assistant messages with multiple content parts.
    pub(crate) fn assistant_with_parts(parts: Vec<ContentPart>) -> Self {
        Self {
            role: MessageRole::Assistant,
            parts,
            name: None,
        }
    }

    /// Creates a user message with a single image URL.
    pub fn user_image_url(url: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            parts: vec![ContentPart::Image(ImagePart {
                data: MediaData::Url(url.into()),
                media_type: None,
                provider_metadata: None,
            })],
            name: None,
        }
    }

    /// Creates a user message with image bytes and MIME type.
    pub fn user_image_bytes(bytes: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            parts: vec![ContentPart::Image(ImagePart {
                data: MediaData::Bytes(bytes),
                media_type: Some(media_type.into()),
                provider_metadata: None,
            })],
            name: None,
        }
    }
}

/// Raw media data for image content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediaData {
    /// Remote URL or data URL.
    Url(String),
    /// Raw base64 string (no `data:` prefix).
    Base64(String),
    /// Raw bytes; adapters will base64-encode as needed.
    Bytes(Vec<u8>),
}

/// Image content block for vision inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePart {
    /// Image data.
    pub data: MediaData,
    /// MIME type (e.g. `"image/jpeg"`).
    /// Required for Bytes/Base64; optional for Url.
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional provider-specific metadata.
    pub provider_metadata: Option<Value>,
}

/// Content block used in a message.
///
/// This enum represents the different types of content that can appear
/// in a message, enabling rich multi-modal conversations with support
/// for text, reasoning traces, tool interactions, and images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    /// Plain text content.
    Text(String),
    /// Image content for vision inputs.
    Image(ImagePart),
    /// Provider reasoning content (chain-of-thought traces).
    Reasoning(ReasoningPart),
    /// Tool call requested by the model.
    ToolCall(ToolCall),
    /// Tool execution result returned to the model.
    ToolResult(ToolResult),
}

/// Reasoning content block.
///
/// This struct holds chain-of-thought or reasoning traces generated by
/// reasoning-capable models. The content may include provider-specific
/// metadata such as signatures for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPart {
    /// Reasoning text content.
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional provider-specific metadata (for example signatures).
    pub provider_metadata: Option<Value>,
}

/// Tool call requested by the model.
///
/// This struct represents a function/tool invocation request emitted
/// by the model. It contains all information needed to execute the tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-generated call identifier for correlating results.
    pub call_id: String,
    /// Tool name to execute.
    pub tool_name: String,
    /// JSON arguments for the tool invocation.
    pub args_json: Value,
}

/// Tool execution result sent back to the model.
///
/// This struct carries the output of a tool execution back to the model,
/// enabling multi-step tool-using conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Matches [`ToolCall::call_id`] for correlation.
    pub call_id: String,
    /// JSON output payload from tool execution.
    pub output_json: Value,
    /// Indicates whether this payload represents a tool error.
    pub is_error: bool,
}

/// Schema specification for structured output generation.
///
/// When set on a [`GenerateTextRequest`], adapters use provider-native structured
/// output mechanisms (or tool-use fallback) to constrain the model to valid JSON
/// matching the given schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSchema {
    /// Short name sent to the provider (e.g. `"output"`).
    pub name: String,
    /// Optional human-readable description for the output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema value (typically generated via [`schemars::JsonSchema`]).
    pub json_schema: Value,
}

/// Request for generation/streaming calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) tools: Option<Vec<ToolDescriptor>>,
    pub(crate) output_schema: Option<OutputSchema>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) provider_options: Option<Value>,
    #[serde(skip)]
    pub(crate) cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

impl GenerateTextRequest {
    /// Returns the optional output schema for structured generation.
    pub fn output_schema(&self) -> Option<&OutputSchema> {
        self.output_schema.as_ref()
    }

    /// Returns the optional provider-specific options.
    pub fn provider_options(&self) -> Option<&Value> {
        self.provider_options.as_ref()
    }

    /// Builds a one-message request from a user prompt.
    pub fn from_user_prompt(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: vec![Message::user_text(prompt)],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stop_sequences: vec![],
            tools: None,
            output_schema: None,
            provider_options: None,
            cancellation_token: None,
        }
    }

    /// Starts a request builder.
    pub fn builder(model: impl Into<String>) -> GenerateTextRequestBuilder {
        GenerateTextRequestBuilder {
            request: Self {
                model: model.into(),
                messages: Vec::new(),
                temperature: None,
                top_p: None,
                max_output_tokens: None,
                stop_sequences: vec![],
                tools: None,
                output_schema: None,
                provider_options: None,
                cancellation_token: None,
            },
        }
    }
}

/// Builder for [`GenerateTextRequest`].
pub struct GenerateTextRequestBuilder {
    request: GenerateTextRequest,
}

impl GenerateTextRequestBuilder {
    /// Appends one message.
    pub fn message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    /// Replaces all messages.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.request.messages = messages.into_iter().collect();
        self
    }

    /// Replaces messages with a single user prompt.
    pub fn user_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.request.messages = vec![Message::user_text(prompt)];
        self
    }

    /// Sets sampling temperature in range `0.0..=2.0`.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.request.temperature = Some(temperature);
        self
    }

    /// Sets nucleus sampling value in range `0.0..=1.0`.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.request.top_p = Some(top_p);
        self
    }

    /// Sets max output token budget.
    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.request.max_output_tokens = Some(max_output_tokens);
        self
    }

    /// Sets stop sequences.
    pub fn stop_sequences<S: Into<String>>(
        mut self,
        stop_sequences: impl IntoIterator<Item = S>,
    ) -> Self {
        self.request.stop_sequences = stop_sequences.into_iter().map(Into::into).collect();
        self
    }

    /// Sets tools available to the model in this request.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDescriptor>) -> Self {
        let tools = tools.into_iter().collect::<Vec<_>>();
        self.request.tools = if tools.is_empty() { None } else { Some(tools) };
        self
    }

    /// Sets the output schema for structured generation.
    ///
    /// When set, providers constrain the model to produce valid JSON matching
    /// the given schema. Prefer [`BoundClient::generate_object`] for
    /// automatic schema derivation from Rust types.
    ///
    /// [`BoundClient::generate_object`]: crate::BoundClient::generate_object
    pub fn output_schema(mut self, output_schema: OutputSchema) -> Self {
        self.request.output_schema = Some(output_schema);
        self
    }

    /// Sets provider-specific options passed through to the adapter.
    ///
    /// The value should be a JSON object keyed by provider slug (e.g. `"anthropic"`,
    /// `"openai"`, `"google"`). Each adapter extracts and merges its own options
    /// into the request payload without validation.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use aquaregia::GenerateTextRequest;
    /// use serde_json::json;
    ///
    /// let req = GenerateTextRequest::builder("claude-sonnet-4-6")
    ///     .user_prompt("Explain Rust ownership")
    ///     .provider_options(json!({
    ///         "anthropic": {
    ///             "thinking": {
    ///                 "type": "enabled",
    ///                 "budget_tokens": 10000
    ///             }
    ///         }
    ///     }))
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn provider_options(mut self, options: Value) -> Self {
        self.request.provider_options = Some(options);
        self
    }

    /// Adds a cancellation token checked by adapters and streams.
    pub fn cancellation_token(mut self, token: tokio_util::sync::CancellationToken) -> Self {
        self.request.cancellation_token = Some(token);
        self
    }

    /// Validates and finalizes the request.
    pub fn build(self) -> Result<GenerateTextRequest, Error> {
        validate_model_ref(&self.request.model)?;
        validate_messages(&self.request.messages)?;
        validate_sampling(self.request.temperature, self.request.top_p)?;
        Ok(self.request)
    }
}

/// Policy for handling tool execution errors inside agent loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ToolErrorPolicy {
    /// Convert tool failures into error-shaped tool results and continue.
    #[default]
    ContinueAsToolResult,
    /// Abort the run immediately when a tool fails.
    FailFast,
}

#[derive(Clone)]
pub(crate) struct RunTools {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Vec<Tool>,
    pub(crate) max_steps: Option<u32>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) prepare_step: Option<PrepareStepHook>,
    pub(crate) on_start: Option<Hook<AgentStart>>,
    pub(crate) on_step_start: Option<Hook<AgentStepStart>>,
    pub(crate) on_tool_call_start: Option<Hook<AgentToolCallStart>>,
    pub(crate) on_tool_call_finish: Option<Hook<AgentToolCallFinish>>,
    pub(crate) on_step_finish: Option<Hook<AgentStep>>,
    pub(crate) on_finish: Option<Hook<AgentFinish>>,
    pub(crate) stop_when: Option<StopPredicate>,
    pub(crate) tool_error_policy: ToolErrorPolicy,
    pub(crate) provider_options: Option<Value>,
    pub(crate) cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

impl RunTools {
    pub(crate) fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            tools: Vec::new(),
            max_steps: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stop_sequences: Vec::new(),
            prepare_step: None,
            on_start: None,
            on_step_start: None,
            on_tool_call_start: None,
            on_tool_call_finish: None,
            on_step_finish: None,
            on_finish: None,
            stop_when: None,
            tool_error_policy: ToolErrorPolicy::ContinueAsToolResult,
            provider_options: None,
            cancellation_token: None,
        }
    }

    pub(crate) fn tools<I, T>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: IntoTool,
    {
        self.tools
            .extend(tools.into_iter().map(IntoTool::into_tool));
        self
    }

    pub(crate) fn max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    pub(crate) fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub(crate) fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub(crate) fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub(crate) fn stop_sequences<S: Into<String>>(
        mut self,
        stop_sequences: impl IntoIterator<Item = S>,
    ) -> Self {
        self.stop_sequences = stop_sequences.into_iter().map(Into::into).collect();
        self
    }

    pub(crate) fn prepare_step<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentPrepareStep) -> AgentPreparedStep + Send + Sync + 'static,
    {
        self.prepare_step = Some(Arc::new(callback));
        self
    }

    pub(crate) fn on_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStart) + Send + Sync + 'static,
    {
        self.on_start = Some(Arc::new(callback));
        self
    }

    pub(crate) fn on_step_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStepStart) + Send + Sync + 'static,
    {
        self.on_step_start = Some(Arc::new(callback));
        self
    }

    pub(crate) fn on_tool_call_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallStart) + Send + Sync + 'static,
    {
        self.on_tool_call_start = Some(Arc::new(callback));
        self
    }

    pub(crate) fn on_tool_call_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallFinish) + Send + Sync + 'static,
    {
        self.on_tool_call_finish = Some(Arc::new(callback));
        self
    }

    pub(crate) fn on_step_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStep) + Send + Sync + 'static,
    {
        self.on_step_finish = Some(Arc::new(callback));
        self
    }

    pub(crate) fn on_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentFinish) + Send + Sync + 'static,
    {
        self.on_finish = Some(Arc::new(callback));
        self
    }

    pub(crate) fn stop_when<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&AgentStep) -> bool + Send + Sync + 'static,
    {
        self.stop_when = Some(Arc::new(predicate));
        self
    }

    pub(crate) fn tool_error_policy(mut self, policy: ToolErrorPolicy) -> Self {
        self.tool_error_policy = policy;
        self
    }

    pub(crate) fn provider_options(mut self, options: Value) -> Self {
        self.provider_options = Some(options);
        self
    }

    pub(crate) fn cancellation_token(mut self, token: tokio_util::sync::CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    pub(crate) fn build(self) -> Result<Self, Error> {
        validate_model_ref(&self.model)?;
        validate_messages(&self.messages)?;
        validate_sampling(self.temperature, self.top_p)?;
        Ok(self)
    }
}

/// Emitted once before the first agent step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStart {
    /// Fully-qualified model id used by the run.
    pub model_id: String,
    /// Initial messages passed into the run.
    pub messages: Vec<Message>,
    /// Number of registered tools.
    pub tool_count: usize,
    /// Effective max step cap for this run. `0` means unlimited.
    pub max_steps: u32,
}

/// Emitted when an agent step begins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepStart {
    /// 1-based step index.
    pub step: u32,
    /// Messages sent to the model for this step.
    pub messages: Vec<Message>,
}

/// Emitted right before executing one tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolCallStart {
    /// 1-based step index.
    pub step: u32,
    /// Tool call about to execute.
    pub tool_call: ToolCall,
}

/// Emitted right after executing one tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolCallFinish {
    /// 1-based step index.
    pub step: u32,
    /// Executed tool call.
    pub tool_call: ToolCall,
    /// Result returned by the tool runtime.
    pub tool_result: ToolResult,
    /// Tool execution duration in milliseconds.
    pub duration_ms: u64,
}

/// Result snapshot for one completed agent step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    /// 1-based step index.
    pub step: u32,
    /// Assistant text output for this step.
    pub output_text: String,
    #[serde(default)]
    /// Flattened reasoning text (legacy convenience field).
    pub reasoning_text: String,
    #[serde(default)]
    /// Structured reasoning parts for this step.
    pub reasoning_parts: Vec<ReasoningPart>,
    /// Provider finish reason.
    pub finish_reason: FinishReason,
    /// Token usage for this single step.
    pub usage: Usage,
    /// Tool calls requested by the model in this step.
    pub tool_calls: Vec<ToolCall>,
    /// Tool results produced in this step.
    pub tool_results: Vec<ToolResult>,
}

/// Emitted once when the agent run ends successfully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFinish {
    /// Final assistant text.
    pub output_text: String,
    /// Number of executed steps.
    pub step_count: u32,
    /// Finish reason from the final model call.
    pub finish_reason: FinishReason,
    /// Accumulated token usage across all steps.
    pub usage_total: Usage,
    /// Full message transcript produced by the run.
    pub transcript: Vec<Message>,
    /// Per-step snapshots.
    pub step_results: Vec<AgentStep>,
}

/// Per-step mutable input passed to `prepare_step`.
#[derive(Debug, Clone)]
pub struct AgentPrepareStep {
    /// 1-based step index to be executed.
    pub step: u32,
    /// Model selected for this step.
    pub model: String,
    /// Messages that will be sent unless changed.
    pub messages: Vec<Message>,
    /// Tools currently available for this step.
    pub tools: Vec<Tool>,
    /// Sampling temperature for this step.
    pub temperature: Option<f32>,
    /// Max output token budget for this step.
    pub max_output_tokens: Option<u32>,
    /// Stop sequences for this step.
    pub stop_sequences: Vec<String>,
    /// Completed previous step snapshots.
    pub previous_steps: Vec<AgentStep>,
}

/// Finalized step input returned by `prepare_step`.
#[derive(Debug, Clone)]
pub struct AgentPreparedStep {
    /// Model selected for this step.
    pub model: String,
    /// Messages to send for this step.
    pub messages: Vec<Message>,
    /// Tools available for this step.
    pub tools: Vec<Tool>,
    /// Sampling temperature for this step.
    pub temperature: Option<f32>,
    /// Max output token budget for this step.
    pub max_output_tokens: Option<u32>,
    /// Stop sequences for this step.
    pub stop_sequences: Vec<String>,
}

// ─────────── Callback type aliases ──────────────────────────────────────────

pub(crate) type Hook<T> = Arc<dyn Fn(&T) + Send + Sync>;
pub(crate) type PrepareStepHook = Arc<dyn Fn(&AgentPrepareStep) -> AgentPreparedStep + Send + Sync>;
pub(crate) type StopPredicate = Arc<dyn Fn(&AgentStep) -> bool + Send + Sync>;

/// Normalized non-streaming generation response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextResponse {
    /// Assistant text output.
    pub output_text: String,
    #[serde(default)]
    /// Flattened reasoning text (legacy convenience field).
    pub reasoning_text: String,
    #[serde(default)]
    /// Structured reasoning parts.
    pub reasoning_parts: Vec<ReasoningPart>,
    /// Provider finish reason.
    pub finish_reason: FinishReason,
    /// Token usage for this request.
    pub usage: Usage,
    /// Tool calls emitted by the model.
    pub tool_calls: Vec<ToolCall>,
    /// Best-effort raw provider response for debugging.
    pub raw_provider_response: Option<Value>,
}

/// Structured output returned by [`BoundClient::generate_object`].
///
/// [`BoundClient::generate_object`]: crate::BoundClient::generate_object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateObjectResponse<T> {
    /// Deserialized structured output.
    pub object: T,
    #[serde(default)]
    /// Reasoning/chain-of-thought text from the model.
    pub reasoning_text: String,
    /// Provider finish reason.
    pub finish_reason: FinishReason,
    /// Token usage for this request.
    pub usage: Usage,
    /// Best-effort raw provider response for debugging.
    pub raw_provider_response: Option<Value>,
}

/// Final response of a completed agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// Final assistant text output.
    pub output_text: String,
    /// Number of executed steps.
    pub steps: u32,
    /// Full transcript (including tool results).
    pub transcript: Vec<Message>,
    /// Accumulated token usage.
    pub usage_total: Usage,
    /// Per-step snapshots.
    pub step_results: Vec<AgentStep>,
}

/// Provider finish reasons normalized across adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// Model stopped naturally.
    Stop,
    /// Output was cut by token limit.
    Length,
    /// Model expects tool execution before final answer.
    ToolCalls,
    /// Content was filtered by provider policy.
    ContentFilter,
    /// Provider paused the turn for continuation (Anthropic `pause_turn`).
    PauseTurn,
    /// Provider refused the request (Anthropic `refusal`).
    Refusal,
    /// Any provider-specific reason not mapped above.
    Unknown(String),
}

/// Token usage counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt/input tokens.
    pub input_tokens: u32,
    #[serde(default)]
    /// Non-cached input tokens (best effort when provider exposes cache split).
    pub input_no_cache_tokens: u32,
    #[serde(default)]
    /// Cached input tokens read from provider cache (best effort).
    pub input_cache_read_tokens: u32,
    #[serde(default)]
    /// Input tokens used to create/write cache entries (best effort).
    pub input_cache_write_tokens: u32,
    /// Completion/output tokens.
    pub output_tokens: u32,
    #[serde(default)]
    /// Output text tokens when provider exposes text/reasoning split.
    pub output_text_tokens: u32,
    #[serde(default)]
    /// Provider-reported reasoning tokens (if available).
    pub reasoning_tokens: u32,
    /// Total tokens (`input + output + reasoning` when reported).
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Raw provider usage payload for debugging and future extensions.
    pub raw_usage: Option<Value>,
}

impl std::ops::Add for Usage {
    type Output = Usage;

    fn add(self, rhs: Self) -> Self::Output {
        let mut usage = Usage {
            input_tokens: self.input_tokens.saturating_add(rhs.input_tokens),
            input_no_cache_tokens: self
                .input_no_cache_tokens
                .saturating_add(rhs.input_no_cache_tokens),
            input_cache_read_tokens: self
                .input_cache_read_tokens
                .saturating_add(rhs.input_cache_read_tokens),
            input_cache_write_tokens: self
                .input_cache_write_tokens
                .saturating_add(rhs.input_cache_write_tokens),
            output_tokens: self.output_tokens.saturating_add(rhs.output_tokens),
            output_text_tokens: self
                .output_text_tokens
                .saturating_add(rhs.output_text_tokens),
            reasoning_tokens: self.reasoning_tokens.saturating_add(rhs.reasoning_tokens),
            total_tokens: self.total_tokens.saturating_add(rhs.total_tokens),
            raw_usage: None,
        };
        usage.normalize_usage_fields();
        usage
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        let rhs_has_raw_usage = rhs.raw_usage.is_some();
        let lhs_was_zero = self.is_zero_numeric();

        self.input_tokens = self.input_tokens.saturating_add(rhs.input_tokens);
        self.input_no_cache_tokens = self
            .input_no_cache_tokens
            .saturating_add(rhs.input_no_cache_tokens);
        self.input_cache_read_tokens = self
            .input_cache_read_tokens
            .saturating_add(rhs.input_cache_read_tokens);
        self.input_cache_write_tokens = self
            .input_cache_write_tokens
            .saturating_add(rhs.input_cache_write_tokens);
        self.output_tokens = self.output_tokens.saturating_add(rhs.output_tokens);
        self.output_text_tokens = self
            .output_text_tokens
            .saturating_add(rhs.output_text_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(rhs.reasoning_tokens);
        self.total_tokens = self.total_tokens.saturating_add(rhs.total_tokens);
        self.raw_usage = if lhs_was_zero {
            rhs.raw_usage.or_else(|| self.raw_usage.take())
        } else if rhs_has_raw_usage {
            None
        } else {
            self.raw_usage.take()
        };
        self.normalize_usage_fields();
    }
}

impl Usage {
    /// Builds usage from provider totals and back-fills derived counters.
    pub(crate) fn from_totals(
        input_tokens: u32,
        output_tokens: u32,
        reasoning_tokens: u32,
        total_tokens: Option<u32>,
    ) -> Self {
        let mut usage = Self {
            input_tokens,
            input_no_cache_tokens: input_tokens,
            input_cache_read_tokens: 0,
            input_cache_write_tokens: 0,
            output_tokens,
            output_text_tokens: output_tokens.saturating_sub(reasoning_tokens),
            reasoning_tokens,
            total_tokens: total_tokens
                .unwrap_or_else(|| input_tokens.saturating_add(output_tokens)),
            raw_usage: None,
        };
        usage.normalize_usage_fields();
        usage
    }

    /// Sets input cache split and recomputes no-cache input tokens.
    pub(crate) fn with_input_cache_split(
        mut self,
        cache_read_tokens: u32,
        cache_write_tokens: u32,
    ) -> Self {
        self.input_cache_read_tokens = cache_read_tokens;
        self.input_cache_write_tokens = cache_write_tokens;
        self.input_no_cache_tokens = self
            .input_tokens
            .saturating_sub(cache_read_tokens.saturating_add(cache_write_tokens));
        self.normalize_usage_fields();
        self
    }

    /// Sets output text/reasoning split and recomputes total output tokens.
    pub(crate) fn with_output_split(
        mut self,
        output_text_tokens: u32,
        reasoning_tokens: u32,
    ) -> Self {
        self.output_text_tokens = output_text_tokens;
        self.reasoning_tokens = reasoning_tokens;
        self.output_tokens = output_text_tokens.saturating_add(reasoning_tokens);
        self.normalize_usage_fields();
        self
    }

    /// Attaches raw provider usage payload.
    pub(crate) fn with_raw_usage(mut self, raw_usage: Value) -> Self {
        self.raw_usage = Some(raw_usage);
        self
    }

    fn normalize_usage_fields(&mut self) {
        let cache_total = self
            .input_cache_read_tokens
            .saturating_add(self.input_cache_write_tokens);
        let no_cache_floor = self.input_tokens.saturating_sub(cache_total);
        self.input_no_cache_tokens = self.input_no_cache_tokens.max(no_cache_floor);

        let output_text_floor = self.output_tokens.saturating_sub(self.reasoning_tokens);
        self.output_text_tokens = self.output_text_tokens.max(output_text_floor);

        let computed_total = self.input_tokens.saturating_add(self.output_tokens);
        if self.total_tokens == 0 {
            self.total_tokens = computed_total;
        }
    }

    fn is_zero_numeric(&self) -> bool {
        self.input_tokens == 0
            && self.input_no_cache_tokens == 0
            && self.input_cache_read_tokens == 0
            && self.input_cache_write_tokens == 0
            && self.output_tokens == 0
            && self.output_text_tokens == 0
            && self.reasoning_tokens == 0
            && self.total_tokens == 0
    }
}

/// Streaming event emitted by [`TextStream`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    /// A reasoning block started.
    ReasoningStarted {
        /// Stable block id to correlate start/delta/done events.
        block_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        /// Optional provider-specific metadata for this block.
        provider_metadata: Option<Value>,
    },
    /// Incremental reasoning text.
    ReasoningDelta {
        /// Stable block id.
        block_id: String,
        /// Text delta payload.
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        /// Optional provider-specific metadata.
        provider_metadata: Option<Value>,
    },
    /// A reasoning block completed.
    ReasoningDone {
        /// Stable block id.
        block_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        /// Optional provider-specific metadata.
        provider_metadata: Option<Value>,
    },
    /// Incremental assistant text.
    TextDelta {
        /// Text delta payload.
        text: String,
    },
    /// Tool call became executable.
    ToolCallReady {
        /// Executable tool call.
        call: ToolCall,
    },
    /// Incremental usage metadata.
    Usage {
        /// Usage counters snapshot.
        usage: Usage,
    },
    /// Stream finished cleanly.
    Done,
}

/// Streaming event emitted by [`ObjectStream`].
///
/// [`ObjectStream`]: type.ObjectStream.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamObjectEvent<T> {
    /// Best-effort partial deserialisation. Fields that have arrived are
    /// populated; the rest use their `Default`.
    Partial { partial: T },
    /// Final complete object, emitted immediately before `Done`.
    Object { object: T },
}

/// Provider-agnostic stream of partially-populated structured output.
pub type ObjectStream<T> = Pin<Box<dyn Stream<Item = Result<StreamObjectEvent<T>, Error>> + Send>>;

/// Provider-agnostic stream of structured generation events.
pub type TextStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>;
/// Convenience stream of text deltas only.
pub type TextDeltaStream = Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>;

fn validate_message_parts(role: MessageRole, parts: &[ContentPart]) -> Result<(), Error> {
    if parts.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "message parts cannot be empty",
        ));
    }
    if role == MessageRole::Tool
        && !parts
            .iter()
            .any(|part| matches!(part, ContentPart::ToolResult(_)))
    {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "tool role message must include a ToolResult part",
        ));
    }
    Ok(())
}

impl AgentPrepareStep {
    /// Converts this event payload into a mutable prepared-step value.
    pub fn to_prepared(&self) -> AgentPreparedStep {
        AgentPreparedStep {
            model: self.model.clone(),
            messages: self.messages.clone(),
            tools: self.tools.clone(),
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            stop_sequences: self.stop_sequences.clone(),
        }
    }
}

pub(crate) fn validate_messages(messages: &[Message]) -> Result<(), Error> {
    if messages.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "messages cannot be empty",
        ));
    }

    for msg in messages {
        validate_message_parts(msg.role.clone(), &msg.parts)?;
    }

    Ok(())
}

pub(crate) fn validate_model_ref(model: &str) -> Result<(), Error> {
    if model.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "model name cannot be empty",
        ));
    }
    Ok(())
}

pub(crate) fn validate_sampling(temperature: Option<f32>, top_p: Option<f32>) -> Result<(), Error> {
    if let Some(temp) = temperature
        && !(0.0..=2.0).contains(&temp)
    {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "temperature must be within 0.0..=2.0",
        ));
    }
    if let Some(p) = top_p
        && !(0.0..=1.0).contains(&p)
    {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "top_p must be within 0.0..=1.0",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Model validation ────────────────────────────────────────────────

    #[test]
    fn rejects_empty_model_name() {
        let err = validate_model_ref("  ").expect_err("empty model should fail");
        assert!(
            err.message.contains("cannot be empty"),
            "unexpected error: {}",
            err.message
        );
    }

    // ─── Message validation ──────────────────────────────────────────────

    #[test]
    fn message_new_rejects_empty_parts() {
        let err = Message::new(MessageRole::User, vec![]).expect_err("empty parts should fail");
        assert!(err.message.contains("cannot be empty"));
    }

    #[test]
    fn message_new_rejects_tool_role_without_tool_result() {
        let err = Message::new(MessageRole::Tool, vec![ContentPart::Text("x".into())])
            .expect_err("tool role without ToolResult should fail");
        assert!(err.message.contains("ToolResult"));
    }

    #[test]
    fn message_tool_role_with_tool_result_is_valid() {
        let msg = Message::new(
            MessageRole::Tool,
            vec![ContentPart::ToolResult(ToolResult {
                call_id: "c1".into(),
                output_json: serde_json::json!({"ok": true}),
                is_error: false,
            })],
        )
        .expect("tool role with ToolResult should be valid");
        assert_eq!(msg.role(), MessageRole::Tool);
    }

    #[test]
    fn message_system_text_constructor() {
        let msg = Message::system_text("be helpful");
        assert_eq!(msg.role(), MessageRole::System);
        assert_eq!(msg.parts().len(), 1);
    }

    #[test]
    fn message_user_text_constructor() {
        let msg = Message::user_text("hello");
        assert_eq!(msg.role(), MessageRole::User);
    }

    #[test]
    fn message_assistant_text_constructor() {
        let msg = Message::assistant_text("hi");
        assert_eq!(msg.role(), MessageRole::Assistant);
    }

    #[test]
    fn message_tool_result_constructor() {
        let result = ToolResult {
            call_id: "c1".into(),
            output_json: serde_json::json!({"temp": 23}),
            is_error: false,
        };
        let msg = Message::tool_result(result.clone());
        assert_eq!(msg.role(), MessageRole::Tool);
    }

    #[test]
    fn message_user_image_url() {
        let msg = Message::user_image_url("https://example.com/img.jpg");
        assert_eq!(msg.role(), MessageRole::User);
    }

    #[test]
    fn message_user_image_bytes() {
        let msg = Message::user_image_bytes(vec![0xFF, 0xD8], "image/jpeg");
        assert_eq!(msg.role(), MessageRole::User);
    }

    #[test]
    fn message_assistant_with_parts() {
        let msg = Message::assistant_with_parts(vec![ContentPart::Text("output".into())]);
        assert_eq!(msg.role(), MessageRole::Assistant);
    }

    // ─── ContentPart / MediaData / ImagePart / ReasoningPart serialization

    #[test]
    fn content_part_text_serialization() {
        let part = ContentPart::Text("hello".into());
        let json = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, ContentPart::Text(ref t) if t == "hello"));
    }

    #[test]
    fn content_part_image_serialization() {
        let part = ContentPart::Image(ImagePart {
            data: MediaData::Url("https://x.com/i.jpg".into()),
            media_type: Some("image/png".into()),
            provider_metadata: None,
        });
        let json = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, ContentPart::Image(_)));
    }

    #[test]
    fn content_part_reasoning_serialization() {
        let part = ContentPart::Reasoning(ReasoningPart {
            text: "chain-of-thought".into(),
            provider_metadata: Some(serde_json::json!({"sig": "abc"})),
        });
        let json = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, ContentPart::Reasoning(ref r) if r.text == "chain-of-thought"));
    }

    #[test]
    fn content_part_tool_call_serialization() {
        let part = ContentPart::ToolCall(ToolCall {
            call_id: "call_1".into(),
            tool_name: "get_weather".into(),
            args_json: serde_json::json!({"city": "NYC"}),
        });
        let json = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, ContentPart::ToolCall(_)));
    }

    #[test]
    fn content_part_tool_result_serialization() {
        let part = ContentPart::ToolResult(ToolResult {
            call_id: "call_1".into(),
            output_json: serde_json::json!({"temp": 23}),
            is_error: false,
        });
        let json = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, ContentPart::ToolResult(_)));
    }

    #[test]
    fn media_data_base64_serialization() {
        let data = MediaData::Base64("aGVsbG8=".into());
        let json = serde_json::to_string(&data).unwrap();
        let back: MediaData = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, MediaData::Base64(ref s) if s == "aGVsbG8="));
    }

    #[test]
    fn media_data_bytes_serialization() {
        let data = MediaData::Bytes(vec![1, 2, 3]);
        let json = serde_json::to_string(&data).unwrap();
        let back: MediaData = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, MediaData::Bytes(ref b) if b == &vec![1u8, 2, 3]));
    }

    // ─── Usage Add / AddAssign ───────────────────────────────────────────

    #[test]
    fn usage_add_accumulates_fields() {
        let a = Usage::from_totals(10, 5, 0, Some(15));
        let b = Usage::from_totals(20, 10, 3, Some(30));
        let total = a + b;
        assert_eq!(total.input_tokens, 30);
        assert_eq!(total.output_tokens, 15);
        assert_eq!(total.total_tokens, 45);
        assert_eq!(total.reasoning_tokens, 3);
    }

    #[test]
    fn usage_add_handles_overflow() {
        let a = Usage {
            input_tokens: u32::MAX,
            output_tokens: 1,
            total_tokens: u32::MAX,
            ..Default::default()
        };
        let b = Usage {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 1,
            ..Default::default()
        };
        let total = a + b;
        assert_eq!(total.input_tokens, u32::MAX);
        assert_eq!(total.output_tokens, 2);
    }

    #[test]
    fn usage_add_drops_raw_usage() {
        let mut a = Usage::from_totals(1, 1, 0, None);
        a.raw_usage = Some(serde_json::json!({"a": 1}));
        let mut b = Usage::from_totals(1, 1, 0, None);
        b.raw_usage = Some(serde_json::json!({"b": 1}));
        let total = a + b;
        assert!(total.raw_usage.is_none());
    }

    #[test]
    fn usage_add_assign_accumulates() {
        let mut a = Usage::from_totals(10, 5, 0, Some(15));
        let b = Usage::from_totals(20, 10, 3, Some(30));
        a += b;
        assert_eq!(a.input_tokens, 30);
        assert_eq!(a.total_tokens, 45);
    }

    #[test]
    fn usage_add_assign_preserves_first_raw_usage() {
        let mut a = Usage::from_totals(10, 5, 0, Some(15));
        a.raw_usage = Some(serde_json::json!({"first": true}));
        let b = Usage::from_totals(20, 10, 0, None);
        a += b;
        assert_eq!(a.raw_usage, Some(serde_json::json!({"first": true})));
    }

    #[test]
    fn usage_add_assign_with_second_raw_usage_drops_to_none() {
        let mut a = Usage::from_totals(10, 5, 0, Some(15));
        a.raw_usage = Some(serde_json::json!({"first": true}));
        let mut b = Usage::from_totals(20, 10, 0, None);
        b.raw_usage = Some(serde_json::json!({"second": true}));
        a += b;
        assert!(a.raw_usage.is_none());
    }

    #[test]
    fn usage_from_totals_backfills_fields() {
        let usage = Usage::from_totals(100, 50, 10, None);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.reasoning_tokens, 10);
        assert_eq!(usage.output_text_tokens, 40);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn usage_with_input_cache_split_recomputes_no_cache() {
        let usage = Usage::from_totals(100, 50, 0, None).with_input_cache_split(30, 20);
        assert_eq!(usage.input_cache_read_tokens, 30);
        assert_eq!(usage.input_cache_write_tokens, 20);
        assert_eq!(usage.input_no_cache_tokens, 50);
    }

    #[test]
    fn usage_with_output_split_recomputes_output_tokens() {
        let usage = Usage::from_totals(100, 50, 0, None).with_output_split(30, 20);
        assert_eq!(usage.output_text_tokens, 30);
        assert_eq!(usage.reasoning_tokens, 20);
        assert_eq!(usage.output_tokens, 50);
    }

    #[test]
    fn usage_with_raw_usage_attaches_payload() {
        let raw = serde_json::json!({"vendor": "custom"});
        let usage = Usage::from_totals(1, 1, 0, None).with_raw_usage(raw.clone());
        assert_eq!(usage.raw_usage, Some(raw));
    }

    #[test]
    fn usage_normalize_enforces_floor() {
        let mut usage = Usage {
            input_tokens: 5,
            input_no_cache_tokens: 0,
            input_cache_read_tokens: 3,
            input_cache_write_tokens: 3,
            output_tokens: 10,
            output_text_tokens: 0,
            reasoning_tokens: 12,
            total_tokens: 0,
            ..Default::default()
        };
        usage.normalize_usage_fields();
        assert_eq!(usage.input_no_cache_tokens, 0);
        assert_eq!(usage.output_text_tokens, 0);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn usage_is_zero_numeric_detects_all_zero() {
        assert!(Usage::default().is_zero_numeric());
    }

    #[test]
    fn usage_is_zero_numeric_detects_nonzero() {
        let usage = Usage::from_totals(1, 0, 0, None);
        assert!(!usage.is_zero_numeric());
    }

    // ─── ToolErrorPolicy / FinishReason ──────────────────────────────────

    #[test]
    fn tool_error_policy_default_is_continue_as_tool_result() {
        assert_eq!(
            ToolErrorPolicy::default(),
            ToolErrorPolicy::ContinueAsToolResult
        );
    }

    #[test]
    fn finish_reason_serialization_roundtrip() {
        let reasons = [
            FinishReason::Stop,
            FinishReason::Length,
            FinishReason::ToolCalls,
            FinishReason::ContentFilter,
            FinishReason::Unknown("custom".into()),
        ];
        for reason in reasons {
            let json = serde_json::to_string(&reason).unwrap();
            let back: FinishReason = serde_json::from_str(&json).unwrap();
            assert_eq!(reason, back);
        }
    }

    // ─── StreamEvent serialization ───────────────────────────────────────

    #[test]
    fn stream_event_text_delta_serialization() {
        let event = StreamEvent::TextDelta {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("TextDelta"));
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, StreamEvent::TextDelta { text } if text == "hello"));
    }

    #[test]
    fn stream_event_done_serialization() {
        let event = StreamEvent::Done;
        let json = serde_json::to_string(&event).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, StreamEvent::Done));
    }

    #[test]
    fn stream_event_reasoning_serialization() {
        let event = StreamEvent::ReasoningStarted {
            block_id: "r1".into(),
            provider_metadata: Some(serde_json::json!({"k": "v"})),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, StreamEvent::ReasoningStarted { .. }));
    }

    // ─── Agent event types ──────────────────────────────────────────────

    #[test]
    fn agent_prepare_step_to_prepared() {
        let step = AgentPrepareStep {
            step: 1,
            model: "gpt-5.5".to_string(),
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            temperature: Some(0.5),
            max_output_tokens: Some(100),
            stop_sequences: vec!["END".into()],
            previous_steps: vec![],
        };
        let prepared = step.to_prepared();
        assert_eq!(prepared.model.as_str(), "gpt-5.5");
        assert_eq!(prepared.temperature, Some(0.5));
        assert_eq!(prepared.max_output_tokens, Some(100));
        assert_eq!(prepared.messages.len(), 1);
    }

    #[test]
    fn agent_start_serialization() {
        let event = AgentStart {
            model_id: "openai/gpt-5.5".into(),
            messages: vec![Message::user_text("hi")],
            tool_count: 3,
            max_steps: 8,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AgentStart = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_id, "openai/gpt-5.5");
        assert_eq!(back.tool_count, 3);
    }

    #[test]
    fn agent_step_start_serialization() {
        let event = AgentStepStart {
            step: 2,
            messages: vec![Message::user_text("hi")],
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AgentStepStart = serde_json::from_str(&json).unwrap();
        assert_eq!(back.step, 2);
    }

    #[test]
    fn agent_tool_call_start_serialization() {
        let event = AgentToolCallStart {
            step: 1,
            tool_call: ToolCall {
                call_id: "c1".into(),
                tool_name: "weather".into(),
                args_json: serde_json::json!({"city": "NYC"}),
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AgentToolCallStart = serde_json::from_str(&json).unwrap();
        assert_eq!(back.step, 1);
    }

    #[test]
    fn agent_tool_call_finish_serialization() {
        let event = AgentToolCallFinish {
            step: 1,
            tool_call: ToolCall {
                call_id: "c1".into(),
                tool_name: "weather".into(),
                args_json: serde_json::json!({}),
            },
            tool_result: ToolResult {
                call_id: "c1".into(),
                output_json: serde_json::json!({"temp": 23}),
                is_error: false,
            },
            duration_ms: 42,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AgentToolCallFinish = serde_json::from_str(&json).unwrap();
        assert_eq!(back.duration_ms, 42);
    }

    #[test]
    fn agent_step_serialization() {
        let step = AgentStep {
            step: 1,
            output_text: "done".into(),
            reasoning_text: "think".into(),
            reasoning_parts: vec![],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            tool_calls: vec![],
            tool_results: vec![],
        };
        let json = serde_json::to_string(&step).unwrap();
        let back: AgentStep = serde_json::from_str(&json).unwrap();
        assert_eq!(back.step, 1);
        assert_eq!(back.output_text, "done");
    }

    #[test]
    fn agent_finish_serialization() {
        let finish = AgentFinish {
            output_text: "final".into(),
            step_count: 3,
            finish_reason: FinishReason::Stop,
            usage_total: Usage::default(),
            transcript: vec![],
            step_results: vec![],
        };
        let json = serde_json::to_string(&finish).unwrap();
        let back: AgentFinish = serde_json::from_str(&json).unwrap();
        assert_eq!(back.step_count, 3);
    }

    #[test]
    fn agent_response_serialization() {
        let response = AgentResponse {
            output_text: "answer".into(),
            steps: 2,
            transcript: vec![],
            usage_total: Usage::default(),
            step_results: vec![],
        };
        let json = serde_json::to_string(&response).unwrap();
        let back: AgentResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.steps, 2);
    }

    // ─── GenerateTextRequest / GenerateTextResponse ─────────────────────

    #[test]
    fn builds_request_from_prompt() {
        let request = GenerateTextRequest::from_user_prompt("gpt-5.4-mini", "hello");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.model.as_str(), "gpt-5.4-mini");
    }

    #[test]
    fn generate_text_response_serialization() {
        let response = GenerateTextResponse {
            output_text: "hello".into(),
            reasoning_text: String::new(),
            reasoning_parts: vec![],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            tool_calls: vec![],
            raw_provider_response: Some(serde_json::json!({"raw": true})),
        };
        let json = serde_json::to_string(&response).unwrap();
        let back: GenerateTextResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.output_text, "hello");
    }

    // ─── Request builder ─────────────────────────────────────────────────

    #[test]
    fn request_builder_rejects_invalid_top_p() {
        let err = GenerateTextRequest::builder("gpt-5.4-mini")
            .message(Message::user_text("hello"))
            .top_p(1.1)
            .build()
            .expect_err("invalid top_p should fail");
        assert!(err.message.contains("top_p"));
    }

    #[test]
    fn request_builder_rejects_invalid_temperature() {
        let err = GenerateTextRequest::builder("gpt-5.4-mini")
            .message(Message::user_text("hello"))
            .temperature(2.1)
            .build()
            .expect_err("invalid temperature should fail");
        assert!(err.message.contains("temperature"));
    }

    #[test]
    fn request_builder_rejects_empty_messages() {
        let err = GenerateTextRequest::builder("gpt-5.4-mini")
            .build()
            .expect_err("empty messages should fail");
        assert!(err.message.contains("cannot be empty"));
    }

    #[test]
    fn request_builder_accepts_messages_method() {
        let req = GenerateTextRequest::builder("gpt-5.4-mini")
            .messages(vec![Message::user_text("h1"), Message::user_text("h2")])
            .build()
            .expect("request should build");
        assert_eq!(req.messages.len(), 2);
    }

    #[test]
    fn request_builder_user_prompt_replaces_messages() {
        let req = GenerateTextRequest::builder("gpt-5.4-mini")
            .message(Message::user_text("old"))
            .user_prompt("replaced")
            .build()
            .expect("request should build");
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn request_builder_sets_all_fields() {
        let req = GenerateTextRequest::builder("gpt-5.4-mini")
            .message(Message::user_text("hi"))
            .temperature(0.7)
            .top_p(0.9)
            .max_output_tokens(100)
            .stop_sequences(["END", "STOP"])
            .cancellation_token(tokio_util::sync::CancellationToken::new())
            .build()
            .expect("request should build");
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.max_output_tokens, Some(100));
        assert_eq!(req.stop_sequences.len(), 2);
        assert!(req.cancellation_token.is_some());
    }

    #[test]
    fn request_builder_empty_tools_is_none() {
        let req = GenerateTextRequest::builder("gpt-5.4-mini")
            .message(Message::user_text("hi"))
            .tools(Vec::<crate::tool::ToolDescriptor>::new())
            .build()
            .expect("request should build");
        assert!(req.tools.is_none());
    }

    #[test]
    fn request_builder_valid_temperature_boundary() {
        let req = GenerateTextRequest::builder("gpt-5.4-mini")
            .message(Message::user_text("hi"))
            .temperature(2.0)
            .top_p(0.0)
            .build()
            .expect("request should build");
        assert_eq!(req.temperature, Some(2.0));
        assert_eq!(req.top_p, Some(0.0));
    }

    #[test]
    fn request_builder_sets_provider_options() {
        let options = serde_json::json!({
            "anthropic": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            }
        });
        let req = GenerateTextRequest::builder("claude-sonnet-4-6")
            .message(Message::user_text("hi"))
            .provider_options(options.clone())
            .build()
            .expect("request should build");
        assert_eq!(req.provider_options, Some(options));
    }

    #[test]
    fn request_builder_provider_options_defaults_to_none() {
        let req = GenerateTextRequest::builder("gpt-5.4-mini")
            .message(Message::user_text("hi"))
            .build()
            .expect("request should build");
        assert!(req.provider_options.is_none());
    }

    // ─── Validate functions ──────────────────────────────────────────────

    #[test]
    fn validate_sampling_rejects_negative_temperature() {
        let err = validate_sampling(Some(-0.1), None).expect_err("negative temp should fail");
        assert!(err.message.contains("temperature"));
    }

    #[test]
    fn validate_sampling_rejects_negative_top_p() {
        let err = validate_sampling(None, Some(-0.1)).expect_err("negative top_p should fail");
        assert!(err.message.contains("top_p"));
    }

    #[test]
    fn validate_sampling_accepts_none() {
        assert!(validate_sampling(None, None).is_ok());
    }

    #[test]
    fn validate_messages_rejects_empty_list() {
        let err = validate_messages(&[]).expect_err("empty list should fail");
        assert!(err.message.contains("cannot be empty"));
    }

    #[test]
    fn validate_messages_rejects_message_with_empty_parts() {
        let err = validate_messages(&[Message {
            role: MessageRole::User,
            parts: vec![],
            name: None,
        }])
        .expect_err("message with empty parts should fail");
        assert!(err.message.contains("cannot be empty"));
    }

    // ─── RunTools builder ────────────────────────────────────────────────

    fn run_tools_with_messages(model: &str) -> RunTools {
        let mut rt = RunTools::new(model);
        rt.messages = vec![Message::user_text("hello")];
        rt
    }

    #[test]
    fn run_tools_builds_with_valid_config() {
        let rt = run_tools_with_messages("gpt-5.5")
            .tools(Vec::<crate::tool::Tool>::new())
            .max_steps(5)
            .temperature(0.5)
            .top_p(0.9)
            .max_output_tokens(256)
            .stop_sequences(["END"])
            .tool_error_policy(ToolErrorPolicy::FailFast)
            .cancellation_token(tokio_util::sync::CancellationToken::new())
            .build()
            .expect("run tools should build");
        assert_eq!(rt.max_steps, Some(5));
    }

    #[test]
    fn run_tools_build_accepts_zero_max_steps_as_unlimited() {
        let req = run_tools_with_messages("gpt-5.5")
            .max_steps(0)
            .build()
            .expect("0 max_steps is allowed and means unlimited");
        assert_eq!(req.max_steps, Some(0));
    }

    #[test]
    fn run_tools_build_rejects_invalid_model() {
        match run_tools_with_messages("  ").build() {
            Err(err) => assert!(err.message.contains("cannot be empty")),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn run_tools_default_max_steps_is_none() {
        let rt = run_tools_with_messages("gpt-5.5")
            .build()
            .expect("run tools should build");
        assert_eq!(rt.max_steps, None);
    }

    #[test]
    fn run_tools_default_tool_error_policy() {
        let rt = run_tools_with_messages("gpt-5.5")
            .build()
            .expect("run tools should build");
        assert_eq!(rt.tool_error_policy, ToolErrorPolicy::ContinueAsToolResult);
    }
}
