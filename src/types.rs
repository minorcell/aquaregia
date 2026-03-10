use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, ErrorCode};
use crate::tool::{IntoTool, Tool, ToolDescriptor};

/// Supported provider families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    /// OpenAI provider family.
    OpenAi,
    /// Anthropic provider family.
    Anthropic,
    /// Google provider family.
    Google,
    /// OpenAI-compatible provider family.
    OpenAiCompatible,
}

impl ProviderKind {
    /// Parses a provider slug (case-insensitive).
    pub fn from_slug(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            "google" => Some(Self::Google),
            "openai-compatible" => Some(Self::OpenAiCompatible),
            _ => None,
        }
    }

    /// Returns the canonical provider slug.
    pub fn as_slug(&self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
            Self::OpenAiCompatible => "openai-compatible",
        }
    }
}

/// Type marker for provider-specific request/response typing.
pub trait ProviderMarker: Clone + Copy + Send + Sync + 'static {
    /// Provider family represented by this marker type.
    const KIND: ProviderKind;
}

/// OpenAI provider marker.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct OpenAi;

impl ProviderMarker for OpenAi {
    const KIND: ProviderKind = ProviderKind::OpenAi;
}

/// Anthropic provider marker.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Anthropic;

impl ProviderMarker for Anthropic {
    const KIND: ProviderKind = ProviderKind::Anthropic;
}

/// Google provider marker.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Google;

impl ProviderMarker for Google {
    const KIND: ProviderKind = ProviderKind::Google;
}

/// OpenAI-compatible provider marker.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct OpenAiCompatible;

impl ProviderMarker for OpenAiCompatible {
    const KIND: ProviderKind = ProviderKind::OpenAiCompatible;
}

/// Strongly-typed model reference carrying provider information at compile time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef<P: ProviderMarker> {
    model: String,
    #[serde(skip)]
    _marker: PhantomData<P>,
}

impl<P: ProviderMarker> ModelRef<P> {
    /// Creates a model reference from a provider-local model id.
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        Self {
            model,
            _marker: PhantomData,
        }
    }

    /// Returns a fully-qualified model id (`<provider>/<model>`).
    pub fn id(&self) -> String {
        format!("{}/{}", P::KIND.as_slug(), self.model)
    }

    /// Returns the provider family marker for this model.
    pub fn provider_kind(&self) -> ProviderKind {
        P::KIND
    }

    /// Returns the provider slug for this model.
    pub fn provider_slug(&self) -> &'static str {
        P::KIND.as_slug()
    }

    /// Returns the provider-local model id.
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl<P: ProviderMarker> std::fmt::Display for ModelRef<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

/// Converts values into a typed [`ModelRef`].
pub trait IntoModelRef<P: ProviderMarker> {
    /// Performs the conversion.
    fn into_model_ref(self) -> ModelRef<P>;
}

impl<P: ProviderMarker> IntoModelRef<P> for ModelRef<P> {
    fn into_model_ref(self) -> ModelRef<P> {
        self
    }
}

impl<P: ProviderMarker> IntoModelRef<P> for &str {
    fn into_model_ref(self) -> ModelRef<P> {
        ModelRef::new(self)
    }
}

impl<P: ProviderMarker> IntoModelRef<P> for String {
    fn into_model_ref(self) -> ModelRef<P> {
        ModelRef::new(self)
    }
}

/// Chat message role used across providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    /// System/instruction message.
    System,
    /// User message.
    User,
    /// Assistant/model message.
    Assistant,
    /// Tool result message.
    Tool,
}

/// Provider-agnostic chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub(crate) role: MessageRole,
    pub(crate) parts: Vec<ContentPart>,
    pub(crate) name: Option<String>,
}

impl Message {
    /// Creates a message with custom role and parts.
    ///
    /// Prefer the named constructors ([`Message::system_text`], [`Message::user_text`],
    /// [`Message::assistant_text`], [`Message::tool_result`]) for common cases. Use `new` only
    /// when you need to build a message with custom [`ContentPart`] combinations.
    pub fn new(role: MessageRole, parts: Vec<ContentPart>) -> Result<Self, Error> {
        validate_message_parts(role.clone(), &parts)?;
        Ok(Self {
            role,
            parts,
            name: None,
        })
    }

    /// Creates a system message containing one text part.
    pub fn system_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            parts: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    /// Creates a user message containing one text part.
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            parts: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    /// Creates an assistant message containing one text part.
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            parts: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    /// Creates a tool-role message containing one tool result part.
    pub fn tool_result(result: ToolResult) -> Self {
        Self {
            role: MessageRole::Tool,
            parts: vec![ContentPart::ToolResult(result)],
            name: None,
        }
    }

    /// Attaches an optional author/tool name to the message.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the message role.
    pub fn role(&self) -> MessageRole {
        self.role.clone()
    }

    /// Returns message content parts.
    pub fn parts(&self) -> &[ContentPart] {
        &self.parts
    }

    /// Returns optional message name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub(crate) fn assistant_with_parts(parts: Vec<ContentPart>) -> Self {
        Self {
            role: MessageRole::Assistant,
            parts,
            name: None,
        }
    }
}

/// Content block used in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    /// Plain text content.
    Text(String),
    /// Provider reasoning content.
    Reasoning(ReasoningPart),
    /// Tool call requested by the model.
    ToolCall(ToolCall),
    /// Tool execution result.
    ToolResult(ToolResult),
}

/// Reasoning content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPart {
    /// Reasoning text content.
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional provider-specific metadata (for example signatures).
    pub provider_metadata: Option<Value>,
}

/// Tool call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-generated call identifier.
    pub call_id: String,
    /// Tool name to execute.
    pub tool_name: String,
    /// JSON arguments for the tool.
    pub args_json: Value,
}

/// Tool execution result sent back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Matches [`ToolCall::call_id`].
    pub call_id: String,
    /// JSON output payload.
    pub output_json: Value,
    /// Indicates this payload represents a tool error.
    pub is_error: bool,
}

/// Provider-typed request for generation/streaming calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextRequest<P: ProviderMarker> {
    pub(crate) model: ModelRef<P>,
    pub(crate) messages: Vec<Message>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) tools: Option<Vec<ToolDescriptor>>,
    #[serde(skip)]
    pub(crate) cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

impl<P: ProviderMarker> GenerateTextRequest<P> {
    /// Builds a one-message request from a user prompt.
    pub fn from_user_prompt(model: impl IntoModelRef<P>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into_model_ref(),
            messages: vec![Message::user_text(prompt)],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stop_sequences: vec![],
            tools: None,
            cancellation_token: None,
        }
    }

    /// Starts a request builder.
    pub fn builder(model: impl IntoModelRef<P>) -> GenerateTextRequestBuilder<P> {
        GenerateTextRequestBuilder {
            request: Self {
                model: model.into_model_ref(),
                messages: Vec::new(),
                temperature: None,
                top_p: None,
                max_output_tokens: None,
                stop_sequences: vec![],
                tools: None,
                cancellation_token: None,
            },
        }
    }
}

/// Builder for [`GenerateTextRequest`].
pub struct GenerateTextRequestBuilder<P: ProviderMarker> {
    request: GenerateTextRequest<P>,
}

impl<P: ProviderMarker> GenerateTextRequestBuilder<P> {
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

    /// Adds a cancellation token checked by adapters and streams.
    pub fn cancellation_token(mut self, token: tokio_util::sync::CancellationToken) -> Self {
        self.request.cancellation_token = Some(token);
        self
    }

    /// Validates and finalizes the request.
    pub fn build(self) -> Result<GenerateTextRequest<P>, Error> {
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
pub(crate) struct RunTools<P: ProviderMarker> {
    pub(crate) model: ModelRef<P>,
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Vec<Tool>,
    pub(crate) max_steps: Option<u8>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) prepare_step: Option<PrepareStepHook<P>>,
    pub(crate) on_start: Option<Hook<AgentStart>>,
    pub(crate) on_step_start: Option<Hook<AgentStepStart>>,
    pub(crate) on_tool_call_start: Option<Hook<AgentToolCallStart>>,
    pub(crate) on_tool_call_finish: Option<Hook<AgentToolCallFinish>>,
    pub(crate) on_step_finish: Option<Hook<AgentStep>>,
    pub(crate) on_finish: Option<Hook<AgentFinish>>,
    pub(crate) stop_when: Option<StopPredicate>,
    pub(crate) tool_error_policy: ToolErrorPolicy,
    pub(crate) cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

impl<P: ProviderMarker> RunTools<P> {
    pub(crate) fn new(model: impl IntoModelRef<P>) -> Self {
        Self {
            model: model.into_model_ref(),
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
            cancellation_token: None,
        }
    }

    pub(crate) fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages = messages.into_iter().collect();
        self
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

    pub(crate) fn max_steps(mut self, max_steps: u8) -> Self {
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
        F: Fn(&AgentPrepareStep<P>) -> AgentPreparedStep<P> + Send + Sync + 'static,
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

    pub(crate) fn cancellation_token(mut self, token: tokio_util::sync::CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    pub(crate) fn build(self) -> Result<Self, Error> {
        validate_model_ref(&self.model)?;
        validate_messages(&self.messages)?;
        validate_sampling(self.temperature, self.top_p)?;
        if let Some(max_steps) = self.max_steps {
            validate_max_steps(max_steps)?;
        }
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
    /// Effective max step cap for this run.
    pub max_steps: u8,
}

/// Emitted when an agent step begins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepStart {
    /// 1-based step index.
    pub step: u8,
    /// Messages sent to the model for this step.
    pub messages: Vec<Message>,
}

/// Emitted right before executing one tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolCallStart {
    /// 1-based step index.
    pub step: u8,
    /// Tool call about to execute.
    pub tool_call: ToolCall,
}

/// Emitted right after executing one tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolCallFinish {
    /// 1-based step index.
    pub step: u8,
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
    pub step: u8,
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
    pub step_count: u8,
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
pub struct AgentPrepareStep<P: ProviderMarker> {
    /// 1-based step index to be executed.
    pub step: u8,
    /// Model selected for this step.
    pub model: ModelRef<P>,
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
pub struct AgentPreparedStep<P: ProviderMarker> {
    /// Model selected for this step.
    pub model: ModelRef<P>,
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
pub(crate) type PrepareStepHook<P> =
    Arc<dyn Fn(&AgentPrepareStep<P>) -> AgentPreparedStep<P> + Send + Sync>;
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

/// Final response of a completed agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// Final assistant text output.
    pub output_text: String,
    /// Number of executed steps.
    pub steps: u8,
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
    pub fn from_totals(
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
    pub fn with_input_cache_split(
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
    pub fn with_output_split(mut self, output_text_tokens: u32, reasoning_tokens: u32) -> Self {
        self.output_text_tokens = output_text_tokens;
        self.reasoning_tokens = reasoning_tokens;
        self.output_tokens = output_text_tokens.saturating_add(reasoning_tokens);
        self.normalize_usage_fields();
        self
    }

    /// Attaches raw provider usage payload.
    pub fn with_raw_usage(mut self, raw_usage: Value) -> Self {
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

impl<P: ProviderMarker> AgentPrepareStep<P> {
    /// Converts this event payload into a mutable prepared-step value.
    pub fn to_prepared(&self) -> AgentPreparedStep<P> {
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

pub(crate) fn validate_model_ref<P: ProviderMarker>(model: &ModelRef<P>) -> Result<(), Error> {
    if model.model().trim().is_empty() {
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

pub(crate) fn validate_max_steps(max_steps: u8) -> Result<(), Error> {
    if !(1..=32).contains(&max_steps) {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "max_steps must be in 1..=32",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        GenerateTextRequest, Message, ModelRef, OpenAi, ProviderKind, validate_max_steps,
        validate_model_ref,
    };

    #[test]
    fn builds_openai_model() {
        let model = ModelRef::<OpenAi>::new("gpt-4o-mini");
        assert_eq!(model.provider_kind(), ProviderKind::OpenAi);
        assert_eq!(model.model(), "gpt-4o-mini");
    }

    #[test]
    fn rejects_empty_model_name() {
        let model = ModelRef::<OpenAi>::new("  ");
        let err = validate_model_ref(&model).expect_err("empty model should fail");
        assert!(
            err.message.contains("cannot be empty"),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn builds_request_from_prompt() {
        let request =
            GenerateTextRequest::from_user_prompt(ModelRef::<OpenAi>::new("gpt-4o-mini"), "hello");

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.model.model(), "gpt-4o-mini");
    }

    #[test]
    fn rejects_invalid_max_steps() {
        let err = validate_max_steps(0).expect_err("0 should fail");
        assert!(err.message.contains("1..=32"));
    }

    #[test]
    fn model_ref_display_matches_model_id() {
        let model = ModelRef::<OpenAi>::new("gpt-4o-mini");

        assert_eq!(model.to_string(), "openai/gpt-4o-mini");
    }

    #[test]
    fn request_builder_rejects_invalid_top_p() {
        let err = GenerateTextRequest::builder(ModelRef::<OpenAi>::new("gpt-4o-mini"))
            .message(Message::user_text("hello"))
            .top_p(1.1)
            .build()
            .expect_err("invalid top_p should fail");

        assert!(err.message.contains("top_p"));
    }
}
