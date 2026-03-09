use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, ErrorCode};
use crate::tool::{IntoTool, Tool, ToolDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    OpenAi,
    Anthropic,
    Google,
    OpenAiCompatible,
}

impl ProviderKind {
    pub fn from_slug(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            "google" => Some(Self::Google),
            "openai-compatible" => Some(Self::OpenAiCompatible),
            _ => None,
        }
    }

    pub fn as_slug(&self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
            Self::OpenAiCompatible => "openai-compatible",
        }
    }
}

pub trait ProviderMarker: Clone + Copy + Send + Sync + 'static {
    const KIND: ProviderKind;
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct OpenAi;

impl ProviderMarker for OpenAi {
    const KIND: ProviderKind = ProviderKind::OpenAi;
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Anthropic;

impl ProviderMarker for Anthropic {
    const KIND: ProviderKind = ProviderKind::Anthropic;
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Google;

impl ProviderMarker for Google {
    const KIND: ProviderKind = ProviderKind::Google;
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct OpenAiCompatible;

impl ProviderMarker for OpenAiCompatible {
    const KIND: ProviderKind = ProviderKind::OpenAiCompatible;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef<P: ProviderMarker> {
    model: String,
    #[serde(skip)]
    _marker: PhantomData<P>,
}

impl<P: ProviderMarker> ModelRef<P> {
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        Self {
            model,
            _marker: PhantomData,
        }
    }

    pub fn id(&self) -> String {
        format!("{}/{}", P::KIND.as_slug(), self.model)
    }

    pub fn provider_kind(&self) -> ProviderKind {
        P::KIND
    }

    pub fn provider_slug(&self) -> &'static str {
        P::KIND.as_slug()
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

pub trait IntoModelRef<P: ProviderMarker> {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub(crate) role: MessageRole,
    pub(crate) parts: Vec<ContentPart>,
    pub(crate) name: Option<String>,
}

impl Message {
    pub fn new(role: MessageRole, parts: Vec<ContentPart>) -> Result<Self, Error> {
        validate_message_parts(role.clone(), &parts)?;
        Ok(Self {
            role,
            parts,
            name: None,
        })
    }

    pub fn system_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            parts: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            parts: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            parts: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    pub fn tool_result(result: ToolResult) -> Self {
        Self {
            role: MessageRole::Tool,
            parts: vec![ContentPart::ToolResult(result)],
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn role(&self) -> MessageRole {
        self.role.clone()
    }

    pub fn parts(&self) -> &[ContentPart] {
        &self.parts
    }

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    Text(String),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub args_json: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub output_json: Value,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextRequest<P: ProviderMarker> {
    pub(crate) model: ModelRef<P>,
    pub(crate) messages: Vec<Message>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) tools: Option<Vec<ToolDescriptor>>,
}

impl<P: ProviderMarker> GenerateTextRequest<P> {
    pub fn from_user_prompt(model: impl IntoModelRef<P>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into_model_ref(),
            messages: vec![Message::user_text(prompt)],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stop_sequences: vec![],
            tools: None,
        }
    }

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
            },
        }
    }
}

pub struct GenerateTextRequestBuilder<P: ProviderMarker> {
    request: GenerateTextRequest<P>,
}

impl<P: ProviderMarker> GenerateTextRequestBuilder<P> {
    pub fn message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.request.messages = messages.into_iter().collect();
        self
    }

    pub fn user_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.request.messages = vec![Message::user_text(prompt)];
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.request.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.request.top_p = Some(top_p);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.request.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn stop_sequences<S: Into<String>>(
        mut self,
        stop_sequences: impl IntoIterator<Item = S>,
    ) -> Self {
        self.request.stop_sequences = stop_sequences.into_iter().map(Into::into).collect();
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDescriptor>) -> Self {
        let tools = tools.into_iter().collect::<Vec<_>>();
        self.request.tools = if tools.is_empty() { None } else { Some(tools) };
        self
    }

    pub fn build(self) -> Result<GenerateTextRequest<P>, Error> {
        validate_model_ref(&self.request.model)?;
        validate_messages(&self.request.messages)?;
        validate_sampling(self.request.temperature, self.request.top_p)?;
        Ok(self.request)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ToolErrorPolicy {
    #[default]
    ContinueAsToolResult,
    FailFast,
}

#[derive(Debug, Clone)]
pub(crate) struct RunTools<P: ProviderMarker> {
    pub(crate) model: ModelRef<P>,
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Vec<Tool>,
    pub(crate) max_steps: Option<u8>,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) prepare_step: Option<PrepareStepCallback<P>>,
    pub(crate) on_start: Option<StartCallback<P>>,
    pub(crate) on_step_start: Option<StepStartCallback>,
    pub(crate) on_tool_call_start: Option<ToolCallStartCallback>,
    pub(crate) on_tool_call_finish: Option<ToolCallFinishCallback>,
    pub(crate) on_step_finish: Option<StepCallback>,
    pub(crate) on_finish: Option<FinishCallback>,
    pub(crate) stop_when: Option<StopWhen>,
    pub(crate) tool_error_policy: ToolErrorPolicy,
}

impl<P: ProviderMarker> RunTools<P> {
    pub(crate) fn new(model: impl IntoModelRef<P>) -> Self {
        Self {
            model: model.into_model_ref(),
            messages: Vec::new(),
            tools: Vec::new(),
            max_steps: None,
            temperature: None,
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
        self.prepare_step = Some(PrepareStepCallback::new(callback));
        self
    }

    pub(crate) fn on_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStart<P>) + Send + Sync + 'static,
    {
        self.on_start = Some(StartCallback::new(callback));
        self
    }

    pub(crate) fn on_step_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStepStart) + Send + Sync + 'static,
    {
        self.on_step_start = Some(StepStartCallback::new(callback));
        self
    }

    pub(crate) fn on_tool_call_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallStart) + Send + Sync + 'static,
    {
        self.on_tool_call_start = Some(ToolCallStartCallback::new(callback));
        self
    }

    pub(crate) fn on_tool_call_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallFinish) + Send + Sync + 'static,
    {
        self.on_tool_call_finish = Some(ToolCallFinishCallback::new(callback));
        self
    }

    pub(crate) fn on_step_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStep) + Send + Sync + 'static,
    {
        self.on_step_finish = Some(StepCallback::new(callback));
        self
    }

    pub(crate) fn on_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentFinish) + Send + Sync + 'static,
    {
        self.on_finish = Some(FinishCallback::new(callback));
        self
    }

    pub(crate) fn stop_when<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&AgentStep) -> bool + Send + Sync + 'static,
    {
        self.stop_when = Some(StopWhen::new(predicate));
        self
    }

    pub(crate) fn tool_error_policy(mut self, policy: ToolErrorPolicy) -> Self {
        self.tool_error_policy = policy;
        self
    }

    pub(crate) fn build(self) -> Result<Self, Error> {
        validate_model_ref(&self.model)?;
        validate_messages(&self.messages)?;
        validate_sampling(self.temperature, None)?;
        if let Some(max_steps) = self.max_steps {
            validate_max_steps(max_steps)?;
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStart<P: ProviderMarker> {
    pub model: ModelRef<P>,
    pub messages: Vec<Message>,
    pub tool_count: usize,
    pub max_steps: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepStart {
    pub step: u8,
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolCallStart {
    pub step: u8,
    pub tool_call: ToolCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolCallFinish {
    pub step: u8,
    pub tool_call: ToolCall,
    pub tool_result: ToolResult,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub step: u8,
    pub output_text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFinish {
    pub output_text: String,
    pub step_count: u8,
    pub finish_reason: FinishReason,
    pub usage_total: Usage,
    pub transcript: Vec<Message>,
    pub step_results: Vec<AgentStep>,
}

#[derive(Debug, Clone)]
pub struct AgentPrepareStep<P: ProviderMarker> {
    pub step: u8,
    pub model: ModelRef<P>,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub stop_sequences: Vec<String>,
    pub previous_steps: Vec<AgentStep>,
}

#[derive(Debug, Clone)]
pub struct AgentPreparedStep<P: ProviderMarker> {
    pub model: ModelRef<P>,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub stop_sequences: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct PrepareStepCallback<P: ProviderMarker> {
    inner: Arc<dyn Fn(&AgentPrepareStep<P>) -> AgentPreparedStep<P> + Send + Sync>,
}

impl<P: ProviderMarker> std::fmt::Debug for PrepareStepCallback<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PrepareStepCallback(<fn>)")
    }
}

impl<P: ProviderMarker> PrepareStepCallback<P> {
    pub(crate) fn new<F>(callback: F) -> Self
    where
        F: Fn(&AgentPrepareStep<P>) -> AgentPreparedStep<P> + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(callback),
        }
    }

    pub(crate) fn call(&self, event: &AgentPrepareStep<P>) -> AgentPreparedStep<P> {
        (self.inner)(event)
    }
}

#[derive(Clone)]
pub(crate) struct StartCallback<P: ProviderMarker> {
    inner: Arc<dyn Fn(&AgentStart<P>) + Send + Sync>,
}

impl<P: ProviderMarker> std::fmt::Debug for StartCallback<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StartCallback(<fn>)")
    }
}

impl<P: ProviderMarker> StartCallback<P> {
    pub(crate) fn new<F>(callback: F) -> Self
    where
        F: Fn(&AgentStart<P>) + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(callback),
        }
    }

    pub(crate) fn call(&self, event: &AgentStart<P>) {
        (self.inner)(event);
    }
}

#[derive(Clone)]
pub(crate) struct StepStartCallback {
    inner: Arc<dyn Fn(&AgentStepStart) + Send + Sync>,
}

impl std::fmt::Debug for StepStartCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StepStartCallback(<fn>)")
    }
}

impl StepStartCallback {
    pub(crate) fn new<F>(callback: F) -> Self
    where
        F: Fn(&AgentStepStart) + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(callback),
        }
    }

    pub(crate) fn call(&self, event: &AgentStepStart) {
        (self.inner)(event);
    }
}

#[derive(Clone)]
pub(crate) struct ToolCallStartCallback {
    inner: Arc<dyn Fn(&AgentToolCallStart) + Send + Sync>,
}

impl std::fmt::Debug for ToolCallStartCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ToolCallStartCallback(<fn>)")
    }
}

impl ToolCallStartCallback {
    pub(crate) fn new<F>(callback: F) -> Self
    where
        F: Fn(&AgentToolCallStart) + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(callback),
        }
    }

    pub(crate) fn call(&self, event: &AgentToolCallStart) {
        (self.inner)(event);
    }
}

#[derive(Clone)]
pub(crate) struct ToolCallFinishCallback {
    inner: Arc<dyn Fn(&AgentToolCallFinish) + Send + Sync>,
}

impl std::fmt::Debug for ToolCallFinishCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ToolCallFinishCallback(<fn>)")
    }
}

impl ToolCallFinishCallback {
    pub(crate) fn new<F>(callback: F) -> Self
    where
        F: Fn(&AgentToolCallFinish) + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(callback),
        }
    }

    pub(crate) fn call(&self, event: &AgentToolCallFinish) {
        (self.inner)(event);
    }
}

#[derive(Clone)]
pub(crate) struct StepCallback {
    inner: Arc<dyn Fn(&AgentStep) + Send + Sync>,
}

impl std::fmt::Debug for StepCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StepCallback(<fn>)")
    }
}

impl StepCallback {
    pub(crate) fn new<F>(callback: F) -> Self
    where
        F: Fn(&AgentStep) + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(callback),
        }
    }

    pub(crate) fn call(&self, step: &AgentStep) {
        (self.inner)(step);
    }
}

#[derive(Clone)]
pub(crate) struct FinishCallback {
    inner: Arc<dyn Fn(&AgentFinish) + Send + Sync>,
}

impl std::fmt::Debug for FinishCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FinishCallback(<fn>)")
    }
}

impl FinishCallback {
    pub(crate) fn new<F>(callback: F) -> Self
    where
        F: Fn(&AgentFinish) + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(callback),
        }
    }

    pub(crate) fn call(&self, event: &AgentFinish) {
        (self.inner)(event);
    }
}

#[derive(Clone)]
pub(crate) struct StopWhen {
    inner: Arc<dyn Fn(&AgentStep) -> bool + Send + Sync>,
}

impl std::fmt::Debug for StopWhen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StopWhen(<fn>)")
    }
}

impl StopWhen {
    pub(crate) fn new<F>(predicate: F) -> Self
    where
        F: Fn(&AgentStep) -> bool + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(predicate),
        }
    }

    pub(crate) fn should_stop(&self, step: &AgentStep) -> bool {
        (self.inner)(step)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextResponse {
    pub output_text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub tool_calls: Vec<ToolCall>,
    pub raw_provider_response: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub output_text: String,
    pub steps: u8,
    pub transcript: Vec<Message>,
    pub usage_total: Usage,
    pub step_results: Vec<AgentStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Unknown(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

impl std::ops::Add for Usage {
    type Output = Usage;

    fn add(self, rhs: Self) -> Self::Output {
        Usage {
            input_tokens: self.input_tokens.saturating_add(rhs.input_tokens),
            output_tokens: self.output_tokens.saturating_add(rhs.output_tokens),
            total_tokens: self.total_tokens.saturating_add(rhs.total_tokens),
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens = self.input_tokens.saturating_add(rhs.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(rhs.output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(rhs.total_tokens);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    TextDelta { text: String },
    ToolCallReady { call: ToolCall },
    Usage { usage: Usage },
    Done,
}

pub type TextStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>;
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
    if let Some(temp) = temperature {
        if !(0.0..=2.0).contains(&temp) {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                "temperature must be within 0.0..=2.0",
            ));
        }
    }
    if let Some(p) = top_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                "top_p must be within 0.0..=1.0",
            ));
        }
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
        GenerateTextRequest, ModelRef, OpenAi, ProviderKind, validate_max_steps, validate_model_ref,
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
}
