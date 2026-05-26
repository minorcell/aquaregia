//! Provider-bound client types and retry behavior for Aquaregia.
//!
//! This module provides the core client abstractions for making LLM requests:
//!
//! - [`LlmClient`]: Entry point for creating provider-specific clients
//! - [`ClientBuilder`]: Builder for configuring HTTP/runtime behavior
//! - [`BoundClient`]: Reusable client for generate/stream/agent operations
//!
//! ## Architecture
//!
//! 1. [`LlmClient`] constructors return a [`ClientBuilder`] parameterised on the settings type
//! 2. [`ClientBuilder`] configures settings and HTTP behavior
//! 3. [`ClientBuilder::build()`] produces a [`BoundClient`]
//! 4. [`BoundClient`] is used for all subsequent operations
//!
//! ## Example
//!
//! ```rust,no_run
//! use aquaregia::{GenerateTextRequest, LlmClient};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create and build client
//! let client = LlmClient::openai()
//!     .api_key("api-key")
//!     .timeout(std::time::Duration::from_secs(60))
//!     .max_retries(3)
//!     .build()?;
//!
//! // Use client for generation
//! let response = client
//!     .generate(GenerateTextRequest::from_user_prompt("gpt-4o", "Hello!"))
//!     .await?;
//!
//! println!("{}", response.output_text);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::join_all;
use tokio::time::sleep;

use crate::error::{Error, ErrorCode};
use crate::model_adapters::ModelAdapter;
use crate::model_adapters::anthropic::{AnthropicAdapter, AnthropicAdapterSettings};
use crate::model_adapters::google::{GoogleAdapter, GoogleAdapterSettings};
use crate::model_adapters::openai::{OpenAiAdapter, OpenAiAdapterSettings};
use crate::model_adapters::openai_compatible::{
    OpenAiCompatibleAdapter, OpenAiCompatibleAdapterSettings,
};
use crate::partial_json::repair_json;
use crate::tool::{ToolExecError, ToolRegistry};
use crate::types::{
    AgentFinish, AgentPrepareStep, AgentPreparedStep, AgentResponse, AgentStart, AgentStep,
    AgentStepStart, AgentToolCallFinish, AgentToolCallStart, ContentPart, GenerateObjectResponse,
    GenerateTextRequest, GenerateTextResponse, Message, ObjectStream, OutputSchema, RunTools,
    StreamEvent, StreamObjectEvent, TextStream, ToolCall, ToolErrorPolicy, ToolResult, Usage,
    validate_messages, validate_model_ref, validate_sampling,
};

#[doc(hidden)]
pub trait BuildProvider {
    fn validate(&self) -> Result<(), Error>;
    fn into_adapter(self, http: Arc<reqwest::Client>) -> Arc<dyn ModelAdapter>;
}

impl BuildProvider for OpenAiAdapterSettings {
    fn validate(&self) -> Result<(), Error> {
        if self.api_key.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::AuthFailed,
                "api_key must not be empty",
            ));
        }
        Ok(())
    }
    fn into_adapter(self, http: Arc<reqwest::Client>) -> Arc<dyn ModelAdapter> {
        Arc::new(OpenAiAdapter::from_settings(self, http))
    }
}

impl BuildProvider for AnthropicAdapterSettings {
    fn validate(&self) -> Result<(), Error> {
        if self.api_key.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::AuthFailed,
                "api_key must not be empty",
            ));
        }
        Ok(())
    }
    fn into_adapter(self, http: Arc<reqwest::Client>) -> Arc<dyn ModelAdapter> {
        Arc::new(AnthropicAdapter::from_settings(self, http))
    }
}

impl BuildProvider for GoogleAdapterSettings {
    fn validate(&self) -> Result<(), Error> {
        if self.api_key.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::AuthFailed,
                "api_key must not be empty",
            ));
        }
        Ok(())
    }
    fn into_adapter(self, http: Arc<reqwest::Client>) -> Arc<dyn ModelAdapter> {
        Arc::new(GoogleAdapter::from_settings(self, http))
    }
}

impl BuildProvider for OpenAiCompatibleAdapterSettings {
    fn validate(&self) -> Result<(), Error> {
        if self.base_url.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                "base_url must not be empty",
            ));
        }
        Ok(())
    }
    fn into_adapter(self, http: Arc<reqwest::Client>) -> Arc<dyn ModelAdapter> {
        Arc::new(OpenAiCompatibleAdapter::from_settings(self, http))
    }
}

/// Entry point for creating provider-bound clients.
///
/// `LlmClient` only provides constructors; call `.build()` on the returned
/// [`ClientBuilder`] to obtain a reusable [`BoundClient`].
pub struct LlmClient;

impl LlmClient {
    /// Creates an OpenAI client builder.
    ///
    /// Set the API key with [`ClientBuilder::api_key`] (required) and optionally
    /// override the endpoint with [`ClientBuilder::base_url`].
    pub fn openai() -> ClientBuilder<OpenAiAdapterSettings> {
        ClientBuilder::new(OpenAiAdapterSettings::new())
    }

    /// Creates an Anthropic client builder.
    ///
    /// Set the API key with [`ClientBuilder::api_key`] (required) and optionally
    /// override the endpoint with [`ClientBuilder::base_url`] or the version
    /// header with [`ClientBuilder::api_version`].
    pub fn anthropic() -> ClientBuilder<AnthropicAdapterSettings> {
        ClientBuilder::new(AnthropicAdapterSettings::new())
    }

    /// Creates a Google client builder.
    ///
    /// Set the API key with [`ClientBuilder::api_key`] (required) and optionally
    /// override the endpoint with [`ClientBuilder::base_url`].
    pub fn google() -> ClientBuilder<GoogleAdapterSettings> {
        ClientBuilder::new(GoogleAdapterSettings::new())
    }

    /// Creates an OpenAI-compatible client builder.
    ///
    /// Set the base URL with [`ClientBuilder::base_url`] (required). The bearer
    /// token is optional and configured with [`ClientBuilder::api_key`] (or
    /// disabled with [`ClientBuilder::no_api_key`], which is the default).
    pub fn openai_compatible() -> ClientBuilder<OpenAiCompatibleAdapterSettings> {
        ClientBuilder::new(OpenAiCompatibleAdapterSettings::new())
    }
}

/// Configures HTTP/runtime behavior before building a [`BoundClient`].
pub struct ClientBuilder<S> {
    timeout: Duration,
    max_retries: u8,
    default_max_steps: u32,
    user_agent: String,
    settings: S,
}

impl<S: BuildProvider> ClientBuilder<S> {
    fn new(settings: S) -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            default_max_steps: 0,
            user_agent: format!("aquaregia-ai-sdk/{}", env!("CARGO_PKG_VERSION")),
            settings,
        }
    }

    /// Sets request timeout for all requests sent by this client.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the maximum number of retries for retryable errors.
    pub fn max_retries(mut self, retries: u8) -> Self {
        self.max_retries = retries;
        self
    }

    /// Sets the default max step count used by agent tool loops.
    ///
    /// `0` (the default) means unlimited: agents loop until the model returns a
    /// final answer, a `stop_when` predicate matches, or the run is cancelled.
    pub fn default_max_steps(mut self, max_steps: u32) -> Self {
        self.default_max_steps = max_steps;
        self
    }

    /// Overrides the default SDK `User-Agent` header value.
    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Builds a provider-bound client with validated settings.
    pub fn build(self) -> Result<BoundClient, Error> {
        self.settings.validate()?;
        let http = Arc::new(
            reqwest::Client::builder()
                .timeout(self.timeout)
                .user_agent(self.user_agent)
                .build()
                .map_err(|e| Error::new(ErrorCode::Transport, e.to_string()))?,
        );

        Ok(BoundClient {
            max_retries: self.max_retries,
            default_max_steps: self.default_max_steps,
            adapter: self.settings.into_adapter(http),
        })
    }
}

impl ClientBuilder<OpenAiAdapterSettings> {
    /// Sets the OpenAI API key (required).
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.settings.api_key = api_key.into();
        self
    }

    /// Overrides the OpenAI API base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }
}

impl ClientBuilder<AnthropicAdapterSettings> {
    /// Sets the Anthropic API key (required).
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.settings.api_key = api_key.into();
        self
    }

    /// Overrides the Anthropic API base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }

    /// Overrides the Anthropic API version header.
    pub fn api_version(mut self, api_version: impl Into<String>) -> Self {
        self.settings.api_version = api_version.into();
        self
    }
}

impl ClientBuilder<GoogleAdapterSettings> {
    /// Sets the Google API key (required).
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.settings.api_key = api_key.into();
        self
    }

    /// Overrides the Google Generative Language API base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }
}

impl ClientBuilder<OpenAiCompatibleAdapterSettings> {
    /// Sets the OpenAI-compatible endpoint base URL (required).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }

    /// Sets a bearer token for OpenAI-compatible requests.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.settings.set_api_key(api_key);
        self
    }

    /// Sends requests without an `Authorization` bearer token.
    pub fn no_api_key(mut self) -> Self {
        self.settings.clear_api_key();
        self
    }

    /// Adds or replaces a custom HTTP header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.insert_header(name, value);
        self
    }

    /// Adds or replaces a query parameter on the chat completions endpoint.
    pub fn query_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.insert_query_param(name, value);
        self
    }

    /// Overrides the chat completions path (default: `/v1/chat/completions`).
    pub fn chat_completions_path(mut self, path: impl Into<String>) -> Self {
        self.settings.set_chat_completions_path(path);
        self
    }
}

/// Reusable provider-bound client used for `generate`, `stream`, and agent loops.
pub struct BoundClient {
    max_retries: u8,
    default_max_steps: u32,
    adapter: Arc<dyn ModelAdapter>,
}

impl BoundClient {
    /// Runs a non-streaming generation request.
    ///
    /// The request is validated locally and retried on retryable failures.
    pub async fn generate(&self, req: GenerateTextRequest) -> Result<GenerateTextResponse, Error> {
        validate_model_ref(&req.model)?;
        validate_messages(&req.messages)?;
        validate_sampling(req.temperature, req.top_p)?;
        self.call_with_retry(|| async { self.adapter.generate_text(&req).await })
            .await
    }

    /// Runs a streaming generation request.
    ///
    /// The request is validated locally and retried on retryable failures.
    pub async fn stream(&self, req: GenerateTextRequest) -> Result<TextStream, Error> {
        validate_model_ref(&req.model)?;
        validate_messages(&req.messages)?;
        validate_sampling(req.temperature, req.top_p)?;
        self.call_with_retry(|| async { self.adapter.stream_text(&req).await })
            .await
    }

    fn parse_final_buffer<T: serde::de::DeserializeOwned>(buffer: &str) -> Result<T, Error> {
        let repaired = repair_json(buffer);
        serde_json::from_str::<T>(&repaired).map_err(|e| {
            let mut err = Error::new(
                ErrorCode::InvalidResponse,
                format!(
                    "failed to parse streamed object as {}: {}",
                    std::any::type_name::<T>(),
                    e
                ),
            );
            err.raw_body = Some(buffer.to_string());
            err
        })
    }

    fn inject_output_schema<T: schemars::JsonSchema>(
        req: &mut GenerateTextRequest,
    ) -> Result<(), Error> {
        let schema = schemars::schema_for!(T);
        let json_schema = serde_json::to_value(&schema)
            .map_err(|e| Error::new(ErrorCode::InvalidRequest, e.to_string()))?;
        let raw = std::any::type_name::<T>();
        // Strip generic suffixes (`Vec<Foo>` → `Foo`) and path segments.
        let name = raw
            .split("::")
            .last()
            .unwrap_or("output")
            .trim_end_matches(|c: char| !c.is_alphanumeric())
            .to_string();
        req.output_schema = Some(OutputSchema {
            name,
            description: None,
            json_schema,
        });
        Ok(())
    }

    /// Performs a non-streaming generation that returns deserialized structured output.
    ///
    /// The JSON Schema is derived automatically from `T` via [`schemars::JsonSchema`].
    /// Providers that lack native structured-output support (Anthropic, Google) use a
    /// tool-use fallback: the adapter injects a forced tool call and extracts its arguments.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::InvalidResponse`] if the deserialization from JSON fails.
    pub async fn generate_object<T: serde::de::DeserializeOwned + schemars::JsonSchema>(
        &self,
        mut req: GenerateTextRequest,
    ) -> Result<GenerateObjectResponse<T>, Error> {
        Self::inject_output_schema::<T>(&mut req)?;
        let response = self.generate(req).await?;
        let object: T = match serde_json::from_str(&response.output_text) {
            Ok(obj) => obj,
            Err(e) => {
                return Err({
                    let mut err = Error::new(
                        ErrorCode::InvalidResponse,
                        format!(
                            "failed to parse structured output as {}: {}",
                            std::any::type_name::<T>(),
                            e
                        ),
                    );
                    err.raw_body = Some(response.output_text);
                    err
                });
            }
        };
        Ok(GenerateObjectResponse {
            object,
            reasoning_text: response.reasoning_text,
            finish_reason: response.finish_reason,
            usage: response.usage,
            raw_provider_response: response.raw_provider_response,
        })
    }

    /// Streams a generation that emits progressively-populated structured output.
    ///
    /// As the model streams JSON tokens, each chunk is repaired and deserialised
    /// into a partial `T`. Downstream consumers receive [`StreamObjectEvent::Partial`]
    /// events as fields arrive, and a final [`StreamObjectEvent::Object`] when the
    /// stream completes.
    ///
    /// Fields not yet emitted by the model are left at their `Default`. For this
    /// reason `T` should use `#[serde(default)]` on fields.
    pub async fn stream_object<
        T: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static,
    >(
        &self,
        mut req: GenerateTextRequest,
    ) -> Result<ObjectStream<T>, Error> {
        Self::inject_output_schema::<T>(&mut req)?;

        let mut stream = self.stream(req).await?;
        let mut buffer = String::new();
        let mut last_emitted_len = 0usize;

        let object_stream = async_stream::try_stream! {
            let mut saw_done = false;
            while let Some(event) = futures_util::StreamExt::next(&mut stream).await {
                match event? {
                    StreamEvent::TextDelta { text } => {
                        buffer.push_str(&text);
                        // Only attempt partial parse when we have meaningful new
                        // content.  A few leading bytes (e.g. `{"city":`) can't
                        // produce a useful partial before the value arrives.
                        if buffer.len() - last_emitted_len < 8 {
                            continue;
                        }
                        let repaired = repair_json(&buffer);
                        if let Ok(partial) = serde_json::from_str::<T>(&repaired) {
                            yield StreamObjectEvent::Partial { partial };
                            last_emitted_len = buffer.len();
                        }
                    }
                    StreamEvent::Done => {
                        yield StreamObjectEvent::Object {
                            object: Self::parse_final_buffer::<T>(&buffer)?,
                        };
                        saw_done = true;
                        break;
                    }
                    StreamEvent::Usage { .. }
                    | StreamEvent::ReasoningStarted { .. }
                    | StreamEvent::ReasoningDelta { .. }
                    | StreamEvent::ReasoningDone { .. }
                    | StreamEvent::ToolCallReady { .. } => {}
                }
            }

            // Stream ended without Done — flush whatever remains.
            if !saw_done && !buffer.is_empty() {
                yield StreamObjectEvent::Object {
                    object: Self::parse_final_buffer::<T>(&buffer)?,
                };
            }
        };

        Ok(Box::pin(object_stream))
    }

    pub(crate) async fn run_tools(&self, req: RunTools) -> Result<AgentResponse, Error> {
        let RunTools {
            model,
            messages,
            tools,
            max_steps,
            temperature,
            top_p,
            max_output_tokens,
            stop_sequences,
            prepare_step,
            on_start,
            on_step_start,
            on_tool_call_start,
            on_tool_call_finish,
            on_step_finish,
            on_finish,
            stop_when,
            tool_error_policy,
            tool_concurrency,
            cancellation_token,
        } = req;

        // `0` means unlimited. Otherwise the loop returns MaxStepsExceeded once
        // it would start a step past the cap.
        let resolved_max_steps = max_steps.unwrap_or(self.default_max_steps);

        let mut messages = messages;
        let mut usage_total = Usage::default();
        let mut step_results = Vec::new();
        let mut tool_registry = ToolRegistry::from_tools(tools.clone())?;
        let mut cached_tools: Vec<crate::tool::Tool> = tools.clone();

        if let Some(callback) = &on_start {
            callback(&AgentStart {
                model_id: model.model().to_string(),
                messages: messages.clone(),
                tool_count: tools.len(),
                max_steps: resolved_max_steps,
            });
        }

        let mut step: u32 = 0;
        loop {
            step += 1;
            if resolved_max_steps != 0 && step > resolved_max_steps {
                return Err(Error::new(
                    ErrorCode::MaxStepsExceeded,
                    format!(
                        "agent reached max_steps ({}) without final answer",
                        resolved_max_steps
                    ),
                ));
            }
            if cancellation_token
                .as_ref()
                .map(|t| t.is_cancelled())
                .unwrap_or(false)
            {
                return Err(Error::new(ErrorCode::Cancelled, "agent cancelled"));
            }

            let mut prepared_step = AgentPreparedStep {
                model: model.clone(),
                messages: messages.clone(),
                tools: tools.clone(),
                temperature,
                max_output_tokens,
                stop_sequences: stop_sequences.clone(),
            };
            if let Some(callback) = &prepare_step {
                prepared_step = callback(&AgentPrepareStep {
                    step,
                    model: model.clone(),
                    messages: messages.clone(),
                    tools: tools.clone(),
                    temperature,
                    max_output_tokens,
                    stop_sequences: stop_sequences.clone(),
                    previous_steps: step_results.clone(),
                });
                // Rebuild registry only when prepare_step changed the tool list.
                if prepared_step.tools.len() != cached_tools.len()
                    || prepared_step
                        .tools
                        .iter()
                        .zip(cached_tools.iter())
                        .any(|(a, b)| a.descriptor.name != b.descriptor.name)
                {
                    tool_registry = ToolRegistry::from_tools(prepared_step.tools.clone())?;
                    cached_tools = prepared_step.tools.clone();
                }
            }

            validate_messages(&prepared_step.messages)?;

            if let Some(callback) = &on_step_start {
                callback(&AgentStepStart {
                    step,
                    messages: prepared_step.messages.clone(),
                });
            }

            let response = self
                .generate(GenerateTextRequest {
                    model: prepared_step.model.clone(),
                    messages: prepared_step.messages.clone(),
                    temperature: prepared_step.temperature,
                    top_p,
                    max_output_tokens: prepared_step.max_output_tokens,
                    stop_sequences: prepared_step.stop_sequences.clone(),
                    tools: if prepared_step.tools.is_empty() {
                        None
                    } else {
                        Some(
                            prepared_step
                                .tools
                                .iter()
                                .map(|tool| tool.descriptor.clone())
                                .collect(),
                        )
                    },
                    output_schema: None,
                    cancellation_token: cancellation_token.clone(),
                })
                .await?;
            usage_total += response.usage.clone();
            let mut next_messages = prepared_step.messages.clone();
            next_messages.push(assistant_message_from_response(&response));

            if response.tool_calls.is_empty() {
                let step_state = AgentStep {
                    step,
                    output_text: response.output_text.clone(),
                    reasoning_text: response.reasoning_text.clone(),
                    reasoning_parts: response.reasoning_parts.clone(),
                    finish_reason: response.finish_reason.clone(),
                    usage: response.usage.clone(),
                    tool_calls: Vec::new(),
                    tool_results: Vec::new(),
                };
                step_results.push(step_state.clone());
                if let Some(callback) = &on_step_finish {
                    callback(&step_state);
                }
                let final_response = AgentResponse {
                    output_text: response.output_text,
                    steps: step,
                    transcript: next_messages,
                    usage_total,
                    step_results: step_results.clone(),
                };
                emit_on_finish(
                    on_finish.as_ref(),
                    &final_response,
                    &step_state.finish_reason,
                    &step_results,
                );
                return Ok(final_response);
            }

            let executed_tool_calls = execute_tool_calls(
                &tool_registry,
                &response.tool_calls,
                step,
                tool_error_policy,
                tool_concurrency,
                on_tool_call_start.as_ref(),
                on_tool_call_finish.as_ref(),
            )
            .await?;
            let mut tool_messages = executed_tool_calls
                .iter()
                .map(|entry| Message::tool_result(entry.result.clone()))
                .collect::<Vec<_>>();
            let step_state = AgentStep {
                step,
                output_text: response.output_text.clone(),
                reasoning_text: response.reasoning_text.clone(),
                reasoning_parts: response.reasoning_parts.clone(),
                finish_reason: response.finish_reason.clone(),
                usage: response.usage.clone(),
                tool_calls: response.tool_calls.clone(),
                tool_results: executed_tool_calls
                    .iter()
                    .map(|entry| entry.result.clone())
                    .collect(),
            };
            step_results.push(step_state.clone());
            next_messages.append(&mut tool_messages);
            if let Some(callback) = &on_step_finish {
                callback(&step_state);
            }
            if stop_when
                .as_ref()
                .is_some_and(|predicate| predicate(&step_state))
            {
                let final_response = AgentResponse {
                    output_text: response.output_text,
                    steps: step,
                    transcript: next_messages,
                    usage_total,
                    step_results: step_results.clone(),
                };
                emit_on_finish(
                    on_finish.as_ref(),
                    &final_response,
                    &step_state.finish_reason,
                    &step_results,
                );
                return Ok(final_response);
            }

            messages = next_messages;
        }
    }

    async fn call_with_retry<T, F, Fut>(&self, mut op: F) -> Result<T, Error>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, Error>>,
    {
        let mut attempt = 0u8;
        loop {
            match op().await {
                Ok(v) => return Ok(v),
                Err(err) => {
                    if !err.retryable || attempt >= self.max_retries {
                        return Err(err);
                    }
                    attempt = attempt.saturating_add(1);
                    let delay = err
                        .retry_after_secs
                        .map(Duration::from_secs)
                        .unwrap_or_else(|| backoff_delay(attempt));
                    sleep(delay).await;
                }
            }
        }
    }
}

fn backoff_delay(attempt: u8) -> Duration {
    let base_ms = 200u64;
    let cap_ms = 2_000u64;
    let exp = 2u64.saturating_pow(attempt as u32);
    let ms = (base_ms.saturating_mul(exp)).min(cap_ms);
    // Simple jitter: stagger by ±25% to avoid thundering herd.
    let jitter = (ms as f64 * 0.25) as i64;
    let offset = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as i64)
        % (jitter * 2 + 1)
        - jitter;
    Duration::from_millis((ms as i64 + offset).max(0) as u64)
}

fn assistant_message_from_response(response: &GenerateTextResponse) -> Message {
    let mut parts = Vec::new();
    for reasoning in &response.reasoning_parts {
        parts.push(ContentPart::Reasoning(reasoning.clone()));
    }
    if !response.output_text.is_empty() {
        parts.push(ContentPart::Text(response.output_text.clone()));
    }
    for call in &response.tool_calls {
        parts.push(ContentPart::ToolCall(call.clone()));
    }
    if parts.is_empty() {
        parts.push(ContentPart::Text(String::new()));
    }
    Message::assistant_with_parts(parts)
}

fn emit_on_finish(
    callback: Option<&crate::types::Hook<AgentFinish>>,
    response: &AgentResponse,
    finish_reason: &crate::types::FinishReason,
    step_results: &[AgentStep],
) {
    let Some(callback) = callback else {
        return;
    };

    callback(&AgentFinish {
        output_text: response.output_text.clone(),
        step_count: response.steps,
        finish_reason: finish_reason.clone(),
        usage_total: response.usage_total.clone(),
        transcript: response.transcript.clone(),
        step_results: step_results.to_vec(),
    });
}

#[derive(Debug, Clone)]
struct ExecutedToolCall {
    result: ToolResult,
}

async fn execute_tool_calls(
    registry: &ToolRegistry,
    calls: &[ToolCall],
    step: u32,
    policy: ToolErrorPolicy,
    concurrency: usize,
    on_tool_call_start: Option<&crate::types::Hook<AgentToolCallStart>>,
    on_tool_call_finish: Option<&crate::types::Hook<AgentToolCallFinish>>,
) -> Result<Vec<ExecutedToolCall>, Error> {
    let mut executions = Vec::with_capacity(calls.len());
    let mut tasks = Vec::with_capacity(calls.len());
    for call in calls {
        let Some(registered) = registry.resolve(&call.tool_name) else {
            return Err(Error::new(
                ErrorCode::UnknownTool,
                format!("unknown tool `{}`", call.tool_name),
            ));
        };

        if let Some(callback) = on_tool_call_start {
            callback(&AgentToolCallStart {
                step,
                tool_call: call.clone(),
            });
        }

        let executor = Arc::clone(&registered.tool.executor);
        let call = call.clone();
        let args_json = call.args_json.clone();
        tasks.push(async move {
            let started_at = Instant::now();
            let result = executor.execute(args_json).await;
            let duration_ms = started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
            (call, result, duration_ms)
        });
    }

    let results = if concurrency == 1 {
        let mut sequential = Vec::with_capacity(tasks.len());
        for task in tasks {
            sequential.push(task.await);
        }
        sequential
    } else {
        join_all(tasks).await
    };
    for (call, result, duration_ms) in results {
        let (output_json, is_error) = match result {
            Ok(output_json) => (output_json, false),
            Err(ToolExecError::Execution(message)) => {
                if policy == ToolErrorPolicy::FailFast {
                    return Err(Error::new(
                        ErrorCode::ToolExecutionFailed,
                        format!(
                            "tool `{}` execution failed for call `{}`: {}",
                            call.tool_name, call.call_id, message
                        ),
                    ));
                }
                (serde_json::json!({ "error": message }), true)
            }
            Err(ToolExecError::Timeout) => {
                if policy == ToolErrorPolicy::FailFast {
                    return Err(Error::new(
                        ErrorCode::ToolExecutionFailed,
                        format!(
                            "tool `{}` timed out for call `{}`",
                            call.tool_name, call.call_id
                        ),
                    ));
                }
                (serde_json::json!({ "error": "timeout" }), true)
            }
        };

        let tool_result = ToolResult {
            call_id: call.call_id.clone(),
            output_json,
            is_error,
        };

        if let Some(callback) = on_tool_call_finish {
            callback(&AgentToolCallFinish {
                step,
                tool_call: call.clone(),
                tool_result: tool_result.clone(),
                duration_ms,
            });
        }

        executions.push(ExecutedToolCall {
            result: tool_result,
        });
    }

    Ok(executions)
}

#[cfg(test)]
mod tests {
    use super::LlmClient;

    #[test]
    fn openai_builder_builds() {
        let client = LlmClient::openai()
            .api_key("key")
            .build()
            .expect("client should build");
        let _ = client;
    }

    #[test]
    fn anthropic_builder_builds() {
        let client = LlmClient::anthropic()
            .api_key("key")
            .build()
            .expect("client should build");
        let _ = client;
    }
}
