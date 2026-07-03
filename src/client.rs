//! Provider-bound client types and retry behavior for Aquaregia.
//!
//! This module provides the core client abstractions for making LLM requests:
//!
//! - [`Client`]: Provider-bound client for generate/stream/agent operations
//! - [`ClientBuilder`]: Builder for configuring HTTP/runtime behavior
//!
//! ## Architecture
//!
//! 1. Public provider modules create provider-specific clients.
//! 2. Provider builders configure settings and HTTP behavior.
//! 3. Internally, provider builders produce a [`Client`].
//! 4. [`Client`] is used for all subsequent operations.
//!
//! ## Example
//!
//! ```rust,ignore
//! use aquaregia::providers::openai;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create and build client
//! let client = openai::Client::builder()
//!     .api_key("api-key")
//!     .timeout(std::time::Duration::from_secs(60))
//!     .max_retries(3)
//!     .build()?;
//!
//! // Use client for generation
//! let response = client
//!     .generate(aquaregia::ChatRequest::from_prompt("gpt-5.5", "Hello!"))
//!     .await?;
//!
//! println!("{}", response.output_text);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::join_all;
use futures_util::stream::{FuturesUnordered, StreamExt};
use tokio::time::sleep;

use crate::adapters::ModelAdapter;
#[cfg(feature = "anthropic")]
use crate::adapters::anthropic::{AnthropicAdapter, AnthropicAdapterSettings};
#[cfg(feature = "google")]
use crate::adapters::google::{GoogleAdapter, GoogleAdapterSettings};
#[cfg(feature = "openai")]
use crate::adapters::openai::{OpenAiAdapter, OpenAiAdapterSettings};
#[cfg(feature = "openai-compatible")]
use crate::adapters::openai_compatible::{
    OpenAiCompatibleAdapter, OpenAiCompatibleAdapterSettings,
};
use crate::embed::{EmbedRequest, EmbedResponse, validate_embed_request};
use crate::error::{Error, ErrorCode};
use crate::partial_json::repair_json;
use crate::tool::{Tool, ToolExecError, ToolRegistry};
use crate::types::{
    AgentFinish, AgentOutput, AgentPrepareStep, AgentPreparedStep, AgentStart, AgentStep,
    AgentStepStart, AgentStream, AgentStreamEvent, AgentToolCallFinish, AgentToolCallStart,
    ChatRequest, ChatResponse, ContentPart, FinishReason, Message, ObjectResponse, ObjectStream,
    OutputSchema, ReasoningPart, RunTools, StreamEvent, StreamObjectEvent, TextPart, TextStream,
    ToolCall, ToolErrorPolicy, ToolResult, ToolSourceRef, Usage, validate_messages,
    validate_model_ref, validate_sampling,
};

mod sealed {
    pub trait Sealed {}
    #[cfg(feature = "openai")]
    impl Sealed for super::OpenAiAdapterSettings {}
    #[cfg(feature = "anthropic")]
    impl Sealed for super::AnthropicAdapterSettings {}
    #[cfg(feature = "google")]
    impl Sealed for super::GoogleAdapterSettings {}
    #[cfg(feature = "openai-compatible")]
    impl Sealed for super::OpenAiCompatibleAdapterSettings {}
}

/// Provider-settings contract consumed by [`ClientBuilder`].
///
/// This trait is sealed: it is implemented for enabled built-in
/// `*AdapterSettings` types and cannot be implemented downstream. External
/// providers should add an adapter to the `adapters` module rather
/// than implementing this trait.
pub trait BuildProvider: sealed::Sealed {
    fn validate(&self) -> Result<(), Error>;
    fn into_adapter(self, http: Arc<reqwest::Client>) -> Arc<dyn ModelAdapter>;
}

#[cfg(feature = "openai")]
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

#[cfg(feature = "anthropic")]
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

#[cfg(feature = "google")]
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

#[cfg(feature = "openai-compatible")]
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

/// Configures HTTP/runtime behavior before building a [`Client`].
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
            user_agent: format!("aquaregia/{}", env!("CARGO_PKG_VERSION")),
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

    /// Overrides the default Aquaregia `User-Agent` header value.
    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Builds a provider-bound client with validated settings.
    pub fn build(self) -> Result<Client, Error> {
        self.settings.validate()?;
        let http = Arc::new(
            reqwest::Client::builder()
                .timeout(self.timeout)
                .user_agent(self.user_agent)
                .build()
                .map_err(|e| Error::new(ErrorCode::Transport, e.to_string()))?,
        );

        Ok(Client {
            max_retries: self.max_retries,
            default_max_steps: self.default_max_steps,
            adapter: self.settings.into_adapter(http),
        })
    }
}

#[cfg(feature = "openai")]
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

#[cfg(feature = "anthropic")]
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

#[cfg(feature = "google")]
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

#[cfg(feature = "openai-compatible")]
impl ClientBuilder<OpenAiCompatibleAdapterSettings> {
    /// Sets the OpenAI-compatible endpoint base URL (required).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }

    /// Sets a bearer token for OpenAI-compatible requests.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.settings.api_key = Some(api_key.into());
        self
    }

    /// Sends requests without an `Authorization` bearer token.
    pub fn no_api_key(mut self) -> Self {
        self.settings.api_key = None;
        self
    }

    /// Adds or replaces a custom HTTP header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.headers.insert(name.into(), value.into());
        self
    }

    /// Adds or replaces a query parameter on the chat completions endpoint.
    pub fn query_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.query_params.insert(name.into(), value.into());
        self
    }

    /// Overrides the chat completions path (default: `/v1/chat/completions`).
    pub fn chat_completions_path(mut self, path: impl Into<String>) -> Self {
        self.settings.chat_completions_path = path.into();
        self
    }
}

/// Reusable provider-bound client used for `generate`, `stream`, and agent loops.
///
/// ## Constructing a Client
///
/// Provider-specific public builders produce this internal client:
///
/// ```rust,ignore
/// use aquaregia::providers::openai;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = openai::Client::builder()
///     .api_key("api-key")
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct Client {
    max_retries: u8,
    default_max_steps: u32,
    adapter: Arc<dyn ModelAdapter>,
}

impl Client {
    /// Creates an OpenAI client builder.
    ///
    /// Set the API key with `ClientBuilder::api_key` (required) and optionally
    /// override the endpoint with `ClientBuilder::base_url`.
    #[cfg(feature = "openai")]
    pub fn openai() -> ClientBuilder<OpenAiAdapterSettings> {
        ClientBuilder::new(OpenAiAdapterSettings::new())
    }

    /// Creates an Anthropic client builder.
    ///
    /// Set the API key with `ClientBuilder::api_key` (required) and optionally
    /// override the endpoint with `ClientBuilder::base_url` or the version
    /// header with `ClientBuilder::api_version`.
    #[cfg(feature = "anthropic")]
    pub fn anthropic() -> ClientBuilder<AnthropicAdapterSettings> {
        ClientBuilder::new(AnthropicAdapterSettings::new())
    }

    /// Creates a Google client builder.
    ///
    /// Set the API key with `ClientBuilder::api_key` (required) and optionally
    /// override the endpoint with `ClientBuilder::base_url`.
    #[cfg(feature = "google")]
    pub fn google() -> ClientBuilder<GoogleAdapterSettings> {
        ClientBuilder::new(GoogleAdapterSettings::new())
    }

    /// Creates an OpenAI-compatible client builder.
    ///
    /// Set the base URL with `ClientBuilder::base_url` (required). The bearer
    /// token is optional and configured with `ClientBuilder::api_key` (or
    /// disabled with `ClientBuilder::no_api_key`, which is the default).
    #[cfg(feature = "openai-compatible")]
    pub fn openai_compatible() -> ClientBuilder<OpenAiCompatibleAdapterSettings> {
        ClientBuilder::new(OpenAiCompatibleAdapterSettings::new())
    }

    /// Runs a non-streaming generation request.
    ///
    /// The request is validated locally and retried on retryable failures.
    pub async fn generate(&self, req: ChatRequest) -> Result<ChatResponse, Error> {
        validate_model_ref(&req.model)?;
        validate_messages(&req.messages)?;
        validate_sampling(req.temperature, req.top_p)?;
        self.call_with_retry(|| async { self.adapter.generate_text(&req).await })
            .await
    }

    /// Runs a streaming generation request.
    ///
    /// The request is validated locally and retried on retryable failures.
    pub async fn stream(&self, req: ChatRequest) -> Result<TextStream, Error> {
        validate_model_ref(&req.model)?;
        validate_messages(&req.messages)?;
        validate_sampling(req.temperature, req.top_p)?;
        self.call_with_retry(|| async { self.adapter.stream_text(&req).await })
            .await
    }

    /// Generates embeddings for text values.
    ///
    /// The request is validated locally and retried on retryable failures.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::UnsupportedOperation`] if the provider does not
    /// support embeddings (e.g., Anthropic).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use aquaregia::embed::EmbedRequest;
    /// use aquaregia::providers::openai;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = openai::Client::builder()
    ///     .api_key(std::env::var("OPENAI_API_KEY")?)
    ///     .build()?;
    ///
    /// let response = client.embed(
    ///     EmbedRequest::new("text-embedding-3-small", vec!["Hello, world!"])
    /// ).await?;
    ///
    /// println!("Dimension: {}", response.embeddings[0].len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse, Error> {
        validate_embed_request(&req)?;
        self.call_with_retry(|| async { self.adapter.embed(&req).await })
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

    fn inject_output_schema<T: schemars::JsonSchema>(req: &mut ChatRequest) -> Result<(), Error> {
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
        mut req: ChatRequest,
    ) -> Result<ObjectResponse<T>, Error> {
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
        Ok(ObjectResponse {
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
        mut req: ChatRequest,
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
                    StreamEvent::Done { .. } => {
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

    async fn resolve_tools(
        tools: &[Tool],
        tool_sources: &[ToolSourceRef],
    ) -> Result<Vec<Tool>, Error> {
        let mut resolved = tools.to_vec();
        for source in tool_sources {
            resolved.extend(source.tools().await?);
        }
        Ok(resolved)
    }

    pub(crate) async fn run_tools(&self, req: RunTools) -> Result<AgentOutput, Error> {
        let RunTools {
            model,
            messages,
            tools,
            tool_sources,
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
            provider_options,
            cancellation_token,
        } = req;

        // `0` means unlimited. Otherwise the loop returns MaxStepsExceeded once
        // it would start a step past the cap.
        let resolved_max_steps = max_steps.unwrap_or(self.default_max_steps);

        let mut messages = messages;
        let mut usage_total = Usage::default();
        let mut step_results = Vec::new();
        let start_tools = Self::resolve_tools(&tools, &tool_sources).await?;

        if let Some(callback) = &on_start {
            callback(&AgentStart {
                model_id: model.clone(),
                messages: messages.clone(),
                tool_count: start_tools.len(),
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
                tools: Self::resolve_tools(&tools, &tool_sources).await?,
                temperature,
                max_output_tokens,
                stop_sequences: stop_sequences.clone(),
            };
            if let Some(callback) = &prepare_step {
                prepared_step = callback(&AgentPrepareStep {
                    step,
                    model: model.clone(),
                    messages: messages.clone(),
                    tools: prepared_step.tools.clone(),
                    temperature,
                    max_output_tokens,
                    stop_sequences: stop_sequences.clone(),
                    previous_steps: step_results.clone(),
                });
            }
            let tool_registry = ToolRegistry::from_tools(prepared_step.tools.clone())?;

            validate_messages(&prepared_step.messages)?;

            if let Some(callback) = &on_step_start {
                callback(&AgentStepStart {
                    step,
                    messages: prepared_step.messages.clone(),
                });
            }

            let response = self
                .generate(ChatRequest {
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
                    provider_options: provider_options.clone(),
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
                let final_response = AgentOutput {
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
                on_tool_call_start.as_ref(),
                on_tool_call_finish.as_ref(),
            )
            .await?;
            let mut tool_messages = executed_tool_calls
                .iter()
                .map(|r| Message::tool_result(r.clone()))
                .collect::<Vec<_>>();
            let step_state = AgentStep {
                step,
                output_text: response.output_text.clone(),
                reasoning_text: response.reasoning_text.clone(),
                reasoning_parts: response.reasoning_parts.clone(),
                finish_reason: response.finish_reason.clone(),
                usage: response.usage.clone(),
                tool_calls: response.tool_calls.clone(),
                tool_results: executed_tool_calls.clone(),
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
                let final_response = AgentOutput {
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

    pub(crate) fn stream_tools(self: Arc<Self>, req: RunTools) -> Result<AgentStream, Error> {
        let RunTools {
            model,
            messages,
            tools,
            tool_sources,
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
            provider_options,
            cancellation_token,
        } = req;

        let resolved_max_steps = max_steps.unwrap_or(self.default_max_steps);
        let client = self;

        let stream = async_stream::try_stream! {
            let mut messages = messages;
            let mut usage_total = Usage::default();
            let mut step_results = Vec::new();
            let start_tools = Self::resolve_tools(&tools, &tool_sources).await?;

            let start_event = AgentStart {
                model_id: model.clone(),
                messages: messages.clone(),
                tool_count: start_tools.len(),
                max_steps: resolved_max_steps,
            };
            if let Some(callback) = &on_start {
                callback(&start_event);
            }
            yield AgentStreamEvent::Start { event: start_event };

            let mut step: u32 = 0;
            loop {
                step += 1;
                if resolved_max_steps != 0 && step > resolved_max_steps {
                    Err(Error::new(
                        ErrorCode::MaxStepsExceeded,
                        format!(
                            "agent reached max_steps ({}) without final answer",
                            resolved_max_steps
                        ),
                    ))?;
                }
                if cancellation_token
                    .as_ref()
                    .map(|t| t.is_cancelled())
                    .unwrap_or(false)
                {
                    Err(Error::new(ErrorCode::Cancelled, "agent cancelled"))?;
                }

                let mut prepared_step = AgentPreparedStep {
                    model: model.clone(),
                    messages: messages.clone(),
                    tools: Self::resolve_tools(&tools, &tool_sources).await?,
                    temperature,
                    max_output_tokens,
                    stop_sequences: stop_sequences.clone(),
                };
                if let Some(callback) = &prepare_step {
                    prepared_step = callback(&AgentPrepareStep {
                        step,
                        model: model.clone(),
                        messages: messages.clone(),
                        tools: prepared_step.tools.clone(),
                        temperature,
                        max_output_tokens,
                        stop_sequences: stop_sequences.clone(),
                        previous_steps: step_results.clone(),
                    });
                }
                let tool_registry = ToolRegistry::from_tools(prepared_step.tools.clone())?;

                validate_messages(&prepared_step.messages)?;

                let step_start = AgentStepStart {
                    step,
                    messages: prepared_step.messages.clone(),
                };
                if let Some(callback) = &on_step_start {
                    callback(&step_start);
                }
                yield AgentStreamEvent::StepStart { event: step_start };

                let mut model_stream = client
                    .stream(ChatRequest {
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
                        provider_options: provider_options.clone(),
                        cancellation_token: cancellation_token.clone(),
                    })
                    .await?;

                let mut output_text = String::new();
                let mut reasoning_text = String::new();
                let mut reasoning_parts = Vec::new();
                let mut usage = Usage::default();
                let mut tool_calls = Vec::new();
                let mut finish_reason = FinishReason::Stop;

                while let Some(event) = model_stream.next().await {
                    let event = event?;
                    if let StreamEvent::Done { finish_reason: reason } = &event {
                        finish_reason = reason.clone();
                    }
                    collect_stream_event(
                        &event,
                        &mut output_text,
                        &mut reasoning_text,
                        &mut reasoning_parts,
                        &mut usage,
                        &mut tool_calls,
                    );
                    yield AgentStreamEvent::Model { step, event };
                }

                let response = ChatResponse {
                    output_text,
                    reasoning_text,
                    reasoning_parts: reasoning_parts
                        .into_iter()
                        .map(|(_, part)| part)
                        .collect(),
                    finish_reason,
                    usage,
                    tool_calls,
                    raw_provider_response: None,
                };

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
                    yield AgentStreamEvent::StepFinish {
                        event: step_state.clone(),
                    };

                    let final_response = AgentOutput {
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
                    yield AgentStreamEvent::Done {
                        output: final_response,
                    };
                    break;
                }

                let executed_tool_calls = execute_tool_calls_for_stream(
                    &tool_registry,
                    &response.tool_calls,
                    step,
                    tool_error_policy,
                    on_tool_call_start.as_ref(),
                    on_tool_call_finish.as_ref(),
                )
                .await?;
                for event in executed_tool_calls.events {
                    yield event;
                }

                let mut tool_messages = executed_tool_calls
                    .results
                    .iter()
                    .map(|r| Message::tool_result(r.clone()))
                    .collect::<Vec<_>>();
                let step_state = AgentStep {
                    step,
                    output_text: response.output_text.clone(),
                    reasoning_text: response.reasoning_text.clone(),
                    reasoning_parts: response.reasoning_parts.clone(),
                    finish_reason: response.finish_reason.clone(),
                    usage: response.usage.clone(),
                    tool_calls: response.tool_calls.clone(),
                    tool_results: executed_tool_calls.results.clone(),
                };
                step_results.push(step_state.clone());
                next_messages.append(&mut tool_messages);
                if let Some(callback) = &on_step_finish {
                    callback(&step_state);
                }
                yield AgentStreamEvent::StepFinish {
                    event: step_state.clone(),
                };

                if stop_when
                    .as_ref()
                    .is_some_and(|predicate| predicate(&step_state))
                {
                    let final_response = AgentOutput {
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
                    yield AgentStreamEvent::Done {
                        output: final_response,
                    };
                    break;
                }

                messages = next_messages;
            }
        };

        Ok(Box::pin(stream))
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
    Duration::from_millis(ms)
}

fn assistant_message_from_response(response: &ChatResponse) -> Message {
    let mut parts = Vec::new();
    for reasoning in &response.reasoning_parts {
        parts.push(ContentPart::Reasoning(reasoning.clone()));
    }
    if !response.output_text.is_empty() {
        parts.push(ContentPart::Text(TextPart::new(
            response.output_text.clone(),
        )));
    }
    for call in &response.tool_calls {
        parts.push(ContentPart::ToolCall(call.clone()));
    }
    if parts.is_empty() {
        parts.push(ContentPart::Text(TextPart::new(String::new())));
    }
    Message::assistant_with_parts(parts)
}

fn emit_on_finish(
    callback: Option<&crate::types::Hook<AgentFinish>>,
    response: &AgentOutput,
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

fn collect_stream_event(
    event: &StreamEvent,
    output_text: &mut String,
    reasoning_text: &mut String,
    reasoning_parts: &mut Vec<(String, ReasoningPart)>,
    usage: &mut Usage,
    tool_calls: &mut Vec<ToolCall>,
) {
    match event {
        StreamEvent::ReasoningStarted {
            block_id,
            provider_metadata,
        } => {
            upsert_reasoning_part(reasoning_parts, block_id, "", provider_metadata.clone());
        }
        StreamEvent::ReasoningDelta {
            block_id,
            text,
            provider_metadata,
        } => {
            reasoning_text.push_str(text);
            upsert_reasoning_part(reasoning_parts, block_id, text, provider_metadata.clone());
        }
        StreamEvent::ReasoningDone {
            block_id,
            provider_metadata,
        } => {
            upsert_reasoning_part(reasoning_parts, block_id, "", provider_metadata.clone());
        }
        StreamEvent::TextDelta { text } => {
            output_text.push_str(text);
        }
        StreamEvent::ToolCallReady { call } => {
            tool_calls.push(call.clone());
        }
        StreamEvent::Usage {
            usage: stream_usage,
        } => {
            *usage = stream_usage.clone();
        }
        StreamEvent::Done { .. } => {}
    }
}

fn upsert_reasoning_part(
    parts: &mut Vec<(String, ReasoningPart)>,
    block_id: &str,
    text_delta: &str,
    provider_metadata: Option<serde_json::Value>,
) {
    if let Some((_, part)) = parts.iter_mut().find(|(id, _)| id == block_id) {
        part.text.push_str(text_delta);
        if provider_metadata.is_some() {
            part.provider_metadata = provider_metadata;
        }
        return;
    }

    parts.push((
        block_id.to_string(),
        ReasoningPart {
            text: text_delta.to_string(),
            provider_metadata,
        },
    ));
}

struct StreamToolCalls {
    results: Vec<ToolResult>,
    events: Vec<AgentStreamEvent>,
}

async fn execute_tool_calls_for_stream(
    registry: &ToolRegistry,
    calls: &[ToolCall],
    step: u32,
    policy: ToolErrorPolicy,
    on_tool_call_start: Option<&crate::types::Hook<AgentToolCallStart>>,
    on_tool_call_finish: Option<&crate::types::Hook<AgentToolCallFinish>>,
) -> Result<StreamToolCalls, Error> {
    for call in calls {
        if registry.resolve(&call.tool_name).is_none() {
            return Err(Error::new(
                ErrorCode::UnknownTool,
                format!("unknown tool `{}`", call.tool_name),
            ));
        }
    }

    let mut events = Vec::with_capacity(calls.len() * 2);
    let mut tasks = FuturesUnordered::new();
    for (index, call) in calls.iter().cloned().enumerate() {
        let registered = registry
            .resolve(&call.tool_name)
            .expect("tool existence was checked above");

        let start_event = AgentToolCallStart {
            step,
            tool_call: call.clone(),
        };
        if let Some(callback) = on_tool_call_start {
            callback(&start_event);
        }
        events.push(AgentStreamEvent::ToolCallStart { event: start_event });

        let executor = Arc::clone(&registered.executor);
        let args_json = call.args_json.clone();
        tasks.push(async move {
            let started_at = Instant::now();
            let result = executor.execute(args_json).await;
            let duration_ms = started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
            (index, call, result, duration_ms)
        });
    }

    let mut indexed_results = vec![None; calls.len()];
    while let Some((index, call, result, duration_ms)) = tasks.next().await {
        let tool_result = tool_result_from_execution(&call, result, policy)?;

        let finish_event = AgentToolCallFinish {
            step,
            tool_call: call,
            tool_result: tool_result.clone(),
            duration_ms,
        };
        if let Some(callback) = on_tool_call_finish {
            callback(&finish_event);
        }
        events.push(AgentStreamEvent::ToolCallFinish {
            event: finish_event,
        });
        indexed_results[index] = Some(tool_result);
    }

    let results = indexed_results
        .into_iter()
        .map(|result| {
            result.ok_or_else(|| {
                Error::new(
                    ErrorCode::ToolExecutionFailed,
                    "tool task did not produce a result",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(StreamToolCalls { results, events })
}

fn tool_result_from_execution(
    call: &ToolCall,
    result: Result<serde_json::Value, ToolExecError>,
    policy: ToolErrorPolicy,
) -> Result<ToolResult, Error> {
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

    Ok(ToolResult {
        call_id: call.call_id.clone(),
        output_json,
        is_error,
    })
}

async fn execute_tool_calls(
    registry: &ToolRegistry,
    calls: &[ToolCall],
    step: u32,
    policy: ToolErrorPolicy,
    on_tool_call_start: Option<&crate::types::Hook<AgentToolCallStart>>,
    on_tool_call_finish: Option<&crate::types::Hook<AgentToolCallFinish>>,
) -> Result<Vec<ToolResult>, Error> {
    let mut results_out = Vec::with_capacity(calls.len());
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

        let executor = Arc::clone(&registered.executor);
        let call = call.clone();
        let args_json = call.args_json.clone();
        tasks.push(async move {
            let started_at = Instant::now();
            let result = executor.execute(args_json).await;
            let duration_ms = started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
            (call, result, duration_ms)
        });
    }

    let results = join_all(tasks).await;
    for (call, result, duration_ms) in results {
        let tool_result = tool_result_from_execution(&call, result, policy)?;

        if let Some(callback) = on_tool_call_finish {
            callback(&AgentToolCallFinish {
                step,
                tool_call: call.clone(),
                tool_result: tool_result.clone(),
                duration_ms,
            });
        }

        results_out.push(tool_result);
    }

    Ok(results_out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::json;

    use crate::error::Error;
    use crate::tool::Tool;
    use crate::types::{ToolSource, ToolSourceRef};

    use super::Client;

    struct StaticToolSource(Vec<Tool>);

    #[async_trait]
    impl ToolSource for StaticToolSource {
        async fn tools(&self) -> Result<Vec<Tool>, Error> {
            Ok(self.0.clone())
        }
    }

    #[cfg(feature = "openai")]
    #[test]
    fn openai_builder_builds() {
        let client = Client::openai()
            .api_key("key")
            .build()
            .expect("client should build");
        let _ = client;
    }

    #[cfg(feature = "anthropic")]
    #[test]
    fn anthropic_builder_builds() {
        let client = Client::anthropic()
            .api_key("key")
            .build()
            .expect("client should build");
        let _ = client;
    }

    #[tokio::test]
    async fn resolve_tools_merges_static_tools_and_sources() {
        let static_tool = crate::tool("static_tool")
            .description("static")
            .execute_raw(|_| async { Ok(json!({"ok": true})) });
        let dynamic_tool = crate::tool("dynamic_tool")
            .description("dynamic")
            .execute_raw(|_| async { Ok(json!({"ok": true})) });
        let source: ToolSourceRef = Arc::new(StaticToolSource(vec![dynamic_tool]));

        let tools = Client::resolve_tools(&[static_tool], &[source])
            .await
            .expect("tools should resolve");

        let names = tools
            .into_iter()
            .map(|tool| tool.descriptor.name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["static_tool", "dynamic_tool"]);
    }
}
