use std::marker::PhantomData;
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
use crate::tool::{ToolExecError, ToolRegistry};
use crate::types::{
    AgentFinish, AgentPrepareStep, AgentPreparedStep, AgentResponse, AgentStart, AgentStep,
    AgentStepStart, AgentToolCallFinish, AgentToolCallStart, Anthropic, ContentPart,
    GenerateTextRequest, GenerateTextResponse, Google, Message, OpenAi, OpenAiCompatible,
    ProviderMarker, RunTools, TextStream, ToolCall, ToolErrorPolicy, ToolResult, Usage,
    validate_max_steps, validate_messages, validate_model_ref, validate_sampling,
};

/// Binds a provider marker type to its adapter settings and adapter implementation.
///
/// This trait is mainly an internal extension seam used by [`ClientBuilder`].
pub trait ProviderBinding: ProviderMarker {
    /// Provider-specific client configuration payload.
    type Settings;

    /// Converts provider settings into a runtime adapter.
    fn into_adapter(
        settings: Self::Settings,
        http: Arc<reqwest::Client>,
    ) -> Arc<dyn ModelAdapter<Self>>;
}

impl ProviderBinding for OpenAi {
    type Settings = OpenAiAdapterSettings;

    fn into_adapter(
        settings: Self::Settings,
        http: Arc<reqwest::Client>,
    ) -> Arc<dyn ModelAdapter<Self>> {
        Arc::new(OpenAiAdapter::from_settings(settings, http))
    }
}

impl ProviderBinding for Anthropic {
    type Settings = AnthropicAdapterSettings;

    fn into_adapter(
        settings: Self::Settings,
        http: Arc<reqwest::Client>,
    ) -> Arc<dyn ModelAdapter<Self>> {
        Arc::new(AnthropicAdapter::from_settings(settings, http))
    }
}

impl ProviderBinding for Google {
    type Settings = GoogleAdapterSettings;

    fn into_adapter(
        settings: Self::Settings,
        http: Arc<reqwest::Client>,
    ) -> Arc<dyn ModelAdapter<Self>> {
        Arc::new(GoogleAdapter::from_settings(settings, http))
    }
}

impl ProviderBinding for OpenAiCompatible {
    type Settings = OpenAiCompatibleAdapterSettings;

    fn into_adapter(
        settings: Self::Settings,
        http: Arc<reqwest::Client>,
    ) -> Arc<dyn ModelAdapter<Self>> {
        Arc::new(OpenAiCompatibleAdapter::from_settings(settings, http))
    }
}

/// Entry point for creating provider-bound clients.
///
/// `LlmClient` only provides constructors; call `.build()` on the returned
/// [`ClientBuilder`] to obtain a reusable [`BoundClient`].
pub struct LlmClient;

impl LlmClient {
    /// Creates an OpenAI client builder.
    pub fn openai(api_key: impl Into<String>) -> ClientBuilder<OpenAi> {
        ClientBuilder::new(OpenAiAdapterSettings::new(api_key))
    }

    /// Creates an Anthropic client builder.
    pub fn anthropic(api_key: impl Into<String>) -> ClientBuilder<Anthropic> {
        ClientBuilder::new(AnthropicAdapterSettings::new(api_key))
    }

    /// Creates a Google client builder.
    pub fn google(api_key: impl Into<String>) -> ClientBuilder<Google> {
        ClientBuilder::new(GoogleAdapterSettings::new(api_key))
    }

    /// Creates an OpenAI-compatible client builder.
    pub fn openai_compatible(base_url: impl Into<String>) -> ClientBuilder<OpenAiCompatible> {
        ClientBuilder::new(OpenAiCompatibleAdapterSettings::new(base_url))
    }
}

/// Configures HTTP/runtime behavior before building a [`BoundClient`].
pub struct ClientBuilder<P: ProviderBinding> {
    timeout: Duration,
    max_retries: u8,
    default_max_steps: u8,
    user_agent: String,
    settings: P::Settings,
}

impl<P: ProviderBinding> ClientBuilder<P> {
    fn new(settings: P::Settings) -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            default_max_steps: 8,
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
    pub fn default_max_steps(mut self, max_steps: u8) -> Self {
        self.default_max_steps = max_steps;
        self
    }

    /// Overrides the default SDK `User-Agent` header value.
    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Builds a provider-bound client with validated settings.
    pub fn build(self) -> Result<BoundClient<P>, Error> {
        validate_max_steps(self.default_max_steps)?;
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
            adapter: P::into_adapter(self.settings, http),
            _marker: PhantomData,
        })
    }
}

impl ClientBuilder<OpenAi> {
    /// Overrides the OpenAI API base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }
}

impl ClientBuilder<Anthropic> {
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

impl ClientBuilder<Google> {
    /// Overrides the Google Generative Language API base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }
}

impl ClientBuilder<OpenAiCompatible> {
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
pub struct BoundClient<P: ProviderMarker> {
    max_retries: u8,
    default_max_steps: u8,
    adapter: Arc<dyn ModelAdapter<P>>,
    _marker: PhantomData<P>,
}

impl<P: ProviderMarker> BoundClient<P> {
    #[cfg_attr(
        feature = "telemetry",
        tracing::instrument(
            skip_all,
            fields(model = %req.model.model(), provider = %P::KIND.as_slug())
        )
    )]
    /// Runs a non-streaming generation request.
    ///
    /// The request is validated locally and retried on retryable failures.
    pub async fn generate(
        &self,
        req: GenerateTextRequest<P>,
    ) -> Result<GenerateTextResponse, Error> {
        validate_model_ref(&req.model)?;
        validate_messages(&req.messages)?;
        validate_sampling(req.temperature, req.top_p)?;
        self.call_with_retry(|| async { self.adapter.generate_text(&req).await })
            .await
    }

    #[cfg_attr(
        feature = "telemetry",
        tracing::instrument(
            skip_all,
            fields(model = %req.model.model())
        )
    )]
    /// Runs a streaming generation request.
    ///
    /// The request is validated locally and retried on retryable failures.
    pub async fn stream(&self, req: GenerateTextRequest<P>) -> Result<TextStream, Error> {
        validate_model_ref(&req.model)?;
        validate_messages(&req.messages)?;
        validate_sampling(req.temperature, req.top_p)?;
        self.call_with_retry(|| async { self.adapter.stream_text(&req).await })
            .await
    }

    pub(crate) async fn run_tools(&self, req: RunTools<P>) -> Result<AgentResponse, Error> {
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
            cancellation_token,
        } = req;

        let resolved_max_steps = max_steps.unwrap_or(self.default_max_steps);

        let mut messages = messages;
        let mut usage_total = Usage::default();
        let mut step_results = Vec::new();

        if let Some(callback) = &on_start {
            callback(&AgentStart {
                model_id: model.id(),
                messages: messages.clone(),
                tool_count: tools.len(),
                max_steps: resolved_max_steps,
            });
        }

        for step in 1..=resolved_max_steps {
            if cancellation_token
                .as_ref()
                .map(|t| t.is_cancelled())
                .unwrap_or(false)
            {
                return Err(Error::new(ErrorCode::Cancelled, "agent cancelled"));
            }

            #[cfg(feature = "telemetry")]
            let _step_span = tracing::info_span!("agent_step", step = step).entered();

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

            let tool_registry = ToolRegistry::from_tools(prepared_step.tools.clone())?;
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

        Err(Error::new(
            ErrorCode::MaxStepsExceeded,
            format!(
                "agent reached max_steps ({}) without final answer",
                resolved_max_steps
            ),
        ))
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
                    let delay = backoff_delay(attempt);
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
    Duration::from_millis((base_ms.saturating_mul(exp)).min(cap_ms))
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
    step: u8,
    policy: ToolErrorPolicy,
    on_tool_call_start: Option<&crate::types::Hook<AgentToolCallStart>>,
    on_tool_call_finish: Option<&crate::types::Hook<AgentToolCallFinish>>,
) -> Result<Vec<ExecutedToolCall>, Error> {
    let mut tasks = Vec::with_capacity(calls.len());
    for call in calls {
        let Some(registered) = registry.resolve(&call.tool_name) else {
            return Err(Error::new(
                ErrorCode::UnknownTool,
                format!("unknown tool `{}`", call.tool_name),
            ));
        };

        registered
            .validator
            .validate(&call.args_json)
            .map_err(|e| {
                Error::new(
                    ErrorCode::InvalidToolArgs,
                    format!(
                        "tool args for `{}` failed schema validation: {}",
                        call.tool_name, e
                    ),
                )
            })?;

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
            #[cfg(feature = "telemetry")]
            let _span = tracing::info_span!("tool_call", tool.name = %call.tool_name).entered();
            let started_at = Instant::now();
            let result = executor.execute(args_json).await;
            let duration_ms = started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
            (call, result, duration_ms)
        });
    }

    let results = join_all(tasks).await;
    let mut executions = Vec::with_capacity(results.len());
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
        let client = LlmClient::openai("key")
            .build()
            .expect("client should build");
        let _ = client;
    }

    #[test]
    fn anthropic_builder_builds() {
        let client = LlmClient::anthropic("key")
            .build()
            .expect("client should build");
        let _ = client;
    }
}
