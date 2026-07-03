//! Agent runtime and builder APIs for Aquaregia.
//!
//! This module provides the multi-step tool-using agent abstraction:
//!
//! - [`Agent`]: Main agent runtime for tool loops
//! - [`AgentBuilder`]: Builder for configuring agent behavior
//!
//! ## Agent Architecture
//!
//! The agent implements a tool-use loop:
//! 1. Send messages to the LLM with available tools
//! 2. If the model requests tool calls, execute them
//! 3. Send tool results back to the model
//! 4. Repeat until the model produces a final answer
//!
//! ## Example
//!
//! ```rust,ignore
//! use aquaregia::{providers::openai, tool};
//! use serde_json::json;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let get_weather = tool("get_weather")
//!     .description("Get weather by city")
//!     .execute(|city: String| async move {
//!         json!({ "city": city, "temp_c": 23, "condition": "sunny" })
//!     });
//!
//! let agent = openai::Client::builder()
//!     .api_key("api-key")
//!     .build()?
//!     .agent("gpt-5.5")
//!     .instructions("You can call tools before answering.")
//!     .tool(get_weather)
//!     .max_steps(4)
//!     .build()?;
//!
//! let out = agent.prompt("What is the weather in Shanghai?").await?;
//! println!("{out}");
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use serde_json::Value;

use crate::client::Client;
use crate::tool::IntoTool;
use crate::types::{
    AgentFinish, AgentOutput, AgentPrepareStep, AgentPreparedStep, AgentStart, AgentStep,
    AgentStepStart, AgentStream, AgentToolCallFinish, AgentToolCallStart, Message, RunTools,
    ToolErrorPolicy, validate_model_ref, validate_sampling,
};

mod sealed {
    pub trait Sealed {}

    impl<I, T> Sealed for I
    where
        I: IntoIterator<Item = T>,
        T: crate::tool::IntoTool,
    {
    }

    #[cfg(feature = "mcp")]
    impl Sealed for crate::mcp::McpConnection {}
}

/// Values that can be registered through [`AgentBuilder::tools`].
///
/// Normal tool collections are registered as static tools. With the `mcp`
/// feature enabled, `McpConnection` is registered as a live MCP tool source
/// that updates when the server sends
/// `notifications/tools/list_changed`.
pub trait IntoAgentTools: sealed::Sealed {
    #[doc(hidden)]
    fn __apply_to_agent_builder(self, builder: AgentBuilder) -> AgentBuilder;
}

impl<I, T> IntoAgentTools for I
where
    I: IntoIterator<Item = T>,
    T: IntoTool,
{
    fn __apply_to_agent_builder(self, mut builder: AgentBuilder) -> AgentBuilder {
        builder.template = builder.template.tools(self);
        builder
    }
}

#[cfg(feature = "mcp")]
impl IntoAgentTools for crate::mcp::McpConnection {
    fn __apply_to_agent_builder(self, mut builder: AgentBuilder) -> AgentBuilder {
        builder.template = builder.template.tool_source(Arc::new(self));
        builder
    }
}

/// Multi-step tool-using agent bound to one provider and one default model.
///
/// The agent implements a tool-use loop that:
/// 1. Sends messages to the LLM with available tools
/// 2. Executes tool calls requested by the model
/// 3. Sends tool results back to the model
/// 4. Repeats until the model produces a final answer or max_steps is reached
///
/// ## Features
///
/// - **Configurable hooks**: Callbacks for run start, step start/finish, tool call start/finish
/// - **Dynamic planning**: `prepare_step` callback for runtime per-step adjustments
/// - **Early stopping**: `stop_when` predicate for custom termination conditions
/// - **Cancellation**: Bind a `CancellationToken` at builder time to cancel running agents
/// - **Error policies**: Configurable tool error handling (`ContinueAsToolResult` or `FailFast`)
pub struct Agent {
    client: Arc<Client>,
    instructions: Option<String>,
    template: RunTools,
}

impl Agent {
    /// Starts building an [`Agent`] from an internal provider-bound client and model.
    pub(crate) fn builder(
        client: impl Into<Arc<Client>>,
        model: impl Into<String>,
    ) -> AgentBuilder {
        AgentBuilder::new(client.into(), model.into())
    }

    /// Returns the fully qualified model id (`<provider>/<model>`).
    pub fn model_id(&self) -> String {
        self.template.model.clone()
    }

    /// Prepends instructions as a system message if configured and no system
    /// message already exists in the message list.
    fn inject_instructions(&self, mut messages: Vec<Message>) -> Vec<Message> {
        let has_system = messages
            .first()
            .map(|m| m.role() == crate::types::MessageRole::System)
            .unwrap_or(false);
        if !has_system && let Some(instructions) = &self.instructions {
            messages.insert(0, Message::system_text(instructions.clone()));
        }
        messages
    }

    /// Runs the agent with a single user prompt.
    ///
    /// If `instructions` were configured, they are inserted as an initial system message.
    pub async fn run(&self, prompt: impl Into<String>) -> Result<AgentOutput, crate::error::Error> {
        let messages = vec![Message::user_text(prompt)];
        self.run_messages_inner(self.inject_instructions(messages))
            .await
    }

    /// Prompts the agent and returns only the visible output text.
    ///
    /// Use [`Agent::run`] when you need run metadata such as steps, transcript,
    /// tool results, or token usage.
    pub async fn prompt(&self, prompt: impl Into<String>) -> Result<String, crate::error::Error> {
        self.run(prompt).await.map(|output| output.output_text)
    }

    /// Streams the full agent execution for a single user prompt.
    ///
    /// The returned stream includes model deltas, tool execution events, step
    /// snapshots, and the final [`AgentOutput`].
    pub async fn stream(
        &self,
        prompt: impl Into<String>,
    ) -> Result<AgentStream, crate::error::Error> {
        let messages = vec![Message::user_text(prompt)];
        self.stream_messages_inner(self.inject_instructions(messages))
            .await
    }

    /// Runs the agent with an explicit message list.
    ///
    /// If `instructions` were configured and the message list does not already
    /// contain a system message, the instructions are inserted as a system
    /// message at the front of the list.
    pub async fn run_messages(
        &self,
        messages: Vec<Message>,
    ) -> Result<AgentOutput, crate::error::Error> {
        self.run_messages_inner(self.inject_instructions(messages))
            .await
    }

    /// Streams the full agent execution for an explicit message list.
    ///
    /// If `instructions` were configured and the message list does not already
    /// contain a system message, the instructions are inserted as a system
    /// message at the front of the list.
    pub async fn stream_messages(
        &self,
        messages: Vec<Message>,
    ) -> Result<AgentStream, crate::error::Error> {
        self.stream_messages_inner(self.inject_instructions(messages))
            .await
    }

    async fn run_messages_inner(
        &self,
        messages: Vec<Message>,
    ) -> Result<AgentOutput, crate::error::Error> {
        let mut request = self.template.clone();
        request.messages = messages;
        self.client.run_tools(request.build()?).await
    }

    async fn stream_messages_inner(
        &self,
        messages: Vec<Message>,
    ) -> Result<AgentStream, crate::error::Error> {
        let mut request = self.template.clone();
        request.messages = messages;
        Arc::clone(&self.client).stream_tools(request.build()?)
    }
}

/// Builder for configuring an [`Agent`].
pub struct AgentBuilder {
    client: Arc<Client>,
    instructions: Option<String>,
    template: RunTools,
}

impl AgentBuilder {
    pub(crate) fn new(client: Arc<Client>, model: String) -> Self {
        Self {
            client,
            instructions: None,
            template: RunTools::new(model),
        }
    }

    /// Sets default system instructions prepended for prompt-based runs.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Registers one tool available to the model.
    pub fn tool<T>(mut self, tool: T) -> Self
    where
        T: IntoTool,
    {
        self.template = self.template.tool(tool);
        self
    }

    /// Registers tools available to the model.
    ///
    /// Pass a collection of local tools for a static snapshot, or an MCP
    /// connection for a live tool set when the `mcp` feature is enabled.
    pub fn tools<T>(self, tools: T) -> Self
    where
        T: IntoAgentTools,
    {
        tools.__apply_to_agent_builder(self)
    }

    /// Sets the max number of agent loop steps.
    ///
    /// `0` means unlimited (the loop continues until the model returns a final
    /// answer, an explicit `stop_when` predicate matches, or the run is
    /// cancelled). When not set, falls back to the client's `default_max_steps`
    /// (which is `0` / unlimited by default).
    pub fn max_steps(mut self, max_steps: u32) -> Self {
        self.template = self.template.max_steps(max_steps);
        self
    }

    /// Sets default sampling temperature in range `0.0..=2.0`.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.template = self.template.temperature(temperature);
        self
    }

    /// Sets default nucleus sampling value in range `0.0..=1.0`.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.template = self.template.top_p(top_p);
        self
    }

    /// Sets default maximum output token budget per step.
    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.template = self.template.max_output_tokens(max_output_tokens);
        self
    }

    /// Appends default stop sequences for each model call.
    pub fn stop_sequences<S: Into<String>>(
        mut self,
        stop_sequences: impl IntoIterator<Item = S>,
    ) -> Self {
        self.template = self.template.stop_sequences(stop_sequences);
        self
    }

    /// Binds a [`CancellationToken`] checked during agent execution.
    ///
    /// When the token is cancelled, the agent stops before the next step and returns
    /// [`crate::ErrorCode::Cancelled`]. To cancel different runs independently, build
    /// a separate agent per token.
    pub fn cancellation_token(mut self, token: CancellationToken) -> Self {
        self.template = self.template.cancellation_token(token);
        self
    }

    /// Registers a callback to mutate per-step inputs right before each model call.
    pub fn prepare_step<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentPrepareStep) -> AgentPreparedStep + Send + Sync + 'static,
    {
        self.template = self.template.prepare_step(callback);
        self
    }

    /// Registers a callback fired after each completed step.
    pub fn on_step_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStep) + Send + Sync + 'static,
    {
        self.template = self.template.on_step_finish(callback);
        self
    }

    /// Registers a callback fired once before step 1 starts.
    pub fn on_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStart) + Send + Sync + 'static,
    {
        self.template = self.template.on_start(callback);
        self
    }

    /// Registers a callback fired at the beginning of each step.
    pub fn on_step_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStepStart) + Send + Sync + 'static,
    {
        self.template = self.template.on_step_start(callback);
        self
    }

    /// Registers a callback fired right before each tool execution.
    pub fn on_tool_call_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallStart) + Send + Sync + 'static,
    {
        self.template = self.template.on_tool_call_start(callback);
        self
    }

    /// Registers a callback fired right after each tool execution.
    pub fn on_tool_call_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallFinish) + Send + Sync + 'static,
    {
        self.template = self.template.on_tool_call_finish(callback);
        self
    }

    /// Registers a callback fired once when the run finishes successfully.
    pub fn on_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentFinish) + Send + Sync + 'static,
    {
        self.template = self.template.on_finish(callback);
        self
    }

    /// Registers an early-stop predicate evaluated after each completed step.
    pub fn stop_when<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&AgentStep) -> bool + Send + Sync + 'static,
    {
        self.template = self.template.stop_when(predicate);
        self
    }

    /// Controls how tool execution errors are handled inside the loop.
    pub fn tool_error_policy(mut self, policy: ToolErrorPolicy) -> Self {
        self.template = self.template.tool_error_policy(policy);
        self
    }

    /// Sets provider-specific options passed through on every step.
    ///
    /// Same shape as [`ChatRequestBuilder::provider_options`]: a JSON
    /// object keyed by provider slug (e.g. `"anthropic"`, `"openai"`). Each
    /// adapter extracts its own block and merges it into the request payload
    /// for every step in the loop.
    ///
    /// [`ChatRequestBuilder::provider_options`]: crate::ChatRequest::builder
    pub fn provider_options(mut self, options: Value) -> Self {
        self.template = self.template.provider_options(options);
        self
    }

    /// Validates configuration and builds the [`Agent`].
    pub fn build(self) -> Result<Agent, crate::error::Error> {
        validate_model_ref(&self.template.model)?;
        validate_sampling(self.template.temperature, self.template.top_p)?;

        Ok(Agent {
            client: self.client,
            instructions: self.instructions,
            template: self.template,
        })
    }
}

#[cfg(all(test, feature = "openai"))]
mod tests {
    use crate::providers;
    use serde_json::json;

    #[test]
    fn builder_accepts_provider_options() {
        let client = providers::openai::Client::builder()
            .api_key("test-key")
            .base_url("https://api.openai.com")
            .build()
            .expect("client should build");
        let options = json!({ "anthropic": { "thinking": { "budget_tokens": 1024 } } });
        let agent = client
            .agent("claude-sonnet-4-6")
            .provider_options(options.clone())
            .build()
            .expect("agent should build");

        assert_eq!(agent.template.provider_options.as_ref(), Some(&options));
    }

    #[test]
    fn builder_accepts_typed_model() {
        let client = providers::openai::Client::builder()
            .api_key("test-key")
            .base_url("https://api.openai.com")
            .build()
            .expect("client should build");
        let agent = client
            .agent("gpt-5.4-mini")
            .max_steps(3)
            .build()
            .expect("agent should build");

        assert_eq!(agent.model_id(), "gpt-5.4-mini");
    }

    #[test]
    fn builder_rejects_invalid_top_p() {
        let client = providers::openai::Client::builder()
            .api_key("test-key")
            .base_url("https://api.openai.com")
            .build()
            .expect("client should build");
        let err = match client.agent("gpt-5.4-mini").top_p(1.5).build() {
            Ok(_) => panic!("agent build should fail"),
            Err(err) => err,
        };

        assert!(err.message.contains("top_p"));
    }
}
