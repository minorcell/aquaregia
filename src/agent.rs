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
//! ```rust,no_run
//! use aquaregia::{Agent, LlmClient, ToolBuilder};
//! use serde_json::{Value, json};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let get_weather = ToolBuilder::new("get_weather")
//!     .description("Get weather by city")
//!     .execute(|city: String| async move {
//!         Ok(json!({ "city": city, "temp_c": 23, "condition": "sunny" }))
//!     });
//!
//! let client = LlmClient::openai().api_key("api-key").build()?;
//!
//! let agent = Agent::builder(client, "gpt-5.5")
//!     .instructions("You can call tools before answering.")
//!     .tools([get_weather])
//!     .max_steps(4)
//!     .build()?;
//!
//! let out = agent.run("What is the weather in Shanghai?").await?;
//! println!("{}", out.output_text);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::client::BoundClient;
use crate::tool::IntoTool;
use crate::types::{
    AgentFinish, AgentPrepareStep, AgentPreparedStep, AgentResponse, AgentStart, AgentStep,
    AgentStepStart, AgentToolCallFinish, AgentToolCallStart, Message, RunTools, ToolErrorPolicy,
    validate_model_ref, validate_sampling,
};

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
    client: Arc<BoundClient>,
    instructions: Option<String>,
    template: RunTools,
}

impl Agent {
    /// Starts building an [`Agent`] from a provider-bound client and model.
    pub fn builder(
        client: impl Into<Arc<BoundClient>>,
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
    pub async fn run(
        &self,
        prompt: impl Into<String>,
    ) -> Result<AgentResponse, crate::error::Error> {
        let messages = vec![Message::user_text(prompt)];
        self.run_messages_inner(self.inject_instructions(messages))
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
    ) -> Result<AgentResponse, crate::error::Error> {
        self.run_messages_inner(self.inject_instructions(messages))
            .await
    }

    async fn run_messages_inner(
        &self,
        messages: Vec<Message>,
    ) -> Result<AgentResponse, crate::error::Error> {
        let mut request = self.template.clone();
        request.messages = messages;
        self.client.run_tools(request.build()?).await
    }
}

/// Builder for configuring an [`Agent`].
pub struct AgentBuilder {
    client: Arc<BoundClient>,
    instructions: Option<String>,
    template: RunTools,
}

impl AgentBuilder {
    fn new(client: Arc<BoundClient>, model: String) -> Self {
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

    /// Registers tools available to the model.
    pub fn tools<I, T>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: IntoTool,
    {
        self.template = self.template.tools(tools);
        self
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

#[cfg(test)]
mod tests {
    use super::Agent;
    use crate::LlmClient;

    #[test]
    fn builder_accepts_typed_model() {
        let client = LlmClient::openai()
            .api_key("test-key")
            .base_url("https://api.openai.com")
            .build()
            .expect("client should build");
        let agent = Agent::builder(client, "gpt-5.4-mini")
            .max_steps(3)
            .build()
            .expect("agent should build");

        assert_eq!(agent.model_id(), "gpt-5.4-mini");
    }

    #[test]
    fn builder_rejects_invalid_top_p() {
        let client = LlmClient::openai()
            .api_key("test-key")
            .base_url("https://api.openai.com")
            .build()
            .expect("client should build");
        let err = match Agent::builder(client, "gpt-5.4-mini").top_p(1.5).build() {
            Ok(_) => panic!("agent build should fail"),
            Err(err) => err,
        };

        assert!(err.message.contains("top_p"));
    }
}
