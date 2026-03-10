use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::client::BoundClient;
use crate::tool::{IntoTool, Tool};
use crate::types::{
    AgentFinish, AgentPrepareStep, AgentPreparedStep, AgentResponse, AgentStart, AgentStep,
    AgentStepStart, AgentToolCallFinish, AgentToolCallStart, Hook, IntoModelRef, Message, ModelRef,
    PrepareStepHook, ProviderMarker, RunTools, StopPredicate, ToolErrorPolicy, validate_max_steps,
    validate_model_ref, validate_sampling,
};

pub(crate) type PrepareCallHook<P> = Arc<dyn Fn(&mut AgentRunPlan<P>) + Send + Sync>;

/// Mutable plan for one agent run before execution starts.
///
/// This is passed to `prepare_call` so callers can adjust request-level settings
/// (model, messages, tools, sampling, limits) per invocation.
#[derive(Debug, Clone)]
pub struct AgentRunPlan<P: ProviderMarker> {
    /// Model selected for this run.
    pub model: ModelRef<P>,
    /// Initial message list to start the loop with.
    pub messages: Vec<Message>,
    /// Tools available to the model.
    pub tools: Vec<Tool>,
    /// Step cap for this run. `None` falls back to client default.
    pub max_steps: Option<u8>,
    /// Sampling temperature in range `0.0..=2.0`.
    pub temperature: Option<f32>,
    /// Nucleus sampling value in range `0.0..=1.0`.
    pub top_p: Option<f32>,
    /// Maximum output token budget per model call.
    pub max_output_tokens: Option<u32>,
    /// Stop sequences forwarded to provider requests.
    pub stop_sequences: Vec<String>,
}

/// Multi-step tool-using agent bound to one provider and one default model.
pub struct Agent<P: ProviderMarker> {
    client: Arc<BoundClient<P>>,
    model: ModelRef<P>,
    instructions: Option<String>,
    tools: Vec<Tool>,
    max_steps: Option<u8>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_output_tokens: Option<u32>,
    stop_sequences: Vec<String>,
    prepare_call: Option<PrepareCallHook<P>>,
    prepare_step: Option<PrepareStepHook<P>>,
    on_start: Option<Hook<AgentStart>>,
    on_step_start: Option<Hook<AgentStepStart>>,
    on_tool_call_start: Option<Hook<AgentToolCallStart>>,
    on_tool_call_finish: Option<Hook<AgentToolCallFinish>>,
    on_step_finish: Option<Hook<AgentStep>>,
    on_finish: Option<Hook<AgentFinish>>,
    stop_when: Option<StopPredicate>,
    tool_error_policy: ToolErrorPolicy,
}

impl<P: ProviderMarker> Agent<P> {
    /// Starts building an [`Agent`] from a provider-bound client and model.
    pub fn builder(
        client: impl Into<Arc<BoundClient<P>>>,
        model: impl IntoModelRef<P>,
    ) -> AgentBuilder<P> {
        AgentBuilder::new(client.into(), model.into_model_ref())
    }

    /// Returns the fully qualified model id (`<provider>/<model>`).
    pub fn model_id(&self) -> String {
        self.model.id()
    }

    /// Runs the agent with a single user prompt.
    ///
    /// If `instructions` were configured, they are inserted as an initial system message.
    pub async fn run(
        &self,
        prompt: impl Into<String>,
    ) -> Result<AgentResponse, crate::error::Error> {
        let mut messages = Vec::new();
        if let Some(instructions) = &self.instructions {
            messages.push(Message::system_text(instructions.clone()));
        }
        messages.push(Message::user_text(prompt));
        self.run_messages_inner(messages, None).await
    }

    /// Runs the agent with an explicit message list.
    pub async fn run_messages(
        &self,
        messages: Vec<Message>,
    ) -> Result<AgentResponse, crate::error::Error> {
        self.run_messages_inner(messages, None).await
    }

    /// Runs the agent with a prompt and cancellation support.
    pub async fn run_cancellable(
        &self,
        prompt: impl Into<String>,
        token: CancellationToken,
    ) -> Result<AgentResponse, crate::error::Error> {
        let mut messages = Vec::new();
        if let Some(instructions) = &self.instructions {
            messages.push(Message::system_text(instructions.clone()));
        }
        messages.push(Message::user_text(prompt));
        self.run_messages_cancellable(messages, token).await
    }

    /// Runs the agent with explicit messages and cancellation support.
    pub async fn run_messages_cancellable(
        &self,
        messages: Vec<Message>,
        token: CancellationToken,
    ) -> Result<AgentResponse, crate::error::Error> {
        self.run_messages_inner(messages, Some(token)).await
    }

    async fn run_messages_inner(
        &self,
        messages: Vec<Message>,
        token: Option<CancellationToken>,
    ) -> Result<AgentResponse, crate::error::Error> {
        let mut call_plan = AgentRunPlan {
            model: self.model.clone(),
            messages,
            tools: self.tools.clone(),
            max_steps: self.max_steps,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_output_tokens,
            stop_sequences: self.stop_sequences.clone(),
        };
        if let Some(callback) = &self.prepare_call {
            callback(&mut call_plan);
        }

        let mut request = RunTools::new(call_plan.model)
            .messages(call_plan.messages)
            .tools(call_plan.tools)
            .tool_error_policy(self.tool_error_policy)
            .stop_sequences(call_plan.stop_sequences);

        if let Some(t) = token {
            request = request.cancellation_token(t);
        }
        if let Some(max_steps) = call_plan.max_steps {
            request = request.max_steps(max_steps);
        }
        if let Some(temperature) = call_plan.temperature {
            request = request.temperature(temperature);
        }
        if let Some(top_p) = call_plan.top_p {
            request = request.top_p(top_p);
        }
        if let Some(max_output_tokens) = call_plan.max_output_tokens {
            request = request.max_output_tokens(max_output_tokens);
        }

        if let Some(prepare_step) = &self.prepare_step {
            let prepare_step = prepare_step.clone();
            request = request.prepare_step(move |event| prepare_step(event));
        }
        if let Some(on_start) = &self.on_start {
            let on_start = on_start.clone();
            request = request.on_start(move |event: &AgentStart| on_start(event));
        }
        if let Some(on_step_start) = &self.on_step_start {
            let on_step_start = on_step_start.clone();
            request = request.on_step_start(move |event: &AgentStepStart| on_step_start(event));
        }
        if let Some(on_tool_call_start) = &self.on_tool_call_start {
            let on_tool_call_start = on_tool_call_start.clone();
            request = request
                .on_tool_call_start(move |event: &AgentToolCallStart| on_tool_call_start(event));
        }
        if let Some(on_tool_call_finish) = &self.on_tool_call_finish {
            let on_tool_call_finish = on_tool_call_finish.clone();
            request = request
                .on_tool_call_finish(move |event: &AgentToolCallFinish| on_tool_call_finish(event));
        }
        if let Some(on_step_finish) = &self.on_step_finish {
            let on_step_finish = on_step_finish.clone();
            request = request.on_step_finish(move |event: &AgentStep| on_step_finish(event));
        }
        if let Some(on_finish) = &self.on_finish {
            let on_finish = on_finish.clone();
            request = request.on_finish(move |event: &AgentFinish| on_finish(event));
        }
        if let Some(stop_when) = &self.stop_when {
            let stop_when = stop_when.clone();
            request = request.stop_when(move |step: &AgentStep| stop_when(step));
        }

        self.client.run_tools(request.build()?).await
    }
}

/// Builder for configuring an [`Agent`].
pub struct AgentBuilder<P: ProviderMarker> {
    client: Arc<BoundClient<P>>,
    model: ModelRef<P>,
    instructions: Option<String>,
    tools: Vec<Tool>,
    max_steps: Option<u8>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_output_tokens: Option<u32>,
    stop_sequences: Vec<String>,
    prepare_call: Option<PrepareCallHook<P>>,
    prepare_step: Option<PrepareStepHook<P>>,
    on_start: Option<Hook<AgentStart>>,
    on_step_start: Option<Hook<AgentStepStart>>,
    on_tool_call_start: Option<Hook<AgentToolCallStart>>,
    on_tool_call_finish: Option<Hook<AgentToolCallFinish>>,
    on_step_finish: Option<Hook<AgentStep>>,
    on_finish: Option<Hook<AgentFinish>>,
    stop_when: Option<StopPredicate>,
    tool_error_policy: ToolErrorPolicy,
}

impl<P: ProviderMarker> AgentBuilder<P> {
    fn new(client: Arc<BoundClient<P>>, model: ModelRef<P>) -> Self {
        Self {
            client,
            model,
            instructions: None,
            tools: Vec::new(),
            max_steps: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stop_sequences: Vec::new(),
            prepare_call: None,
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
        self.tools
            .extend(tools.into_iter().map(IntoTool::into_tool));
        self
    }

    /// Sets the max number of agent loop steps.
    pub fn max_steps(mut self, max_steps: u8) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Sets default sampling temperature in range `0.0..=2.0`.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets default nucleus sampling value in range `0.0..=1.0`.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets default maximum output token budget per step.
    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    /// Appends default stop sequences for each model call.
    pub fn stop_sequences<S: Into<String>>(
        mut self,
        stop_sequences: impl IntoIterator<Item = S>,
    ) -> Self {
        self.stop_sequences
            .extend(stop_sequences.into_iter().map(Into::into));
        self
    }

    /// Registers a callback to mutate [`AgentRunPlan`] before execution starts.
    pub fn prepare_call<F>(mut self, callback: F) -> Self
    where
        F: Fn(&mut AgentRunPlan<P>) + Send + Sync + 'static,
    {
        self.prepare_call = Some(Arc::new(callback));
        self
    }

    /// Registers a callback to mutate per-step inputs right before each model call.
    pub fn prepare_step<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentPrepareStep<P>) -> AgentPreparedStep<P> + Send + Sync + 'static,
    {
        self.prepare_step = Some(Arc::new(callback));
        self
    }

    /// Registers a callback fired after each completed step.
    pub fn on_step_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStep) + Send + Sync + 'static,
    {
        self.on_step_finish = Some(Arc::new(callback));
        self
    }

    /// Registers a callback fired once before step 1 starts.
    pub fn on_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStart) + Send + Sync + 'static,
    {
        self.on_start = Some(Arc::new(callback));
        self
    }

    /// Registers a callback fired at the beginning of each step.
    pub fn on_step_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStepStart) + Send + Sync + 'static,
    {
        self.on_step_start = Some(Arc::new(callback));
        self
    }

    /// Registers a callback fired right before each tool execution.
    pub fn on_tool_call_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallStart) + Send + Sync + 'static,
    {
        self.on_tool_call_start = Some(Arc::new(callback));
        self
    }

    /// Registers a callback fired right after each tool execution.
    pub fn on_tool_call_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallFinish) + Send + Sync + 'static,
    {
        self.on_tool_call_finish = Some(Arc::new(callback));
        self
    }

    /// Registers a callback fired once when the run finishes successfully.
    pub fn on_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentFinish) + Send + Sync + 'static,
    {
        self.on_finish = Some(Arc::new(callback));
        self
    }

    /// Registers an early-stop predicate evaluated after each completed step.
    pub fn stop_when<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&AgentStep) -> bool + Send + Sync + 'static,
    {
        self.stop_when = Some(Arc::new(predicate));
        self
    }

    /// Controls how tool execution errors are handled inside the loop.
    pub fn tool_error_policy(mut self, policy: ToolErrorPolicy) -> Self {
        self.tool_error_policy = policy;
        self
    }

    /// Validates configuration and builds the [`Agent`].
    pub fn build(self) -> Result<Agent<P>, crate::error::Error> {
        validate_model_ref(&self.model)?;
        if let Some(max_steps) = self.max_steps {
            validate_max_steps(max_steps)?;
        }
        validate_sampling(self.temperature, self.top_p)?;

        Ok(Agent {
            client: self.client,
            model: self.model,
            instructions: self.instructions,
            tools: self.tools,
            max_steps: self.max_steps,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_output_tokens,
            stop_sequences: self.stop_sequences,
            prepare_call: self.prepare_call,
            prepare_step: self.prepare_step,
            on_start: self.on_start,
            on_step_start: self.on_step_start,
            on_tool_call_start: self.on_tool_call_start,
            on_tool_call_finish: self.on_tool_call_finish,
            on_step_finish: self.on_step_finish,
            on_finish: self.on_finish,
            stop_when: self.stop_when,
            tool_error_policy: self.tool_error_policy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Agent;
    use crate::{LlmClient, openai};

    #[test]
    fn builder_accepts_typed_model() {
        let client = LlmClient::openai("test-key")
            .base_url("https://api.openai.com")
            .build()
            .expect("client should build");
        let agent = Agent::builder(client, openai("gpt-4o-mini"))
            .max_steps(3)
            .build()
            .expect("agent should build");

        assert_eq!(agent.model_id(), "openai/gpt-4o-mini");
    }

    #[test]
    fn builder_rejects_invalid_top_p() {
        let client = LlmClient::openai("test-key")
            .base_url("https://api.openai.com")
            .build()
            .expect("client should build");
        let err = match Agent::builder(client, openai("gpt-4o-mini"))
            .top_p(1.5)
            .build()
        {
            Ok(_) => panic!("agent build should fail"),
            Err(err) => err,
        };

        assert!(err.message.contains("top_p"));
    }
}
