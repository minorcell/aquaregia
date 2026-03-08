use std::sync::Arc;

use crate::client::BoundClient;
use crate::tool::{IntoTool, Tool};
use crate::types::{
    AgentFinish, AgentPrepareStep, AgentPreparedStep, AgentResponse, AgentStart, AgentStep,
    AgentStepStart, AgentToolCallFinish, AgentToolCallStart, FinishCallback, IntoModelRef,
    Message, ModelRef, PrepareStepCallback, ProviderMarker, RunTools, StartCallback, StepCallback,
    StepStartCallback, StopWhen, ToolCallFinishCallback, ToolCallStartCallback, ToolErrorPolicy,
    validate_max_steps, validate_model_ref, validate_sampling,
};

#[derive(Debug, Clone)]
pub struct AgentCallPlan<P: ProviderMarker> {
    pub model: ModelRef<P>,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub max_steps: Option<u8>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub stop_sequences: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct PrepareCallCallback<P: ProviderMarker> {
    inner: Arc<dyn Fn(&mut AgentCallPlan<P>) + Send + Sync>,
}

impl<P: ProviderMarker> std::fmt::Debug for PrepareCallCallback<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PrepareCallCallback(<fn>)")
    }
}

impl<P: ProviderMarker> PrepareCallCallback<P> {
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(&mut AgentCallPlan<P>) + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(callback),
        }
    }

    pub fn call(&self, event: &mut AgentCallPlan<P>) {
        (self.inner)(event)
    }
}

pub struct Agent<P: ProviderMarker> {
    client: Arc<BoundClient<P>>,
    model: ModelRef<P>,
    instructions: Option<String>,
    tools: Vec<Tool>,
    max_steps: Option<u8>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    stop_sequences: Vec<String>,
    prepare_call: Option<PrepareCallCallback<P>>,
    prepare_step: Option<PrepareStepCallback<P>>,
    on_start: Option<StartCallback<P>>,
    on_step_start: Option<StepStartCallback>,
    on_tool_call_start: Option<ToolCallStartCallback>,
    on_tool_call_finish: Option<ToolCallFinishCallback>,
    on_step_finish: Option<StepCallback>,
    on_finish: Option<FinishCallback>,
    stop_when: Option<StopWhen>,
    tool_error_policy: ToolErrorPolicy,
}

impl<P: ProviderMarker> Agent<P> {
    pub fn builder(client: impl Into<Arc<BoundClient<P>>>, model: impl IntoModelRef<P>) -> AgentBuilder<P> {
        AgentBuilder::new(client.into(), model.into_model_ref())
    }

    pub fn model_id(&self) -> String {
        self.model.id()
    }

    pub async fn run(
        &self,
        prompt: impl Into<String>,
    ) -> Result<AgentResponse, crate::error::AiError> {
        let mut messages = Vec::new();
        if let Some(instructions) = &self.instructions {
            messages.push(Message::system_text(instructions.clone()));
        }
        messages.push(Message::user_text(prompt));
        self.run_messages(messages).await
    }

    pub async fn run_messages(
        &self,
        messages: Vec<Message>,
    ) -> Result<AgentResponse, crate::error::AiError> {
        let mut call_plan = AgentCallPlan {
            model: self.model.clone(),
            messages,
            tools: self.tools.clone(),
            max_steps: self.max_steps,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            stop_sequences: self.stop_sequences.clone(),
        };
        if let Some(callback) = &self.prepare_call {
            callback.call(&mut call_plan);
        }

        let mut request = RunTools::new(call_plan.model)
            .messages(call_plan.messages)
            .tools(call_plan.tools)
            .tool_error_policy(self.tool_error_policy)
            .stop_sequences(call_plan.stop_sequences);

        if let Some(max_steps) = call_plan.max_steps {
            request = request.max_steps(max_steps);
        }
        if let Some(temperature) = call_plan.temperature {
            request = request.temperature(temperature);
        }
        if let Some(max_output_tokens) = call_plan.max_output_tokens {
            request = request.max_output_tokens(max_output_tokens);
        }

        if let Some(prepare_step) = &self.prepare_step {
            let prepare_step = prepare_step.clone();
            request = request.prepare_step(move |event| prepare_step.call(event));
        }
        if let Some(on_start) = &self.on_start {
            let on_start = on_start.clone();
            request = request.on_start(move |event: &AgentStart<P>| on_start.call(event));
        }
        if let Some(on_step_start) = &self.on_step_start {
            let on_step_start = on_step_start.clone();
            request = request.on_step_start(move |event: &AgentStepStart| on_step_start.call(event));
        }
        if let Some(on_tool_call_start) = &self.on_tool_call_start {
            let on_tool_call_start = on_tool_call_start.clone();
            request = request.on_tool_call_start(move |event: &AgentToolCallStart| on_tool_call_start.call(event));
        }
        if let Some(on_tool_call_finish) = &self.on_tool_call_finish {
            let on_tool_call_finish = on_tool_call_finish.clone();
            request = request.on_tool_call_finish(move |event: &AgentToolCallFinish| on_tool_call_finish.call(event));
        }
        if let Some(on_step_finish) = &self.on_step_finish {
            let on_step_finish = on_step_finish.clone();
            request = request.on_step_finish(move |event: &AgentStep| on_step_finish.call(event));
        }
        if let Some(on_finish) = &self.on_finish {
            let on_finish = on_finish.clone();
            request = request.on_finish(move |event: &AgentFinish| on_finish.call(event));
        }
        if let Some(stop_when) = &self.stop_when {
            let stop_when = stop_when.clone();
            request = request.stop_when(move |step: &AgentStep| stop_when.should_stop(step));
        }

        self.client.run_tools(request.build()?).await
    }
}

pub struct AgentBuilder<P: ProviderMarker> {
    client: Arc<BoundClient<P>>,
    model: ModelRef<P>,
    instructions: Option<String>,
    tools: Vec<Tool>,
    max_steps: Option<u8>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    stop_sequences: Vec<String>,
    prepare_call: Option<PrepareCallCallback<P>>,
    prepare_step: Option<PrepareStepCallback<P>>,
    on_start: Option<StartCallback<P>>,
    on_step_start: Option<StepStartCallback>,
    on_tool_call_start: Option<ToolCallStartCallback>,
    on_tool_call_finish: Option<ToolCallFinishCallback>,
    on_step_finish: Option<StepCallback>,
    on_finish: Option<FinishCallback>,
    stop_when: Option<StopWhen>,
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

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn tools<I, T>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: IntoTool,
    {
        self.tools
            .extend(tools.into_iter().map(IntoTool::into_tool));
        self
    }

    pub fn max_steps(mut self, max_steps: u8) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn stop_sequences(mut self, stop_sequences: impl IntoIterator<Item = String>) -> Self {
        self.stop_sequences.extend(stop_sequences);
        self
    }

    pub fn prepare_call<F>(mut self, callback: F) -> Self
    where
        F: Fn(&mut AgentCallPlan<P>) + Send + Sync + 'static,
    {
        self.prepare_call = Some(PrepareCallCallback::new(callback));
        self
    }

    pub fn prepare_step<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentPrepareStep<P>) -> AgentPreparedStep<P> + Send + Sync + 'static,
    {
        self.prepare_step = Some(PrepareStepCallback::new(callback));
        self
    }

    pub fn on_step_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStep) + Send + Sync + 'static,
    {
        self.on_step_finish = Some(StepCallback::new(callback));
        self
    }

    pub fn on_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStart<P>) + Send + Sync + 'static,
    {
        self.on_start = Some(StartCallback::new(callback));
        self
    }

    pub fn on_step_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentStepStart) + Send + Sync + 'static,
    {
        self.on_step_start = Some(StepStartCallback::new(callback));
        self
    }

    pub fn on_tool_call_start<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallStart) + Send + Sync + 'static,
    {
        self.on_tool_call_start = Some(ToolCallStartCallback::new(callback));
        self
    }

    pub fn on_tool_call_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentToolCallFinish) + Send + Sync + 'static,
    {
        self.on_tool_call_finish = Some(ToolCallFinishCallback::new(callback));
        self
    }

    pub fn on_finish<F>(mut self, callback: F) -> Self
    where
        F: Fn(&AgentFinish) + Send + Sync + 'static,
    {
        self.on_finish = Some(FinishCallback::new(callback));
        self
    }

    pub fn stop_when<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&AgentStep) -> bool + Send + Sync + 'static,
    {
        self.stop_when = Some(StopWhen::new(predicate));
        self
    }

    pub fn tool_error_policy(mut self, policy: ToolErrorPolicy) -> Self {
        self.tool_error_policy = policy;
        self
    }

    pub fn build(self) -> Result<Agent<P>, crate::error::AiError> {
        validate_model_ref(&self.model)?;
        if let Some(max_steps) = self.max_steps {
            validate_max_steps(max_steps)?;
        }
        validate_sampling(self.temperature, None)?;

        Ok(Agent {
            client: self.client,
            model: self.model,
            instructions: self.instructions,
            tools: self.tools,
            max_steps: self.max_steps,
            temperature: self.temperature,
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
    use crate::{LlmClient, openai_model};

    #[test]
    fn builder_accepts_typed_model() {
        let client = LlmClient::openai("test-key")
            .base_url("https://api.openai.com")
            .build()
            .expect("client should build");
        let agent = Agent::builder(client, openai_model("gpt-4o-mini"))
            .max_steps(3)
            .build()
            .expect("agent should build");

        assert_eq!(agent.model_id(), "openai/gpt-4o-mini");
    }
}
