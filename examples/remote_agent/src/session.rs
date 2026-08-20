use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;

use aquaregia::{Agent, AgentOutput, AgentStreamEvent, Message};
use futures_util::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::{Mutex, broadcast};
use uuid::Uuid;

use crate::mcp_bridge::{McpBridge, ToolInfo};
use crate::store::{RunRecord, SessionRecord, Store, StoreError, StoredEvent};
use crate::telemetry;

#[derive(Clone)]
pub struct SessionManager {
    store: Arc<Store>,
    provider: Arc<aquaregia::providers::openai_compatible::Client>,
    mcp: McpBridge,
    default_model: String,
    default_instructions: String,
    active_runs: Arc<Mutex<HashSet<String>>>,
}

pub enum RunMode {
    Stream,
    Blocking,
}

pub enum StartRunResult {
    Stream {
        stream: Pin<Box<dyn Stream<Item = SessionRunEvent> + Send>>,
    },
    Blocking {
        run: RunRecord,
    },
}

#[derive(Debug, Clone)]
pub enum SessionRunEvent {
    Accepted { run_id: String, session_id: String },
    Agent(AgentStreamEvent),
    Error { code: String, message: String },
}

#[derive(Debug)]
pub enum SessionError {
    NotFound,
    Conflict(String),
    Internal(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::NotFound => write!(f, "not found"),
            SessionError::Conflict(message) => write!(f, "{message}"),
            SessionError::Internal(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<StoreError> for SessionError {
    fn from(err: StoreError) -> Self {
        SessionError::Internal(err.to_string())
    }
}

impl From<aquaregia::Error> for SessionError {
    fn from(err: aquaregia::Error) -> Self {
        SessionError::Internal(err.to_string())
    }
}

impl SessionManager {
    pub fn new(
        store: Arc<Store>,
        provider: Arc<aquaregia::providers::openai_compatible::Client>,
        mcp: McpBridge,
        default_model: String,
        default_instructions: String,
    ) -> Self {
        Self {
            store,
            provider,
            mcp,
            default_model,
            default_instructions,
            active_runs: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub async fn create_session(
        &self,
        model: Option<String>,
        instructions: Option<String>,
        metadata: Value,
    ) -> Result<SessionRecord, SessionError> {
        let id = format!("sess_{}", short_id());
        let model = model.unwrap_or_else(|| self.default_model.clone());
        let instructions = instructions.unwrap_or_else(|| self.default_instructions.clone());
        Ok(self
            .store
            .create_session(&id, &model, &instructions, &metadata)
            .await?)
    }

    pub async fn list_sessions(&self) -> Result<Vec<SessionRecord>, SessionError> {
        Ok(self.store.list_sessions().await?)
    }

    pub async fn get_session(&self, id: &str) -> Result<Option<SessionRecord>, SessionError> {
        Ok(self.store.get_session(id).await?)
    }

    pub async fn archive_session(&self, id: &str) -> Result<bool, SessionError> {
        Ok(self.store.archive_session(id).await?)
    }

    pub async fn list_messages(
        &self,
        session_id: &str,
    ) -> Result<Option<Vec<Message>>, SessionError> {
        if self.store.get_session(session_id).await?.is_none() {
            return Ok(None);
        }
        Ok(Some(self.store.load_messages(session_id).await?))
    }

    pub async fn get_run(&self, run_id: &str) -> Result<Option<RunRecord>, SessionError> {
        Ok(self.store.get_run(run_id).await?)
    }

    pub async fn list_runs(
        &self,
        session_id: &str,
    ) -> Result<Option<Vec<RunRecord>>, SessionError> {
        if self.store.get_session(session_id).await?.is_none() {
            return Ok(None);
        }
        Ok(Some(self.store.list_runs(session_id).await?))
    }

    pub async fn list_run_events(&self, run_id: &str) -> Result<Vec<StoredEvent>, SessionError> {
        Ok(self.store.list_events(run_id).await?)
    }

    pub fn tools(&self) -> Vec<ToolInfo> {
        self.mcp.tool_info()
    }

    pub async fn health(&self) -> Result<usize, SessionError> {
        self.mcp
            .health()
            .await
            .map_err(|err| SessionError::Internal(err.to_string()))
    }

    pub async fn start_run(
        &self,
        session_id: &str,
        input: String,
        mode: RunMode,
    ) -> Result<StartRunResult, SessionError> {
        let session = self
            .store
            .get_session(session_id)
            .await?
            .ok_or(SessionError::NotFound)?;
        self.claim_session(session_id).await?;

        let result = match mode {
            RunMode::Stream => self.start_streaming_run(session, input).await,
            RunMode::Blocking => self.start_blocking_run(session, input).await,
        };

        if result.is_err() {
            self.release_session(session_id).await;
        }
        result
    }

    async fn start_streaming_run(
        &self,
        session: SessionRecord,
        input: String,
    ) -> Result<StartRunResult, SessionError> {
        let run_id = format!("run_{}", short_id());
        self.store.create_run(&run_id, &session.id, &input).await?;
        let mut messages = self.store.load_messages(&session.id).await?;
        messages.push(Message::user_text(input));
        let agent = Arc::new(self.build_agent(&session)?);

        let (tx, rx) = broadcast::channel(256);
        let session_id = session.id.clone();
        let task = RunTask {
            store: Arc::clone(&self.store),
            active_runs: Arc::clone(&self.active_runs),
            session_id: session_id.clone(),
            run_id: run_id.clone(),
            agent,
            messages,
            tx,
        };
        tokio::spawn(async move {
            task.run().await;
        });

        Ok(StartRunResult::Stream {
            stream: Box::pin(broadcast_stream(rx, run_id, session_id)),
        })
    }

    async fn start_blocking_run(
        &self,
        session: SessionRecord,
        input: String,
    ) -> Result<StartRunResult, SessionError> {
        let run_id = format!("run_{}", short_id());
        self.store.create_run(&run_id, &session.id, &input).await?;
        let mut messages = self.store.load_messages(&session.id).await?;
        messages.push(Message::user_text(input));
        let agent = self.build_agent(&session)?;

        let result = run_to_completion(
            Arc::clone(&self.store),
            session.id.clone(),
            run_id.clone(),
            Arc::new(agent),
            messages,
            None,
        )
        .await;
        self.release_session(&session.id).await;
        result?;

        let run = self.store.get_run(&run_id).await?.ok_or_else(|| {
            SessionError::Internal("run disappeared after completion".to_string())
        })?;
        Ok(StartRunResult::Blocking { run })
    }

    fn build_agent(&self, session: &SessionRecord) -> Result<Agent, SessionError> {
        Ok(self
            .provider
            .agent(session.model.clone())
            .instructions(session.instructions.clone())
            .tools(self.mcp.tools())
            .max_steps(12)
            .temperature(0.2)
            .max_output_tokens(2_000)
            .build()?)
    }

    async fn claim_session(&self, session_id: &str) -> Result<(), SessionError> {
        let mut active = self.active_runs.lock().await;
        if active.contains(session_id) {
            return Err(SessionError::Conflict(format!(
                "session {session_id} already has an active run"
            )));
        }
        active.insert(session_id.to_string());
        Ok(())
    }

    async fn release_session(&self, session_id: &str) {
        self.active_runs.lock().await.remove(session_id);
    }
}

struct RunTask {
    store: Arc<Store>,
    active_runs: Arc<Mutex<HashSet<String>>>,
    session_id: String,
    run_id: String,
    agent: Arc<Agent>,
    messages: Vec<Message>,
    tx: broadcast::Sender<SessionRunEvent>,
}

impl RunTask {
    async fn run(self) {
        let result = run_to_completion(
            Arc::clone(&self.store),
            self.session_id.clone(),
            self.run_id.clone(),
            self.agent,
            self.messages,
            Some(self.tx.clone()),
        )
        .await;

        if let Err(err) = result {
            let message = err.to_string();
            let _ = self.store.fail_run(&self.run_id, &message).await;
            let _ = self.tx.send(SessionRunEvent::Error {
                code: "run_failed".to_string(),
                message,
            });
        }

        self.active_runs.lock().await.remove(&self.session_id);
    }
}

async fn run_to_completion(
    store: Arc<Store>,
    session_id: String,
    run_id: String,
    agent: Arc<Agent>,
    messages: Vec<Message>,
    tx: Option<broadcast::Sender<SessionRunEvent>>,
) -> Result<(), SessionError> {
    let mut final_output = None::<AgentOutput>;
    let mut event_seq = 0_i64;
    let mut stream = agent.stream_messages(messages).await?;

    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => {
                if let AgentStreamEvent::Done { output } = &event {
                    final_output = Some(output.clone());
                }
                if let Err(err) = telemetry::record(&store, &run_id, event_seq, &event).await {
                    eprintln!("failed to store run event for {run_id}: {err}");
                }
                event_seq += 1;
                if let Some(tx) = &tx {
                    let _ = tx.send(SessionRunEvent::Agent(event));
                }
            }
            Err(err) => {
                return Err(SessionError::Internal(err.to_string()));
            }
        }
    }

    let output = final_output
        .ok_or_else(|| SessionError::Internal("agent stream ended without run_done".to_string()))?;
    store
        .replace_messages(&session_id, &output.transcript)
        .await?;
    store
        .finish_run(
            &run_id,
            output.steps,
            &output.output_text,
            &serde_json::to_value(&output.usage_total)
                .map_err(|err| SessionError::Internal(err.to_string()))?,
        )
        .await?;
    Ok(())
}

fn broadcast_stream(
    mut rx: broadcast::Receiver<SessionRunEvent>,
    run_id: String,
    session_id: String,
) -> impl Stream<Item = SessionRunEvent> + Send + 'static {
    async_stream::stream! {
        yield SessionRunEvent::Accepted { run_id, session_id };
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let done = matches!(
                        event,
                        SessionRunEvent::Agent(AgentStreamEvent::Done { .. })
                            | SessionRunEvent::Error { .. }
                    );
                    yield event;
                    if done {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    yield SessionRunEvent::Error {
                        code: "event_lagged".to_string(),
                        message: "SSE receiver lagged behind the run event stream".to_string(),
                    };
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    }
}

fn short_id() -> String {
    Uuid::new_v4().simple().to_string()
}
