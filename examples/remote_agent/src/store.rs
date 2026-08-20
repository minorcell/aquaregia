use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aquaregia::Message;
use serde::Serialize;
use serde_json::Value;
use tokio::sync::Mutex;

#[derive(Clone, Default)]
pub struct Store {
    state: Arc<Mutex<StoreState>>,
}

#[derive(Default)]
struct StoreState {
    sessions: HashMap<String, SessionRecord>,
    messages: HashMap<String, Vec<Message>>,
    runs: HashMap<String, RunRecord>,
    run_events: HashMap<String, Vec<StoredEvent>>,
}

#[derive(Debug)]
pub struct StoreError {
    message: String,
}

impl StoreError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StoreError {}

pub type StoreResult<T> = Result<T, StoreError>;

#[derive(Debug, Clone, Serialize)]
pub struct SessionRecord {
    pub id: String,
    pub model: String,
    pub instructions: String,
    pub status: String,
    pub metadata: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunRecord {
    pub id: String,
    pub session_id: String,
    pub status: String,
    pub input: String,
    pub steps: i64,
    pub output_text: Option<String>,
    pub usage_total: Option<Value>,
    pub started_at: i64,
    pub finished_at: Option<i64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoredEvent {
    pub seq: i64,
    pub ts: i64,
    pub event_type: String,
    pub payload: Value,
}

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn create_session(
        &self,
        id: &str,
        model: &str,
        instructions: &str,
        metadata: &Value,
    ) -> StoreResult<SessionRecord> {
        let now = now_ms();
        let record = SessionRecord {
            id: id.to_string(),
            model: model.to_string(),
            instructions: instructions.to_string(),
            status: "active".to_string(),
            metadata: metadata.clone(),
            created_at: now,
            updated_at: now,
        };

        let mut state = self.state.lock().await;
        state.sessions.insert(id.to_string(), record.clone());
        state.messages.entry(id.to_string()).or_default();
        Ok(record)
    }

    pub async fn list_sessions(&self) -> StoreResult<Vec<SessionRecord>> {
        let state = self.state.lock().await;
        let mut sessions = state.sessions.values().cloned().collect::<Vec<_>>();
        sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(sessions)
    }

    pub async fn get_session(&self, id: &str) -> StoreResult<Option<SessionRecord>> {
        let state = self.state.lock().await;
        Ok(state.sessions.get(id).cloned())
    }

    pub async fn archive_session(&self, id: &str) -> StoreResult<bool> {
        let mut state = self.state.lock().await;
        let Some(session) = state.sessions.get_mut(id) else {
            return Ok(false);
        };
        if session.status == "archived" {
            return Ok(false);
        }
        session.status = "archived".to_string();
        session.updated_at = now_ms();
        Ok(true)
    }

    pub async fn load_messages(&self, session_id: &str) -> StoreResult<Vec<Message>> {
        let state = self.state.lock().await;
        Ok(state.messages.get(session_id).cloned().unwrap_or_default())
    }

    pub async fn replace_messages(
        &self,
        session_id: &str,
        messages: &[Message],
    ) -> StoreResult<()> {
        let mut state = self.state.lock().await;
        let Some(session) = state.sessions.get_mut(session_id) else {
            return Err(StoreError::new(format!("session {session_id} not found")));
        };
        session.updated_at = now_ms();
        state
            .messages
            .insert(session_id.to_string(), messages.to_vec());
        Ok(())
    }

    pub async fn create_run(
        &self,
        id: &str,
        session_id: &str,
        input: &str,
    ) -> StoreResult<RunRecord> {
        let mut state = self.state.lock().await;
        if !state.sessions.contains_key(session_id) {
            return Err(StoreError::new(format!("session {session_id} not found")));
        }

        let now = now_ms();
        let record = RunRecord {
            id: id.to_string(),
            session_id: session_id.to_string(),
            status: "running".to_string(),
            input: input.to_string(),
            steps: 0,
            output_text: None,
            usage_total: None,
            started_at: now,
            finished_at: None,
            error: None,
        };
        state.runs.insert(id.to_string(), record.clone());
        state.run_events.entry(id.to_string()).or_default();
        Ok(record)
    }

    pub async fn get_run(&self, id: &str) -> StoreResult<Option<RunRecord>> {
        let state = self.state.lock().await;
        Ok(state.runs.get(id).cloned())
    }

    pub async fn list_runs(&self, session_id: &str) -> StoreResult<Vec<RunRecord>> {
        let state = self.state.lock().await;
        let mut runs = state
            .runs
            .values()
            .filter(|run| run.session_id == session_id)
            .cloned()
            .collect::<Vec<_>>();
        runs.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        Ok(runs)
    }

    pub async fn finish_run(
        &self,
        id: &str,
        steps: u32,
        output_text: &str,
        usage_total: &Value,
    ) -> StoreResult<()> {
        let mut state = self.state.lock().await;
        let Some(run) = state.runs.get_mut(id) else {
            return Err(StoreError::new(format!("run {id} not found")));
        };
        run.status = "completed".to_string();
        run.steps = steps as i64;
        run.output_text = Some(output_text.to_string());
        run.usage_total = Some(usage_total.clone());
        run.finished_at = Some(now_ms());
        Ok(())
    }

    pub async fn fail_run(&self, id: &str, error: &str) -> StoreResult<()> {
        let mut state = self.state.lock().await;
        let Some(run) = state.runs.get_mut(id) else {
            return Err(StoreError::new(format!("run {id} not found")));
        };
        run.status = "failed".to_string();
        run.error = Some(error.to_string());
        run.finished_at = Some(now_ms());
        Ok(())
    }

    pub async fn insert_event(
        &self,
        run_id: &str,
        seq: i64,
        event_type: &str,
        payload: &Value,
    ) -> StoreResult<()> {
        let mut state = self.state.lock().await;
        if !state.runs.contains_key(run_id) {
            return Err(StoreError::new(format!("run {run_id} not found")));
        }
        state
            .run_events
            .entry(run_id.to_string())
            .or_default()
            .push(StoredEvent {
                seq,
                ts: now_ms(),
                event_type: event_type.to_string(),
                payload: payload.clone(),
            });
        Ok(())
    }

    pub async fn list_events(&self, run_id: &str) -> StoreResult<Vec<StoredEvent>> {
        let state = self.state.lock().await;
        let mut events = state.run_events.get(run_id).cloned().unwrap_or_default();
        events.sort_by(|a, b| a.seq.cmp(&b.seq));
        Ok(events)
    }
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
