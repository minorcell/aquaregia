use std::convert::Infallible;
use std::sync::Arc;

use aquaregia::AgentStreamEvent;
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::session::{RunMode, SessionManager, SessionRunEvent};
use crate::store::{RunRecord, SessionRecord, StoredEvent};

#[derive(Clone)]
pub struct AppState {
    manager: Arc<SessionManager>,
}

impl AppState {
    pub fn new(manager: SessionManager) -> Self {
        Self {
            manager: Arc::new(manager),
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/favicon.ico", get(favicon))
        .route("/static/app.js", get(app_js))
        .route("/static/style.css", get(style_css))
        .route("/healthz", get(healthz))
        .route("/v1/tools", get(list_tools))
        .route("/v1/sessions", post(create_session).get(list_sessions))
        .route(
            "/v1/sessions/{session_id}",
            get(get_session).delete(delete_session),
        )
        .route("/v1/sessions/{session_id}/messages", get(list_messages))
        .route(
            "/v1/sessions/{session_id}/runs",
            get(list_runs).post(start_run),
        )
        .route("/v1/sessions/{session_id}/runs/{run_id}", get(get_run))
        .route(
            "/v1/sessions/{session_id}/runs/{run_id}/events",
            get(list_run_events),
        )
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../index.html"))
}

async fn favicon() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn app_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("../static/app.js"),
    )
}

async fn style_css() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../static/style.css"),
    )
}

#[derive(Debug, Deserialize)]
struct CreateSessionRequest {
    model: Option<String>,
    instructions: Option<String>,
    metadata: Option<Value>,
}

#[derive(Debug, Serialize)]
struct CreateSessionResponse {
    session_id: String,
    session: SessionRecord,
}

#[derive(Debug, Deserialize)]
struct StartRunRequest {
    input: String,
    #[serde(default = "default_stream")]
    stream: bool,
}

fn default_stream() -> bool {
    true
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
}

async fn healthz(State(state): State<AppState>) -> Response {
    match state.manager.health().await {
        Ok(tools) => Json(json!({
            "status": "ok",
            "sandbox": "ok",
            "tool_count": tools,
        }))
        .into_response(),
        Err(err) => api_error(StatusCode::BAD_GATEWAY, "sandbox_unavailable", err),
    }
}

async fn list_tools(State(state): State<AppState>) -> Response {
    Json(state.manager.tools()).into_response()
}

async fn create_session(
    State(state): State<AppState>,
    Json(payload): Json<CreateSessionRequest>,
) -> Response {
    match state
        .manager
        .create_session(
            payload.model,
            payload.instructions,
            payload.metadata.unwrap_or_else(|| json!({})),
        )
        .await
    {
        Ok(session) => Json(CreateSessionResponse {
            session_id: session.id.clone(),
            session,
        })
        .into_response(),
        Err(err) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "create_session_failed",
            err,
        ),
    }
}

async fn list_sessions(State(state): State<AppState>) -> Response {
    match state.manager.list_sessions().await {
        Ok(sessions) => Json(sessions).into_response(),
        Err(err) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "list_sessions_failed",
            err,
        ),
    }
}

async fn get_session(State(state): State<AppState>, Path(session_id): Path<String>) -> Response {
    match state.manager.get_session(&session_id).await {
        Ok(Some(session)) => Json(session).into_response(),
        Ok(None) => api_error_text(StatusCode::NOT_FOUND, "not_found", "session not found"),
        Err(err) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "get_session_failed", err),
    }
}

async fn delete_session(State(state): State<AppState>, Path(session_id): Path<String>) -> Response {
    match state.manager.archive_session(&session_id).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => api_error_text(StatusCode::NOT_FOUND, "not_found", "session not found"),
        Err(err) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "archive_session_failed",
            err,
        ),
    }
}

async fn list_messages(State(state): State<AppState>, Path(session_id): Path<String>) -> Response {
    match state.manager.list_messages(&session_id).await {
        Ok(Some(messages)) => Json(messages).into_response(),
        Ok(None) => api_error_text(StatusCode::NOT_FOUND, "not_found", "session not found"),
        Err(err) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "list_messages_failed",
            err,
        ),
    }
}

async fn list_runs(State(state): State<AppState>, Path(session_id): Path<String>) -> Response {
    match state.manager.list_runs(&session_id).await {
        Ok(Some(runs)) => Json(runs).into_response(),
        Ok(None) => api_error_text(StatusCode::NOT_FOUND, "not_found", "session not found"),
        Err(err) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "list_runs_failed", err),
    }
}

async fn start_run(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(payload): Json<StartRunRequest>,
) -> Response {
    if payload.input.trim().is_empty() {
        return api_error_text(
            StatusCode::BAD_REQUEST,
            "invalid_input",
            "input must not be empty",
        );
    }

    let mode = if payload.stream {
        RunMode::Stream
    } else {
        RunMode::Blocking
    };

    match state
        .manager
        .start_run(&session_id, payload.input, mode)
        .await
    {
        Ok(crate::session::StartRunResult::Stream { stream }) => Sse::new(to_sse_stream(stream))
            .keep_alive(KeepAlive::default())
            .into_response(),
        Ok(crate::session::StartRunResult::Blocking { run }) => Json(run).into_response(),
        Err(crate::session::SessionError::NotFound) => {
            api_error_text(StatusCode::NOT_FOUND, "not_found", "session not found")
        }
        Err(crate::session::SessionError::Conflict(message)) => {
            api_error_text(StatusCode::CONFLICT, "run_conflict", &message)
        }
        Err(err) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "start_run_failed", err),
    }
}

async fn get_run(
    State(state): State<AppState>,
    Path((_session_id, run_id)): Path<(String, String)>,
) -> Response {
    match state.manager.get_run(&run_id).await {
        Ok(Some(run)) => Json(run).into_response(),
        Ok(None) => api_error_text(StatusCode::NOT_FOUND, "not_found", "run not found"),
        Err(err) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "get_run_failed", err),
    }
}

async fn list_run_events(
    State(state): State<AppState>,
    Path((_session_id, run_id)): Path<(String, String)>,
) -> Response {
    match state.manager.list_run_events(&run_id).await {
        Ok(events) => Json(events).into_response(),
        Err(err) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "list_events_failed", err),
    }
}

fn to_sse_stream(
    stream: impl Stream<Item = SessionRunEvent> + Send + 'static,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        futures_util::pin_mut!(stream);
        while let Some(event) = futures_util::StreamExt::next(&mut stream).await {
            yield Ok(to_sse_event(event));
        }
    }
}

fn to_sse_event(event: SessionRunEvent) -> Event {
    match event {
        SessionRunEvent::Accepted { run_id, session_id } => {
            Event::default().event("run_accepted").data(
                json!({
                    "run_id": run_id,
                    "session_id": session_id,
                })
                .to_string(),
            )
        }
        SessionRunEvent::Agent(event) => Event::default()
            .event(agent_event_name(&event))
            .json_data(event)
            .unwrap_or_else(|err| Event::default().event("error").data(err.to_string())),
        SessionRunEvent::Error { code, message } => Event::default().event("error").data(
            json!({
                "code": code,
                "message": message,
            })
            .to_string(),
        ),
    }
}

pub fn agent_event_name(event: &AgentStreamEvent) -> &'static str {
    match event {
        AgentStreamEvent::Start { .. } => "run_start",
        AgentStreamEvent::StepStart { .. } => "step_start",
        AgentStreamEvent::Model { .. } => "model_delta",
        AgentStreamEvent::ToolCallStart { .. } => "tool_call_start",
        AgentStreamEvent::ToolCallFinish { .. } => "tool_call_finish",
        AgentStreamEvent::StepFinish { .. } => "step_finish",
        AgentStreamEvent::Done { .. } => "run_done",
    }
}

fn api_error(status: StatusCode, code: &'static str, err: impl std::fmt::Display) -> Response {
    api_error_text(status, code, &err.to_string())
}

fn api_error_text(status: StatusCode, code: &'static str, message: &str) -> Response {
    (
        status,
        Json(ErrorBody {
            code,
            message: message.to_string(),
        }),
    )
        .into_response()
}

#[allow(dead_code)]
fn _assert_response_types(_: RunRecord, _: StoredEvent) {}
