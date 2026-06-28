use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use aquaregia::{ChatRequest, Message, StreamEvent};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use serde::Deserialize;

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_MODEL: &str = "deepseek-v4-pro";
const SYSTEM_PROMPT: &str = "You are a concise, practical assistant in a web chat app.";

#[derive(Clone)]
struct AppState {
    client: Arc<aquaregia::providers::openai_compatible::Client>,
    model: Arc<str>,
}

#[derive(Debug, Deserialize)]
struct ChatPayload {
    messages: Vec<BrowserMessage>,
}

#[derive(Debug, Deserialize)]
struct BrowserMessage {
    role: String,
    content: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let base_url =
        std::env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let client = aquaregia::providers::openai_compatible::Client::builder()
        .base_url(base_url)
        .api_key_from_env("DEEPSEEK_API_KEY")
        .build()?;
    let state = AppState {
        client: Arc::new(client),
        model: Arc::from(model),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/chat", post(chat))
        .route("/static/app.js", get(app_js))
        .route("/static/style.css", get(style_css))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("chatgpt example listening on http://{addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../index.html"))
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

async fn chat(State(state): State<AppState>, Json(payload): Json<ChatPayload>) -> Response {
    let request = match build_request(&state.model, payload) {
        Ok(request) => request,
        Err((status, message)) => return (status, message).into_response(),
    };

    match state.client.stream(request).await {
        Ok(stream) => Sse::new(stream.map(to_sse_event))
            .keep_alive(KeepAlive::default())
            .into_response(),
        Err(err) => (StatusCode::BAD_GATEWAY, err.to_string()).into_response(),
    }
}

fn build_request(model: &str, payload: ChatPayload) -> Result<ChatRequest, (StatusCode, String)> {
    if payload.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "messages must not be empty".to_string(),
        ));
    }

    let mut messages = Vec::with_capacity(payload.messages.len() + 1);
    messages.push(Message::system_text(SYSTEM_PROMPT));

    for message in payload.messages {
        let content = message.content.trim();
        if content.is_empty() {
            continue;
        }

        match message.role.as_str() {
            "user" => messages.push(Message::user_text(content.to_string())),
            "assistant" => messages.push(Message::assistant_text(content.to_string())),
            role => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("unsupported message role: {role}"),
                ));
            }
        }
    }

    ChatRequest::builder(model)
        .messages(messages)
        .temperature(0.7)
        .max_output_tokens(1_200)
        .build()
        .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))
}

fn to_sse_event(item: Result<StreamEvent, aquaregia::Error>) -> Result<Event, Infallible> {
    let event = match item {
        Ok(StreamEvent::TextDelta { text }) => Event::default().event("text_delta").data(text),
        Ok(StreamEvent::ReasoningDelta { text, .. }) => {
            Event::default().event("reasoning_delta").data(text)
        }
        Ok(StreamEvent::Usage { usage }) => Event::default().event("usage").data(format!(
            r#"{{"input":{},"output":{},"total":{}}}"#,
            usage.input_tokens, usage.output_tokens, usage.total_tokens
        )),
        Ok(StreamEvent::Done { finish_reason }) => Event::default()
            .event("done")
            .data(format!(r#"{{"finish_reason":"{finish_reason:?}"}}"#)),
        Ok(_) => Event::default().event("meta").data("{}"),
        Err(err) => Event::default().event("error").data(err.to_string()),
    };

    Ok(event)
}
