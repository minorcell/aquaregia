mod api;
mod mcp_bridge;
mod session;
mod store;
mod telemetry;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use tokio_util::sync::CancellationToken;

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_MODEL: &str = "deepseek-v4-pro";
const DEFAULT_SANDBOX_URL: &str = "http://127.0.0.1:8931/mcp";
const DEFAULT_GATEWAY_ADDR: &str = "127.0.0.1:3000";
const DEFAULT_INSTRUCTIONS: &str = r#"
You are a remote coding agent.

Use the sandbox tools for repository, shell, Node.js, and file-system work.
Keep commands scoped to the sandbox working directory. Explain what you did and
include concise command results in your final answer.
"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = AppConfig::from_env();
    let store = store::Store::new();

    let provider = aquaregia::providers::openai_compatible::Client::builder()
        .base_url(config.provider_base_url.clone())
        .api_key_from_env("DEEPSEEK_API_KEY")
        .build()?;

    let mcp = mcp_bridge::McpBridge::connect(&config.sandbox_url, &config.sandbox_token).await?;
    let manager = session::SessionManager::new(
        Arc::new(store),
        Arc::new(provider),
        mcp,
        config.default_model.clone(),
        DEFAULT_INSTRUCTIONS.trim().to_string(),
    );

    let app = api::router(api::AppState::new(manager));
    serve(app, config.gateway_addr).await?;
    Ok(())
}

async fn serve(
    app: Router,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let token = CancellationToken::new();
    let shutdown_token = token.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        shutdown_token.cancel();
    });

    println!("remote_agent gateway listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(token.cancelled_owned())
        .await?;
    Ok(())
}

struct AppConfig {
    gateway_addr: SocketAddr,
    default_model: String,
    provider_base_url: String,
    sandbox_url: String,
    sandbox_token: String,
}

impl AppConfig {
    fn from_env() -> Self {
        Self {
            gateway_addr: env_or("REMOTE_AGENT_ADDR", DEFAULT_GATEWAY_ADDR)
                .parse()
                .expect("REMOTE_AGENT_ADDR must be a socket address"),
            default_model: env_or("DEEPSEEK_MODEL", DEFAULT_MODEL),
            provider_base_url: env_or("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            sandbox_url: env_or("SANDBOX_MCP_URL", DEFAULT_SANDBOX_URL),
            sandbox_token: env_or("SANDBOX_MCP_TOKEN", "dev-token"),
        }
    }
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}
