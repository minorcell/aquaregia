use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::Response;
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

const DEFAULT_ADDR: &str = "0.0.0.0:8931";
const DEFAULT_WORKDIR: &str = "/workspace";
const MAX_OUTPUT_BYTES: usize = 16 * 1024;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = env_or("SANDBOX_MCP_ADDR", DEFAULT_ADDR).parse()?;
    let token = Arc::new(env_or("SANDBOX_MCP_TOKEN", "dev-token"));
    let workdir = PathBuf::from(env_or("SANDBOX_WORKDIR", DEFAULT_WORKDIR));
    tokio::fs::create_dir_all(&workdir).await?;

    let ct = CancellationToken::new();
    let service = StreamableHttpService::new(
        move || Ok(SandboxServer::new(workdir.clone())),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default()
            .with_allowed_hosts(allowed_hosts(addr))
            .with_cancellation_token(ct.child_token()),
    );

    let app =
        axum::Router::new()
            .nest_service("/mcp", service)
            .layer(middleware::from_fn_with_state(
                Arc::clone(&token),
                check_bearer,
            ));

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let shutdown = ct.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        shutdown.cancel();
    });

    println!("remote_agent sandbox MCP listening on http://{addr}/mcp");
    axum::serve(listener, app)
        .with_graceful_shutdown(ct.cancelled_owned())
        .await?;
    Ok(())
}

#[derive(Debug, Clone)]
struct SandboxServer {
    workdir: PathBuf,
    tool_router: ToolRouter<Self>,
}

impl SandboxServer {
    fn new(workdir: PathBuf) -> Self {
        Self {
            workdir,
            tool_router: Self::tool_router(),
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct BashArgs {
    #[schemars(description = "Shell command to run inside the sandbox working directory.")]
    cmd: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct FilePathArgs {
    #[schemars(description = "Path relative to the sandbox working directory.")]
    path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct WriteFileArgs {
    #[schemars(description = "Path relative to the sandbox working directory.")]
    path: String,
    #[schemars(description = "UTF-8 file contents to write.")]
    content: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GitArgs {
    #[schemars(description = "Arguments passed after `git`, for example `status --short`.")]
    args: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct NodeEvalArgs {
    #[schemars(description = "JavaScript source to run with Node.js.")]
    script: String,
}

#[tool_router]
impl SandboxServer {
    #[tool(description = "Run a shell command inside the sandbox working directory.")]
    async fn bash(&self, Parameters(BashArgs { cmd }): Parameters<BashArgs>) -> CallToolResult {
        command_result(run_shell(&self.workdir, &cmd).await)
    }

    #[tool(description = "Read a UTF-8 file from the sandbox working directory.")]
    async fn read_file(
        &self,
        Parameters(FilePathArgs { path }): Parameters<FilePathArgs>,
    ) -> CallToolResult {
        match resolve_path(&self.workdir, &path) {
            Ok(path) => match tokio::fs::read_to_string(&path).await {
                Ok(content) => CallToolResult::structured(json!({
                    "path": path,
                    "content": content,
                })),
                Err(err) => structured_error("read_failed", err.to_string()),
            },
            Err(err) => structured_error("invalid_path", err),
        }
    }

    #[tool(description = "Write a UTF-8 file inside the sandbox working directory.")]
    async fn write_file(
        &self,
        Parameters(WriteFileArgs { path, content }): Parameters<WriteFileArgs>,
    ) -> CallToolResult {
        match resolve_path(&self.workdir, &path) {
            Ok(path) => {
                if let Some(parent) = path.parent()
                    && let Err(err) = tokio::fs::create_dir_all(parent).await
                {
                    return structured_error("mkdir_failed", err.to_string());
                }
                match tokio::fs::write(&path, content).await {
                    Ok(()) => CallToolResult::structured(json!({
                        "path": path,
                        "written": true,
                    })),
                    Err(err) => structured_error("write_failed", err.to_string()),
                }
            }
            Err(err) => structured_error("invalid_path", err),
        }
    }

    #[tool(description = "Run a git command inside the sandbox working directory.")]
    async fn git(&self, Parameters(GitArgs { args }): Parameters<GitArgs>) -> CallToolResult {
        command_result(run_shell(&self.workdir, &format!("git {args}")).await)
    }

    #[tool(description = "Run JavaScript with Node.js inside the sandbox working directory.")]
    async fn node_eval(
        &self,
        Parameters(NodeEvalArgs { script }): Parameters<NodeEvalArgs>,
    ) -> CallToolResult {
        command_result(run_node(&self.workdir, &script).await)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SandboxServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("Tools execute inside the remote_agent Docker sandbox.")
    }
}

async fn check_bearer(
    State(token): State<Arc<String>>,
    req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let expected = format!("Bearer {}", token.as_str());
    let authorized = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == expected);

    if authorized {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

async fn run_shell(workdir: &Path, command: &str) -> Result<Value, String> {
    let child = Command::new("sh")
        .arg("-lc")
        .arg(command)
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| err.to_string())?;

    let output = timeout(COMMAND_TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| "command timed out".to_string())?
        .map_err(|err| err.to_string())?;

    Ok(json!({
        "exit_code": output.status.code(),
        "success": output.status.success(),
        "stdout": truncate_output(&output.stdout),
        "stderr": truncate_output(&output.stderr),
    }))
}

async fn run_node(workdir: &Path, script: &str) -> Result<Value, String> {
    let child = Command::new("node")
        .arg("-e")
        .arg(script)
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| err.to_string())?;

    let output = timeout(COMMAND_TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| "node execution timed out".to_string())?
        .map_err(|err| err.to_string())?;

    Ok(json!({
        "exit_code": output.status.code(),
        "success": output.status.success(),
        "stdout": truncate_output(&output.stdout),
        "stderr": truncate_output(&output.stderr),
    }))
}

fn command_result(result: Result<Value, String>) -> CallToolResult {
    match result {
        Ok(value) if value.get("success").and_then(Value::as_bool) == Some(true) => {
            CallToolResult::structured(value)
        }
        Ok(value) => CallToolResult::structured_error(value),
        Err(message) => structured_error("command_failed", message),
    }
}

fn structured_error(code: &str, message: impl Into<String>) -> CallToolResult {
    CallToolResult::structured_error(json!({
        "code": code,
        "message": message.into(),
    }))
}

fn resolve_path(root: &Path, input: &str) -> Result<PathBuf, String> {
    let path = Path::new(input);
    if path.is_absolute() {
        return Err("path must be relative to the sandbox working directory".to_string());
    }

    let mut resolved = root.to_path_buf();
    for component in path.components() {
        match component {
            Component::Normal(part) => resolved.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err("path must not contain `..`".to_string());
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err("path must be relative to the sandbox working directory".to_string());
            }
        }
    }
    Ok(resolved)
}

fn truncate_output(bytes: &[u8]) -> String {
    let truncated = bytes.len() > MAX_OUTPUT_BYTES;
    let end = bytes.len().min(MAX_OUTPUT_BYTES);
    let mut text = String::from_utf8_lossy(&bytes[..end]).to_string();
    if truncated {
        text.push_str("\n[output truncated]");
    }
    text
}

fn allowed_hosts(addr: SocketAddr) -> Vec<String> {
    let port = addr.port();
    env_or(
        "SANDBOX_ALLOWED_HOSTS",
        &format!("127.0.0.1:{port},localhost:{port}"),
    )
    .split(',')
    .map(str::trim)
    .filter(|host| !host.is_empty())
    .map(ToString::to_string)
    .collect()
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}
