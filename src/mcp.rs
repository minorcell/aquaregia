//! MCP (Model Context Protocol) client integration.
//!
//! Feature-gated behind `mcp`. Connects to an MCP server, lists its tools, and
//! adapts each one into an aquaregia [`Tool`] that plugs straight into an
//! [`Agent`](crate::Agent) via `.tools(...)`. Prompts and resources are exposed
//! as thin pass-through APIs that return `rmcp` model types. The protocol,
//! transport, and handshake are handled by the official [`rmcp`] SDK; this
//! module only maps `rmcp` tools to aquaregia tools and flattens MCP tool
//! results.
//!
//! ```rust,no_run
//! use aquaregia::providers;
//! use tokio::process::Command;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mcp = aquaregia::mcp::connect_stdio(Command::new("mcp-server-fs")).await?;
//! let agent = providers::openai_compatible::Client::builder()
//!     .base_url("https://api.example.com")
//!     .api_key("api-key")
//!     .build()?
//!     .agent("gpt-5.4-mini")
//!     .tools(mcp.clone())
//!     .build()?;
//! let snapshot = mcp.list_tools().await?;
//! # let _ = (agent, snapshot);
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use rmcp::handler::client::ClientHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, GetPromptRequestParams, PaginatedRequestParams,
    ReadResourceRequestParams,
};
use rmcp::service::{RoleClient, RunningService, ServerSink, ServiceError, ServiceExt};
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use serde_json::{Value, json};
use tokio::process::Command;

use crate::error::Error;
use crate::tool::{Tool, ToolDescriptor, ToolExecError, ToolExecutor};
use crate::types::ToolSource;

pub use rmcp::model::{
    ContentBlock, GetPromptResult, Prompt, PromptArgument, PromptMessage, ReadResourceResult,
    Resource, ResourceContents, ResourceTemplate, Role,
};

/// MCP request metadata passed through as `_meta`.
///
/// This is hidden from the model and sent only to the MCP server, which makes it
/// suitable for auth tokens, session ids, or other call-scoped transport data.
pub use rmcp::model::Meta;

/// Default per-call timeout. A never-answered MCP response would otherwise hang
/// the agent forever; this bounds it into a recoverable [`ToolExecError`].
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(300);

/// Error from establishing or querying an MCP connection.
///
/// Errors from individual tool *calls* surface as [`ToolExecError`] inside the
/// agent loop, not here.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// Failed to spawn the server or complete the MCP handshake.
    #[error("MCP connect failed: {0}")]
    Connect(String),
    /// Failed to fetch the tool list from the server.
    #[error("MCP list tools failed: {0}")]
    ListTools(String),
    /// Failed to fetch the prompt list from the server.
    #[error("MCP list prompts failed: {0}")]
    ListPrompts(String),
    /// Failed to fetch a prompt from the server.
    #[error("MCP get prompt failed: {0}")]
    GetPrompt(String),
    /// Failed to fetch the resource list from the server.
    #[error("MCP list resources failed: {0}")]
    ListResources(String),
    /// Failed to fetch the resource template list from the server.
    #[error("MCP list resource templates failed: {0}")]
    ListResourceTemplates(String),
    /// Failed to read a resource from the server.
    #[error("MCP read resource failed: {0}")]
    ReadResource(String),
    /// The provided request arguments cannot be encoded as MCP arguments.
    #[error("{0}")]
    InvalidArguments(String),
    /// The connection has already closed.
    #[error("MCP connection is disconnected")]
    Disconnected,
    /// A tool call exceeded the configured timeout.
    #[error("MCP tool call timed out")]
    CallTimeout,
}

impl McpError {
    fn into_tool_error(self) -> ToolExecError {
        match self {
            McpError::CallTimeout => ToolExecError::Timeout,
            other => ToolExecError::Execution(other.to_string()),
        }
    }
}

type ToolCache = Arc<tokio::sync::RwLock<Vec<Tool>>>;
type SharedToolConfig = Arc<RwLock<McpToolConfig>>;

#[derive(Clone)]
struct McpToolConfig {
    timeout: Option<Duration>,
    meta: Option<Meta>,
}

impl Default for McpToolConfig {
    fn default() -> Self {
        Self {
            timeout: Some(DEFAULT_CALL_TIMEOUT),
            meta: None,
        }
    }
}

#[derive(Clone)]
struct McpClientHandler {
    tools: ToolCache,
    config: SharedToolConfig,
}

impl McpClientHandler {
    fn new() -> Self {
        Self {
            tools: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            config: Arc::new(RwLock::new(McpToolConfig::default())),
        }
    }

    async fn refresh_tools(&self, peer: ServerSink) -> Result<Vec<Tool>, ServiceError> {
        let raw_tools = peer.list_all_tools().await?;
        let tools = raw_tools
            .into_iter()
            .map(|tool| build_tool_from_mcp(peer.clone(), tool, Arc::clone(&self.config)))
            .collect::<Vec<_>>();
        *self.tools.write().await = tools.clone();
        Ok(tools)
    }
}

impl ClientHandler for McpClientHandler {
    async fn on_tool_list_changed(&self, context: rmcp::service::NotificationContext<RoleClient>) {
        let _ = self.refresh_tools(context.peer).await;
    }
}

/// An active MCP connection.
///
/// Must be kept alive while the [`Tool`]s it produces are in use: dropping it
/// closes the connection, after which those tools' calls fail with a
/// [`ToolExecError`].
#[derive(Clone)]
pub struct McpConnection {
    service: Arc<RunningService<RoleClient, McpClientHandler>>,
    tools: ToolCache,
    config: SharedToolConfig,
}

/// Connect to a local MCP server launched as a child process (stdio transport).
pub async fn connect_stdio(command: Command) -> Result<McpConnection, McpError> {
    let transport =
        TokioChildProcess::new(command).map_err(|e| McpError::Connect(e.to_string()))?;
    let handler = McpClientHandler::new();
    let tools = Arc::clone(&handler.tools);
    let config = Arc::clone(&handler.config);
    let service = handler
        .serve(transport)
        .await
        .map_err(|e| McpError::Connect(e.to_string()))?;
    let connection = McpConnection {
        service: Arc::new(service),
        tools,
        config,
    };
    let _ = connection.refresh_tools().await;
    Ok(connection)
}

/// Connect to a remote MCP server over Streamable HTTP.
pub async fn connect_http(url: impl Into<String>) -> Result<McpConnection, McpError> {
    let transport = StreamableHttpClientTransport::from_uri(url.into());
    let handler = McpClientHandler::new();
    let tools = Arc::clone(&handler.tools);
    let config = Arc::clone(&handler.config);
    let service = handler
        .serve(transport)
        .await
        .map_err(|e| McpError::Connect(e.to_string()))?;
    let connection = McpConnection {
        service: Arc::new(service),
        tools,
        config,
    };
    let _ = connection.refresh_tools().await;
    Ok(connection)
}

impl McpConnection {
    /// Set (or clear with `None`) the per-call timeout applied to tools built by
    /// subsequent [`tools`](Self::tools) calls. Defaults to
    /// [`DEFAULT_CALL_TIMEOUT`].
    pub fn with_call_timeout(self, timeout: impl Into<Option<Duration>>) -> Self {
        self.config
            .write()
            .expect("MCP tool config lock poisoned")
            .timeout = timeout.into();
        self
    }

    /// Set (or clear with `None`) MCP `_meta` attached to subsequent tool,
    /// prompt, and resource requests.
    ///
    /// `_meta` is part of the MCP request envelope, not the model-visible tool
    /// arguments. Use it for values the server needs but the model should not
    /// see, such as auth tokens or session ids.
    pub fn with_call_meta(self, meta: impl Into<Option<Meta>>) -> Self {
        self.config
            .write()
            .expect("MCP tool config lock poisoned")
            .meta = meta.into();
        self
    }

    /// Fetch the server's current tools, update the live tool cache, and adapt
    /// them into aquaregia [`Tool`]s.
    ///
    /// Use [`AgentBuilder::tools`](crate::AgentBuilder::tools) with a
    /// [`McpConnection`] to register MCP tools as a live tool source. This
    /// method is for explicit listing or static snapshots.
    pub async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        self.refresh_tools().await
    }

    /// Return the currently cached MCP tools without a network round trip.
    pub async fn current_tools(&self) -> Vec<Tool> {
        self.tools.read().await.clone()
    }

    async fn refresh_tools(&self) -> Result<Vec<Tool>, McpError> {
        self.ensure_connected()?;

        self.service
            .service()
            .refresh_tools(self.service.peer().clone())
            .await
            .map_err(map_list_tools_error)
    }

    /// Fetch all prompts currently advertised by the server.
    pub async fn prompts(&self) -> Result<Vec<Prompt>, McpError> {
        self.ensure_connected()?;

        let meta = self.config_snapshot().meta;
        let mut prompts = Vec::new();
        let mut cursor = None;
        loop {
            let result = self
                .service
                .peer()
                .list_prompts(Some(build_paginated_request(cursor, meta.clone())))
                .await
                .map_err(map_list_prompts_error)?;
            prompts.extend(result.prompts);
            cursor = result.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        Ok(prompts)
    }

    /// Fetch a prompt by name, optionally passing JSON-object arguments.
    pub async fn get_prompt(
        &self,
        name: impl Into<String>,
        arguments: impl Into<Option<Value>>,
    ) -> Result<GetPromptResult, McpError> {
        self.ensure_connected()?;

        let request =
            build_get_prompt_request(name, arguments.into(), self.config_snapshot().meta)?;
        self.service
            .peer()
            .get_prompt(request)
            .await
            .map_err(map_get_prompt_error)
    }

    /// Fetch all resources currently advertised by the server.
    pub async fn resources(&self) -> Result<Vec<Resource>, McpError> {
        self.ensure_connected()?;

        let meta = self.config_snapshot().meta;
        let mut resources = Vec::new();
        let mut cursor = None;
        loop {
            let result = self
                .service
                .peer()
                .list_resources(Some(build_paginated_request(cursor, meta.clone())))
                .await
                .map_err(map_list_resources_error)?;
            resources.extend(result.resources);
            cursor = result.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        Ok(resources)
    }

    /// Fetch all resource templates currently advertised by the server.
    pub async fn resource_templates(&self) -> Result<Vec<ResourceTemplate>, McpError> {
        self.ensure_connected()?;

        let meta = self.config_snapshot().meta;
        let mut resource_templates = Vec::new();
        let mut cursor = None;
        loop {
            let result = self
                .service
                .peer()
                .list_resource_templates(Some(build_paginated_request(cursor, meta.clone())))
                .await
                .map_err(map_list_resource_templates_error)?;
            resource_templates.extend(result.resource_templates);
            cursor = result.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        Ok(resource_templates)
    }

    /// Read the current contents for a resource URI.
    pub async fn read_resource(
        &self,
        uri: impl Into<String>,
    ) -> Result<ReadResourceResult, McpError> {
        self.ensure_connected()?;

        let request = build_read_resource_request(uri, self.config_snapshot().meta);
        self.service
            .peer()
            .read_resource(request)
            .await
            .map_err(map_read_resource_error)
    }

    fn ensure_connected(&self) -> Result<(), McpError> {
        if self.service.peer().is_transport_closed() {
            return Err(McpError::Disconnected);
        }
        Ok(())
    }

    fn config_snapshot(&self) -> McpToolConfig {
        self.config
            .read()
            .expect("MCP tool config lock poisoned")
            .clone()
    }
}

#[async_trait]
impl ToolSource for McpConnection {
    async fn tools(&self) -> Result<Vec<Tool>, Error> {
        Ok(self.current_tools().await)
    }
}

fn build_tool_from_mcp(
    peer: ServerSink,
    tool: rmcp::model::Tool,
    config: SharedToolConfig,
) -> Tool {
    let descriptor = ToolDescriptor {
        name: tool.name.to_string(),
        description: tool.description.as_deref().unwrap_or_default().to_string(),
        input_schema: serde_json::to_value(&tool.input_schema)
            .unwrap_or_else(|_| json!({ "type": "object" })),
    };
    let executor = Arc::new(McpToolExecutor {
        peer,
        tool_name: tool.name.to_string(),
        config,
    });
    Tool::from_parts(descriptor, executor)
}

struct McpToolExecutor {
    peer: ServerSink,
    tool_name: String,
    config: SharedToolConfig,
}

#[async_trait]
impl ToolExecutor for McpToolExecutor {
    async fn execute(&self, args: Value) -> Result<Value, ToolExecError> {
        if self.peer.is_transport_closed() {
            return Err(McpError::Disconnected.into_tool_error());
        }

        let config = self
            .config
            .read()
            .expect("MCP tool config lock poisoned")
            .clone();
        let request = build_call_request(&self.tool_name, args, config.meta)?;
        let call = self.peer.call_tool(request);
        let result = match config.timeout {
            Some(timeout) => tokio::time::timeout(timeout, call)
                .await
                .map_err(|_| McpError::CallTimeout.into_tool_error())?
                .map_err(map_call_error)?,
            None => call.await.map_err(map_call_error)?,
        };
        map_result(result)
    }
}

fn build_call_request(
    tool_name: &str,
    args: Value,
    meta: Option<Meta>,
) -> Result<CallToolRequestParams, ToolExecError> {
    let mut request = match args {
        Value::Object(obj) => CallToolRequestParams::new(tool_name.to_string()).with_arguments(obj),
        Value::Null => CallToolRequestParams::new(tool_name.to_string()),
        _ => {
            return Err(ToolExecError::Execution(
                "MCP tool arguments must be a JSON object".to_string(),
            ));
        }
    };
    request.meta = meta;
    Ok(request)
}

fn build_paginated_request(cursor: Option<String>, meta: Option<Meta>) -> PaginatedRequestParams {
    let mut request = PaginatedRequestParams::default();
    request.meta = meta;
    request.cursor = cursor;
    request
}

fn build_get_prompt_request(
    name: impl Into<String>,
    arguments: Option<Value>,
    meta: Option<Meta>,
) -> Result<GetPromptRequestParams, McpError> {
    let name = name.into();
    let mut request = match arguments {
        Some(Value::Object(arguments)) => {
            GetPromptRequestParams::new(name).with_arguments(arguments)
        }
        Some(Value::Null) | None => GetPromptRequestParams::new(name),
        _ => {
            return Err(McpError::InvalidArguments(
                "MCP prompt arguments must be a JSON object".to_string(),
            ));
        }
    };
    request.meta = meta;
    Ok(request)
}

fn build_read_resource_request(
    uri: impl Into<String>,
    meta: Option<Meta>,
) -> ReadResourceRequestParams {
    let mut request = ReadResourceRequestParams::new(uri);
    request.meta = meta;
    request
}

fn map_list_tools_error(error: ServiceError) -> McpError {
    map_service_error(error, McpError::ListTools)
}

fn map_list_prompts_error(error: ServiceError) -> McpError {
    map_service_error(error, McpError::ListPrompts)
}

fn map_get_prompt_error(error: ServiceError) -> McpError {
    map_service_error(error, McpError::GetPrompt)
}

fn map_list_resources_error(error: ServiceError) -> McpError {
    map_service_error(error, McpError::ListResources)
}

fn map_list_resource_templates_error(error: ServiceError) -> McpError {
    map_service_error(error, McpError::ListResourceTemplates)
}

fn map_read_resource_error(error: ServiceError) -> McpError {
    map_service_error(error, McpError::ReadResource)
}

fn map_service_error(error: ServiceError, fallback: impl FnOnce(String) -> McpError) -> McpError {
    match error {
        ServiceError::TransportClosed => McpError::Disconnected,
        other => fallback(other.to_string()),
    }
}

fn map_call_error(error: ServiceError) -> ToolExecError {
    match error {
        ServiceError::TransportClosed => McpError::Disconnected.into_tool_error(),
        other => ToolExecError::Execution(format!("MCP call failed: {other}")),
    }
}

/// Map an MCP tool result into an aquaregia tool result value.
///
/// - `is_error: true` becomes a [`ToolExecError::Execution`].
/// - a single text block collapses to a JSON string (most ergonomic for the model).
/// - anything else becomes a JSON array of typed content objects.
fn map_result(result: CallToolResult) -> Result<Value, ToolExecError> {
    if result.is_error == Some(true) {
        let text = collect_text(&result.content);
        return Err(ToolExecError::Execution(if text.is_empty() {
            "MCP tool returned an error".to_string()
        } else {
            text
        }));
    }

    if let [ContentBlock::Text(text)] = result.content.as_slice() {
        return Ok(Value::String(text.text.clone()));
    }

    let mut parts = Vec::with_capacity(result.content.len());
    for item in result.content {
        parts.push(content_to_value(item)?);
    }
    Ok(Value::Array(parts))
}

fn collect_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn content_to_value(raw: ContentBlock) -> Result<Value, ToolExecError> {
    Ok(match raw {
        ContentBlock::Text(t) => json!({ "type": "text", "text": t.text }),
        ContentBlock::Image(img) => json!({
            "type": "image",
            "mime_type": img.mime_type,
            "data": img.data,
        }),
        ContentBlock::Resource(res) => match res.resource {
            ResourceContents::TextResourceContents {
                uri,
                mime_type,
                text,
                ..
            } => json!({
                "type": "resource",
                "uri": uri,
                "mime_type": mime_type,
                "text": text,
            }),
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } => json!({
                "type": "resource",
                "uri": uri,
                "mime_type": mime_type,
                "blob": blob,
            }),
            _ => {
                return Err(ToolExecError::Execution(
                    "unsupported MCP resource content".to_string(),
                ));
            }
        },
        ContentBlock::ResourceLink(resource) => json!({
            "type": "resource_link",
            "uri": resource.uri,
            "name": resource.name,
            "title": resource.title,
            "description": resource.description,
            "mime_type": resource.mime_type,
            "size": resource.size,
        }),
        ContentBlock::Audio(_) => {
            return Err(ToolExecError::Execution(
                "unsupported MCP content: audio".to_string(),
            ));
        }
        _ => {
            return Err(ToolExecError::Execution(
                "unsupported MCP content".to_string(),
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use rmcp::model::{CallToolResult, ContentBlock, Resource, ResourceContents};
    use serde_json::json;

    use super::{
        Meta, build_call_request, build_get_prompt_request, build_paginated_request,
        build_read_resource_request, map_result,
    };

    #[test]
    fn build_call_request_attaches_meta_outside_arguments() {
        let mut meta = Meta::new();
        meta.0
            .insert("authorization".to_string(), json!("Bearer token"));

        let request = build_call_request("search", json!({ "query": "rust" }), Some(meta.clone()))
            .expect("request should build");

        assert_eq!(request.meta, Some(meta));
        assert_eq!(
            request.arguments.expect("arguments should be set")["query"],
            "rust"
        );
    }

    #[test]
    fn build_get_prompt_request_attaches_meta_and_arguments() {
        let mut meta = Meta::new();
        meta.0.insert("session".to_string(), json!("abc"));

        let request = build_get_prompt_request(
            "summarize",
            Some(json!({ "topic": "rust" })),
            Some(meta.clone()),
        )
        .expect("request should build");

        assert_eq!(request.meta, Some(meta));
        assert_eq!(request.name, "summarize");
        assert_eq!(
            request.arguments.expect("arguments should be set")["topic"],
            "rust"
        );
    }

    #[test]
    fn build_get_prompt_request_rejects_non_object_arguments() {
        let err = build_get_prompt_request("summarize", Some(json!("rust")), None)
            .expect_err("string arguments should be rejected");

        assert_eq!(
            err.to_string(),
            "MCP prompt arguments must be a JSON object"
        );
    }

    #[test]
    fn build_paginated_request_attaches_meta_and_cursor() {
        let mut meta = Meta::new();
        meta.0.insert("session".to_string(), json!("abc"));

        let request = build_paginated_request(Some("next".to_string()), Some(meta.clone()));

        assert_eq!(request.meta, Some(meta));
        assert_eq!(request.cursor.as_deref(), Some("next"));
    }

    #[test]
    fn build_read_resource_request_attaches_meta() {
        let mut meta = Meta::new();
        meta.0.insert("session".to_string(), json!("abc"));

        let request = build_read_resource_request("file:///note.txt", Some(meta.clone()));

        assert_eq!(request.meta, Some(meta));
        assert_eq!(request.uri, "file:///note.txt");
    }

    #[test]
    fn maps_single_text_to_json_string() {
        let value = map_result(CallToolResult::success(vec![ContentBlock::text("ok")]))
            .expect("text result should map");

        assert_eq!(value, json!("ok"));
    }

    #[test]
    fn maps_multiple_blocks_to_typed_array() {
        let value = map_result(CallToolResult::success(vec![
            ContentBlock::text("first"),
            ContentBlock::image("aW1n", "image/png"),
            ContentBlock::resource(ResourceContents::TextResourceContents {
                uri: "file:///note.txt".to_string(),
                mime_type: Some("text/plain".to_string()),
                text: "note".to_string(),
                meta: None,
            }),
        ]))
        .expect("mixed result should map");

        assert_eq!(
            value,
            json!([
                { "type": "text", "text": "first" },
                { "type": "image", "mime_type": "image/png", "data": "aW1n" },
                {
                    "type": "resource",
                    "uri": "file:///note.txt",
                    "mime_type": "text/plain",
                    "text": "note",
                },
            ])
        );
    }

    #[test]
    fn maps_resource_link_to_typed_object() {
        let value = map_result(CallToolResult::success(vec![ContentBlock::resource_link(
            Resource::new("file:///note.txt", "note").with_mime_type("text/plain"),
        )]))
        .expect("resource link should map");

        assert_eq!(
            value,
            json!([
                {
                    "type": "resource_link",
                    "uri": "file:///note.txt",
                    "name": "note",
                    "title": null,
                    "description": null,
                    "mime_type": "text/plain",
                    "size": null,
                },
            ])
        );
    }

    #[test]
    fn maps_mcp_error_result_to_tool_error() {
        let err = map_result(CallToolResult::error(vec![
            ContentBlock::text("first failure"),
            ContentBlock::text("second failure"),
        ]))
        .expect_err("MCP error result should fail");

        assert_eq!(
            err.to_string(),
            "execution failed: first failure\nsecond failure"
        );
    }

    #[test]
    fn rejects_audio_content() {
        let err = map_result(CallToolResult::success(vec![ContentBlock::audio(
            "YXVkaW8=",
            "audio/wav",
        )]))
        .expect_err("audio should be unsupported in the first MCP bridge version");

        assert_eq!(
            err.to_string(),
            "execution failed: unsupported MCP content: audio"
        );
    }
}
