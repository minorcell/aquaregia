use std::sync::Arc;

use aquaregia::tool::ToolExecError;
use aquaregia::{Tool, tool};
use rmcp::model::{CallToolRequestParams, ClientInfo};
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::{
    StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
};
use serde::Serialize;
use serde_json::{Value, json};

type McpClient = RunningService<RoleClient, ClientInfo>;

#[derive(Clone)]
pub struct McpBridge {
    client: Arc<McpClient>,
    tools: Vec<Tool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

impl McpBridge {
    pub async fn connect(
        url: &str,
        token: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = StreamableHttpClientTransportConfig::with_uri(url.to_string())
            .auth_header(token.to_string());
        let transport = StreamableHttpClientTransport::from_config(config);
        let client = ClientInfo::default().serve(transport).await?;
        let client = Arc::new(client);
        let tools = bridge_all(Arc::clone(&client)).await?;
        Ok(Self { client, tools })
    }

    pub fn tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    pub fn tool_info(&self) -> Vec<ToolInfo> {
        self.tools
            .iter()
            .map(|tool| ToolInfo {
                name: tool.descriptor.name.clone(),
                description: tool.descriptor.description.clone(),
                input_schema: tool.descriptor.input_schema.clone(),
            })
            .collect()
    }

    pub async fn health(&self) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.client.list_all_tools().await?.len())
    }
}

async fn bridge_all(client: Arc<McpClient>) -> Result<Vec<Tool>, ToolExecError> {
    let remote_tools = client
        .list_all_tools()
        .await
        .map_err(|err| ToolExecError::Execution(err.to_string()))?;

    Ok(remote_tools
        .into_iter()
        .map(|remote_tool| {
            let name = remote_tool.name.to_string();
            let call_name = name.clone();
            let description = remote_tool
                .description
                .map(|description| description.to_string())
                .unwrap_or_default();
            let input_schema = Value::Object((*remote_tool.input_schema).clone());
            let client = Arc::clone(&client);

            tool(name)
                .description(description)
                .raw_schema(input_schema)
                .execute_raw(move |args| {
                    let client = Arc::clone(&client);
                    let call_name = call_name.clone();
                    async move { call_remote_tool(client, call_name, args).await }
                })
        })
        .collect())
}

async fn call_remote_tool(
    client: Arc<McpClient>,
    name: String,
    args: Value,
) -> Result<Value, ToolExecError> {
    let mut params = CallToolRequestParams::new(name);
    if let Value::Object(arguments) = args {
        params = params.with_arguments(arguments);
    }

    let result = client
        .call_tool(params)
        .await
        .map_err(|err| ToolExecError::Execution(err.to_string()))?;
    let output = result_output_value(&result);

    if result.is_error == Some(true) {
        return Err(ToolExecError::Execution(error_message(output)));
    }

    Ok(output)
}

fn result_output_value(result: &rmcp::model::CallToolResult) -> Value {
    if let Some(value) = &result.structured_content {
        return value.clone();
    }

    let text = result
        .content
        .first()
        .and_then(|content| content.as_text())
        .map(|text| text.text.clone())
        .unwrap_or_default();
    json!({ "text": text })
}

fn error_message(value: Value) -> String {
    value
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| value.get("error").and_then(Value::as_str))
        .map(ToString::to_string)
        .unwrap_or_else(|| value.to_string())
}
