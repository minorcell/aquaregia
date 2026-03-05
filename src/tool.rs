use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use jsonschema::{Validator, validator_for};
use regex::Regex;
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{AiError, AiErrorCode};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, args: Value) -> Result<Value, ToolExecError>;
}

#[derive(Clone)]
pub struct Tool {
    pub descriptor: ToolDescriptor,
    pub executor: Arc<dyn ToolExecutor>,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("descriptor", &self.descriptor)
            .field("executor", &"<dyn ToolExecutor>")
            .finish()
    }
}

impl Tool {
    pub fn from_parts(descriptor: ToolDescriptor, executor: Arc<dyn ToolExecutor>) -> Self {
        Self {
            descriptor,
            executor,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolExecError {
    #[error("execution failed: {0}")]
    Execution(String),
    #[error("timeout")]
    Timeout,
}

pub fn tool(name: impl Into<String>) -> ToolBuilder {
    ToolBuilder::new(name)
}

pub trait IntoTool {
    fn into_tool(self) -> Tool;
}

impl IntoTool for Tool {
    fn into_tool(self) -> Tool {
        self
    }
}

impl<F> IntoTool for F
where
    F: FnOnce() -> Tool,
{
    fn into_tool(self) -> Tool {
        self()
    }
}

pub struct ToolBuilder {
    descriptor: ToolDescriptor,
}

impl ToolBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            descriptor: ToolDescriptor {
                name: name.into(),
                description: String::new(),
                input_schema: json!({
                    "type": "object",
                    "additionalProperties": true
                }),
            },
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.descriptor.description = description.into();
        self
    }

    pub fn raw_schema(mut self, input_schema: Value) -> Self {
        self.descriptor.input_schema = input_schema;
        self
    }

    pub fn execute<Args, F, Fut>(mut self, executor: F) -> Tool
    where
        Args: DeserializeOwned + JsonSchema + Send + Sync + 'static,
        F: Fn(Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, ToolExecError>> + Send + 'static,
    {
        let schema = schema_for!(Args);
        self.descriptor.input_schema =
            serde_json::to_value(schema).expect("typed tool schema should serialize to JSON");
        Tool::from_parts(
            self.descriptor,
            Arc::new(TypedFnToolExecutor {
                executor,
                _args: PhantomData,
            }),
        )
    }

    pub fn execute_raw<F, Fut>(self, executor: F) -> Tool
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, ToolExecError>> + Send + 'static,
    {
        Tool::from_parts(self.descriptor, Arc::new(RawFnToolExecutor { executor }))
    }
}

struct RawFnToolExecutor<F> {
    executor: F,
}

#[async_trait]
impl<F, Fut> ToolExecutor for RawFnToolExecutor<F>
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Value, ToolExecError>> + Send + 'static,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolExecError> {
        (self.executor)(args).await
    }
}

struct TypedFnToolExecutor<F, Args> {
    executor: F,
    _args: PhantomData<fn() -> Args>,
}

#[async_trait]
impl<F, Fut, Args> ToolExecutor for TypedFnToolExecutor<F, Args>
where
    Args: DeserializeOwned + Send + Sync + 'static,
    F: Fn(Args) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Value, ToolExecError>> + Send + 'static,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolExecError> {
        let typed = serde_json::from_value::<Args>(args).map_err(|e| {
            ToolExecError::Execution(format!("tool args deserialization failed: {}", e))
        })?;
        (self.executor)(typed).await
    }
}

pub(crate) struct RegisteredTool {
    pub tool: Tool,
    pub validator: Validator,
}

pub struct ToolRegistry {
    entries: HashMap<String, RegisteredTool>,
}

impl ToolRegistry {
    pub fn from_tools(tools: Vec<Tool>) -> Result<Self, AiError> {
        let mut entries = HashMap::new();
        let name_re = Regex::new(r"^[a-zA-Z0-9_-]{1,64}$")
            .expect("tool name regex must be valid at compile time");

        for tool in tools {
            let name = tool.descriptor.name.clone();
            if !name_re.is_match(&name) {
                return Err(AiError::new(
                    AiErrorCode::InvalidRequest,
                    format!("invalid tool name `{}`", name),
                ));
            }
            if entries.contains_key(&name) {
                return Err(AiError::new(
                    AiErrorCode::InvalidRequest,
                    format!("duplicate tool name `{}`", name),
                ));
            }

            let validator = validator_for(&tool.descriptor.input_schema).map_err(|e| {
                AiError::new(
                    AiErrorCode::InvalidRequest,
                    format!("invalid JSON schema for tool `{}`: {}", name, e),
                )
            })?;

            entries.insert(name, RegisteredTool { tool, validator });
        }

        Ok(Self { entries })
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.entries.get(name).map(|entry| &entry.tool)
    }

    pub fn names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    pub(crate) fn resolve(&self, name: &str) -> Option<&RegisteredTool> {
        self.entries.get(name)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use serde_json::json;

    use super::{Tool, ToolDescriptor, ToolExecError, ToolExecutor, ToolRegistry, tool};

    struct EchoTool;

    #[async_trait]
    impl ToolExecutor for EchoTool {
        async fn execute(
            &self,
            args: serde_json::Value,
        ) -> Result<serde_json::Value, ToolExecError> {
            Ok(args)
        }
    }

    fn make_tool(name: &str) -> Tool {
        Tool {
            descriptor: ToolDescriptor {
                name: name.to_string(),
                description: "echo".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "x": { "type": "number" } },
                    "required": ["x"]
                }),
            },
            executor: Arc::new(EchoTool),
        }
    }

    #[test]
    fn rejects_duplicate_tool_names() {
        let tools = vec![make_tool("echo"), make_tool("echo")];
        let result = ToolRegistry::from_tools(tools);
        assert!(result.is_err());
    }

    #[test]
    fn compiles_and_validates_schema() {
        let registry = ToolRegistry::from_tools(vec![make_tool("echo")]).unwrap();
        let entry = registry.resolve("echo").unwrap();
        assert!(entry.validator.validate(&json!({"x": 1})).is_ok());
        assert!(entry.validator.validate(&json!({"y": 1})).is_err());
    }

    #[tokio::test]
    async fn tool_builder_executes_closure() {
        #[derive(Debug, Deserialize, JsonSchema)]
        struct EchoArgs {
            x: String,
        }

        let echo_tool = tool("echo")
            .description("Echo input")
            .execute(|args: EchoArgs| async move { Ok(json!({ "x": args.x })) });

        let output = echo_tool
            .executor
            .execute(json!({ "x": "ok" }))
            .await
            .expect("tool should execute");
        assert_eq!(output, json!({ "x": "ok" }));
    }

    #[tokio::test]
    async fn raw_tool_builder_executes_closure() {
        let echo_tool = tool("echo")
            .description("Echo input")
            .raw_schema(json!({
                "type": "object",
                "properties": {
                    "x": { "type": "string" }
                },
                "required": ["x"]
            }))
            .execute_raw(|args| async move { Ok(args) });

        let output = echo_tool
            .executor
            .execute(json!({ "x": "ok" }))
            .await
            .expect("tool should execute");
        assert_eq!(output, json!({ "x": "ok" }));
    }
}
