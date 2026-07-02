//! Provider clients for Aquaregia.
//!
//! Provider modules are the public entry point for building clients:
//!
//! ```rust,no_run
//! use aquaregia::providers::openai;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = openai::Client::from_env()?
//!     .agent("gpt-5.5")
//!     .build()?;
//!
//! let text = agent.prompt("Who are you?").await?;
//! println!("{text}");
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::Duration;

use crate::agent::AgentBuilder;
use crate::client as core;
use crate::embed::{EmbedRequest, EmbedResponse};
use crate::error::{Error, ErrorCode};
use crate::types::{ChatRequest, ChatResponse, ObjectResponse, ObjectStream, TextStream};

#[derive(Clone)]
enum ApiKeySource {
    Missing,
    Value(String),
    Env(String),
}

impl ApiKeySource {
    fn resolve_required(&self) -> Result<String, Error> {
        match self {
            Self::Missing => Err(Error::new(
                ErrorCode::AuthFailed,
                "api key must be set with api_key(...) or api_key_from_env(...)",
            )),
            Self::Value(value) => validate_api_key("api_key", value),
            Self::Env(name) => {
                let value = std::env::var(name).map_err(|_| {
                    Error::new(
                        ErrorCode::AuthFailed,
                        format!("{name} environment variable must be set"),
                    )
                })?;
                validate_api_key(name, value)
            }
        }
    }

    fn resolve_optional(&self) -> Result<Option<String>, Error> {
        match self {
            Self::Missing => Ok(None),
            Self::Value(value) => validate_api_key("api_key", value).map(Some),
            Self::Env(name) => {
                let value = std::env::var(name).map_err(|_| {
                    Error::new(
                        ErrorCode::AuthFailed,
                        format!("{name} environment variable must be set"),
                    )
                })?;
                validate_api_key(name, value).map(Some)
            }
        }
    }
}

fn validate_api_key(label: &str, value: impl Into<String>) -> Result<String, Error> {
    let value = value.into();
    if value.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::AuthFailed,
            format!("{label} must not be empty"),
        ));
    }
    Ok(value)
}

#[derive(Clone)]
struct RuntimeConfig {
    timeout: Duration,
    max_retries: u8,
    default_max_steps: u32,
    user_agent: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            default_max_steps: 0,
            user_agent: None,
        }
    }
}

macro_rules! runtime_setters {
    () => {
        /// Sets request timeout for all requests sent by this client.
        pub fn timeout(mut self, timeout: Duration) -> Self {
            self.runtime.timeout = timeout;
            self
        }

        /// Sets the maximum number of retries for retryable errors.
        pub fn max_retries(mut self, retries: u8) -> Self {
            self.runtime.max_retries = retries;
            self
        }

        /// Sets the default max step count used by agent tool loops.
        ///
        /// `0` means unlimited.
        pub fn default_max_steps(mut self, max_steps: u32) -> Self {
            self.runtime.default_max_steps = max_steps;
            self
        }

        /// Overrides the default Aquaregia `User-Agent` header value.
        pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
            self.runtime.user_agent = Some(user_agent.into());
            self
        }
    };
}

macro_rules! provider_client_methods {
    () => {
        /// Starts building a model-bound agent.
        pub fn agent(&self, model: impl Into<String>) -> AgentBuilder {
            crate::agent::Agent::builder(Arc::clone(&self.inner), model)
        }

        /// Runs a non-streaming generation request.
        pub async fn generate(&self, req: ChatRequest) -> Result<ChatResponse, Error> {
            self.inner.generate(req).await
        }

        /// Runs a streaming generation request.
        pub async fn stream(&self, req: ChatRequest) -> Result<TextStream, Error> {
            self.inner.stream(req).await
        }

        /// Generates embeddings for text values.
        pub async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse, Error> {
            self.inner.embed(req).await
        }

        /// Performs a non-streaming generation that returns deserialized structured output.
        pub async fn generate_object<T: serde::de::DeserializeOwned + schemars::JsonSchema>(
            &self,
            req: ChatRequest,
        ) -> Result<ObjectResponse<T>, Error> {
            self.inner.generate_object(req).await
        }

        /// Streams a generation that emits progressively-populated structured output.
        pub async fn stream_object<
            T: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static,
        >(
            &self,
            req: ChatRequest,
        ) -> Result<ObjectStream<T>, Error> {
            self.inner.stream_object(req).await
        }
    };
}

fn apply_runtime<S: core::BuildProvider>(
    mut builder: core::ClientBuilder<S>,
    runtime: RuntimeConfig,
) -> core::ClientBuilder<S> {
    builder = builder
        .timeout(runtime.timeout)
        .max_retries(runtime.max_retries)
        .default_max_steps(runtime.default_max_steps);
    if let Some(user_agent) = runtime.user_agent {
        builder = builder.user_agent(user_agent);
    }
    builder
}

/// OpenAI provider client.
pub mod openai {
    use super::*;

    /// OpenAI client.
    #[derive(Clone)]
    pub struct Client {
        pub(crate) inner: Arc<core::Client>,
    }

    impl Client {
        /// Creates a builder for an OpenAI client.
        pub fn builder() -> Builder {
            Builder {
                api_key: ApiKeySource::Missing,
                base_url: crate::adapters::openai::DEFAULT_BASE_URL.to_string(),
                runtime: RuntimeConfig::default(),
            }
        }

        /// Builds an OpenAI client from `OPENAI_API_KEY`.
        pub fn from_env() -> Result<Self, Error> {
            Self::builder().api_key_from_env("OPENAI_API_KEY").build()
        }

        provider_client_methods!();
    }

    /// Configures an OpenAI client.
    pub struct Builder {
        api_key: ApiKeySource,
        base_url: String,
        runtime: RuntimeConfig,
    }

    impl Builder {
        /// Sets the OpenAI API key.
        pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
            self.api_key = ApiKeySource::Value(api_key.into());
            self
        }

        /// Reads the OpenAI API key from an environment variable during `build()`.
        pub fn api_key_from_env(mut self, env_var: impl Into<String>) -> Self {
            self.api_key = ApiKeySource::Env(env_var.into());
            self
        }

        /// Overrides the OpenAI API base URL.
        pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }

        runtime_setters!();

        /// Builds the client.
        pub fn build(self) -> Result<Client, Error> {
            let core = apply_runtime(
                core::Client::openai()
                    .api_key(self.api_key.resolve_required()?)
                    .base_url(self.base_url),
                self.runtime,
            )
            .build()?;
            Ok(Client {
                inner: Arc::new(core),
            })
        }
    }
}

/// Anthropic provider client.
pub mod anthropic {
    use super::*;

    /// Anthropic client.
    #[derive(Clone)]
    pub struct Client {
        pub(crate) inner: Arc<core::Client>,
    }

    impl Client {
        /// Creates a builder for an Anthropic client.
        pub fn builder() -> Builder {
            Builder {
                api_key: ApiKeySource::Missing,
                base_url: crate::adapters::anthropic::DEFAULT_BASE_URL.to_string(),
                api_version: crate::adapters::anthropic::DEFAULT_API_VERSION.to_string(),
                runtime: RuntimeConfig::default(),
            }
        }

        /// Builds an Anthropic client from `ANTHROPIC_API_KEY`.
        pub fn from_env() -> Result<Self, Error> {
            Self::builder()
                .api_key_from_env("ANTHROPIC_API_KEY")
                .build()
        }

        provider_client_methods!();
    }

    /// Configures an Anthropic client.
    pub struct Builder {
        api_key: ApiKeySource,
        base_url: String,
        api_version: String,
        runtime: RuntimeConfig,
    }

    impl Builder {
        /// Sets the Anthropic API key.
        pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
            self.api_key = ApiKeySource::Value(api_key.into());
            self
        }

        /// Reads the Anthropic API key from an environment variable during `build()`.
        pub fn api_key_from_env(mut self, env_var: impl Into<String>) -> Self {
            self.api_key = ApiKeySource::Env(env_var.into());
            self
        }

        /// Overrides the Anthropic API base URL.
        pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }

        /// Overrides the Anthropic API version header.
        pub fn api_version(mut self, api_version: impl Into<String>) -> Self {
            self.api_version = api_version.into();
            self
        }

        runtime_setters!();

        /// Builds the client.
        pub fn build(self) -> Result<Client, Error> {
            let core = apply_runtime(
                core::Client::anthropic()
                    .api_key(self.api_key.resolve_required()?)
                    .base_url(self.base_url)
                    .api_version(self.api_version),
                self.runtime,
            )
            .build()?;
            Ok(Client {
                inner: Arc::new(core),
            })
        }
    }
}

/// Google provider client.
pub mod google {
    use super::*;

    /// Google client.
    #[derive(Clone)]
    pub struct Client {
        pub(crate) inner: Arc<core::Client>,
    }

    impl Client {
        /// Creates a builder for a Google client.
        pub fn builder() -> Builder {
            Builder {
                api_key: ApiKeySource::Missing,
                base_url: crate::adapters::google::DEFAULT_BASE_URL.to_string(),
                runtime: RuntimeConfig::default(),
            }
        }

        /// Builds a Google client from `GOOGLE_API_KEY`.
        pub fn from_env() -> Result<Self, Error> {
            Self::builder().api_key_from_env("GOOGLE_API_KEY").build()
        }

        provider_client_methods!();
    }

    /// Configures a Google client.
    pub struct Builder {
        api_key: ApiKeySource,
        base_url: String,
        runtime: RuntimeConfig,
    }

    impl Builder {
        /// Sets the Google API key.
        pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
            self.api_key = ApiKeySource::Value(api_key.into());
            self
        }

        /// Reads the Google API key from an environment variable during `build()`.
        pub fn api_key_from_env(mut self, env_var: impl Into<String>) -> Self {
            self.api_key = ApiKeySource::Env(env_var.into());
            self
        }

        /// Overrides the Google Generative Language API base URL.
        pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }

        runtime_setters!();

        /// Builds the client.
        pub fn build(self) -> Result<Client, Error> {
            let core = apply_runtime(
                core::Client::google()
                    .api_key(self.api_key.resolve_required()?)
                    .base_url(self.base_url),
                self.runtime,
            )
            .build()?;
            Ok(Client {
                inner: Arc::new(core),
            })
        }
    }
}

/// OpenAI-compatible provider client.
pub mod openai_compatible {
    use super::*;

    /// OpenAI-compatible client.
    #[derive(Clone)]
    pub struct Client {
        pub(crate) inner: Arc<core::Client>,
    }

    impl Client {
        /// Creates a builder for an OpenAI-compatible client.
        pub fn builder() -> Builder {
            Builder {
                base_url: String::new(),
                api_key: ApiKeySource::Missing,
                headers: Vec::new(),
                query_params: Vec::new(),
                chat_completions_path: crate::adapters::openai_compatible::DEFAULT_PATH.to_string(),
                runtime: RuntimeConfig::default(),
            }
        }

        /// Builds an OpenAI-compatible client from environment variables.
        ///
        /// Reads `OPENAI_COMPATIBLE_BASE_URL`. If `OPENAI_COMPATIBLE_API_KEY`
        /// is set, it is sent as a bearer token; otherwise requests are sent
        /// without an `Authorization` header.
        pub fn from_env() -> Result<Self, Error> {
            let base_url = std::env::var("OPENAI_COMPATIBLE_BASE_URL").map_err(|_| {
                Error::new(
                    ErrorCode::InvalidRequest,
                    "OPENAI_COMPATIBLE_BASE_URL environment variable must be set",
                )
            })?;

            let mut builder = Self::builder().base_url(base_url);
            if let Ok(api_key) = std::env::var("OPENAI_COMPATIBLE_API_KEY") {
                builder = builder.api_key(api_key);
            } else {
                builder = builder.no_api_key();
            }
            builder.build()
        }

        provider_client_methods!();
    }

    /// Configures an OpenAI-compatible client.
    pub struct Builder {
        base_url: String,
        api_key: ApiKeySource,
        headers: Vec<(String, String)>,
        query_params: Vec<(String, String)>,
        chat_completions_path: String,
        runtime: RuntimeConfig,
    }

    impl Builder {
        /// Sets the OpenAI-compatible endpoint base URL.
        pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }

        /// Sets a bearer token for OpenAI-compatible requests.
        pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
            self.api_key = ApiKeySource::Value(api_key.into());
            self
        }

        /// Reads the bearer token from an environment variable during `build()`.
        pub fn api_key_from_env(mut self, env_var: impl Into<String>) -> Self {
            self.api_key = ApiKeySource::Env(env_var.into());
            self
        }

        /// Sends requests without an `Authorization` bearer token.
        pub fn no_api_key(mut self) -> Self {
            self.api_key = ApiKeySource::Missing;
            self
        }

        /// Adds or replaces a custom HTTP header.
        pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
            self.headers.push((name.into(), value.into()));
            self
        }

        /// Adds or replaces a query parameter on the chat completions endpoint.
        pub fn query_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
            self.query_params.push((name.into(), value.into()));
            self
        }

        /// Overrides the chat completions path.
        pub fn chat_completions_path(mut self, path: impl Into<String>) -> Self {
            self.chat_completions_path = path.into();
            self
        }

        runtime_setters!();

        /// Builds the client.
        pub fn build(self) -> Result<Client, Error> {
            let mut builder = core::Client::openai_compatible()
                .base_url(self.base_url)
                .chat_completions_path(self.chat_completions_path);
            if let Some(api_key) = self.api_key.resolve_optional()? {
                builder = builder.api_key(api_key);
            } else {
                builder = builder.no_api_key();
            }
            for (name, value) in self.headers {
                builder = builder.header(name, value);
            }
            for (name, value) in self.query_params {
                builder = builder.query_param(name, value);
            }

            let core = apply_runtime(builder, self.runtime).build()?;
            Ok(Client {
                inner: Arc::new(core),
            })
        }
    }
}
