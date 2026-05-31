//! Embedding generation types and APIs for Aquaregia.
//!
//! This module provides text embedding functionality following the AI SDK v3
//! embedding model interface design:
//!
//! - [`EmbedRequest`]: Request for generating embeddings
//! - [`EmbedResponse`]: Response containing generated embeddings
//! - [`EmbedUsage`]: Token usage information
//!
//! ## Example
//!
//! ```rust,no_run
//! use aquaregia::embed::EmbedRequest;
//! use aquaregia::LlmClient;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = LlmClient::openai()
//!     .api_key(std::env::var("OPENAI_API_KEY")?)
//!     .build()?;
//!
//! let response = client.embed(
//!     EmbedRequest::new("text-embedding-3-small", vec!["Hello, world!"])
//! ).await?;
//!
//! println!("Dimension: {}", response.embeddings[0].len());
//! println!("Tokens: {}", response.usage.tokens);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, ErrorCode};

/// Request for generating text embeddings.
///
/// This struct represents a request to generate vector embeddings for one or
/// more text inputs. The request is provider-agnostic and will be translated
/// to the appropriate provider-specific format by the adapter.
///
/// # Example
///
/// ```rust
/// use aquaregia::embed::EmbedRequest;
///
/// // Single text
/// let req = EmbedRequest::new("text-embedding-3-small", vec!["Hello"]);
///
/// // Multiple texts (batch)
/// let req = EmbedRequest::new(
///     "text-embedding-3-small",
///     vec!["First text", "Second text", "Third text"]
/// );
///
/// // With provider options
/// let req = EmbedRequest::builder("text-embedding-3-small")
///     .values(vec!["Hello"])
///     .provider_options(serde_json::json!({
///         "openai": { "dimensions": 256 }
///     }))
///     .build()?;
/// # Ok::<(), aquaregia::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct EmbedRequest {
    /// Model identifier (e.g., "text-embedding-3-small").
    pub model: String,
    /// List of text values to generate embeddings for.
    pub values: Vec<String>,
    /// Provider-specific options passed through to the adapter.
    ///
    /// Same JSON-by-slug shape as `GenerateTextRequest::provider_options`.
    /// Each adapter extracts its own slug and merges those fields into the
    /// request body it sends.
    ///
    /// Example:
    /// ```json
    /// {
    ///   "openai": { "dimensions": 256, "encoding_format": "float" },
    ///   "google": { "outputDimensionality": 256 }
    /// }
    /// ```
    pub provider_options: Option<Value>,
}

impl EmbedRequest {
    /// Creates a new embedding request with the given model and text values.
    ///
    /// # Arguments
    ///
    /// * `model` - Model identifier (e.g., "text-embedding-3-small")
    /// * `values` - List of text strings to embed
    ///
    /// # Example
    ///
    /// ```rust
    /// use aquaregia::embed::EmbedRequest;
    ///
    /// let req = EmbedRequest::new(
    ///     "text-embedding-3-small",
    ///     vec!["Hello", "World"]
    /// );
    /// ```
    pub fn new(model: impl Into<String>, values: Vec<impl Into<String>>) -> Self {
        Self {
            model: model.into(),
            values: values.into_iter().map(Into::into).collect(),
            provider_options: None,
        }
    }

    /// Creates a builder for constructing an embedding request.
    ///
    /// # Arguments
    ///
    /// * `model` - Model identifier
    ///
    /// # Example
    ///
    /// ```rust
    /// use aquaregia::embed::EmbedRequest;
    ///
    /// let req = EmbedRequest::builder("text-embedding-3-small")
    ///     .values(vec!["Hello"])
    ///     .provider_options(serde_json::json!({
    ///         "openai": { "dimensions": 256 }
    ///     }))
    ///     .build()?;
    /// # Ok::<(), aquaregia::Error>(())
    /// ```
    pub fn builder(model: impl Into<String>) -> EmbedRequestBuilder {
        EmbedRequestBuilder {
            model: model.into(),
            values: None,
            provider_options: None,
        }
    }
}

/// Builder for constructing [`EmbedRequest`] instances.
#[derive(Debug, Clone)]
pub struct EmbedRequestBuilder {
    model: String,
    values: Option<Vec<String>>,
    provider_options: Option<Value>,
}

impl EmbedRequestBuilder {
    /// Sets the text values to embed.
    ///
    /// # Arguments
    ///
    /// * `values` - List of text strings
    pub fn values(mut self, values: Vec<impl Into<String>>) -> Self {
        self.values = Some(values.into_iter().map(Into::into).collect());
        self
    }

    /// Sets provider-specific options.
    ///
    /// # Arguments
    ///
    /// * `options` - JSON object keyed by provider slug
    pub fn provider_options(mut self, options: Value) -> Self {
        self.provider_options = Some(options);
        self
    }

    /// Builds the [`EmbedRequest`].
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing or invalid.
    pub fn build(self) -> Result<EmbedRequest, Error> {
        let values = self
            .values
            .ok_or_else(|| Error::new(ErrorCode::InvalidRequest, "values must be provided"))?;

        if values.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                "values must not be empty",
            ));
        }

        Ok(EmbedRequest {
            model: self.model,
            values,
            provider_options: self.provider_options,
        })
    }
}

/// Response from an embedding generation request.
///
/// Contains the generated embeddings in the same order as the input values,
/// along with token usage information.
#[derive(Debug, Clone)]
pub struct EmbedResponse {
    /// Generated embeddings, in the same order as input values.
    ///
    /// Each embedding is a vector of floating-point numbers. The dimension
    /// depends on the model used.
    pub embeddings: Vec<Vec<f32>>,
    /// Model identifier that generated these embeddings.
    pub model: String,
    /// Token usage information.
    pub usage: EmbedUsage,
    /// Optional provider-specific metadata.
    ///
    /// This field carries provider-specific information that doesn't fit
    /// into the standard response structure.
    pub provider_metadata: Option<Value>,
}

/// Token usage information for embedding generation.
///
/// Unlike text generation, embeddings only consume input tokens.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmbedUsage {
    /// Number of tokens in the input text(s).
    pub tokens: u32,
}

impl EmbedUsage {
    /// Creates a new usage record.
    pub fn new(tokens: u32) -> Self {
        Self { tokens }
    }
}

/// Validates an embedding request before sending to the provider.
pub(crate) fn validate_embed_request(req: &EmbedRequest) -> Result<(), Error> {
    if req.model.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "model must not be empty",
        ));
    }

    if req.values.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "values must not be empty",
        ));
    }

    for (idx, value) in req.values.iter().enumerate() {
        if value.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                format!("value at index {} must not be empty", idx),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_request_new() {
        let req = EmbedRequest::new("model", vec!["hello", "world"]);
        assert_eq!(req.model, "model");
        assert_eq!(req.values, vec!["hello", "world"]);
        assert!(req.provider_options.is_none());
    }

    #[test]
    fn test_embed_request_builder() {
        let req = EmbedRequest::builder("model")
            .values(vec!["test"])
            .build()
            .unwrap();
        assert_eq!(req.model, "model");
        assert_eq!(req.values, vec!["test"]);
    }

    #[test]
    fn test_embed_request_builder_missing_values() {
        let result = EmbedRequest::builder("model").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_request_builder_empty_values() {
        let result = EmbedRequest::builder("model")
            .values(Vec::<String>::new())
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_embed_request_empty_model() {
        let req = EmbedRequest::new("", vec!["test"]);
        assert!(validate_embed_request(&req).is_err());
    }

    #[test]
    fn test_validate_embed_request_empty_values() {
        let req = EmbedRequest {
            model: "model".to_string(),
            values: vec![],
            provider_options: None,
        };
        assert!(validate_embed_request(&req).is_err());
    }

    #[test]
    fn test_validate_embed_request_empty_value() {
        let req = EmbedRequest::new("model", vec!["hello", "", "world"]);
        let result = validate_embed_request(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("index 1"));
    }

    #[test]
    fn test_validate_embed_request_valid() {
        let req = EmbedRequest::new("model", vec!["hello", "world"]);
        assert!(validate_embed_request(&req).is_ok());
    }

    #[test]
    fn test_embed_usage_new() {
        let usage = EmbedUsage::new(100);
        assert_eq!(usage.tokens, 100);
    }

    #[test]
    fn test_embed_response_construction() {
        let response = EmbedResponse {
            embeddings: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            model: "test-model".to_string(),
            usage: EmbedUsage::new(50),
            provider_metadata: Some(serde_json::json!({"test": "data"})),
        };
        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.model, "test-model");
        assert_eq!(response.usage.tokens, 50);
        assert!(response.provider_metadata.is_some());
    }

    #[test]
    fn test_embed_request_with_provider_options() {
        let req = EmbedRequest::builder("model")
            .values(vec!["test"])
            .provider_options(serde_json::json!({
                "openai": { "dimensions": 256 }
            }))
            .build()
            .unwrap();
        assert!(req.provider_options.is_some());
    }

    #[test]
    fn test_validate_embed_request_whitespace_model() {
        let req = EmbedRequest {
            model: "   ".to_string(),
            values: vec!["test".to_string()],
            provider_options: None,
        };
        assert!(validate_embed_request(&req).is_err());
    }
}
