//! OpenAI embedding example with dimension control.
//!
//! This example demonstrates OpenAI's embedding API with dimension reduction
//! using provider-specific options.
//!
//! ## Usage
//!
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example openai_embed
//! ```

use aquaregia::embed::EmbedRequest;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let client = aquaregia::providers::openai::Client::builder()
        .api_key(api_key)
        .build()?;

    println!("OpenAI Embedding Examples\n");

    // Example 1: Default embedding (full dimension)
    println!("1. Default embedding (text-embedding-3-small):");
    let response = client
        .embed(EmbedRequest::new(
            "text-embedding-3-small",
            vec!["The quick brown fox jumps over the lazy dog"],
        ))
        .await?;

    println!("   Dimension: {}", response.embeddings[0].len());
    println!("   Tokens: {}", response.usage.tokens);
    println!();

    // Example 2: Reduced dimension using provider options
    println!("2. Reduced dimension (256) using provider options:");
    let response = client
        .embed(
            EmbedRequest::builder("text-embedding-3-small")
                .values(vec!["The quick brown fox jumps over the lazy dog"])
                .provider_options(json!({
                    "openai": {
                        "dimensions": 256
                    }
                }))
                .build()?,
        )
        .await?;

    println!("   Dimension: {}", response.embeddings[0].len());
    println!("   Tokens: {}", response.usage.tokens);
    println!();

    // Example 3: Batch with large model
    println!("3. Batch embedding (text-embedding-3-large):");
    let texts = vec![
        "Artificial intelligence",
        "Machine learning",
        "Deep learning",
        "Neural networks",
    ];

    let response = client
        .embed(EmbedRequest::new("text-embedding-3-large", texts.clone()))
        .await?;

    println!("   Model: {}", response.model);
    println!("   Count: {}", response.embeddings.len());
    println!("   Dimension: {}", response.embeddings[0].len());
    println!("   Tokens: {}", response.usage.tokens);
    println!();

    // Calculate similarity matrix
    println!("4. Similarity matrix:");
    for (i, text_i) in texts.iter().enumerate() {
        for (j, text_j) in texts.iter().enumerate() {
            if i < j {
                let sim = cosine_similarity(&response.embeddings[i], &response.embeddings[j]);
                println!("   \"{}\" <-> \"{}\"", text_i, text_j);
                println!("   Similarity: {:.4}", sim);
                println!();
            }
        }
    }

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}
