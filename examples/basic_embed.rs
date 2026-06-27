//! Basic embedding generation example.
//!
//! This example demonstrates how to generate text embeddings using OpenAI.
//!
//! ## Usage
//!
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example basic_embed
//! ```

use aquaregia::embed::EmbedRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = aquaregia::providers::openai::Client::from_env()?;

    println!("Generating embeddings...\n");

    // Single text embedding
    let response = client
        .embed(EmbedRequest::new(
            "text-embedding-3-small",
            vec!["Hello, world!"],
        ))
        .await?;

    println!("Single embedding:");
    println!("  Model: {}", response.model);
    println!("  Dimension: {}", response.embeddings[0].len());
    println!("  First 5 values: {:?}", &response.embeddings[0][..5]);
    println!("  Tokens used: {}", response.usage.tokens);
    println!();

    // Batch embeddings
    let texts = vec![
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Rust is a systems programming language",
    ];

    let response = client
        .embed(EmbedRequest::new("text-embedding-3-small", texts.clone()))
        .await?;

    println!("Batch embeddings:");
    println!("  Model: {}", response.model);
    println!("  Count: {}", response.embeddings.len());
    println!("  Tokens used: {}", response.usage.tokens);
    println!();

    for (i, text) in texts.iter().enumerate() {
        println!("  [{}] \"{}\"", i, text);
        println!("      Dimension: {}", response.embeddings[i].len());
        println!("      First 3 values: {:?}", &response.embeddings[i][..3]);
    }

    // Calculate cosine similarity between first two embeddings
    let similarity = cosine_similarity(&response.embeddings[0], &response.embeddings[1]);
    println!();
    println!("Cosine similarity between text 0 and 1: {:.4}", similarity);

    Ok(())
}

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}
