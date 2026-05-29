//! Demonstrates sending a PDF to Claude via the unified `FilePart` interface.
//!
//! Aquaregia dispatches on the IANA media_type: `application/pdf` becomes an
//! Anthropic `document` block (or an OpenAI `input_file` block, depending on
//! the provider), without any extra type needed on the caller side.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=<key> PDF_PATH=<path-to-pdf> cargo run --example multimodal_pdf

use std::fs;
use std::path::PathBuf;

use aquaregia::{
    ContentPart, FilePart, GenerateTextRequest, LlmClient, MediaData, Message, MessageRole,
    TextPart,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pdf_path = PathBuf::from(std::env::var("PDF_PATH")?);
    let pdf_bytes = fs::read(&pdf_path)?;

    let filename = pdf_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("document.pdf")
        .to_string();

    let pdf_part =
        FilePart::new(MediaData::Bytes(pdf_bytes), "application/pdf").with_filename(filename);

    let message = Message::new(
        MessageRole::User,
        vec![
            ContentPart::Text(TextPart::new("Summarise this document in 5 bullets.")),
            ContentPart::File(pdf_part),
        ],
    )?;

    let client = LlmClient::anthropic()
        .api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .build()?;

    let response = client
        .generate(
            GenerateTextRequest::builder("claude-sonnet-4-6")
                .message(message)
                .max_output_tokens(1024)
                .build()?,
        )
        .await?;

    println!("{}", response.output_text);
    Ok(())
}
