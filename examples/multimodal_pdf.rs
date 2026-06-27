//! Demonstrates sending a PDF to OpenAI via the unified `FilePart` interface.
//!
//! Aquaregia dispatches on the IANA media_type: `application/pdf` becomes an
//! the provider-specific file block without any extra type needed on the caller side.
//!
//! Run with:
//!   OPENAI_API_KEY=<key> PDF_PATH=<path-to-pdf> cargo run --example multimodal_pdf

use std::fs;
use std::path::PathBuf;

use aquaregia::{ChatRequest, ContentPart, FilePart, MediaData, Message, MessageRole, TextPart};

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
    );

    let client = aquaregia::providers::openai::Client::from_env()?;

    let response = client
        .generate(
            ChatRequest::builder("gpt-5.5")
                .message(message)
                .max_output_tokens(1024)
                .build()?,
        )
        .await?;

    println!("{}", response.output_text);
    Ok(())
}
