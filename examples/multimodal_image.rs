//! Demonstrates sending an image URL to OpenAI and asking about its content.
//!
//! Run with:
//!   OPENAI_API_KEY=<key> cargo run --example multimodal_image

use aquaregia::{ChatRequest, ContentPart, FilePart, MediaData, Message, MessageRole, TextPart};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = aquaregia::providers::openai::Client::from_env()?;

    let message = Message::new(
        MessageRole::User,
        vec![
            ContentPart::Text(TextPart::new("What's in this image?")),
            ContentPart::File(FilePart::new(
                MediaData::Url(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
                        .into(),
                ),
                "image/jpeg",
            )),
        ],
    );

    let response = client
        .generate(ChatRequest::builder("gpt-5.5").message(message).build()?)
        .await?;

    println!("{}", response.output_text);

    // Example: image from raw bytes (e.g. read from a file).
    let _bytes_example = Message::user_file_bytes(vec![/* raw JPEG bytes */], "image/jpeg");

    // Example: image from URL (mediaType is now mandatory and must match the
    // resource — adapters no longer guess "image/jpeg" when it is omitted).
    let _url_example = Message::user_file_url("https://example.com/diagram.png", "image/png");

    // Example: explicit FilePart with base64 data and an attached filename.
    let _explicit_example = ContentPart::File(
        FilePart::new(
            MediaData::Base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".into()),
            "image/png",
        )
        .with_filename("dot.png"),
    );

    Ok(())
}
