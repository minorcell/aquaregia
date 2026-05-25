//! Demonstrates sending an image URL to Claude and asking about its content.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=<key> cargo run --example multimodal_image

use aquaregia::{
    ContentPart, GenerateTextRequest, ImagePart, LlmClient, MediaData, Message, MessageRole,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::anthropic()
        .api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .build()?;

    let message = Message::new(
        MessageRole::User,
        vec![
            ContentPart::Text("What's in this image?".into()),
            ContentPart::Image(ImagePart {
                data: MediaData::Url(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
                        .into(),
                ),
                media_type: None,
                provider_metadata: None,
            }),
        ],
    )?;

    let response = client
        .generate(
            GenerateTextRequest::builder("claude-sonnet-4-5")
                .message(message)
                .build()?,
        )
        .await?;

    println!("{}", response.output_text);

    // Example: image from raw bytes (e.g. read from a file)
    let _image_bytes_example = Message::user_image_bytes(vec![/* raw JPEG bytes */], "image/jpeg");

    // Example: image from base64
    let _image_b64_example = Message::user_image_url(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    );

    // Example: explicit ImagePart with base64
    let _explicit_example = ContentPart::Image(ImagePart {
        data: MediaData::Base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".to_string()),
        media_type: Some("image/png".to_string()),
        provider_metadata: None,
    });

    Ok(())
}
