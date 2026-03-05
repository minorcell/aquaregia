use aquaregia::tool;
use serde_json::{Value, json};

#[tool(description = "Echo weather args")]
async fn macro_weather(city: String) -> Result<Value, String> {
    Ok(json!({ "city": city, "temp_c": 23 }))
}

#[tokio::test]
async fn tool_macro_builds_typed_tool() {
    let tool = macro_weather();
    let output = tool
        .executor
        .execute(json!({ "city": "Shanghai" }))
        .await
        .expect("macro tool should execute");
    assert_eq!(output, json!({ "city": "Shanghai", "temp_c": 23 }));
}
