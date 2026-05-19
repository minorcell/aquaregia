use aquaregia::tool;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

fn macro_weather() -> aquaregia::Tool {
    tool("macro_weather")
        .description("Echo weather args")
        .execute(|args: WeatherArgs| async move {
            Ok(json!({ "city": args.city, "temp_c": 23 }))
        })
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
