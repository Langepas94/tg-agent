//! Live integration test against a real MCP endpoint.
//! Ignored by default (needs network + a running server).
//! Run: `cargo test --test live_connect -- --ignored --nocapture`

use tg_agent::mcp_client::{ConnectParams, McpClient};

#[tokio::test]
#[ignore]
async fn connects_and_lists_tools() {
    let (tx, _rx) = tokio::sync::broadcast::channel(8);
    let params = ConnectParams {
        name: "open-meteo".into(),
        url: "http://5.129.234.9:3000/mcp".into(),
        auth: None,
        headers: vec![],
    };
    let client = McpClient::connect(params, tx)
        .await
        .expect("connect should succeed");
    let tools = client.tools().await;
    println!("got {} tools:", tools.len());
    for t in &tools {
        println!("  - {}", t.name);
    }
    assert!(!tools.is_empty(), "expected at least one tool");
}
