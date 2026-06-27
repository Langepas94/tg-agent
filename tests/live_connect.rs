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
        command: vec![],
        env: vec![],
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

    // Call a real tool and print the result.
    let mut args = serde_json::Map::new();
    args.insert("name".into(), serde_json::json!("Moscow"));
    let out = client
        .call_tool("geocode", Some(args))
        .await
        .expect("geocode call should succeed");
    println!("geocode(Moscow) ->\n{out}");
    assert!(!out.is_empty(), "expected a non-empty geocode result");
}

/// Reproduces the "connect then /tools says nothing connected" report:
/// the registry must keep the connection across separate operations.
#[tokio::test]
#[ignore]
async fn registry_keeps_connection_across_calls() {
    // isolate the state file for this test
    let tmp = std::env::temp_dir().join("tg_agent_test_state.json");
    std::env::set_var("STATE_FILE", &tmp);
    let _ = std::fs::remove_file(&tmp);

    let (tx, _rx) = tokio::sync::broadcast::channel(8);
    let state = tg_agent::state::BotState::new(tx);

    let params = ConnectParams {
        name: "weather".into(),
        url: "http://5.129.234.9:3000/mcp".into(),
        auth: None,
        headers: vec![],
        command: vec![],
        env: vec![],
    };
    let n = state.connect_mcp(params).await.expect("connect");
    assert_eq!(n, 18, "expected 18 tools");

    // Separate operation — must still see the server.
    let names = state.mcp_names().await;
    assert_eq!(
        names,
        vec!["weather".to_string()],
        "registry lost the server!"
    );

    // Persistence must have written the server to disk.
    let saved = tg_agent::persist::load();
    assert_eq!(saved.servers.len(), 1, "server not persisted to disk");
    assert_eq!(saved.servers[0].name, "weather");

    // Simulate a restart: fresh state, reload from disk, reconnect.
    drop(state);
    let (tx2, _rx2) = tokio::sync::broadcast::channel(8);
    let restored = tg_agent::state::BotState::new(tx2);
    for params in tg_agent::persist::load().servers {
        restored
            .connect_mcp(params)
            .await
            .expect("reconnect on restart");
    }
    assert_eq!(
        restored.mcp_names().await,
        vec!["weather".to_string()],
        "restart did not restore the connection"
    );

    let _ = std::fs::remove_file(&tmp);
}
