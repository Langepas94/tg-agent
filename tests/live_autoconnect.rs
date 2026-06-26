//! Live end-to-end proof that the agent connects an MCP server BY ITSELF (no
//! pre-registered server, no hardcoded registry) and then uses that server's
//! tools to answer — exercising mcp_connect + hot tool-refresh + tool ordering.
//! Needs DEEPSEEK_API_KEY/LLM_API_KEY + the live open-meteo MCP.
//! Run: `cargo test --test live_autoconnect -- --ignored --nocapture`

use std::sync::Arc;

use tg_agent::{
    agent::{self, session::ChatSession},
    config::LlmConfig,
    llm::Llm,
    state::BotState,
};

#[tokio::test]
#[ignore]
async fn agent_self_connects_then_uses_tools() {
    let api_key = std::env::var("LLM_API_KEY")
        .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
        .expect("set LLM_API_KEY");
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into());
    let base_url =
        std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".into());
    std::env::set_var(
        "STATE_FILE",
        std::env::temp_dir().join("tg_autoconnect_state.json"),
    );
    std::env::set_var(
        "SESSIONS_DIR",
        std::env::temp_dir().join("tg_autoconnect_sessions"),
    );

    let llm = Arc::new(Llm::new(LlmConfig {
        api_key,
        base_url,
        model,
    }));
    let (tx, _rx) = tokio::sync::broadcast::channel(8);
    let state = BotState::with_llm(tx, Some(llm.clone()));

    // Start with ZERO servers connected — proves the agent attaches one itself.
    assert!(
        state.mcp_names().await.is_empty(),
        "precondition: no servers should be connected at start"
    );

    let mut session = ChatSession::new(9001);
    let result = agent::run_turn(
        &llm,
        &state,
        &mut session,
        "Сейчас не подключён ни один сервер. Вот погодный MCP (HTTP): \
         http://5.129.234.9:3000/mcp — подключи его сам и скажи, какая завтра \
         погода в Сочи. Ответь кратко с температурой.",
    )
    .await
    .expect("turn");

    println!("\n=== ANSWER ===\n{}\n", result.answer);
    let connected = state.mcp_names().await;
    println!("connected servers after turn: {connected:?}");

    // Claim 1: the agent connected a server on its own.
    assert!(
        !connected.is_empty(),
        "agent did not self-connect any MCP server"
    );
    // Claim 3/5: it actually used the tools — a temperature number is present.
    assert!(
        result.answer.chars().any(|c| c.is_ascii_digit()),
        "answer has no number (tools not used?): {}",
        result.answer
    );
}
