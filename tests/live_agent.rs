//! Live end-to-end agent test: natural-language question -> MCP tools ->
//! human answer. Needs DEEPSEEK_API_KEY/LLM_API_KEY + the live MCP.
//! Run: `cargo test --test live_agent -- --ignored --nocapture`

use std::sync::Arc;

use tg_agent::{config::LlmConfig, llm::Llm, mcp_client::ConnectParams, state::BotState};

#[tokio::test]
#[ignore]
async fn answers_weather_in_natural_language() {
    let api_key = std::env::var("LLM_API_KEY")
        .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
        .expect("set LLM_API_KEY");
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into());
    let base_url =
        std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".into());

    std::env::set_var(
        "STATE_FILE",
        std::env::temp_dir().join("tg_agent_agent_test.json"),
    );

    let llm = Arc::new(Llm::new(LlmConfig {
        api_key,
        base_url,
        model,
    }));
    let (tx, _rx) = tokio::sync::broadcast::channel(8);
    let state = BotState::with_llm(tx, Some(llm.clone()));

    state
        .connect_mcp(ConnectParams {
            name: "weather".into(),
            url: "http://5.129.234.9:3000/mcp".into(),
            auth: None,
            headers: vec![],
        })
        .await
        .expect("connect MCP");

    let answer = llm
        .answer(&state, "Какая сейчас погода в Волгограде? Кратко.")
        .await
        .expect("agent answer");

    println!("\n=== AGENT ANSWER ===\n{answer}\n====================\n");
    assert!(!answer.is_empty());
    // crude sanity: a weather answer should mention temperature or degrees
    let low = answer.to_lowercase();
    assert!(
        low.contains('°')
            || low.contains("градус")
            || low.contains("температ")
            || low.contains("°c")
            || low.chars().any(|c| c.is_ascii_digit()),
        "answer doesn't look like weather: {answer}"
    );
}
