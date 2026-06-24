//! Live end-to-end: orchestrated turn (memory+profile+invariants+MCP) and the
//! travel-weather multi-agent flow. Needs DEEPSEEK_API_KEY + the live MCP.
//! Run: `cargo test --test live_orchestrator -- --ignored --nocapture`

use std::sync::Arc;

use tg_agent::{
    agent::{self, session::ChatSession},
    config::LlmConfig,
    llm::Llm,
    mcp_client::ConnectParams,
    state::BotState,
};

async fn setup() -> (Arc<Llm>, BotState) {
    let api_key = std::env::var("LLM_API_KEY")
        .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
        .expect("set LLM_API_KEY");
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into());
    std::env::set_var(
        "STATE_FILE",
        std::env::temp_dir().join("tg_orch_state.json"),
    );
    std::env::set_var(
        "SESSIONS_DIR",
        std::env::temp_dir().join("tg_orch_sessions"),
    );

    let llm = Arc::new(Llm::new(LlmConfig {
        api_key,
        base_url: "https://api.deepseek.com".into(),
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
    (llm, state)
}

#[tokio::test]
#[ignore]
async fn orchestrated_turn_uses_profile_and_invariants() {
    let (llm, state) = setup().await;
    let mut session = ChatSession::new(101);
    session.profile.set("home_city", "Волгоград");

    let result = agent::run_turn(
        &llm,
        &state,
        &mut session,
        "Какая погода у меня дома? Кратко.",
    )
    .await
    .expect("turn");

    println!("\n=== ANSWER ===\n{}\n", result.answer);
    println!(
        "facts_learned={}, invariant={:?}",
        result.facts_learned, result.invariant_status
    );

    // invariant: must contain a number (temperature)
    assert!(
        result.answer.chars().any(|c| c.is_ascii_digit()),
        "answer has no number: {}",
        result.answer
    );
    assert_ne!(
        result.invariant_status,
        agent::invariants::InvariantStatus::Failed
    );
}

#[tokio::test]
#[ignore]
async fn agent_self_subscribes_on_collect_request() {
    let (llm, state) = setup().await;
    // a bot handle is required for start_watch; use a dummy token (no network call
    // happens because the watch's first tick is one interval away).
    state.set_bot(teloxide::Bot::new("123:dummy")).await;

    let mut session = ChatSession::new(303);
    let result = agent::run_turn(
        &llm,
        &state,
        &mut session,
        "Собирай погоду в Волгограде каждые 2 минуты и присылай мне сводку.",
    )
    .await
    .expect("turn");

    println!("\n=== ANSWER ===\n{}\n", result.answer);
    let watches = state.list_watches().await;
    println!("watches registered: {watches:?}");
    assert!(
        !watches.is_empty(),
        "agent did not self-subscribe via schedule_summary"
    );
}

#[tokio::test]
#[ignore]
async fn travel_flow_pipeline_runs() {
    let (llm, state) = setup().await;
    let session = ChatSession::new(202);

    let report = agent::flow::run(&llm, &state, &session, "Москва и Сочи на эти выходные")
        .await
        .expect("flow");

    println!("\n=== PLAN === {:?}", report.plan);
    for r in &report.records {
        println!("[{:?}] {}", r.stage, r.output);
    }
    println!("\n=== FINAL ===\n{}\n", report.answer);

    assert!(!report.plan.cities.is_empty(), "planner found no cities");
    assert!(
        report.answer.chars().any(|c| c.is_ascii_digit()),
        "no weather numbers"
    );
}
