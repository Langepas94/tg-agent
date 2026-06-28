//! Live end-to-end trip-flow driver. Runs the FULL stateful swarm against the
//! real LLM + real MCP servers (weather over HTTP, maps/osmmcp over stdio) and
//! prints every turn, so the flow can be verified by actually running it instead
//! of guessing. Drives two activities — a water trip and a NON-water (cycling)
//! trip — to prove the stages are activity-agnostic (no river/water hardcode).
//!
//! Needs on the host: LLM_API_KEY (+ optional LLM_BASE_URL / LLM_MODEL), the
//! weather MCP reachable, and the osmmcp binary (OSM_BIN, default
//! `/opt/osmmcp/osmmcp`). Run on the VPS:
//!   LLM_API_KEY=… cargo test --test live_trip_flow -- --ignored --nocapture

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
    let base_url =
        std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".into());
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "deepseek-chat".into());
    std::env::set_var(
        "STATE_FILE",
        std::env::temp_dir().join("tg_trip_state.json"),
    );
    std::env::set_var(
        "SESSIONS_DIR",
        std::env::temp_dir().join("tg_trip_sessions"),
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
            url: std::env::var("WEATHER_MCP_URL")
                .unwrap_or_else(|_| "http://5.129.234.9:3000/mcp".into()),
            auth: None,
            headers: vec![],
            command: vec![],
            env: vec![],
        })
        .await
        .expect("connect weather MCP");

    let osm_bin = std::env::var("OSM_BIN").unwrap_or_else(|_| "/opt/osmmcp/osmmcp".into());
    state
        .connect_mcp(ConnectParams {
            name: "maps".into(),
            url: String::new(),
            auth: None,
            headers: vec![],
            command: vec![
                osm_bin,
                "-overpass-rps".into(),
                "2".into(),
                "-overpass-burst".into(),
                "2".into(),
            ],
            env: vec![(
                "OSM_OVERPASS_URL".into(),
                "https://maps.mail.ru/osm/tools/overpass/api/interpreter".into(),
            )],
        })
        .await
        .expect("connect maps MCP");

    (llm, state)
}

/// Feed one user message, print the swarm trace + reply, return (reply, done).
async fn step(
    llm: &Llm,
    state: &BotState,
    session: &mut ChatSession,
    user: &str,
) -> (String, bool) {
    println!("\n──────── USER ────────\n{user}");
    let r = agent::run_turn(llm, state, session, user, None)
        .await
        .expect("turn");
    if !r.trace.is_empty() {
        println!("──── trace ────\n{}", r.trace.join("\n"));
    }
    println!("──────── BOT ────────\n{}", r.answer);
    let done = session.trip.is_none();
    (r.answer, done)
}

/// Drive the flow to completion: after each pause, pick the first option / push
/// forward, capped so a stuck flow can't hang the test forever.
async fn drive(llm: &Llm, state: &BotState, session: &mut ChatSession, opener: &str) {
    let (_r, mut done) = step(llm, state, session, opener).await;
    let mut guard = 0;
    while !done && guard < 8 {
        guard += 1;
        // Generic forward push that works at any checkpoint (pick option / confirm).
        let next = "Давай первый вариант, выходные 11-12 июля. Двигаемся дальше.";
        let (_r, d) = step(llm, state, session, next).await;
        done = d;
    }
    println!(
        "\n==== FLOW {} after {guard} follow-ups ====",
        if done { "DONE" } else { "STILL OPEN" }
    );
}

#[tokio::test]
#[ignore]
async fn live_water_trip_end_to_end() {
    let (llm, state) = setup().await;
    let mut session = ChatSession::new(9001);
    session.profile.set("home_city", "Волгоград");
    drive(
        &llm,
        &state,
        &mut session,
        "Хотим в поход на байдарках с одной ночёвкой в ближайшие 2 недели, только выходные. \
         Команда больше про шашлык, чем грести. Ночёвка в палатках, минимум 1 км до турбаз/сёл, \
         вода максимум в 30 м. Дай план с точками и стоянкой.",
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn live_cycling_trip_end_to_end_no_water_hardcode() {
    let (llm, state) = setup().await;
    let mut session = ChatSession::new(9002);
    session.profile.set("home_city", "Волгоград");
    // A cycling trip never mentions water — the stages must NOT invent a water
    // requirement or talk about rivers/banks/put-in.
    drive(
        &llm,
        &state,
        &mut session,
        "Хотим велопоход на выходных в ближайшие 2 недели, с одной ночёвкой в палатках. \
         Команда любительская, спокойный темп. Дай маршрут с точками и местом ночёвки.",
    )
    .await;
}
