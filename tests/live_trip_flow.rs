//! Live end-to-end trip-flow driver. Runs the FULL stateful swarm against the
//! real LLM + real MCP servers (weather over HTTP, maps/osmmcp over stdio) and
//! prints every turn, so the flow can be verified by actually running it instead
//! of guessing. Drives three activities — a water (kayak) trip, a NON-water
//! (cycling) trip, and a light day-walk — to prove the swarm is
//! activity-agnostic (no river/water/overnight hardcode).
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

    let google_env_keys = [
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
        "OAUTHLIB_INSECURE_TRANSPORT",
        "USER_GOOGLE_EMAIL",
        "WORKSPACE_MCP_BASE_URI",
        "WORKSPACE_MCP_PORT",
        "GOOGLE_OAUTH_REDIRECT_URI",
        "WORKSPACE_MCP_HOST",
        "WORKSPACE_MCP_PORT_FALLBACK_COUNT",
    ];
    let google_env = google_env_keys
        .iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| ((*k).to_string(), v)))
        .collect::<Vec<_>>();
    assert!(
        google_env
            .iter()
            .any(|(k, _)| k == "GOOGLE_OAUTH_CLIENT_ID"),
        "set Google MCP env from the VPS state before running this live suite"
    );
    state
        .connect_mcp(ConnectParams {
            name: "google".into(),
            url: String::new(),
            auth: None,
            headers: vec![],
            command: std::env::var("GOOGLE_MCP_COMMAND")
                .ok()
                .map(|s| s.split_whitespace().map(str::to_string).collect())
                .unwrap_or_else(|| {
                    vec!["uvx".into(), "workspace-mcp".into(), "--single-user".into()]
                }),
            env: google_env,
        })
        .await
        .expect("connect Google MCP");

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
    assert_plain_telegram(&r.answer);
    let done = session.trip.is_none();
    (r.answer, done)
}

fn assert_plain_telegram(answer: &str) {
    assert!(
        !answer.contains("⟦profile:"),
        "profile marker leaked: {answer}"
    );
    assert!(!answer.contains("**"), "markdown bold leaked: {answer}");
    assert!(!answer.contains("|---"), "markdown table leaked: {answer}");
    assert!(!answer.contains("```"), "code fence leaked: {answer}");
    let lower = answer.to_lowercase();
    assert!(
        !lower.contains("пятница") && !lower.contains("пт,") && !lower.contains("пт "),
        "weekend-only trip offered a Friday: {answer}"
    );
    assert!(
        !lower.contains("не удалось проверить") && !lower.contains("не проверено"),
        "flow exposed an unverified hard constraint as a normal answer: {answer}"
    );
}

fn assert_final_concrete(answer: &str) {
    let lower = answer.to_lowercase();
    assert!(
        !lower.contains("не зафиксирован")
            && !lower.contains("не зафиксирована")
            && !lower.contains("требует уточнения")
            && !lower.contains("маршрутная стадия не заверш")
            && !lower.contains("детализированный трек"),
        "final answer is not concrete enough for a completed flow: {answer}"
    );
}

async fn drive_assert_done(
    llm: &Llm,
    state: &BotState,
    session: &mut ChatSession,
    opener: &str,
) -> String {
    let (mut answer, mut done) = step(llm, state, session, opener).await;
    let mut guard = 0;
    while !done && guard < 8 {
        guard += 1;
        let next = "Давай первый вариант. Подтверждаю, двигайся дальше.";
        let (r, d) = step(llm, state, session, next).await;
        answer = r;
        done = d;
    }
    assert!(done, "flow did not finish after {guard} follow-ups");
    answer
}

#[tokio::test]
#[ignore]
async fn live_full_kayak_trip_creates_google_artifacts() {
    let (llm, state) = setup().await;
    let mut session = ChatSession::new(9101);
    session.profile.set("home_city", "Волгоград");
    let email = std::env::var("USER_GOOGLE_EMAIL")
        .unwrap_or_else(|_| "artyom.tyurmorezov@gmail.com".into());
    session.profile.set("email", &email);
    session.profile.set("google_email", &email);

    let final_answer = drive_assert_done(
        &llm,
        &state,
        &mut session,
        "Хотим сходить в поход на байдарках, посмотри в какой день лучше это сделать в течение следующих двух недель (мы можем только на выходных). И в каких местах.\n\n\
         Сплав с одной ночевкой, команда больше хочет шашлыки, чем грести. Ночлег будет в палатках, чтобы в радиусе минимум 1км не было турбаз,сел и так далее. Вода должна быть рядом, максимум в 30 метрах от ночлега.\n\n\
         Составь конкретный план с точками остановки и ночлега, событие в календаре, потом из гугл док поделюсь с друзьями.",
    )
    .await;

    assert_final_concrete(&final_answer);
    let lower = final_answer.to_lowercase();
    assert!(
        lower.contains("calendar")
            || lower.contains("календар")
            || lower.contains("event")
            || lower.contains("событ"),
        "final answer does not mention calendar/event: {final_answer}"
    );
    assert!(
        final_answer.contains("https://docs.google.com/"),
        "final answer does not include the shareable Google Doc link: {final_answer}"
    );
    assert!(
        !final_answer.contains("2025"),
        "trip was scheduled in the past: {final_answer}"
    );
    assert!(
        final_answer.contains("2026"),
        "final answer should include the resolved 2026 trip year: {final_answer}"
    );
    assert!(
        lower.contains("байдар")
            || lower.contains("сплав")
            || lower.contains("греб")
            || lower.contains("put-in"),
        "final answer lost the kayaking/paddling activity: {final_answer}"
    );
    assert!(
        !lower.contains("авто-пеш") && !lower.contains("пеший выезд"),
        "final answer converted the kayaking trip into a walking/car trip: {final_answer}"
    );
    assert!(
        (final_answer.contains('4') && final_answer.contains('5'))
            || (final_answer.contains("11") && final_answer.contains("12")),
        "final answer should use a Saturday-Sunday pair in the next two weeks: {final_answer}"
    );
}

#[tokio::test]
#[ignore]
async fn live_water_trip_end_to_end() {
    let (llm, state) = setup().await;
    let mut session = ChatSession::new(9001);
    session.profile.set("home_city", "Волгоград");
    let final_answer = drive_assert_done(
        &llm,
        &state,
        &mut session,
        "Хотим в поход на байдарках с одной ночёвкой в ближайшие 2 недели, только выходные. \
         Команда больше про шашлык, чем грести. Ночёвка в палатках, минимум 1 км до турбаз/сёл, \
         вода максимум в 30 м. Дай план с точками и стоянкой.",
    )
    .await;
    assert_final_concrete(&final_answer);
    let lower = final_answer.to_lowercase();
    assert!(
        lower.contains("байдар")
            || lower.contains("каяк")
            || lower.contains("сплав")
            || lower.contains("греб"),
        "water trip lost paddling semantics: {final_answer}"
    );
    assert!(
        lower.contains("30") && (lower.contains("вод") || lower.contains("берег")),
        "water-distance constraint missing from final answer: {final_answer}"
    );
    assert!(
        (lower.contains("1") && lower.contains("км"))
            && (lower.contains("турбаз") || lower.contains("сел") || lower.contains("посел")),
        "isolation constraint missing from final answer: {final_answer}"
    );
}

#[tokio::test]
#[ignore]
async fn live_cycling_trip_end_to_end_no_water_hardcode() {
    let (llm, state) = setup().await;
    let mut session = ChatSession::new(9002);
    session.profile.set("home_city", "Волгоград");
    // A cycling trip never mentions water — the stages must NOT invent a water
    // requirement or talk about rivers/banks/put-in.
    let final_answer = drive_assert_done(
        &llm,
        &state,
        &mut session,
        "Хотим велопоход на выходных в ближайшие 2 недели, с одной ночёвкой в палатках. \
         Команда любительская, спокойный темп. Дай маршрут с точками и местом ночёвки.",
    )
    .await;
    assert_final_concrete(&final_answer);
    let lower = final_answer.to_lowercase();
    assert!(
        lower.contains("вел") || lower.contains("bike") || lower.contains("cycling"),
        "cycling trip lost cycling semantics: {final_answer}"
    );
    assert!(
        !lower.contains("байдар")
            && !lower.contains("каяк")
            && !lower.contains("сплав")
            && !lower.contains("put-in")
            && !lower.contains("take-out"),
        "cycling trip was converted into a paddling route: {final_answer}"
    );
}

#[tokio::test]
#[ignore]
async fn live_walk_to_see_new_place_end_to_end_no_overnight_hardcode() {
    let (llm, state) = setup().await;
    let mut session = ChatSession::new(9003);
    session.profile.set("home_city", "Волгоград");
    // A light day-walk to discover a new spot. No overnight, no paddling, no
    // cycling. The swarm must NOT invent a campsite / water-distance / route
    // track requirement, nor convert it into a kayak or bike trip.
    let final_answer = drive_assert_done(
        &llm,
        &state,
        &mut session,
        "Хочу просто прогуляться в эти выходные и посмотреть какое-нибудь новое \
         интересное место недалеко от города. Без ночёвки, пешком на пару часов. \
         Подскажи куда сходить.",
    )
    .await;
    assert_final_concrete(&final_answer);
    let lower = final_answer.to_lowercase();
    assert!(
        lower.contains("прогул")
            || lower.contains("пеш")
            || lower.contains("walk")
            || lower.contains("место")
            || lower.contains("парк"),
        "walk lost its sightseeing/walking semantics: {final_answer}"
    );
    assert!(
        !lower.contains("байдар")
            && !lower.contains("каяк")
            && !lower.contains("сплав")
            && !lower.contains("велопоход")
            && !lower.contains("put-in"),
        "walk was converted into a paddling or cycling trip: {final_answer}"
    );
    // a 2-hour day walk must not have an invented overnight campsite
    assert!(
        !lower.contains("ночёвк") && !lower.contains("ночлег") && !lower.contains("палатк"),
        "day walk invented an overnight stay it was never asked for: {final_answer}"
    );
}
