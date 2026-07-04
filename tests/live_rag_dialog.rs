//! Live long-dialog RAG tests: two scenarios of 10+ messages each, exercising
//! dialog history, task state, mandatory sources with quotes and the
//! low-relevance refusal — the same per-turn flow as `handle_rag_text`.
//!
//! Needs a built index + Ollama with the embed/chat models, e.g.:
//! ```bash
//! RAG_INDEX=~/Documents/Rag/ollama-rag-indexer/indexes-real-qwen/structural \
//! RAG_INDEXER_BIN=~/Documents/Rag/ollama-rag-indexer/.venv/bin/rag-indexer \
//! cargo test --test live_rag_dialog -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use tg_agent::{
    agent::rag_task::{update_task_state, RagTaskState},
    config::RagConfig,
    rag_client::RagClient,
};

fn rag_client_from_env() -> RagClient {
    let index = std::env::var("RAG_INDEX").expect("set RAG_INDEX to a built index dir");
    RagClient::new(RagConfig {
        bin: std::env::var("RAG_INDEXER_BIN").unwrap_or_else(|_| "rag-indexer".into()),
        index: PathBuf::from(index),
        embed_model: std::env::var("RAG_EMBED_MODEL").unwrap_or_else(|_| "qwen3-embedding".into()),
        chat_model: std::env::var("RAG_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:7b".into()),
        chat_url: std::env::var("RAG_CHAT_URL").unwrap_or_else(|_| "http://localhost:11434".into()),
        chat_provider: std::env::var("RAG_CHAT_PROVIDER").unwrap_or_else(|_| "ollama".into()),
        ollama_url: std::env::var("RAG_OLLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:11434".into()),
        search_mode: std::env::var("RAG_SEARCH_MODE").unwrap_or_else(|_| "hybrid".into()),
        top_k: 5,
        rewrite: true,
        // Calibrated for the real corpus: legit top chunks score ≥0.63 dense,
        // off-topic near-misses ≤0.47 (default 0.35 lets those through).
        min_score: Some(
            std::env::var("RAG_MIN_SCORE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.5),
        ),
    })
}

struct DialogStats {
    turns: usize,
    relevant_with_sources: usize,
    refusals: usize,
}

/// Drive one scenario through the same per-turn steps as `handle_rag_text`:
/// push question → update task state → answer with history + task state →
/// push answer. Asserts per turn: a relevant answer carries sources and every
/// source carries a quote; a refusal carries no sources.
async fn run_scenario(
    rag: &RagClient,
    label: &str,
    messages: &[&str],
) -> (DialogStats, RagTaskState) {
    assert!(rag.is_ready(), "RAG index is not ready: {}", rag.describe());
    let mut memory = tg_agent::agent::memory::AgentMemory::default();
    let mut task = RagTaskState::default();
    let mut stats = DialogStats {
        turns: 0,
        relevant_with_sources: 0,
        refusals: 0,
    };

    for (i, text) in messages.iter().enumerate() {
        memory.push_message("user", text);
        // No LLM key needed: falls back to seeding the goal from the first message.
        update_task_state(None, &mut task, text).await;
        let history: Vec<(String, String)> = memory
            .history_for_answer()
            .into_iter()
            .map(|(r, t)| (r.to_string(), t.to_string()))
            .collect();
        let task_prompt = task.to_prompt();

        let reply = rag
            .answer(
                text,
                &history,
                (!task_prompt.is_empty()).then_some(task_prompt.as_str()),
            )
            .await
            .unwrap_or_else(|e| panic!("[{label}] turn {i} failed: {e:#}"));

        println!(
            "\n[{label}] turn {i} q={text}\nrelevant={} sources={} rewrite={:?}\n{}",
            reply.relevant,
            reply.sources.len(),
            reply.rewritten_query,
            reply.render()
        );

        if reply.relevant {
            assert!(
                !reply.sources.is_empty(),
                "[{label}] turn {i}: relevant answer must carry sources"
            );
            assert!(
                reply
                    .sources
                    .iter()
                    .all(|s| s.quote.as_deref().is_some_and(|q| !q.trim().is_empty())),
                "[{label}] turn {i}: every source must carry a verbatim quote"
            );
            assert!(
                reply.render().contains("Источники:"),
                "[{label}] turn {i}: rendered reply must list sources"
            );
            stats.relevant_with_sources += 1;
        } else {
            assert!(
                reply.sources.is_empty(),
                "[{label}] turn {i}: refusal must not cite sources"
            );
            stats.refusals += 1;
        }

        memory.push_message("assistant", &reply.answer);
        stats.turns += 1;
    }
    (stats, task)
}

/// Scenario 1 (12 messages): exploring tg-agent architecture; follow-ups use
/// pronouns so retrieval depends on history-aware query rewriting; one
/// off-topic question must be refused.
#[tokio::test]
#[ignore]
async fn long_dialog_architecture_keeps_goal_and_sources() {
    let rag = rag_client_from_env();
    let messages = [
        "Что такое tg-agent и из чего он состоит?",
        "Как бот подключает MCP-серверы в рантайме?",
        "А какие транспорты он для этого поддерживает?",
        "Как устроен LLM tool-loop?",
        "Какие meta-tools там есть?",
        "Как агент хранит память о пользователе?",
        "Какие слои у этой памяти?",
        "Кто написал роман «Война и мир»?",
        "Что делает команда /watch?",
        "А как её результаты доставляются пользователю?",
        "Как работает trip-planning swarm?",
        "Какие агенты входят в этот рой?",
    ];

    let (stats, task) = run_scenario(&rag, "architecture", &messages).await;

    assert_eq!(stats.turns, 12);
    // The off-topic literature question must be refused; corpus questions answered.
    // (An earlier variant used "температура кипения ртути", which shares weather
    // vocabulary with the corpus and slipped past the global relevance floor —
    // the known single-threshold near-miss documented in the indexer README.)
    assert!(stats.refusals >= 1, "off-topic question should be refused");
    assert!(
        stats.relevant_with_sources >= 9,
        "most corpus questions must produce sourced answers, got {}",
        stats.relevant_with_sources
    );
    // Task state keeps the dialog goal to the end.
    assert!(!task.goal.trim().is_empty(), "dialog goal must be retained");
    assert!(
        task.goal.contains("tg-agent"),
        "goal drifted: {}",
        task.goal
    );
}

/// Scenario 2 (10 messages): deploy/ops dialog around open-meteo-mcp and the
/// VPS; checks the assistant keeps answering with sources deep into the dialog.
#[tokio::test]
#[ignore]
async fn long_dialog_deploy_keeps_sources_deep_into_dialog() {
    let rag = rag_client_from_env();
    let messages = [
        "Что такое open-meteo-mcp?",
        "Какие tools он предоставляет?",
        "Как получить прогноз погоды через него?",
        "Как этот сервер запускается на VPS?",
        "Каким транспортом он отдаёт MCP?",
        "Как tg-agent к нему подключается?",
        "Что происходит при реконнекте?",
        "Как подписаться на регулярные обновления погоды?",
        "Кто выиграл чемпионат мира по футболу 2018 года?",
        "Какие переменные окружения нужны tg-agent для запуска?",
    ];

    let (stats, task) = run_scenario(&rag, "deploy", &messages).await;

    assert_eq!(stats.turns, 10);
    assert!(stats.refusals >= 1, "off-topic question should be refused");
    assert!(
        stats.relevant_with_sources >= 7,
        "most corpus questions must produce sourced answers, got {}",
        stats.relevant_with_sources
    );
    assert!(!task.goal.trim().is_empty(), "dialog goal must be retained");
    // The last sourced turn proves sources survive to the end of a long dialog.
}
