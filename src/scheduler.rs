use teloxide::{prelude::*, types::ChatId};

use crate::{mcp_client::McpEvent, state::BotState};

/// Re-spawn all persisted watches at startup (bot handle lives in state).
pub async fn restore_watches(state: &BotState) {
    for spec in state.list_watches().await {
        state.start_watch(spec).await;
    }
}

/// Forward real MCP notifications (tool-list changes, server log messages) to
/// subscribers. There is no periodic "digest" — recurring updates are explicit
/// watches, so the agent never spams a fixed heartbeat.
pub fn spawn(bot: Bot, state: BotState) {
    let mut rx = state.events.subscribe();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(McpEvent::ToolsChanged { server }) => {
                    broadcast(&bot, &state, &format!("🔄 {server}: tools updated")).await;
                }
                Ok(McpEvent::LogMessage {
                    server,
                    level,
                    data,
                }) => {
                    broadcast(&bot, &state, &format!("📋 {server} [{level}]: {data}")).await;
                }
                // Server-pushed summary → route to the owning chat (session_id =
                // chat_id, injected on the tool call), humanize, deliver.
                Ok(McpEvent::PushSummary { server: _, data }) => {
                    let Some(chat_id) = data
                        .get("session_id")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse::<i64>().ok())
                    else {
                        continue; // not a chat-scoped session_id; can't route
                    };
                    let body = state.humanize_summary(&data).await;
                    let _ = bot
                        .send_message(ChatId(chat_id), format!("📬 {body}"))
                        .await;
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

async fn broadcast(bot: &Bot, state: &BotState, text: &str) {
    for id in state.subscribers().await {
        let _ = bot.send_message(ChatId(id), text).await;
    }
}
