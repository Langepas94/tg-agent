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
