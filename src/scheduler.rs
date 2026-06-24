use std::time::Duration;

use teloxide::{prelude::*, types::ChatId};
use tokio::time;
use tracing::info;

use crate::state::BotState;

/// Spawns background task that:
/// 1. Listens for MCP events and forwards them to digest_chat_id
/// 2. Sends periodic digest every interval_minutes
pub fn spawn(
    bot: Bot,
    digest_chat_id: i64,
    interval_minutes: u64,
    state: BotState,
    mut mcp_events: crate::mcp_client::EventReceiver,
) {
    let chat = ChatId(digest_chat_id);

    // MCP event listener
    let bot_events = bot.clone();
    tokio::spawn(async move {
        loop {
            match mcp_events.recv().await {
                Ok(crate::mcp_client::McpEvent::ToolsChanged) => {
                    info!("MCP tools changed, notifying digest chat");
                    let _ = bot_events
                        .send_message(chat, "🔄 MCP: tool list updated")
                        .await;
                }
                Ok(crate::mcp_client::McpEvent::LogMessage { level, data }) => {
                    let text = format!("📋 MCP [{level}]: {data}");
                    info!("{text}");
                    let _ = bot_events.send_message(chat, text).await;
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("MCP event channel lagged, missed {n} events");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // Periodic digest
    let bot_digest = bot.clone();
    let state_digest = state.clone();
    let interval = Duration::from_secs(interval_minutes * 60);
    tokio::spawn(async move {
        let mut ticker = time::interval(interval);
        ticker.tick().await; // skip first immediate tick
        loop {
            ticker.tick().await;
            send_digest(&bot_digest, chat, &state_digest).await;
        }
    });
}

async fn send_digest(bot: &Bot, chat: ChatId, state: &BotState) {
    let mcp = state.mcp.lock().await;
    let (connected, tool_count) = match mcp.as_ref() {
        None => (false, 0),
        Some(c) => (true, c.tools().await.len()),
    };
    drop(mcp);

    let status = if connected {
        format!("✅ MCP connected — {tool_count} tools available")
    } else {
        "❌ MCP not connected".to_string()
    };

    let text = format!("🗓 *Digest*\n\n{status}\n\n_Agent is running\\._");
    let _ = bot
        .send_message(chat, text)
        .parse_mode(teloxide::types::ParseMode::MarkdownV2)
        .await;
    info!("Digest sent to {chat}");
}
