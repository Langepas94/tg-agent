use std::time::Duration;

use teloxide::{prelude::*, types::ChatId};
use tokio::time;
use tracing::info;

use crate::{mcp_client::McpEvent, state::BotState};

pub fn spawn(bot: Bot, interval_minutes: u64, state: BotState) {
    // Forward MCP events to subscribers
    let bot_ev = bot.clone();
    let state_ev = state.clone();
    let mut rx = state.events.subscribe();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(McpEvent::ToolsChanged { server }) => {
                    broadcast(&bot_ev, &state_ev, &format!("🔄 {server}: tools updated")).await;
                }
                Ok(McpEvent::LogMessage {
                    server,
                    level,
                    data,
                }) => {
                    broadcast(
                        &bot_ev,
                        &state_ev,
                        &format!("📋 {server} [{level}]: {data}"),
                    )
                    .await;
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // Periodic digest
    let bot_d = bot.clone();
    let state_d = state.clone();
    tokio::spawn(async move {
        let mut ticker = time::interval(Duration::from_secs(interval_minutes * 60));
        ticker.tick().await; // skip immediate first tick
        loop {
            ticker.tick().await;
            send_digest(&bot_d, &state_d).await;
        }
    });
}

async fn broadcast(bot: &Bot, state: &BotState, text: &str) {
    for id in state.subscribers().await {
        let _ = bot.send_message(ChatId(id), text).await;
    }
}

async fn send_digest(bot: &Bot, state: &BotState) {
    let subs = state.subscribers().await;
    if subs.is_empty() {
        return;
    }
    let names = state.mcp_names().await;
    let mut body = String::from("🗓 Digest\n\n");
    if names.is_empty() {
        body.push_str("No MCP servers connected.");
    } else {
        let guard = state.mcps.lock().await;
        for n in &names {
            if let Some(c) = guard.get(n) {
                let tc = c.tools().await.len();
                body.push_str(&format!("• {n}: {tc} tools\n"));
            }
        }
    }
    body.push_str("\n\nAgent is running.");
    for id in subs {
        let _ = bot.send_message(ChatId(id), &body).await;
    }
    info!("Digest sent");
}
