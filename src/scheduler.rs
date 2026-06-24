use std::time::Duration;

use teloxide::{prelude::*, types::ChatId};
use tokio::time;
use tracing::info;

use crate::{mcp_client::McpEvent, persist::WatchSpec, state::BotState};

/// Spawn a periodic watch: every `interval_min`, call the tool and post the
/// result to the watch's chat. Registers the task handle so it can be aborted.
pub async fn spawn_watch(bot: Bot, state: BotState, spec: WatchSpec) {
    let id = spec.id;
    let task_state = state.clone();
    let handle = tokio::spawn(async move {
        let state = task_state;
        let mut ticker = time::interval(Duration::from_secs(spec.interval_min.max(1) * 60));
        // first tick is immediate — run once now, then on each interval
        loop {
            ticker.tick().await;
            let chat = ChatId(spec.chat_id);
            match state
                .call_tool(&spec.server, &spec.tool, spec.args.clone())
                .await
            {
                Ok(text) => {
                    let body = if text.trim().is_empty() {
                        "(empty)".to_string()
                    } else {
                        text
                    };
                    let msg = format!("⏱ {} / {}:\n{}", spec.server, spec.tool, body);
                    for chunk in chunks(&msg, 3900) {
                        let _ = bot.send_message(chat, chunk).await;
                    }
                }
                Err(e) => {
                    let _ = bot
                        .send_message(chat, format!("⏱ watch {} failed: {e}", spec.tool))
                        .await;
                }
            }
        }
    });
    state.watch_tasks.lock().await.insert(id, handle);
}

/// Re-spawn all persisted watches at startup.
pub async fn restore_watches(bot: &Bot, state: &BotState) {
    for spec in state.list_watches().await {
        spawn_watch(bot.clone(), state.clone(), spec).await;
    }
}

fn chunks(text: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        if cur.len() + line.len() + 1 > limit && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

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
