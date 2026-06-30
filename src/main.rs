use std::sync::Arc;

use anyhow::Result;
use teloxide::prelude::*;
use tg_agent::{admin, bot, config, llm::Llm, persist, rag_client::RagClient, scheduler, state};
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tg_agent=info".parse().unwrap()),
        )
        .init();

    dotenvy::dotenv().ok();
    let cfg = config::Config::from_env()?;
    info!("Starting tg-agent");

    // Shared event channel for all (runtime-added) MCP servers
    let (tx, _rx) = tokio::sync::broadcast::channel(256);
    let llm = match &cfg.llm {
        Some(c) => {
            info!("LLM enabled: {} @ {}", c.model, c.base_url);
            Some(Arc::new(Llm::new(c.clone())))
        }
        None => {
            warn!("No LLM configured — free-form questions disabled (set DEEPSEEK_API_KEY)");
            None
        }
    };
    let rag = cfg.rag.clone().map(|c| {
        info!("RAG client enabled: {}", c.index.display());
        Arc::new(RagClient::new(c))
    });
    let state = state::BotState::with_llm_rag_and_password(tx, llm, rag, cfg.bot_password.clone());

    let bot = Bot::new(&cfg.telegram_token);
    bot::set_public_commands(&bot).await?;
    info!("Public bot commands registered");

    // Store the bot handle so watches/agent meta-tools can post to chats.
    state.set_bot(bot.clone()).await;

    // Restore persisted state: reconnect MCP servers, subscribers, watches.
    restore_state(&state).await;
    if let Err(e) = bot::sync_authorized_command_menus(&bot, &state).await {
        warn!("Failed to sync chat command menus: {e:#}");
    }

    scheduler::spawn(bot.clone(), state.clone());
    if let (Some(addr), Some(password)) = (cfg.admin_addr.clone(), cfg.admin_password.clone()) {
        admin::spawn(
            state.clone(),
            admin::AdminConfig {
                addr,
                username: cfg.admin_username.clone(),
                password,
            },
        );
    } else if cfg.admin_addr.is_some() {
        warn!("Admin web disabled: set ADMIN_PASSWORD to enable /admin");
    }

    info!("Dispatcher starting");
    Dispatcher::builder(bot, bot::handler_schema())
        .dependencies(dptree::deps![state])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;

    Ok(())
}

/// Reload persisted state from disk and bring it back to life.
async fn restore_state(state: &state::BotState) {
    let saved = persist::load();
    if saved.servers.is_empty()
        && saved.subscribers.is_empty()
        && saved.watches.is_empty()
        && saved.push_subs.is_empty()
        && saved.access.authorized_chat_ids.is_empty()
        && saved.access.root_chat_id.is_none()
    {
        return;
    }
    info!(
        "Restoring state: {} servers, {} subscribers, {} watches, {} push-subs",
        saved.servers.len(),
        saved.subscribers.len(),
        saved.watches.len(),
        saved.push_subs.len()
    );

    state.set_next_watch_id(saved.next_watch_id);
    state.restore_access(saved.access).await;

    for id in saved.subscribers {
        state.subscribe(id).await;
    }

    // Load push-subs BEFORE connecting so reconnect re-applies them.
    for sub in saved.push_subs {
        state
            .add_push_sub(sub.chat_id, sub.server, sub.period)
            .await;
    }

    for params in saved.servers {
        let name = params.name.clone();
        if let Err(e) = state.connect_mcp(params).await {
            warn!("restore: failed to reconnect '{name}': {e}");
        }
    }

    // Re-register watches, then spawn their tasks.
    for spec in saved.watches {
        state.add_watch(spec).await;
    }
    scheduler::restore_watches(state).await;
}
