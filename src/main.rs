use anyhow::Result;
use teloxide::{prelude::*, utils::command::BotCommands};
use tg_agent::{bot, config, persist, scheduler, state};
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
    let state = state::BotState::new(tx);

    let bot = Bot::new(&cfg.telegram_token);
    bot.set_my_commands(bot::Command::bot_commands()).await?;
    info!("Bot commands registered");

    // Restore persisted state: reconnect MCP servers, subscribers, watches.
    restore_state(&bot, &state).await;

    scheduler::spawn(bot.clone(), cfg.digest_interval_minutes, state.clone());

    info!(
        "Dispatcher starting, digest every {} min",
        cfg.digest_interval_minutes
    );
    Dispatcher::builder(bot, bot::handler_schema())
        .dependencies(dptree::deps![state])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;

    Ok(())
}

/// Reload persisted state from disk and bring it back to life.
async fn restore_state(bot: &Bot, state: &state::BotState) {
    let saved = persist::load();
    if saved.servers.is_empty() && saved.subscribers.is_empty() && saved.watches.is_empty() {
        return;
    }
    info!(
        "Restoring state: {} servers, {} subscribers, {} watches",
        saved.servers.len(),
        saved.subscribers.len(),
        saved.watches.len()
    );

    state.set_next_watch_id(saved.next_watch_id);

    for id in saved.subscribers {
        state.subscribe(id).await;
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
    scheduler::restore_watches(bot, state).await;
}
