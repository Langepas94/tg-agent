mod bot;
mod config;
mod mcp_client;
mod scheduler;
mod state;

use anyhow::Result;
use teloxide::{prelude::*, utils::command::BotCommands};
use tracing::info;

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
