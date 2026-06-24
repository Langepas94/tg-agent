mod bot;
mod config;
mod mcp_client;
mod scheduler;
mod state;

use anyhow::Result;
use teloxide::prelude::*;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tg_agent=info,rmcp=info".parse().unwrap()),
        )
        .init();

    // Load .env if present (dev convenience)
    dotenvy::dotenv().ok();

    let cfg = config::Config::from_env()?;
    info!("Starting tg-agent");

    // Connect to MCP (optional — bot runs without it too)
    let (state, mcp_events) = match mcp_client::McpClient::connect(&cfg.mcp).await {
        Ok((client, rx)) => {
            info!("MCP connected");
            (state::BotState::with_mcp(client), rx)
        }
        Err(e) => {
            tracing::warn!("MCP connection failed: {e} — running without MCP");
            let (tx, rx) = tokio::sync::broadcast::channel(1);
            drop(tx);
            (state::BotState::new(), rx)
        }
    };

    let bot = Bot::new(&cfg.telegram_token);
    use teloxide::utils::command::BotCommands;
    bot.set_my_commands(bot::Command::bot_commands()).await?;
    info!("Bot commands registered");

    // Start digest scheduler + MCP event forwarder
    scheduler::spawn(
        bot.clone(),
        cfg.digest_chat_id,
        cfg.digest_interval_minutes,
        state.clone(),
        mcp_events,
    );

    info!(
        "Dispatcher starting, digest every {} min to chat {}",
        cfg.digest_interval_minutes, cfg.digest_chat_id
    );

    Dispatcher::builder(bot, bot::handler_schema())
        .dependencies(dptree::deps![state])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;

    Ok(())
}
