use teloxide::{
    dispatching::UpdateHandler,
    prelude::*,
    types::ParseMode,
    utils::command::BotCommands,
};

use crate::state::BotState;

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "Available commands:")]
pub enum Command {
    #[command(description = "Show this help")]
    Help,
    #[command(description = "Show bot status and MCP connection")]
    Status,
    #[command(description = "List all MCP tools")]
    Tools,
}

pub fn handler_schema() -> UpdateHandler<anyhow::Error> {
    use teloxide::dispatching::UpdateFilterExt;
    dptree::entry()
        .branch(
            Update::filter_message()
                .filter_command::<Command>()
                .endpoint(handle_command),
        )
}

async fn handle_command(bot: Bot, msg: Message, cmd: Command, state: BotState) -> anyhow::Result<()> {
    match cmd {
        Command::Help => {
            bot.send_message(msg.chat.id, Command::descriptions().to_string())
                .await?;
        }
        Command::Status => {
            let mcp = state.mcp.lock().await;
            let status = if mcp.is_some() {
                "✅ MCP connected"
            } else {
                "❌ MCP not connected"
            };
            bot.send_message(msg.chat.id, status).await?;
        }
        Command::Tools => {
            let mcp = state.mcp.lock().await;
            match mcp.as_ref() {
                None => {
                    bot.send_message(msg.chat.id, "❌ MCP not connected. Check MCP\\_HTTP\\_URL or MCP\\_COMMAND env.")
                        .parse_mode(ParseMode::MarkdownV2)
                        .await?;
                }
                Some(client) => {
                    let tools = client.tools().await;
                    if tools.is_empty() {
                        bot.send_message(msg.chat.id, "MCP connected but no tools found.").await?;
                        return Ok(());
                    }
                    let mut lines = vec![format!("🔧 *MCP Tools* \\({} total\\)\n", tools.len())];
                    for tool in &tools {
                        let desc = tool
                            .description
                            .as_deref()
                            .unwrap_or("no description")
                            .replace('.', "\\.")
                            .replace('-', "\\-")
                            .replace('(', "\\(")
                            .replace(')', "\\)")
                            .replace('!', "\\!");
                        let name = tool.name.replace('_', "\\_");
                        lines.push(format!("• `{name}` — {desc}"));
                    }
                    let text = lines.join("\n");
                    // Split if over 4096 chars
                    for chunk in split_message(&text, 4096) {
                        bot.send_message(msg.chat.id, chunk)
                            .parse_mode(ParseMode::MarkdownV2)
                            .await?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn split_message(text: &str, limit: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        if current.len() + line.len() + 1 > limit {
            chunks.push(current.clone());
            current.clear();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}
