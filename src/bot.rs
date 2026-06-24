use teloxide::{
    dispatching::{HandlerExt, UpdateFilterExt, UpdateHandler},
    prelude::*,
    utils::command::BotCommands,
};

use crate::{mcp_client::ConnectParams, state::BotState};

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "Commands:")]
pub enum Command {
    #[command(description = "start and subscribe to digests")]
    Start,
    #[command(description = "show help")]
    Help,
    #[command(description = "connect MCP: /connect <name> <url> [auth=TOKEN] [Header:Value ...]")]
    Connect(String),
    #[command(description = "list connected MCP servers")]
    Mcps,
    #[command(description = "list tools: /tools [server]")]
    Tools(String),
    #[command(description = "disconnect a server: /disconnect <name>")]
    Disconnect(String),
}

pub fn handler_schema() -> UpdateHandler<anyhow::Error> {
    dptree::entry()
        .branch(
            Update::filter_message()
                .filter_command::<Command>()
                .endpoint(handle_command),
        )
        // Catch-all: any non-command message gets a helpful reply
        .branch(Update::filter_message().endpoint(handle_any))
}

async fn handle_any(bot: Bot, msg: Message) -> anyhow::Result<()> {
    bot.send_message(msg.chat.id, "Send /help to see commands.")
        .await?;
    Ok(())
}

async fn handle_command(
    bot: Bot,
    msg: Message,
    cmd: Command,
    state: BotState,
) -> anyhow::Result<()> {
    let chat = msg.chat.id;
    match cmd {
        Command::Start => {
            state.subscribe(chat.0).await;
            bot.send_message(
                chat,
                "✅ Subscribed to digests.\n\nConnect an MCP server with:\n\
                 /connect <name> <url> [auth=TOKEN] [Header:Value ...]\n\n\
                 Then /mcps and /tools.",
            )
            .await?;
        }
        Command::Help => {
            bot.send_message(chat, Command::descriptions().to_string())
                .await?;
        }
        Command::Connect(args) => match parse_connect(&args) {
            Err(e) => {
                bot.send_message(
                        chat,
                        format!(
                            "❌ {e}\n\nUsage:\n/connect <name> <url> [auth=TOKEN] [Header:Value ...]\n\n\
                             Example:\n/connect tracker https://host/mcp auth=SECRET X-Tracker-Token:abc X-Tracker-Org-Id:42"
                        ),
                    )
                    .await?;
            }
            Ok(params) => {
                let name = params.name.clone();
                bot.send_message(chat, format!("⏳ Connecting '{name}'…"))
                    .await?;
                match state.connect_mcp(params).await {
                    Ok(n) => {
                        bot.send_message(
                            chat,
                            format!("✅ '{name}' connected — {n} tools.\nUse /tools {name}"),
                        )
                        .await?;
                    }
                    Err(e) => {
                        bot.send_message(chat, format!("❌ Connect '{name}' failed: {e}"))
                            .await?;
                    }
                }
            }
        },
        Command::Mcps => {
            let names = state.mcp_names().await;
            if names.is_empty() {
                bot.send_message(chat, "No MCP servers connected. Use /connect.")
                    .await?;
            } else {
                let mut lines = vec![format!("Connected MCP servers ({}):", names.len())];
                let guard = state.mcps.lock().await;
                for n in &names {
                    if let Some(c) = guard.get(n) {
                        let tc = c.tools().await.len();
                        lines.push(format!("• {n} — {tc} tools — {}", c.params.url));
                    }
                }
                bot.send_message(chat, lines.join("\n")).await?;
            }
        }
        Command::Tools(arg) => {
            let target = arg.trim();
            let guard = state.mcps.lock().await;
            if guard.is_empty() {
                bot.send_message(chat, "No MCP connected. Use /connect first.")
                    .await?;
                return Ok(());
            }
            // Pick servers: one named, or all
            let servers: Vec<String> = if target.is_empty() {
                let mut v: Vec<_> = guard.keys().cloned().collect();
                v.sort();
                v
            } else if guard.contains_key(target) {
                vec![target.to_string()]
            } else {
                bot.send_message(chat, format!("Unknown server '{target}'. See /mcps."))
                    .await?;
                return Ok(());
            };

            for sname in servers {
                let client = guard.get(&sname).unwrap();
                let tools = client.tools().await;
                let mut lines = vec![format!("🔧 {sname} ({} tools):", tools.len())];
                for t in &tools {
                    let desc = t.description.as_deref().unwrap_or("");
                    lines.push(format!("• {} — {desc}", t.name));
                }
                for chunk in split_chunks(&lines.join("\n"), 4000) {
                    bot.send_message(chat, chunk).await?;
                }
            }
        }
        Command::Disconnect(arg) => {
            let name = arg.trim();
            if name.is_empty() {
                bot.send_message(chat, "Usage: /disconnect <name>").await?;
            } else if state.disconnect_mcp(name).await {
                bot.send_message(chat, format!("✅ '{name}' disconnected."))
                    .await?;
            } else {
                bot.send_message(chat, format!("No server named '{name}'. See /mcps."))
                    .await?;
            }
        }
    }
    Ok(())
}

/// Parse `/connect` args: `<name> <url> [auth=TOKEN] [Header:Value ...]`
fn parse_connect(args: &str) -> Result<ConnectParams, String> {
    let mut it = args.split_whitespace();
    let name = it.next().ok_or("missing <name>")?.to_string();
    let url = it.next().ok_or("missing <url>")?.to_string();
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("url must start with http:// or https://".into());
    }
    let mut auth = None;
    let mut headers = Vec::new();
    for tok in it {
        if let Some(v) = tok.strip_prefix("auth=") {
            auth = Some(v.to_string());
        } else if let Some((k, v)) = tok.split_once(':') {
            headers.push((k.to_string(), v.to_string()));
        } else {
            return Err(format!("bad token '{tok}' (expected auth=… or Key:Value)"));
        }
    }
    Ok(ConnectParams {
        name,
        url,
        auth,
        headers,
    })
}

fn split_chunks(text: &str, limit: usize) -> Vec<String> {
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
    out
}
