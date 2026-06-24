use teloxide::{
    dispatching::{HandlerExt, UpdateFilterExt, UpdateHandler},
    prelude::*,
    types::{InlineKeyboardButton, InlineKeyboardMarkup},
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
    #[command(
        description = "connect MCP: /connect <url> [name=N] [auth=TOKEN] [Header:Value ...]"
    )]
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
        // Bare text (e.g. a pasted URL) — auto-connect or show help
        .branch(Update::filter_message().endpoint(handle_text))
        // Inline button presses
        .branch(Update::filter_callback_query().endpoint(handle_callback))
}

/// Non-command text: everything is driven by commands, so just point to /help.
async fn handle_text(bot: Bot, msg: Message, _state: BotState) -> anyhow::Result<()> {
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
                "✅ Subscribed to digests.\n\n\
                 Connect an MCP server:\n\
                 /connect <url> [name=N] [auth=TOKEN] [Header:Value ...]\n\n\
                 Then /mcps and /tools (or use the buttons).",
            )
            .await?;
        }
        Command::Help => {
            bot.send_message(chat, Command::descriptions().to_string())
                .await?;
        }
        Command::Connect(args) => {
            do_connect(&bot, chat, &state, &args).await?;
        }
        Command::Mcps => {
            send_mcp_list(&bot, chat, &state).await?;
        }
        Command::Tools(arg) => {
            let target = arg.trim();
            if target.is_empty() {
                // all servers
                for name in state.mcp_names().await {
                    send_tools(&bot, chat, &state, &name).await?;
                }
                if state.mcp_names().await.is_empty() {
                    bot.send_message(chat, "No MCP connected. Use /connect <url>.")
                        .await?;
                }
            } else {
                send_tools(&bot, chat, &state, target).await?;
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

/// Inline button press: `tools:<name>` or `disc:<name>`.
async fn handle_callback(bot: Bot, q: CallbackQuery, state: BotState) -> anyhow::Result<()> {
    bot.answer_callback_query(&q.id).await.ok();
    let (Some(data), Some(msg)) = (q.data.as_deref(), q.message.as_ref()) else {
        return Ok(());
    };
    let chat = msg.chat.id;
    if let Some(name) = data.strip_prefix("tools:") {
        send_tools(&bot, chat, &state, name).await?;
    } else if let Some(name) = data.strip_prefix("disc:") {
        if state.disconnect_mcp(name).await {
            bot.send_message(chat, format!("✅ '{name}' disconnected."))
                .await?;
        } else {
            bot.send_message(chat, format!("'{name}' was not connected."))
                .await?;
        }
    }
    Ok(())
}

/// Parse args, connect, and reply with status + action buttons.
async fn do_connect(bot: &Bot, chat: ChatId, state: &BotState, args: &str) -> anyhow::Result<()> {
    let params = match parse_connect(args) {
        Ok(p) => p,
        Err(e) => {
            bot.send_message(
                chat,
                format!(
                    "❌ {e}\n\nUsage:\n/connect <url> [name=N] [auth=TOKEN] [Header:Value ...]\n\n\
                     Example:\nhttps://host/mcp auth=SECRET X-Tracker-Token:abc X-Tracker-Org-Id:42"
                ),
            )
            .await?;
            return Ok(());
        }
    };
    let name = params.name.clone();
    bot.send_message(chat, format!("⏳ Connecting '{name}'…"))
        .await?;
    match state.connect_mcp(params).await {
        Ok(n) => {
            bot.send_message(chat, format!("✅ '{name}' connected — {n} tools."))
                .reply_markup(server_keyboard(&name))
                .await?;
        }
        Err(e) => {
            bot.send_message(chat, format!("❌ Connect '{name}' failed: {e}"))
                .await?;
        }
    }
    Ok(())
}

async fn send_mcp_list(bot: &Bot, chat: ChatId, state: &BotState) -> anyhow::Result<()> {
    let names = state.mcp_names().await;
    if names.is_empty() {
        bot.send_message(chat, "No MCP servers connected. Use /connect <url>.")
            .await?;
        return Ok(());
    }
    let guard = state.mcps.lock().await;
    for n in &names {
        if let Some(c) = guard.get(n) {
            let tc = c.tools().await.len();
            bot.send_message(chat, format!("• {n} — {tc} tools\n{}", c.params.url))
                .reply_markup(server_keyboard(n))
                .await?;
        }
    }
    Ok(())
}

async fn send_tools(bot: &Bot, chat: ChatId, state: &BotState, name: &str) -> anyhow::Result<()> {
    let tools = {
        let guard = state.mcps.lock().await;
        match guard.get(name) {
            Some(c) => c.tools().await,
            None => {
                bot.send_message(chat, format!("Unknown server '{name}'. See /mcps."))
                    .await?;
                return Ok(());
            }
        }
    };
    let mut blocks = vec![format!(
        "🔧 <b>{}</b> — {} tools",
        html_escape(name),
        tools.len()
    )];
    for (i, t) in tools.iter().enumerate() {
        let desc = t.description.as_deref().unwrap_or("");
        blocks.push(format!(
            "<b>{}. {}</b>\n<i>{}</i>",
            i + 1,
            html_escape(&t.name),
            html_escape(desc)
        ));
    }
    // Blank line between blocks; "\n\n" survives split_chunks (line-based).
    for chunk in split_chunks(&blocks.join("\n\n"), 3500) {
        bot.send_message(chat, chunk)
            .parse_mode(teloxide::types::ParseMode::Html)
            .await?;
    }
    Ok(())
}

/// Escape the 3 characters Telegram's HTML parse mode treats specially.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn server_keyboard(name: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new([[
        InlineKeyboardButton::callback("🔧 Tools", format!("tools:{name}")),
        InlineKeyboardButton::callback("❌ Disconnect", format!("disc:{name}")),
    ]])
}

/// Parse `/connect` args. URL-first and forgiving (newlines, order-agnostic):
/// `/connect <url> [name=NAME] [auth=TOKEN] [Header:Value ...]`
/// Only the URL is required; name defaults to the URL host (sanitized).
fn parse_connect(args: &str) -> Result<ConnectParams, String> {
    let mut url: Option<String> = None;
    let mut name: Option<String> = None;
    let mut auth = None;
    let mut headers = Vec::new();

    for tok in args.split_whitespace() {
        if tok.starts_with("http://") || tok.starts_with("https://") {
            url = Some(tok.to_string());
        } else if let Some(v) = tok.strip_prefix("name=") {
            name = Some(v.to_string());
        } else if let Some(v) = tok.strip_prefix("auth=") {
            auth = Some(v.to_string());
        } else if let Some((k, v)) = tok.split_once(':') {
            headers.push((k.to_string(), v.to_string()));
        } else {
            return Err(format!(
                "unrecognized '{tok}' (expected the URL, name=…, auth=… or Key:Value)"
            ));
        }
    }

    let url = url.ok_or("no URL found — give an http(s):// MCP endpoint")?;
    let name = name.unwrap_or_else(|| default_name(&url));
    Ok(ConnectParams {
        name,
        url,
        auth,
        headers,
    })
}

/// Derive a short server name from a URL: host[:port], sanitized.
fn default_name(url: &str) -> String {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let cleaned: String = authority
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() {
        "mcp".to_string()
    } else {
        trimmed.to_string()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_connect_url_only_autoname() {
        let p = parse_connect("http://5.129.234.9:3000/mcp").unwrap();
        assert_eq!(p.url, "http://5.129.234.9:3000/mcp");
        assert_eq!(p.name, "5_129_234_9_3000");
        assert!(p.auth.is_none());
        assert!(p.headers.is_empty());
    }

    #[test]
    fn parse_connect_url_with_leading_space_and_newline() {
        // Mirrors the user pasting "/connect\n http://..." — args is "\n http://..."
        let p = parse_connect("\n http://5.129.234.9:3000/mcp").unwrap();
        assert_eq!(p.url, "http://5.129.234.9:3000/mcp");
    }

    #[test]
    fn parse_connect_explicit_name_auth_headers() {
        let p = parse_connect(
            "https://h/mcp name=trk auth=SECRET X-Tracker-Token:abc X-Tracker-Org-Id:42",
        )
        .unwrap();
        assert_eq!(p.name, "trk");
        assert_eq!(p.auth.as_deref(), Some("SECRET"));
        assert_eq!(
            p.headers,
            vec![
                ("X-Tracker-Token".to_string(), "abc".to_string()),
                ("X-Tracker-Org-Id".to_string(), "42".to_string()),
            ]
        );
    }

    #[test]
    fn parse_connect_order_agnostic() {
        let p = parse_connect("name=x auth=y https://h/mcp").unwrap();
        assert_eq!(p.name, "x");
        assert_eq!(p.url, "https://h/mcp");
        assert_eq!(p.auth.as_deref(), Some("y"));
    }

    #[test]
    fn parse_connect_header_value_keeps_colon() {
        let p = parse_connect("https://h X-Url:http://a.b/c").unwrap();
        assert_eq!(
            p.headers,
            vec![("X-Url".to_string(), "http://a.b/c".to_string())]
        );
    }

    #[test]
    fn parse_connect_no_url_errors() {
        assert!(parse_connect("name=only auth=tok").is_err());
    }

    #[test]
    fn parse_connect_bare_word_errors() {
        assert!(parse_connect("https://h plainword").is_err());
    }

    #[test]
    fn default_name_variants() {
        assert_eq!(default_name("https://example.com/mcp"), "example_com");
        assert_eq!(default_name("http://1.2.3.4:3000/x"), "1_2_3_4_3000");
        assert_eq!(default_name("https://"), "mcp");
    }

    #[test]
    fn split_chunks_single() {
        assert_eq!(split_chunks("a\nb\nc", 4000), vec!["a\nb\nc".to_string()]);
    }

    #[test]
    fn split_chunks_splits_and_roundtrips() {
        let text = "aaaa\nbbbb\ncccc";
        let v = split_chunks(text, 6);
        assert!(v.len() > 1);
        for c in &v {
            assert!(c.len() <= 6);
        }
        assert_eq!(v.join("\n"), text);
    }

    #[test]
    fn split_chunks_empty() {
        assert!(split_chunks("", 100).is_empty());
    }

    #[test]
    fn html_escape_specials() {
        assert_eq!(html_escape("a & b < c > d"), "a &amp; b &lt; c &gt; d");
        assert_eq!(html_escape("plain"), "plain");
    }
}
