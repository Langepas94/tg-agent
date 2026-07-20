use teloxide::{
    dispatching::{HandlerExt, UpdateFilterExt, UpdateHandler},
    prelude::*,
    types::{BotCommand, BotCommandScope, InlineKeyboardButton, InlineKeyboardMarkup, Recipient},
    utils::command::BotCommands,
};

use crate::{mcp_client::ConnectParams, state::BotState};

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "Commands:")]
pub enum Command {
    #[command(description = "unlock the bot: /start <password>")]
    Start(String),
    #[command(description = "show help")]
    Help,
    #[command(
        description = "connect MCP: /connect <url> [name=N] [auth=TOKEN] [Header:Value] | /connect stdio <program> [args...] [name=N] [env=KEY=VAL]"
    )]
    Connect(String),
    #[command(description = "list connected MCP servers")]
    Mcps,
    #[command(description = "list tools: /tools [server]")]
    Tools(String),
    #[command(description = "call a tool: /call <server> <tool> [json-args]")]
    Call(String),
    #[command(description = "periodic poll: /watch <server> <tool> <minutes> [json-args]")]
    Watch(String),
    #[command(description = "stop a watch: /unwatch <id> | /unwatch all")]
    Unwatch(String),
    #[command(description = "list active watches")]
    Watches,
    #[command(description = "disconnect a server: /disconnect <name>")]
    Disconnect(String),
    #[command(description = "view/set profile: /profile [key value | clear]")]
    Profile(String),
    #[command(description = "extra info the agent uses when relevant: /info [label text | clear]")]
    Info(String),
    #[command(description = "show learned facts (layered memory)")]
    Facts,
    #[command(description = "travel-weather flow: /trip <cities/dates>")]
    Trip(String),
    #[command(description = "product support: /support <question>")]
    Support(String),
    #[command(description = "reset this chat's memory (keeps long-term)")]
    Reset,
    #[command(description = "clear short context and active flow, keep profile and durable facts")]
    Compact,
}

fn cmd(name: &str, description: &str) -> BotCommand {
    BotCommand::new(name, description)
}

fn base_commands() -> Vec<BotCommand> {
    vec![
        cmd("start", "unlock the bot"),
        cmd("help", "show help"),
        cmd("profile", "view or set profile"),
        cmd("info", "view or save extra info"),
        cmd("facts", "show learned facts"),
        cmd("trip", "run the travel-weather flow"),
        cmd("support", "ask product support"),
        cmd("watches", "list active watches"),
        cmd("unwatch", "stop a watch"),
        cmd("compact", "clear short context and active flow"),
        cmd("reset", "reset chat memory"),
    ]
}

fn root_commands() -> Vec<BotCommand> {
    let mut commands = base_commands();
    commands.extend([
        cmd("connect", "connect an MCP server"),
        cmd("mcps", "list MCP servers"),
        cmd("tools", "list MCP tools"),
        cmd("call", "call an MCP tool"),
        cmd("watch", "create an admin watch"),
        cmd("disconnect", "disconnect an MCP server"),
    ]);
    commands
}

pub async fn set_public_commands(bot: &Bot) -> anyhow::Result<()> {
    let commands = base_commands();
    bot.set_my_commands(commands.clone()).await?;
    bot.set_my_commands(commands)
        .scope(BotCommandScope::AllPrivateChats)
        .await?;
    Ok(())
}

pub async fn set_chat_commands(bot: &Bot, chat: ChatId, is_root: bool) -> anyhow::Result<()> {
    let commands = if is_root {
        root_commands()
    } else {
        base_commands()
    };
    bot.set_my_commands(commands)
        .scope(BotCommandScope::Chat {
            chat_id: Recipient::Id(chat),
        })
        .await?;
    Ok(())
}

pub async fn sync_authorized_command_menus(bot: &Bot, state: &BotState) -> anyhow::Result<()> {
    let access = state.access_snapshot().await;
    let mut chat_ids = access.authorized_chat_ids;
    if let Some(root_chat_id) = access.root_chat_id {
        if !chat_ids.contains(&root_chat_id) {
            chat_ids.push(root_chat_id);
        }
    }
    for chat_id in chat_ids {
        if let Err(e) =
            set_chat_commands(bot, ChatId(chat_id), access.root_chat_id == Some(chat_id)).await
        {
            tracing::warn!("sync commands for chat {chat_id}: {e:#}");
        }
    }
    Ok(())
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

/// Keep the Telegram "typing…" indicator alive (it expires after ~5s) by
/// re-sending the chat action every 4s until the returned task is aborted.
fn spawn_typing(bot: Bot, chat: ChatId) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let _ = bot
                .send_chat_action(chat, teloxide::types::ChatAction::Typing)
                .await;
            tokio::time::sleep(std::time::Duration::from_secs(4)).await;
        }
    })
}

/// Non-command text: ask the LLM agent (it uses MCP tools), or point to /help
/// if no LLM is configured.
async fn handle_text(bot: Bot, msg: Message, state: BotState) -> anyhow::Result<()> {
    let chat = msg.chat.id;
    if !state.is_authorized(chat.0).await {
        return Ok(());
    }
    let text = msg.text().unwrap_or("").trim().to_string();
    if text.is_empty() {
        return Ok(());
    }
    let Some(llm) = state.llm.clone() else {
        bot.send_message(
            chat,
            "Send /help to see commands. (No LLM configured for free-form questions.)",
        )
        .await?;
        return Ok(());
    };

    let typing = spawn_typing(bot.clone(), chat);
    bot.send_message(chat, "⏳ Обрабатываю запрос…").await.ok();
    // Live progress: the trip swarm streams "step in progress" lines; relay each
    // to Telegram as it arrives so a multi-minute turn is never silent.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let prog_bot = bot.clone();
    let prog_task = tokio::spawn(async move {
        while let Some(line) = rx.recv().await {
            let _ = prog_bot.send_message(chat, line).await;
        }
    });

    let mut session = crate::agent::session::load(chat.0);
    let outcome = crate::agent::run_turn(&llm, &state, &mut session, &text, Some(tx)).await;
    typing.abort();
    let _ = prog_task.await;
    match outcome {
        Ok(result) => {
            if let Err(e) = crate::agent::session::save(&session) {
                tracing::error!("save session {}: {e}", chat.0);
            }
            // Swarm trace is internal telemetry (agent names, raw stage outputs,
            // timeouts) — log it, never surface it to the user.
            if !result.trace.is_empty() {
                tracing::debug!("swarm trace {}:\n{}", chat.0, result.trace.join("\n"));
            }
            // Never send an empty message (Telegram rejects it -> silent failure).
            let answer = if result.answer.trim().is_empty() {
                "✅ Готово.".to_string()
            } else {
                crate::agent::flow::linkify_geo(&result.answer)
            };
            for chunk in split_chunks(&answer, 3900) {
                if chunk.trim().is_empty() {
                    continue;
                }
                bot.send_message(chat, chunk).await?;
            }
        }
        Err(e) => {
            // `{e:#}` renders the full anyhow cause chain (stage → reason),
            // e.g. "answering failed: LLM request failed: operation timed out".
            // Plain `{e}` would drop everything below the top context and is
            // why hangs/errors used to surface as an unhelpful one-liner.
            tracing::error!("agent turn for chat {}: {e:#}", chat.0);
            bot.send_message(
                chat,
                "❌ Не удалось обработать запрос. Попробуйте ещё раз через минуту.",
            )
            .await?;
        }
    }
    Ok(())
}

async fn handle_command(
    bot: Bot,
    msg: Message,
    cmd: Command,
    state: BotState,
) -> anyhow::Result<()> {
    let chat = msg.chat.id;
    let telegram_user_id = msg.from().map(|user| user.id.0);
    if !allow_command(&bot, chat, &state, &cmd).await? {
        return Ok(());
    }
    match cmd {
        Command::Start(_) => {
            state.subscribe(chat.0).await;
            let is_root = state.is_root(chat.0).await;
            if let Err(e) = set_chat_commands(&bot, chat, is_root).await {
                tracing::warn!("set commands for chat {}: {e:#}", chat.0);
            }
            let role = if is_root { "root" } else { "user" };
            bot.send_message(chat, format!("✅ Authorized as {role}. Send /help."))
                .await?;
        }
        Command::Help => {
            send_help(&bot, chat, state.is_root(chat.0).await).await?;
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
        Command::Call(args) => {
            handle_call(&bot, chat, &state, &args).await?;
        }
        Command::Watch(args) => {
            handle_watch(&bot, chat, &state, &args).await?;
        }
        Command::Unwatch(arg) => {
            let arg = arg.trim();
            if arg.eq_ignore_ascii_case("all") {
                let n = if state.is_root(chat.0).await {
                    state.remove_all_watches().await
                } else {
                    state.remove_watches_for_chat(chat.0).await
                };
                bot.send_message(chat, format!("✅ stopped {n} watch(es)."))
                    .await?;
            } else {
                match arg.parse::<u64>() {
                    Ok(id) => {
                        let removed = if state.is_root(chat.0).await {
                            state.remove_watch(id).await
                        } else {
                            state.remove_watch_for_chat(chat.0, id).await
                        };
                        if removed {
                            bot.send_message(chat, format!("✅ watch #{id} stopped."))
                                .await?;
                        } else {
                            bot.send_message(chat, format!("No watch #{id}. See /watches."))
                                .await?;
                        }
                    }
                    Err(_) => {
                        bot.send_message(chat, "Usage: /unwatch <id> | /unwatch all")
                            .await?;
                    }
                }
            }
        }
        Command::Watches => {
            let watches = if state.is_root(chat.0).await {
                state.list_watches().await
            } else {
                state.list_watches_for_chat(chat.0).await
            };
            if watches.is_empty() {
                let msg = if state.is_root(chat.0).await {
                    "No active watches. Use /watch."
                } else {
                    "No active watches."
                };
                bot.send_message(chat, msg).await?;
            } else {
                let mut lines = vec![format!("Active watches ({}):", watches.len())];
                for w in &watches {
                    lines.push(format!(
                        "#{} — {}/{} every {}m",
                        w.id, w.server, w.tool, w.interval_min
                    ));
                }
                bot.send_message(chat, lines.join("\n")).await?;
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
        Command::Profile(args) => {
            handle_profile(&bot, chat, &args).await?;
        }
        Command::Info(args) => {
            handle_info(&bot, chat, &args).await?;
        }
        Command::Facts => {
            handle_facts(&bot, chat).await?;
        }
        Command::Trip(args) => {
            handle_trip(&bot, chat, &state, &args).await?;
        }
        Command::Support(question) => {
            handle_support(&bot, chat, telegram_user_id, &question).await?;
        }
        Command::Reset => {
            let mut session = crate::agent::session::load(chat.0);
            session.memory.reset_for_new_session();
            session.trip = None;
            let _ = crate::agent::session::save(&session);
            bot.send_message(chat, "✅ Chat memory reset (long-term facts kept).")
                .await?;
        }
        Command::Compact => {
            compact_chat_context(&bot, chat).await?;
        }
    }
    Ok(())
}

async fn allow_command(
    bot: &Bot,
    chat: ChatId,
    state: &BotState,
    cmd: &Command,
) -> anyhow::Result<bool> {
    if let Command::Start(args) = cmd {
        let args = args.trim();
        if state.is_authorized(chat.0).await {
            return Ok(true);
        }
        if state.authorize(chat.0, args).await {
            return Ok(true);
        }
        if args.is_empty() {
            bot.send_message(chat, "🔒 Send /start <password> to unlock.")
                .await?;
        }
        return Ok(false);
    }

    if !state.is_authorized(chat.0).await {
        return Ok(false);
    }

    if is_admin_command(cmd) && !state.is_root(chat.0).await {
        bot.send_message(chat, "⛔ This command is available only to the bot owner.")
            .await?;
        return Ok(false);
    }

    Ok(true)
}

fn is_admin_command(cmd: &Command) -> bool {
    matches!(
        cmd,
        Command::Connect(_)
            | Command::Mcps
            | Command::Tools(_)
            | Command::Call(_)
            | Command::Watch(_)
            | Command::Disconnect(_)
    )
}

async fn send_help(bot: &Bot, chat: ChatId, is_root: bool) -> anyhow::Result<()> {
    let mut lines = vec![
        "Commands:".to_string(),
        "/help — show help".to_string(),
        "/profile [key value | clear] — view/set your profile".to_string(),
        "/info [label text | clear] — save extra preferences".to_string(),
        "/facts — show your learned facts".to_string(),
        "/trip <cities/dates> — travel-weather flow".to_string(),
        "/support <question> — ask product support".to_string(),
        "/watches — list your active watches".to_string(),
        "/unwatch <id> | all — stop your watches".to_string(),
        "/compact — clear short context and active flow, keep profile and durable facts"
            .to_string(),
        "/reset — reset chat memory, keep long-term facts".to_string(),
    ];
    if is_root {
        lines.extend(
            [
                "",
                "Owner commands:",
                "/connect <url> ... | /connect stdio <program> ...",
                "/mcps — list connected MCP servers",
                "/tools [server] — list MCP tools",
                "/call <server> <tool> [json-args] — raw tool call",
                "/watch <server> <tool> <minutes> [json-args] — admin watch",
                "/disconnect <name> — disconnect MCP server",
            ]
            .into_iter()
            .map(str::to_string),
        );
    }
    bot.send_message(chat, lines.join("\n")).await?;
    Ok(())
}

async fn handle_support(
    bot: &Bot,
    chat: ChatId,
    telegram_user_id: Option<u64>,
    question: &str,
) -> anyhow::Result<()> {
    let question = question.trim();
    if question.is_empty() {
        bot.send_message(chat, "Использование: /support <вопрос>")
            .await?;
        return Ok(());
    }
    let Some(telegram_user_id) = telegram_user_id else {
        bot.send_message(chat, "Поддержка доступна только в личном чате.")
            .await?;
        return Ok(());
    };
    let typing = spawn_typing(bot.clone(), chat);
    let response = crate::support::answer(question, telegram_user_id).await;
    typing.abort();
    match response {
        Ok(answer) => {
            for chunk in split_chunks(&answer, 3900) {
                if !chunk.trim().is_empty() {
                    bot.send_message(chat, chunk).await?;
                }
            }
        }
        Err(error) => {
            tracing::error!("support request for chat {}: {error:#}", chat.0);
            bot.send_message(chat, "Поддержка временно недоступна. Попробуйте позже.")
                .await?;
        }
    }
    Ok(())
}

/// `/profile` — show; `/profile <key> <value>` — set; `/profile clear`.
async fn handle_profile(bot: &Bot, chat: ChatId, args: &str) -> anyhow::Result<()> {
    let mut session = crate::agent::session::load(chat.0);
    let args = args.trim();
    if args.is_empty() {
        let p = &session.profile;
        if p.is_empty() {
            bot.send_message(
                chat,
                "Profile is empty. Set with: /profile <key> <value>\nKnown keys: home_city, preferred_units, comfort_temp_min, comfort_temp_max, dislikes_rain, interests, language",
            )
            .await?;
        } else {
            bot.send_message(
                chat,
                format!("👤 Profile:\n{}", p.render_lines().join("\n")),
            )
            .await?;
        }
        return Ok(());
    }
    if args.eq_ignore_ascii_case("clear") {
        session.profile.clear();
        let _ = crate::agent::session::save(&session);
        bot.send_message(chat, "✅ Profile cleared.").await?;
        return Ok(());
    }
    let (key, value) = match args.split_once(char::is_whitespace) {
        Some((k, v)) => (k, v.trim()),
        None => {
            bot.send_message(chat, "Usage: /profile <key> <value>  |  /profile clear")
                .await?;
            return Ok(());
        }
    };
    session.profile.set(key, value);
    let _ = crate::agent::session::save(&session);
    bot.send_message(chat, format!("✅ profile.{key} = {value}"))
        .await?;
    Ok(())
}

/// `/info` — show; `/info <label> <text>` — set; `/info clear`.
/// Free-form labelled preferences ("доп инфа"). Unlike /profile these are only
/// mixed into the prompt when the router judges them relevant to a turn.
async fn handle_info(bot: &Bot, chat: ChatId, args: &str) -> anyhow::Result<()> {
    let mut session = crate::agent::session::load(chat.0);
    let args = args.trim();
    if args.is_empty() {
        let n = &session.notes;
        if n.is_empty() {
            bot.send_message(
                chat,
                "No extra info saved. Add some with: /info <label> <text>\n\
                 e.g. /info files Файлы в формате .docx, имя с датой\n\
                 The agent uses a note only when it's relevant to your request.",
            )
            .await?;
        } else {
            bot.send_message(
                chat,
                format!("📝 Extra info:\n{}", n.render_lines().join("\n")),
            )
            .await?;
        }
        return Ok(());
    }
    if args.eq_ignore_ascii_case("clear") {
        session.notes.clear();
        let _ = crate::agent::session::save(&session);
        bot.send_message(chat, "✅ Extra info cleared.").await?;
        return Ok(());
    }
    let (label, text) = match args.split_once(char::is_whitespace) {
        Some((l, t)) => (l, t.trim()),
        None => {
            bot.send_message(
                chat,
                "Usage: /info <label> <text>  |  /info clear\n(empty text removes a label)",
            )
            .await?;
            return Ok(());
        }
    };
    session.notes.set(label, text);
    let _ = crate::agent::session::save(&session);
    bot.send_message(chat, format!("✅ info.{} saved.", label.to_lowercase()))
        .await?;
    Ok(())
}

/// `/facts` — render the layered memory.
async fn handle_facts(bot: &Bot, chat: ChatId) -> anyhow::Result<()> {
    use crate::agent::memory::MemoryLayer;
    let session = crate::agent::session::load(chat.0);
    if session.memory.facts.is_empty() {
        bot.send_message(chat, "No facts learned yet.").await?;
        return Ok(());
    }
    let mut lines = vec!["🧠 Layered memory:".to_string()];
    for layer in MemoryLayer::ORDERED {
        let facts = session.memory.facts_in_layer(layer);
        if facts.is_empty() {
            continue;
        }
        lines.push(format!("\n[{layer}]"));
        for f in facts {
            lines.push(format!("• {}: {}", f.key, f.value));
        }
    }
    bot.send_message(chat, lines.join("\n")).await?;
    Ok(())
}

async fn compact_chat_context(bot: &Bot, chat: ChatId) -> anyhow::Result<()> {
    let mut session = crate::agent::session::load(chat.0);
    let before_recent = session.memory.recent.len();
    let had_summary = !session.memory.summary.trim().is_empty();
    let had_flow = session.trip.is_some();
    session.memory.clear_short_context();
    session.trip = None;
    let _ = crate::agent::session::save(&session);
    let summary_note = if had_summary {
        " and compacted summary"
    } else {
        ""
    };
    let flow_note = if had_flow {
        " Active trip flow cleared."
    } else {
        ""
    };
    bot.send_message(
        chat,
        format!(
            "✅ Cleared short context ({before_recent} recent message(s){summary_note}).{flow_note} Profile and durable facts kept."
        ),
    )
    .await?;
    Ok(())
}

/// `/trip` — run the multi-agent travel-weather pipeline.
async fn handle_trip(bot: &Bot, chat: ChatId, state: &BotState, args: &str) -> anyhow::Result<()> {
    let Some(llm) = state.llm.clone() else {
        bot.send_message(chat, "No LLM configured (set DEEPSEEK_API_KEY).")
            .await?;
        return Ok(());
    };
    let kick = if args.trim().is_empty() {
        "Хочу спланировать поездку"
    } else {
        args
    };
    // Explicit command just force-starts the same stateful swarm; the normal
    // text path then continues it across turns (clarify → plan → … → doc).
    let typing = spawn_typing(bot.clone(), chat);
    bot.send_message(chat, "⏳ Запускаю travel-flow…")
        .await
        .ok();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let prog_bot = bot.clone();
    let prog_task = tokio::spawn(async move {
        while let Some(line) = rx.recv().await {
            let _ = prog_bot.send_message(chat, line).await;
        }
    });
    let mut session = crate::agent::session::load(chat.0);
    session.trip = Some(crate::agent::flow::TripFlowState::start());
    let outcome = crate::agent::run_turn(&llm, state, &mut session, kick, Some(tx)).await;
    typing.abort();
    let _ = prog_task.await;
    match outcome {
        Ok(result) => {
            if let Err(e) = crate::agent::session::save(&session) {
                tracing::error!("save session {}: {e}", chat.0);
            }
            if !result.trace.is_empty() {
                tracing::debug!("swarm trace {}:\n{}", chat.0, result.trace.join("\n"));
            }
            let answer = if result.answer.trim().is_empty() {
                "✅ Готово.".to_string()
            } else {
                crate::agent::flow::linkify_geo(&result.answer)
            };
            for chunk in split_chunks(&answer, 3900) {
                if chunk.trim().is_empty() {
                    continue;
                }
                bot.send_message(chat, chunk).await?;
            }
        }
        Err(e) => {
            tracing::error!("trip flow for chat {}: {e:#}", chat.0);
            bot.send_message(
                chat,
                "❌ Не удалось обработать поездку. Попробуйте ещё раз через минуту.",
            )
            .await?;
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
    if !state.is_authorized(chat.0).await {
        return Ok(());
    }
    if let Some(name) = data.strip_prefix("tools:") {
        if !state.is_root(chat.0).await {
            bot.send_message(chat, "⛔ MCP tools are available only to the bot owner.")
                .await?;
            return Ok(());
        }
        send_tools(&bot, chat, &state, name).await?;
    } else if let Some(name) = data.strip_prefix("disc:") {
        if !state.is_root(chat.0).await {
            bot.send_message(
                chat,
                "⛔ MCP management is available only to the bot owner.",
            )
            .await?;
            return Ok(());
        }
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
                    "❌ {e}\n\nUsage:\n\
                     HTTP:  /connect <url> [name=N] [auth=TOKEN] [Header:Value ...]\n\
                     stdio: /connect stdio <program> [args...] [name=N] [env=KEY=VAL ...]\n\n\
                     Examples:\n\
                     https://host/mcp auth=SECRET X-Tracker-Token:abc\n\
                     stdio npx -y @cocal/google-calendar-mcp name=gcal"
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

/// `/call <server> <tool> [json-args]`
async fn handle_call(bot: &Bot, chat: ChatId, state: &BotState, args: &str) -> anyhow::Result<()> {
    let (server, tool, json) = match parse_call(args) {
        Ok(t) => t,
        Err(e) => {
            bot.send_message(
                chat,
                format!(
                    "❌ {e}\n\nUsage:\n/call <server> <tool> [json-args]\n\n\
                     Example:\n/call weather geocode {{\"name\":\"Moscow\"}}"
                ),
            )
            .await?;
            return Ok(());
        }
    };
    bot.send_message(chat, format!("⏳ Calling {server}/{tool}…"))
        .await?;
    match state.call_tool(&server, &tool, json).await {
        Ok(text) => {
            let body = if text.is_empty() {
                "(empty result)".to_string()
            } else {
                text
            };
            for chunk in split_chunks(&format!("✅ {tool}:\n{body}"), 3900) {
                bot.send_message(chat, chunk).await?;
            }
        }
        Err(e) => {
            bot.send_message(chat, format!("❌ {e}")).await?;
        }
    }
    Ok(())
}

/// `/watch <server> <tool> <minutes> [json-args]`
async fn handle_watch(bot: &Bot, chat: ChatId, state: &BotState, args: &str) -> anyhow::Result<()> {
    let (server, tool, minutes, json) = match parse_watch(args) {
        Ok(t) => t,
        Err(e) => {
            bot.send_message(
                chat,
                format!(
                    "❌ {e}\n\nUsage:\n/watch <server> <tool> <minutes> [json-args]\n\n\
                     Example:\n/watch weather get_weather_summary 60 {{\"job_id\":\"abc\"}}"
                ),
            )
            .await?;
            return Ok(());
        }
    };
    // Validate the server exists before registering.
    if !state.mcp_names().await.iter().any(|n| n == &server) {
        bot.send_message(chat, format!("Unknown server '{server}'. See /mcps."))
            .await?;
        return Ok(());
    }
    let id = state
        .schedule_summary(chat.0, server.clone(), tool.clone(), json, minutes, None)
        .await;
    bot.send_message(
        chat,
        format!("✅ watch #{id}: {server}/{tool} every {minutes}m. First result shortly. /unwatch {id} to stop."),
    )
    .await?;
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
            bot.send_message(chat, format!("• {n} — {tc} tools\n{}", c.params.target()))
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

/// Parse `/connect` args. Two forms:
///
/// HTTP (URL-first, order-agnostic):
///   `/connect <url> [name=NAME] [auth=TOKEN] [Header:Value ...]`
///
/// stdio (spawn a child process — for npx/uvx servers):
///   `/connect stdio <program> [args...] [name=NAME] [env=KEY=VALUE ...]`
///
/// In stdio form the leading `stdio` keyword switches modes; every token that
/// is not `name=…` / `env=…` becomes part of the spawn command, in order.
fn parse_connect(args: &str) -> Result<ConnectParams, String> {
    let mut toks = args.split_whitespace().peekable();
    if toks.peek().is_some_and(|t| *t == "stdio") {
        toks.next(); // consume the keyword
        return parse_connect_stdio(toks);
    }

    let mut url: Option<String> = None;
    let mut name: Option<String> = None;
    let mut auth = None;
    let mut headers = Vec::new();

    for tok in toks {
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
        command: Vec::new(),
        env: Vec::new(),
    })
}

/// Parse the tail of a `/connect stdio …` command (keyword already consumed).
fn parse_connect_stdio<'a>(toks: impl Iterator<Item = &'a str>) -> Result<ConnectParams, String> {
    let mut name: Option<String> = None;
    let mut command: Vec<String> = Vec::new();
    let mut env: Vec<(String, String)> = Vec::new();

    for tok in toks {
        if let Some(v) = tok.strip_prefix("name=") {
            name = Some(v.to_string());
        } else if let Some(kv) = tok.strip_prefix("env=") {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| format!("bad env '{kv}' (expected env=KEY=VALUE)"))?;
            env.push((k.to_string(), v.to_string()));
        } else {
            // Everything else is part of the spawn command, order preserved.
            command.push(tok.to_string());
        }
    }

    if command.is_empty() {
        return Err("no program after 'stdio' — e.g. stdio npx -y <package>".into());
    }
    let name = name.unwrap_or_else(|| default_stdio_name(&command));
    Ok(ConnectParams {
        name,
        url: String::new(),
        auth: None,
        headers: Vec::new(),
        command,
        env,
    })
}

/// Derive a server name from a stdio command: last path segment of the last
/// argument (usually the package name), sanitized. Falls back to the program.
pub fn default_stdio_name(command: &[String]) -> String {
    let pick = command
        .iter()
        .rev()
        .find(|a| !a.starts_with('-'))
        .map(String::as_str)
        .unwrap_or(&command[0]);
    let seg = pick
        .rsplit(['/', '@'])
        .find(|s| !s.is_empty())
        .unwrap_or(pick);
    let cleaned: String = seg
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

/// Parse `/call <server> <tool> [json-args]`.
/// Returns (server, tool, optional JSON object of arguments).
#[allow(clippy::type_complexity)]
fn parse_call(args: &str) -> Result<(String, String, Option<rmcp::model::JsonObject>), String> {
    let args = args.trim();
    let mut it = args.splitn(3, char::is_whitespace);
    let server = it
        .next()
        .filter(|s| !s.is_empty())
        .ok_or("missing <server>")?;
    let tool = it
        .next()
        .filter(|s| !s.is_empty())
        .ok_or("missing <tool>")?;
    let json = match it.next().map(str::trim).filter(|s| !s.is_empty()) {
        None => None,
        Some(raw) => {
            let val: serde_json::Value =
                serde_json::from_str(raw).map_err(|e| format!("bad JSON args: {e}"))?;
            match val {
                serde_json::Value::Object(map) => Some(map),
                _ => return Err("json-args must be an object, e.g. {\"key\":\"value\"}".into()),
            }
        }
    };
    Ok((server.to_string(), tool.to_string(), json))
}

/// Parse `/watch <server> <tool> <minutes> [json-args]`.
#[allow(clippy::type_complexity)]
fn parse_watch(
    args: &str,
) -> Result<(String, String, u64, Option<rmcp::model::JsonObject>), String> {
    let mut it = args.trim().splitn(4, char::is_whitespace);
    let server = it
        .next()
        .filter(|s| !s.is_empty())
        .ok_or("missing <server>")?;
    let tool = it
        .next()
        .filter(|s| !s.is_empty())
        .ok_or("missing <tool>")?;
    let minutes = it
        .next()
        .filter(|s| !s.is_empty())
        .ok_or("missing <minutes>")?
        .parse::<u64>()
        .map_err(|_| "minutes must be a positive integer")?;
    if minutes == 0 {
        return Err("minutes must be >= 1".into());
    }
    let json = match it.next().map(str::trim).filter(|s| !s.is_empty()) {
        None => None,
        Some(raw) => match serde_json::from_str(raw).map_err(|e| format!("bad JSON args: {e}"))? {
            serde_json::Value::Object(map) => Some(map),
            _ => return Err("json-args must be an object".into()),
        },
    };
    Ok((server.to_string(), tool.to_string(), minutes, json))
}

/// Derive a short server name from a URL: host[:port], sanitized.
pub fn default_name(url: &str) -> String {
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
    fn parse_connect_stdio_basic_autoname() {
        let p = parse_connect("stdio npx -y @cocal/google-calendar-mcp").unwrap();
        assert!(p.is_stdio());
        assert_eq!(p.command, vec!["npx", "-y", "@cocal/google-calendar-mcp"]);
        assert_eq!(p.name, "google_calendar_mcp");
        assert!(p.url.is_empty());
    }

    #[test]
    fn parse_connect_stdio_name_and_env() {
        let p = parse_connect(
            "stdio uvx telegram-mcp name=tg env=TELEGRAM_API_ID=123 env=TELEGRAM_API_HASH=abc",
        )
        .unwrap();
        assert_eq!(p.name, "tg");
        assert_eq!(p.command, vec!["uvx", "telegram-mcp"]);
        assert_eq!(
            p.env,
            vec![
                ("TELEGRAM_API_ID".to_string(), "123".to_string()),
                ("TELEGRAM_API_HASH".to_string(), "abc".to_string()),
            ]
        );
    }

    #[test]
    fn parse_connect_stdio_empty_command_errors() {
        assert!(parse_connect("stdio name=x").is_err());
    }

    #[test]
    fn parse_connect_stdio_bad_env_errors() {
        assert!(parse_connect("stdio npx pkg env=NOEQUALS").is_err());
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
    fn parse_call_with_json() {
        let (s, t, j) = parse_call("weather geocode {\"name\":\"Moscow\"}").unwrap();
        assert_eq!(s, "weather");
        assert_eq!(t, "geocode");
        assert_eq!(j.unwrap().get("name").unwrap(), "Moscow");
    }

    #[test]
    fn parse_call_no_args() {
        let (s, t, j) = parse_call("weather list_jobs").unwrap();
        assert_eq!((s.as_str(), t.as_str()), ("weather", "list_jobs"));
        assert!(j.is_none());
    }

    #[test]
    fn parse_call_missing_tool() {
        assert!(parse_call("weather").is_err());
    }

    #[test]
    fn parse_call_bad_json() {
        assert!(parse_call("w t {not json}").is_err());
    }

    #[test]
    fn parse_call_non_object_json() {
        assert!(parse_call("w t [1,2,3]").is_err());
    }

    #[test]
    fn parse_watch_full() {
        let (s, t, m, j) =
            parse_watch("weather get_weather_summary 60 {\"job_id\":\"x\"}").unwrap();
        assert_eq!(
            (s.as_str(), t.as_str(), m),
            ("weather", "get_weather_summary", 60)
        );
        assert_eq!(j.unwrap().get("job_id").unwrap(), "x");
    }

    #[test]
    fn parse_watch_no_json() {
        let (s, t, m, j) = parse_watch("weather list_jobs 30").unwrap();
        assert_eq!((s.as_str(), t.as_str(), m), ("weather", "list_jobs", 30));
        assert!(j.is_none());
    }

    #[test]
    fn parse_watch_bad_minutes() {
        assert!(parse_watch("w t notnum").is_err());
        assert!(parse_watch("w t 0").is_err());
    }

    #[test]
    fn parse_watch_missing_parts() {
        assert!(parse_watch("w").is_err());
        assert!(parse_watch("w t").is_err());
    }

    #[test]
    fn html_escape_specials() {
        assert_eq!(html_escape("a & b < c > d"), "a &amp; b &lt; c &gt; d");
        assert_eq!(html_escape("plain"), "plain");
    }
}
