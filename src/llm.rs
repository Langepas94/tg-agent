//! Minimal OpenAI-compatible LLM client + an agentic tool-calling loop that
//! lets a user ask in natural language and get a human answer backed by MCP
//! tools (default provider: DeepSeek).

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

use crate::{config::LlmConfig, state::BotState};

/// Default tool-loop budget for an ordinary chat turn.
pub const MAX_STEPS: usize = 12;
/// Budget for a trip-swarm worker stage. A stage needs a handful of OSM queries
/// (geocode + a few small-bbox lookups), then it must COMMIT. A high cap just
/// let a chatty model fire 20 slow queries and blow the wall-clock, so keep it
/// tight and pair it with "commit, don't over-verify" stage instructions.
pub const STAGE_MAX_STEPS: usize = 10;
/// Per-tool-result char cap before feeding it back to the model. Sized for a
/// 7-day hourly forecast (~10–15k chars); `fit_to_context` reclaims the window
/// if many such results accumulate across the loop.
const TOOL_RESULT_CAP: usize = 12_000;
/// Hard ceiling on a single LLM HTTP round. Without this, `reqwest` waits
/// forever on a stalled connection — the cause of the multi-minute hangs.
const LLM_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Clone)]
pub struct Llm {
    cfg: LlmConfig,
    http: reqwest::Client,
}

impl Llm {
    pub fn new(cfg: LlmConfig) -> Self {
        Self {
            cfg,
            http: reqwest::Client::builder()
                .timeout(LLM_TIMEOUT)
                .connect_timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Configured model id (used for context-window budgeting).
    pub fn model(&self) -> &str {
        &self.cfg.model
    }

    /// Single completion with a custom system prompt and no tools.
    /// Used by staged service agents (planner, fact/profile extractor).
    pub async fn complete(&self, system: &str, user: &str) -> Result<String> {
        let messages = vec![
            json!({ "role": "system", "content": system }),
            json!({ "role": "user", "content": user }),
        ];
        let msg = self.chat(&messages, &[]).await?;
        Ok(msg["content"].as_str().unwrap_or("").trim().to_string())
    }

    /// One chat-completions round. Returns the assistant `message` object.
    async fn chat(&self, messages: &[Value], tools: &[Value]) -> Result<Value> {
        let mut body = json!({
            "model": self.cfg.model,
            "messages": messages,
        });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!("auto");
        }

        let url = format!(
            "{}/chat/completions",
            self.cfg.base_url.trim_end_matches('/')
        );
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                // Turn reqwest's opaque error into something a human can act on.
                let why = if e.is_timeout() {
                    format!(
                        "no response from {} within {LLM_TIMEOUT:?} (provider slow or hung)",
                        self.cfg.base_url
                    )
                } else if e.is_connect() {
                    format!(
                        "cannot reach LLM at {} (network/DNS/down)",
                        self.cfg.base_url
                    )
                } else {
                    format!("request to {} failed: {e}", self.cfg.base_url)
                };
                anyhow::anyhow!(why)
            })?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("LLM HTTP {status}: {}", truncate(&text, 400));
        }
        let v: Value = serde_json::from_str(&text).context("LLM returned non-JSON")?;
        v["choices"][0]["message"]
            .as_object()
            .map(|m| Value::Object(m.clone()))
            .context("LLM response missing choices[0].message")
    }

    /// Agentic loop with a default system prompt.
    pub async fn answer(&self, state: &BotState, user_text: &str) -> Result<String> {
        let system = "You are a helpful assistant with access to MCP tools. \
            When a question needs live data, call the appropriate tool(s). \
            Resolve place names to coordinates with a geocode tool before \
            weather tools if required. Answer concisely in the user's language. \
            Never show raw JSON — summarize results in human-readable prose.";
        self.answer_with_system(state, system, user_text, MAX_STEPS)
            .await
    }

    /// Agentic loop with a caller-supplied system prompt and no chat context
    /// (no self-scheduling meta-tool).
    pub async fn answer_with_system(
        &self,
        state: &BotState,
        system: &str,
        user_text: &str,
        max_steps: usize,
    ) -> Result<String> {
        self.answer_in_chat(state, system, user_text, &[], None, max_steps)
            .await
    }

    /// Agentic tool-calling loop. When `chat_id` is set, the agent also gets a
    /// `schedule_summary` meta-tool so it can subscribe the user to periodic
    /// updates itself (no separate /watch needed).
    pub async fn answer_in_chat(
        &self,
        state: &BotState,
        system: &str,
        user_text: &str,
        history: &[(&str, &str)],
        chat_id: Option<i64>,
        max_steps: usize,
    ) -> Result<String> {
        let (mut tool_defs, mut tool_map) = collect_tools(state).await;
        // Tools that accept a `session_id` param — the bot forces it to the
        // chat id so jobs are chat-isolated AND server pushes route back here.
        let mut session_tools = tools_with_session_param(&tool_defs);
        append_meta_defs(&mut tool_defs, chat_id);

        let mut messages = vec![json!({ "role": "system", "content": system })];
        // Prior turns (short-term window) give multi-turn continuity.
        for (role, text) in history {
            let role = if *role == "assistant" {
                "assistant"
            } else {
                "user"
            };
            messages.push(json!({ "role": role, "content": text }));
        }
        messages.push(json!({ "role": "user", "content": user_text }));

        for _ in 0..max_steps {
            // Keep the growing tool transcript under the model's window. MCP
            // results (multi-city forecasts) can overflow it mid-loop; clip the
            // oldest results BEFORE the call so the provider never 400s.
            fit_to_context(&mut messages, &self.cfg.model);
            let msg = self.chat(&messages, &tool_defs).await?;

            let tool_calls = msg.get("tool_calls").and_then(|v| v.as_array()).cloned();
            match tool_calls {
                Some(calls) if !calls.is_empty() => {
                    messages.push(msg.clone()); // assistant turn carrying the calls
                                                // The agent may connect/disconnect servers mid-loop; when it
                                                // does, the live tool set changes and must be rebuilt before
                                                // the next model call so new tools become callable at once.
                    let mut tools_dirty = false;
                    for call in calls {
                        let id = call["id"].as_str().unwrap_or("").to_string();
                        let name = call["function"]["name"].as_str().unwrap_or("").to_string();
                        let args_raw = call["function"]["arguments"].as_str().unwrap_or("{}");
                        let mut args = parse_args(args_raw);

                        // Force session_id = chat id on session-scoped tools.
                        if let (Some(cid), true) = (chat_id, session_tools.contains(&name)) {
                            let map = args.get_or_insert_with(serde_json::Map::new);
                            map.insert("session_id".into(), cid.to_string().into());
                        }

                        let content = if name == "mcp_connect" {
                            let out = handle_mcp_connect(state, args_raw).await;
                            tools_dirty |= !out.starts_with("error");
                            out
                        } else if name == "mcp_disconnect" {
                            let out = handle_mcp_disconnect(state, args_raw).await;
                            tools_dirty |= !out.starts_with("error");
                            out
                        } else if name == "schedule_summary" {
                            handle_schedule_summary(state, chat_id, args_raw).await
                        } else if name == "cancel_summary" {
                            handle_cancel_summary(state, chat_id).await
                        } else {
                            match tool_map.get(&name) {
                                None => format!("error: tool '{name}' is not available"),
                                Some((server, real_tool)) => {
                                    normalize_tool_args(server, real_tool, &mut args);
                                    let period = args
                                        .as_ref()
                                        .and_then(|m| m.get("period"))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("1h")
                                        .to_string();
                                    let res = state.call_tool(server, real_tool, args).await;
                                    // Record/clear durable push-subs so they survive restarts.
                                    if let (Some(cid), Ok(_)) = (chat_id, &res) {
                                        if real_tool == "subscribe_summaries" {
                                            state.add_push_sub(cid, server.clone(), period).await;
                                        } else if real_tool == "unsubscribe_summaries" {
                                            state.remove_push_subs(cid, Some(server)).await;
                                        }
                                    }
                                    match res {
                                        Ok(out) => truncate(&out, TOOL_RESULT_CAP),
                                        Err(e) => format!("error: {e}"),
                                    }
                                }
                            }
                        };
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": id,
                            "content": content,
                        }));
                    }
                    // Rebuild the tool registry if the agent changed connections.
                    if tools_dirty {
                        let (d, m) = collect_tools(state).await;
                        tool_defs = d;
                        tool_map = m;
                        session_tools = tools_with_session_param(&tool_defs);
                        append_meta_defs(&mut tool_defs, chat_id);
                    }
                }
                _ => {
                    let answer = msg["content"].as_str().unwrap_or("").trim().to_string();
                    if answer.is_empty() {
                        return Ok("(no answer)".into());
                    }
                    return Ok(answer);
                }
            }
        }
        Ok("Stopped after too many tool calls. Try rephrasing.".into())
    }
}

/// Clip the running tool-loop transcript back under the model's compaction
/// threshold. MCP tool results (e.g. multi-city forecasts) can blow the context
/// window mid-loop; rather than let the provider reject the request with a 400,
/// we shrink the OLDEST `tool` results first — preserving the system prompt and
/// the most recent exchange the model is actively reasoning over.
fn fit_to_context(messages: &mut [Value], model: &str) {
    use crate::agent::context_budget::{compact_threshold, estimate_tokens};

    let budget = compact_threshold(model);
    let total = |m: &[Value]| -> usize {
        m.iter()
            .map(|v| estimate_tokens(v["content"].as_str().unwrap_or("")))
            .sum()
    };
    if total(messages) <= budget {
        return;
    }

    // Never touch the system prompt (index 0) or the last two messages (the
    // live exchange). Clip from oldest to newest until we fit.
    const CLIP_CHARS: usize = 400;
    let last_keep = messages.len().saturating_sub(2);
    for i in 1..last_keep {
        if total(messages) <= budget {
            break;
        }
        if messages[i]["role"] != "tool" {
            continue;
        }
        let Some(c) = messages[i]["content"].as_str() else {
            continue;
        };
        if c.chars().count() <= CLIP_CHARS {
            continue;
        }
        // char-safe clip (tool output may be Cyrillic/multi-byte).
        let head: String = c.chars().take(CLIP_CHARS).collect();
        messages[i]["content"] =
            format!("{head}…[older tool result trimmed to fit context]").into();
    }
}

/// Build OpenAI tool definitions from every connected server's tools, plus a
/// map from the EXPOSED name to `(server, real_tool)`. Tool names are namespaced
/// `{server}__{tool}` so identically-named tools on different servers never
/// collide — the LLM addresses each unambiguously and we route by the map
/// instead of guessing the owner.
async fn collect_tools(
    state: &BotState,
) -> (
    Vec<Value>,
    std::collections::HashMap<String, (String, String)>,
) {
    let mut defs = Vec::new();
    let mut map: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();
    let guard = state.mcps.lock().await;
    for (server, client) in guard.iter() {
        for t in client.tools().await {
            let real = t.name.to_string();
            let exposed = unique_exposed_name(server, &real, &map);
            defs.push(json!({
                "type": "function",
                "function": {
                    "name": exposed,
                    "description": t.description.as_deref().unwrap_or(""),
                    "parameters": Value::Object((*t.input_schema).clone()),
                }
            }));
            map.insert(exposed, (server.clone(), real));
        }
    }
    (defs, map)
}

/// Sanitize to the OpenAI function-name charset `[A-Za-z0-9_-]`.
fn sanitize_fn(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `{server}__{tool}`, sanitized and capped to 64 chars (OpenAI limit), then
/// de-duplicated against already-collected names with a numeric suffix.
fn unique_exposed_name(
    server: &str,
    tool: &str,
    taken: &std::collections::HashMap<String, (String, String)>,
) -> String {
    let mut base = format!("{}__{}", sanitize_fn(server), sanitize_fn(tool));
    base.truncate(64);
    if !taken.contains_key(&base) {
        return base;
    }
    for i in 2..1000 {
        let suffix = format!("_{i}");
        let keep = 64 - suffix.len();
        let mut cand = base.clone();
        cand.truncate(keep);
        cand.push_str(&suffix);
        if !taken.contains_key(&cand) {
            return cand;
        }
    }
    base
}

/// Meta-tool definition: lets the agent subscribe the user to a recurring
/// summary (it sets up the periodic poll itself).
fn schedule_summary_def() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "schedule_summary",
            "description": "Subscribe the user to a RECURRING update: every <minutes> minutes, \
                call <server>/<tool> with <args> and automatically send the result to the user. \
                Call this whenever the user asks to be kept posted, to receive data periodically, \
                or to get a regular summary — so they don't have to ask each time. \
                For weather collection, FIRST call the list-jobs tool and REUSE an existing running \
                job for the same location/variables instead of creating a duplicate; only schedule a \
                new collection job if none exists. Then call this with the summary tool \
                (e.g. get_weather_summary) and that job_id in args. \
                IMPORTANT — keep the cadences consistent: if the summary tool takes an aggregation \
                window/period argument, set it to MATCH this delivery interval (e.g. minutes=10 → \
                period covering ~10 minutes), not a longer window, otherwise consecutive summaries \
                overlap and repeat. In your confirmation to the user, state all three clearly: how \
                often data is COLLECTED, how often a SUMMARY is sent, and the window each summary \
                covers. \
                If you created (or are reusing) an MCP resource that should be torn down when the \
                user unsubscribes (e.g. a collection cron job), set cleanup_tool/cleanup_args to the \
                tool+args that cancel it (e.g. cancel_job with the job_id) so /unwatch removes both \
                the delivery and the underlying job — no orphans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": { "type": "string", "description": "connected MCP server name" },
                    "tool": { "type": "string", "description": "tool to call each interval" },
                    "minutes": { "type": "integer", "description": "interval in minutes (>=1)" },
                    "args": { "type": "object", "description": "JSON arguments for the tool" },
                    "cleanup_tool": { "type": "string", "description": "tool that cancels the underlying job on unsubscribe (optional)" },
                    "cleanup_args": { "type": "object", "description": "args for cleanup_tool, e.g. {job_id, session_id} (optional)" }
                },
                "required": ["server", "tool", "minutes"]
            }
        }
    })
}

/// Meta-tool: cancel the user's recurring summary subscription(s).
fn cancel_summary_def() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "cancel_summary",
            "description": "Cancel/stop the user's recurring summary subscription(s). \
                Use this whenever the user asks to unsubscribe, stop receiving updates, \
                stop the periodic summary, or 'отмени подписку'. This stops the periodic \
                delivery AND tears down the underlying collection job (no orphans). \
                After calling it, confirm to the user in their language.",
            "parameters": { "type": "object", "properties": {} }
        }
    })
}

/// Append the always-available meta-tools (connect/disconnect MCP servers) and,
/// when running inside a chat, the recurring-summary meta-tools.
fn append_meta_defs(tool_defs: &mut Vec<Value>, chat_id: Option<i64>) {
    tool_defs.push(mcp_connect_def());
    tool_defs.push(mcp_disconnect_def());
    if chat_id.is_some() {
        tool_defs.push(schedule_summary_def());
        tool_defs.push(cancel_summary_def());
    }
}

/// Meta-tool: let the agent connect to ANY MCP server on demand. The agent
/// supplies the launch spec itself — for popular servers it knows the `npx`/
/// `uvx` command from training; for HTTP servers it gives the URL. Credentials
/// are NOT hardcoded: when a server needs a token/key, the agent should ask the
/// user in chat and pass the value here via `auth`/`headers`/`env`.
fn mcp_connect_def() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "mcp_connect",
            "description": "Connect to an MCP server so its tools become available. Use this whenever \
                fulfilling the request needs a capability no currently-connected server provides \
                (e.g. the user asks to add a calendar event, send mail, read files) — connect the \
                right server, then call its tools. Pick the transport: \
                'http' for a remote Streamable-HTTP endpoint (give `url`); \
                'stdio' to launch a local server process (give `command` as an argv array, e.g. \
                [\"npx\",\"-y\",\"@cocal/google-calendar-mcp\"] or [\"uvx\",\"some-mcp\"]). \
                For well-known servers supply the standard command from your own knowledge. \
                If the server needs credentials, FIRST ask the user for them in chat, then pass them \
                via `auth` (bearer token), `headers` (HTTP servers), or `env` (stdio servers). \
                Never invent secrets. After connecting, the new tools are immediately callable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "short server alias (optional; derived if omitted)" },
                    "transport": { "type": "string", "enum": ["http", "stdio"], "description": "connection type" },
                    "url": { "type": "string", "description": "http(s):// endpoint (transport=http)" },
                    "auth": { "type": "string", "description": "bearer token, sent as Authorization: Bearer … (optional)" },
                    "headers": { "type": "object", "description": "extra HTTP headers, e.g. {\"X-Api-Key\":\"…\"} (http, optional)" },
                    "command": { "type": "array", "items": { "type": "string" }, "description": "argv to spawn (transport=stdio), e.g. [\"npx\",\"-y\",\"<pkg>\"]" },
                    "env": { "type": "object", "description": "environment variables for the spawned process (stdio, optional)" }
                },
                "required": ["transport"]
            }
        }
    })
}

/// Meta-tool: disconnect a server the agent (or user) connected.
fn mcp_disconnect_def() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "mcp_disconnect",
            "description": "Disconnect a connected MCP server by its name (frees it / removes its \
                tools). Use when the user asks to remove a server or it is no longer needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "the connected server's name" }
                },
                "required": ["name"]
            }
        }
    })
}

/// Execute `mcp_connect`: build [`ConnectParams`] from the agent's args and open
/// the connection. Returns a token-free status line (never echoes secrets).
async fn handle_mcp_connect(state: &BotState, args_raw: &str) -> String {
    use crate::mcp_client::ConnectParams;
    let v: Value = match serde_json::from_str(args_raw) {
        Ok(v) => v,
        Err(e) => return format!("error: bad args: {e}"),
    };

    let obj_pairs = |val: Option<&Value>| -> Vec<(String, String)> {
        val.and_then(|h| h.as_object())
            .map(|m| {
                m.iter()
                    .map(|(k, v)| {
                        let s = v
                            .as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| v.to_string());
                        (k.clone(), s)
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let transport = v["transport"].as_str().unwrap_or("").to_lowercase();
    let url = v["url"].as_str().unwrap_or("").to_string();
    let command: Vec<String> = v["command"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    // Infer transport if the model omitted/mismatched it.
    let is_stdio = transport == "stdio" || (transport.is_empty() && !command.is_empty());
    if is_stdio && command.is_empty() {
        return "error: stdio transport needs a non-empty command array".into();
    }
    if !is_stdio && url.is_empty() {
        return "error: http transport needs a url".into();
    }

    let name = match v["name"].as_str().map(str::trim).filter(|s| !s.is_empty()) {
        Some(n) => n.to_string(),
        None if is_stdio => crate::bot::default_stdio_name(&command),
        None => crate::bot::default_name(&url),
    };

    let params = ConnectParams {
        name: name.clone(),
        url,
        auth: v["auth"].as_str().map(str::to_string),
        headers: obj_pairs(v.get("headers")),
        command,
        env: obj_pairs(v.get("env")),
    };

    match state.connect_mcp(params).await {
        Ok(n) => format!("connected '{name}' — {n} tools now available"),
        Err(e) => format!("error: connect '{name}' failed: {e}"),
    }
}

/// Execute `mcp_disconnect`.
async fn handle_mcp_disconnect(state: &BotState, args_raw: &str) -> String {
    let v: Value = match serde_json::from_str(args_raw) {
        Ok(v) => v,
        Err(e) => return format!("error: bad args: {e}"),
    };
    let name = v["name"].as_str().unwrap_or("").trim().to_string();
    if name.is_empty() {
        return "error: need the server name".into();
    }
    if state.disconnect_mcp(&name).await {
        format!("disconnected '{name}'")
    } else {
        format!("error: no connected server named '{name}'")
    }
}

/// Execute `cancel_summary`: stop this chat's recurring delivery — both
/// client-side polling watches (with linked job cleanup) and server-push
/// subscriptions (unsubscribe on the MCP + drop the durable record).
async fn handle_cancel_summary(state: &BotState, chat_id: Option<i64>) -> String {
    let Some(chat_id) = chat_id else {
        return "error: cannot cancel outside a chat".into();
    };
    // 1. polling watches (cancels linked collection jobs via cleanup)
    let watches = state.remove_watches_for_chat(chat_id).await;

    // 2. server-push subscriptions: tell each MCP to stop pushing, then forget.
    let servers: Vec<String> = state
        .push_subs
        .lock()
        .await
        .iter()
        .filter(|s| s.chat_id == chat_id)
        .map(|s| s.server.clone())
        .collect();
    for server in &servers {
        let mut args = serde_json::Map::new();
        args.insert("session_id".into(), chat_id.to_string().into());
        let _ = state
            .call_tool(server, "unsubscribe_summaries", Some(args))
            .await;
    }
    let pushes = state.remove_push_subs(chat_id, None).await;

    let total = watches + pushes;
    if total == 0 {
        "no active subscriptions to cancel".into()
    } else {
        format!("cancelled {total} subscription(s)")
    }
}

/// Execute the `schedule_summary` meta-tool: register + start a watch.
async fn handle_schedule_summary(state: &BotState, chat_id: Option<i64>, args_raw: &str) -> String {
    let Some(chat_id) = chat_id else {
        return "error: cannot schedule outside a chat".into();
    };
    let v: Value = match serde_json::from_str(args_raw) {
        Ok(v) => v,
        Err(e) => return format!("error: bad args: {e}"),
    };
    let server = v["server"].as_str().unwrap_or("").to_string();
    let tool = v["tool"].as_str().unwrap_or("").to_string();
    let minutes = v["minutes"].as_u64().unwrap_or(0);
    if server.is_empty() || tool.is_empty() || minutes == 0 {
        return "error: need server, tool, and minutes>=1".into();
    }
    if !state.mcp_names().await.iter().any(|n| n == &server) {
        return format!("error: server '{server}' is not connected");
    }
    let args = v.get("args").and_then(|a| a.as_object().cloned());
    let cleanup = v
        .get("cleanup_tool")
        .and_then(|t| t.as_str())
        .map(|t| crate::persist::Cleanup {
            tool: t.to_string(),
            args: v.get("cleanup_args").and_then(|a| a.as_object().cloned()),
        });
    let id = state
        .schedule_summary(
            chat_id,
            server.clone(),
            tool.clone(),
            args,
            minutes,
            cleanup,
        )
        .await;
    format!("scheduled watch #{id}: {server}/{tool} every {minutes}m (first result shortly; /unwatch {id} stops it and cancels the job)")
}

/// Names of tools whose input schema declares a `session_id` property.
fn tools_with_session_param(defs: &[Value]) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    for d in defs {
        let f = &d["function"];
        let has = f["parameters"]["properties"].get("session_id").is_some();
        if has {
            if let Some(name) = f["name"].as_str() {
                set.insert(name.to_string());
            }
        }
    }
    set
}

fn parse_args(raw: &str) -> Option<rmcp::model::JsonObject> {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(m)) => Some(m),
        _ => None,
    }
}

fn normalize_tool_args(server: &str, tool: &str, args: &mut Option<rmcp::model::JsonObject>) {
    let server = server.to_ascii_lowercase();
    if !(server.contains("map") || server.contains("osm")) {
        return;
    }
    let Some(map) = args.as_mut() else {
        return;
    };

    if tool == "geocode_address" && map.values().any(value_contains_cyrillic) {
        let needs_ru = map
            .get("region")
            .and_then(|v| v.as_str())
            .map(|r| r.eq_ignore_ascii_case("singapore") || r.trim().is_empty())
            .unwrap_or(true);
        if needs_ru {
            map.insert("region".into(), "RU".into());
        }
    }

    if tool == "osm_query_bbox" {
        if let Some(tags) = map.get("tags").cloned().and_then(normalize_osm_tags) {
            map.insert("tags".into(), Value::Object(tags));
        }
        clamp_osm_bbox(map);
    }
}

/// Max bbox side, in degrees, allowed through to Overpass. A broad-tag query
/// over a large box takes tens of seconds and is the main cause of trip stages
/// grinding for minutes; some models ignore the "small bbox" instruction, so we
/// hard-clamp here. A route/campsite lookup only ever needs a small box around a
/// point of interest, so shrinking an over-large box around its centre is safe
/// and makes every query fast.
const MAX_BBOX_SPAN_DEG: f64 = 0.30;

fn clamp_osm_bbox(map: &mut rmcp::model::JsonObject) {
    let Some(bbox) = map.get_mut("bbox").and_then(|v| v.as_object_mut()) else {
        return;
    };
    let get = |b: &rmcp::model::JsonObject, k: &str| b.get(k).and_then(Value::as_f64);
    let (Some(min_lat), Some(max_lat), Some(min_lon), Some(max_lon)) = (
        get(bbox, "minLat"),
        get(bbox, "maxLat"),
        get(bbox, "minLon"),
        get(bbox, "maxLon"),
    ) else {
        return;
    };
    // Shrink a side to MAX_BBOX_SPAN_DEG around its centre when it's too wide.
    let clamp = |lo: f64, hi: f64| -> (f64, f64) {
        if hi - lo <= MAX_BBOX_SPAN_DEG {
            (lo, hi)
        } else {
            let c = (lo + hi) / 2.0;
            let h = MAX_BBOX_SPAN_DEG / 2.0;
            (c - h, c + h)
        }
    };
    let (nlat0, nlat1) = clamp(min_lat, max_lat);
    let (nlon0, nlon1) = clamp(min_lon, max_lon);
    bbox.insert("minLat".into(), json!(nlat0));
    bbox.insert("maxLat".into(), json!(nlat1));
    bbox.insert("minLon".into(), json!(nlon0));
    bbox.insert("maxLon".into(), json!(nlon1));
}

fn value_contains_cyrillic(v: &Value) -> bool {
    match v {
        Value::String(s) => s.chars().any(|c| ('\u{0400}'..='\u{04ff}').contains(&c)),
        Value::Array(a) => a.iter().any(value_contains_cyrillic),
        Value::Object(o) => o.values().any(value_contains_cyrillic),
        _ => false,
    }
}

fn normalize_osm_tags(v: Value) -> Option<serde_json::Map<String, Value>> {
    match v {
        Value::Object(o) => Some(
            o.into_iter()
                .map(|(k, v)| {
                    // A tag VALUE must be a single string. Some models pass an
                    // array (`{"place":["village","hamlet"]}`) which Overpass
                    // rejects with 400; collapse it to the first string element
                    // rather than stringifying the whole array.
                    let val = match &v {
                        Value::String(s) => s.clone(),
                        Value::Array(items) => items
                            .iter()
                            .find_map(|i| i.as_str())
                            .map(str::to_string)
                            .unwrap_or_default(),
                        other => other.to_string(),
                    };
                    (k, Value::String(val))
                })
                .collect(),
        ),
        Value::String(s) => tags_from_selector(&s),
        Value::Array(items) => {
            let mut out = serde_json::Map::new();
            for item in items {
                if let Some(tags) = normalize_osm_tags(item) {
                    out.extend(tags);
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
        _ => None,
    }
}

fn tags_from_selector(selector: &str) -> Option<serde_json::Map<String, Value>> {
    let mut out = serde_json::Map::new();
    for raw in selector.split([',', ';']) {
        let tag = raw
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim();
        if tag.is_empty() {
            continue;
        }
        if let Some((k, v)) = tag.split_once('=') {
            let k = k.trim().trim_matches('"').trim_matches('\'');
            let v = v.trim().trim_matches('"').trim_matches('\'');
            if !k.is_empty() {
                out.insert(k.to_string(), Value::String(v.to_string()));
            }
        } else {
            out.insert(tag.to_string(), Value::String(String::new()));
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    // char-safe: byte-slicing `&s[..max]` panics if `max` lands mid-codepoint
    // (Cyrillic city names in tool JSON would hit this). Back off to a boundary.
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…(truncated)", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn exposed_name_namespaces_and_sanitizes() {
        let taken = std::collections::HashMap::new();
        assert_eq!(
            unique_exposed_name("weather", "get_forecast", &taken),
            "weather__get_forecast"
        );
        // dots/spaces in a server name are sanitized to underscores
        assert_eq!(
            unique_exposed_name("acme.io srv", "search", &taken),
            "acme_io_srv__search"
        );
    }

    fn tool_names(defs: &[Value]) -> Vec<String> {
        defs.iter()
            .filter_map(|d| d["function"]["name"].as_str().map(str::to_string))
            .collect()
    }

    #[test]
    fn meta_defs_always_include_connect_disconnect() {
        let mut defs = Vec::new();
        append_meta_defs(&mut defs, None);
        let names = tool_names(&defs);
        assert!(names.contains(&"mcp_connect".to_string()));
        assert!(names.contains(&"mcp_disconnect".to_string()));
        // outside a chat, no recurring-summary meta-tools
        assert!(!names.contains(&"schedule_summary".to_string()));
    }

    #[test]
    fn meta_defs_add_summary_tools_in_chat() {
        let mut defs = Vec::new();
        append_meta_defs(&mut defs, Some(42));
        let names = tool_names(&defs);
        assert!(names.contains(&"mcp_connect".to_string()));
        assert!(names.contains(&"schedule_summary".to_string()));
        assert!(names.contains(&"cancel_summary".to_string()));
    }

    #[test]
    fn mcp_connect_def_requires_transport() {
        let def = mcp_connect_def();
        let required = def["function"]["parameters"]["required"]
            .as_array()
            .unwrap();
        assert!(required.iter().any(|v| v == "transport"));
    }

    #[test]
    fn exposed_name_dedupes_collisions() {
        // Two servers expose a tool whose namespaced names collide → suffix.
        let mut taken: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();
        let a = unique_exposed_name("svc", "search", &taken);
        taken.insert(a.clone(), ("svc".into(), "search".into()));
        let b = unique_exposed_name("svc", "search", &taken);
        assert_eq!(a, "svc__search");
        assert_eq!(b, "svc__search_2");
        assert_ne!(a, b);
    }

    #[test]
    fn exposed_name_capped_at_64() {
        let taken = std::collections::HashMap::new();
        let n = unique_exposed_name(&"s".repeat(40), &"t".repeat(40), &taken);
        assert!(n.len() <= 64, "len was {}", n.len());
    }

    #[test]
    fn session_param_detection() {
        let defs = vec![
            json!({"function":{"name":"subscribe_summaries","parameters":{"properties":{"session_id":{},"period":{}}}}}),
            json!({"function":{"name":"geocode","parameters":{"properties":{"name":{}}}}}),
        ];
        let set = tools_with_session_param(&defs);
        assert!(set.contains("subscribe_summaries"));
        assert!(!set.contains("geocode"));
    }

    #[test]
    fn map_geocode_cyrillic_defaults_to_ru() {
        let mut args = parse_args(r#"{"address":"Волгоград"}"#);
        normalize_tool_args("maps", "geocode_address", &mut args);
        let region = args.unwrap().remove("region").unwrap();
        assert_eq!(region, "RU");
    }

    #[test]
    fn map_geocode_overrides_singapore_for_cyrillic() {
        let mut args = parse_args(r#"{"query":"Ахтуба","region":"Singapore"}"#);
        normalize_tool_args("osm", "geocode_address", &mut args);
        let region = args.unwrap().remove("region").unwrap();
        assert_eq!(region, "RU");
    }

    #[test]
    fn osm_tags_string_selector_becomes_object() {
        let mut args = parse_args(r#"{"tags":"tourism=camp_site"}"#);
        normalize_tool_args("maps", "osm_query_bbox", &mut args);
        let args = args.unwrap();
        assert_eq!(args["tags"], json!({"tourism":"camp_site"}));
    }

    #[test]
    fn osm_tags_array_becomes_object() {
        let mut args = parse_args(r#"{"tags":["waterway",{"boat":"yes"}]}"#);
        normalize_tool_args("maps", "osm_query_bbox", &mut args);
        let args = args.unwrap();
        assert_eq!(args["tags"], json!({"waterway":"","boat":"yes"}));
    }

    #[test]
    fn truncate_is_char_safe_on_multibyte() {
        // 'ё' is 2 bytes; cutting at an odd byte must not panic.
        let s = "ёёёёё"; // 10 bytes
        let out = truncate(s, 5);
        assert!(out.ends_with("…(truncated)"));
        assert!(out.starts_with("ёё")); // backed off to a char boundary
    }

    #[test]
    fn fit_to_context_clips_oldest_tool_results() {
        // deepseek window 65_536 → threshold ~52_428 tokens (~157k chars).
        std::env::remove_var("LLM_CONTEXT_TOKENS");
        let big = "a".repeat(90_000); // ~30_000 tokens each; 3× overflows badly
        let mut messages = vec![
            json!({ "role": "system", "content": "sys" }),
            json!({ "role": "tool", "tool_call_id": "1", "content": big }),
            json!({ "role": "tool", "tool_call_id": "2", "content": big }),
            json!({ "role": "assistant", "content": "thinking" }),
            json!({ "role": "tool", "tool_call_id": "3", "content": big }),
        ];
        fit_to_context(&mut messages, "deepseek-chat");

        // Oldest two tool results clipped...
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("trimmed to fit context"));
        assert!(messages[2]["content"]
            .as_str()
            .unwrap()
            .contains("trimmed to fit context"));
        // ...the freshest tool result (in the protected last-two window) is intact.
        assert_eq!(messages[4]["content"].as_str().unwrap().len(), 90_000);
    }

    fn obj(v: Value) -> Option<rmcp::model::JsonObject> {
        v.as_object().cloned()
    }

    #[test]
    fn osm_bbox_clamped_when_too_large() {
        // ~0.8° x 1.0° box (the broad sweep the model produced) → clamped to 0.30.
        let mut args = obj(json!({
            "bbox": {"minLat": 48.3, "minLon": 44.0, "maxLat": 49.1, "maxLon": 45.0},
            "tags": {"tourism": "camp_site"}
        }));
        normalize_tool_args("maps", "osm_query_bbox", &mut args);
        let b = &args.unwrap()["bbox"];
        let span = |lo: &str, hi: &str| b[hi].as_f64().unwrap() - b[lo].as_f64().unwrap();
        assert!((span("minLat", "maxLat") - 0.30).abs() < 1e-9);
        assert!((span("minLon", "maxLon") - 0.30).abs() < 1e-9);
        // centred on the original box centre (48.7, 44.5)
        assert!((b["minLat"].as_f64().unwrap() - 48.55).abs() < 1e-9);
        assert!((b["maxLon"].as_f64().unwrap() - 44.65).abs() < 1e-9);
    }

    #[test]
    fn osm_bbox_small_box_untouched() {
        let mut args = obj(json!({
            "bbox": {"minLat": 50.20, "minLon": 43.57, "maxLat": 50.24, "maxLon": 43.65},
            "tags": {"place": "village"}
        }));
        normalize_tool_args("maps", "osm_query_bbox", &mut args);
        let b = &args.unwrap()["bbox"];
        assert_eq!(b["minLat"].as_f64().unwrap(), 50.20);
        assert_eq!(b["maxLon"].as_f64().unwrap(), 43.65);
    }

    #[test]
    fn osm_tag_array_value_collapses_to_first() {
        // An array tag value (`place:[village,hamlet]`) → first element, not the
        // stringified array (which Overpass rejects with 400).
        let mut args = obj(json!({
            "bbox": {"minLat": 50.0, "minLon": 43.0, "maxLat": 50.1, "maxLon": 43.1},
            "tags": {"place": ["village", "hamlet", "town"]}
        }));
        normalize_tool_args("maps", "osm_query_bbox", &mut args);
        assert_eq!(args.unwrap()["tags"]["place"].as_str().unwrap(), "village");
    }
}
