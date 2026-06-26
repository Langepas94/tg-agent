//! Minimal OpenAI-compatible LLM client + an agentic tool-calling loop that
//! lets a user ask in natural language and get a human answer backed by MCP
//! tools (default provider: DeepSeek).

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

use crate::{config::LlmConfig, state::BotState};

const MAX_STEPS: usize = 12;
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
        self.answer_with_system(state, system, user_text).await
    }

    /// Agentic loop with a caller-supplied system prompt and no chat context
    /// (no self-scheduling meta-tool).
    pub async fn answer_with_system(
        &self,
        state: &BotState,
        system: &str,
        user_text: &str,
    ) -> Result<String> {
        self.answer_in_chat(state, system, user_text, &[], None)
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
    ) -> Result<String> {
        let (mut tool_defs, tool_map) = collect_tools(state).await;
        // Tools that accept a `session_id` param — the bot forces it to the
        // chat id so jobs are chat-isolated AND server pushes route back here.
        let session_tools = tools_with_session_param(&tool_defs);
        if chat_id.is_some() {
            tool_defs.push(schedule_summary_def());
            tool_defs.push(cancel_summary_def());
        }

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

        for _ in 0..MAX_STEPS {
            // Keep the growing tool transcript under the model's window. MCP
            // results (multi-city forecasts) can overflow it mid-loop; clip the
            // oldest results BEFORE the call so the provider never 400s.
            fit_to_context(&mut messages, &self.cfg.model);
            let msg = self.chat(&messages, &tool_defs).await?;

            let tool_calls = msg.get("tool_calls").and_then(|v| v.as_array()).cloned();
            match tool_calls {
                Some(calls) if !calls.is_empty() => {
                    messages.push(msg.clone()); // assistant turn carrying the calls
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

                        let content = if name == "schedule_summary" {
                            handle_schedule_summary(state, chat_id, args_raw).await
                        } else if name == "cancel_summary" {
                            handle_cancel_summary(state, chat_id).await
                        } else {
                            match tool_map.get(&name) {
                                None => format!("error: tool '{name}' is not available"),
                                Some((server, real_tool)) => {
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
}
