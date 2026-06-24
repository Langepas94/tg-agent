//! Minimal OpenAI-compatible LLM client + an agentic tool-calling loop that
//! lets a user ask in natural language and get a human answer backed by MCP
//! tools (default provider: DeepSeek).

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::{config::LlmConfig, state::BotState};

const MAX_STEPS: usize = 6;

#[derive(Clone)]
pub struct Llm {
    cfg: LlmConfig,
    http: reqwest::Client,
}

impl Llm {
    pub fn new(cfg: LlmConfig) -> Self {
        Self {
            cfg,
            http: reqwest::Client::new(),
        }
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
            .context("LLM request failed")?;

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

    /// Agentic tool-calling loop with a caller-supplied system prompt
    /// (used by the orchestrator's layered PromptBuilder output).
    pub async fn answer_with_system(
        &self,
        state: &BotState,
        system: &str,
        user_text: &str,
    ) -> Result<String> {
        let (tool_defs, tool_to_server) = collect_tools(state).await;

        let mut messages = vec![
            json!({ "role": "system", "content": system }),
            json!({ "role": "user", "content": user_text }),
        ];

        for _ in 0..MAX_STEPS {
            let msg = self.chat(&messages, &tool_defs).await?;

            let tool_calls = msg.get("tool_calls").and_then(|v| v.as_array()).cloned();
            match tool_calls {
                Some(calls) if !calls.is_empty() => {
                    messages.push(msg.clone()); // assistant turn carrying the calls
                    for call in calls {
                        let id = call["id"].as_str().unwrap_or("").to_string();
                        let name = call["function"]["name"].as_str().unwrap_or("").to_string();
                        let args_raw = call["function"]["arguments"].as_str().unwrap_or("{}");
                        let args = parse_args(args_raw);

                        let content = match tool_to_server.get(&name) {
                            None => format!("error: tool '{name}' is not available"),
                            Some(server) => match state.call_tool(server, &name, args).await {
                                Ok(out) => truncate(&out, 6000),
                                Err(e) => format!("error: {e}"),
                            },
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

/// Build OpenAI tool definitions from every connected server's tools, plus a
/// map from tool name to the server that owns it (last writer wins on clash).
async fn collect_tools(
    state: &BotState,
) -> (Vec<Value>, std::collections::HashMap<String, String>) {
    let mut defs = Vec::new();
    let mut map = std::collections::HashMap::new();
    let guard = state.mcps.lock().await;
    for (server, client) in guard.iter() {
        for t in client.tools().await {
            let name = t.name.to_string();
            defs.push(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.description.as_deref().unwrap_or(""),
                    "parameters": Value::Object((*t.input_schema).clone()),
                }
            }));
            map.insert(name, server.clone());
        }
    }
    (defs, map)
}

fn parse_args(raw: &str) -> Option<rmcp::model::JsonObject> {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(m)) => Some(m),
        _ => None,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…(truncated)", &s[..max])
    }
}
