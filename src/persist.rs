//! On-disk state so the agent survives restarts: connected MCP servers,
//! digest subscribers, and periodic watches.

use std::path::PathBuf;

use rmcp::model::JsonObject;
use serde::{Deserialize, Serialize};

use crate::mcp_client::ConnectParams;

/// One periodic watch: call `server`/`tool` with `args` every `interval_min`,
/// posting the result to `chat_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchSpec {
    pub id: u64,
    pub chat_id: i64,
    pub server: String,
    pub tool: String,
    #[serde(default)]
    pub args: Option<JsonObject>,
    pub interval_min: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Persisted {
    #[serde(default)]
    pub servers: Vec<ConnectParams>,
    #[serde(default)]
    pub subscribers: Vec<i64>,
    #[serde(default)]
    pub watches: Vec<WatchSpec>,
    #[serde(default)]
    pub next_watch_id: u64,
}

/// State file location: `$STATE_FILE` or `./state.json`.
pub fn state_path() -> PathBuf {
    std::env::var("STATE_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("state.json"))
}

/// Load persisted state. Missing/corrupt file -> default (logged by caller).
pub fn load() -> Persisted {
    let path = state_path();
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            tracing::warn!(
                "state file {} is corrupt ({e}); starting fresh",
                path.display()
            );
            Persisted::default()
        }),
        Err(_) => Persisted::default(),
    }
}

/// Atomically write persisted state (write temp + rename).
pub fn save(state: &Persisted) -> anyhow::Result<()> {
    let path = state_path();
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(state)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}
