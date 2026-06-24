use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::sync::Mutex;

use crate::mcp_client::{ConnectParams, EventSender, McpClient};

#[derive(Clone)]
pub struct BotState {
    /// name -> live MCP connection
    pub mcps: Arc<Mutex<HashMap<String, McpClient>>>,
    /// chat IDs subscribed to digests
    pub subscribers: Arc<Mutex<HashSet<i64>>>,
    /// shared event channel for all MCP notifications
    pub events: EventSender,
}

impl BotState {
    pub fn new(events: EventSender) -> Self {
        Self {
            mcps: Arc::new(Mutex::new(HashMap::new())),
            subscribers: Arc::new(Mutex::new(HashSet::new())),
            events,
        }
    }

    /// Connect a new MCP and register it. Returns tool count on success.
    pub async fn connect_mcp(&self, params: ConnectParams) -> anyhow::Result<usize> {
        let name = params.name.clone();
        {
            let guard = self.mcps.lock().await;
            if guard.contains_key(&name) {
                anyhow::bail!("server '{name}' already connected — /disconnect {name} first");
            }
        }
        let client = McpClient::connect(params, self.events.clone()).await?;
        let count = client.tools().await.len();
        self.mcps.lock().await.insert(name, client);
        Ok(count)
    }

    pub async fn disconnect_mcp(&self, name: &str) -> bool {
        self.mcps.lock().await.remove(name).is_some()
    }

    /// Call a tool on a connected server. Errors if the server is unknown.
    pub async fn call_tool(
        &self,
        server: &str,
        tool: &str,
        arguments: Option<rmcp::model::JsonObject>,
    ) -> anyhow::Result<String> {
        let guard = self.mcps.lock().await;
        let client = guard
            .get(server)
            .ok_or_else(|| anyhow::anyhow!("unknown server '{server}' — see /mcps"))?;
        client.call_tool(tool, arguments).await
    }

    pub async fn mcp_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.mcps.lock().await.keys().cloned().collect();
        v.sort();
        v
    }

    pub async fn subscribe(&self, chat_id: i64) {
        self.subscribers.lock().await.insert(chat_id);
    }

    pub async fn subscribers(&self) -> Vec<i64> {
        self.subscribers.lock().await.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> BotState {
        let (tx, _rx) = tokio::sync::broadcast::channel(8);
        BotState::new(tx)
    }

    #[tokio::test]
    async fn subscribe_is_idempotent() {
        let s = state();
        s.subscribe(42).await;
        s.subscribe(42).await;
        s.subscribe(7).await;
        let mut subs = s.subscribers().await;
        subs.sort();
        assert_eq!(subs, vec![7, 42]);
    }

    #[tokio::test]
    async fn disconnect_missing_returns_false() {
        let s = state();
        assert!(!s.disconnect_mcp("nope").await);
    }

    #[tokio::test]
    async fn mcp_names_empty_initially() {
        let s = state();
        assert!(s.mcp_names().await.is_empty());
    }
}
