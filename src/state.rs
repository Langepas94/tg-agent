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
