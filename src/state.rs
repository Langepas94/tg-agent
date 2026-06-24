use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokio::{sync::Mutex, task::JoinHandle};

use crate::{
    mcp_client::{ConnectParams, EventSender, McpClient},
    persist::{self, Persisted, WatchSpec},
};

#[derive(Clone)]
pub struct BotState {
    /// name -> live MCP connection
    pub mcps: Arc<Mutex<HashMap<String, McpClient>>>,
    /// chat IDs subscribed to digests
    pub subscribers: Arc<Mutex<HashSet<i64>>>,
    /// periodic watches (persisted)
    pub watches: Arc<Mutex<Vec<WatchSpec>>>,
    /// running watch tasks, keyed by watch id (not persisted)
    pub watch_tasks: Arc<Mutex<HashMap<u64, JoinHandle<()>>>>,
    /// monotonically increasing watch id
    next_watch_id: Arc<AtomicU64>,
    /// shared event channel for all MCP notifications
    pub events: EventSender,
}

impl BotState {
    pub fn new(events: EventSender) -> Self {
        Self {
            mcps: Arc::new(Mutex::new(HashMap::new())),
            subscribers: Arc::new(Mutex::new(HashSet::new())),
            watches: Arc::new(Mutex::new(Vec::new())),
            watch_tasks: Arc::new(Mutex::new(HashMap::new())),
            next_watch_id: Arc::new(AtomicU64::new(1)),
            events,
        }
    }

    pub fn alloc_watch_id(&self) -> u64 {
        self.next_watch_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn set_next_watch_id(&self, v: u64) {
        self.next_watch_id.store(v.max(1), Ordering::SeqCst);
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
        self.persist().await;
        Ok(count)
    }

    pub async fn disconnect_mcp(&self, name: &str) -> bool {
        let removed = self.mcps.lock().await.remove(name).is_some();
        if removed {
            self.persist().await;
        }
        removed
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
        let inserted = self.subscribers.lock().await.insert(chat_id);
        if inserted {
            self.persist().await;
        }
    }

    pub async fn subscribers(&self) -> Vec<i64> {
        self.subscribers.lock().await.iter().cloned().collect()
    }

    pub async fn add_watch(&self, spec: WatchSpec) {
        self.watches.lock().await.push(spec);
        self.persist().await;
    }

    pub async fn remove_watch(&self, id: u64) -> bool {
        let before = self.watches.lock().await.len();
        self.watches.lock().await.retain(|w| w.id != id);
        let removed = self.watches.lock().await.len() != before;
        if let Some(h) = self.watch_tasks.lock().await.remove(&id) {
            h.abort();
        }
        if removed {
            self.persist().await;
        }
        removed
    }

    pub async fn list_watches(&self) -> Vec<WatchSpec> {
        self.watches.lock().await.clone()
    }

    /// Snapshot current in-memory state into a serializable form.
    pub async fn snapshot(&self) -> Persisted {
        let servers: Vec<ConnectParams> = {
            let guard = self.mcps.lock().await;
            guard.values().map(|c| c.params.clone()).collect()
        };
        let mut subscribers: Vec<i64> = self.subscribers.lock().await.iter().cloned().collect();
        subscribers.sort();
        let watches = self.watches.lock().await.clone();
        Persisted {
            servers,
            subscribers,
            watches,
            next_watch_id: self.next_watch_id.load(Ordering::SeqCst),
        }
    }

    /// Persist current state to disk (best-effort; logs on failure).
    pub async fn persist(&self) {
        let snap = self.snapshot().await;
        if let Err(e) = persist::save(&snap) {
            tracing::error!("failed to persist state: {e}");
        }
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
    async fn disconnect_missing_returns_false() {
        let s = state();
        assert!(!s.disconnect_mcp("nope").await);
    }

    #[tokio::test]
    async fn mcp_names_empty_initially() {
        let s = state();
        assert!(s.mcp_names().await.is_empty());
    }

    #[tokio::test]
    async fn watch_ids_increment() {
        let s = state();
        assert_eq!(s.alloc_watch_id(), 1);
        assert_eq!(s.alloc_watch_id(), 2);
        s.set_next_watch_id(10);
        assert_eq!(s.alloc_watch_id(), 10);
    }

    #[tokio::test]
    async fn remove_missing_watch_false() {
        let s = state();
        assert!(!s.remove_watch(999).await);
    }
}
