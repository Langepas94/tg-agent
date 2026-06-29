use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokio::{sync::Mutex, task::JoinHandle};

use crate::{
    llm::Llm,
    mcp_client::{ConnectParams, EventSender, McpClient},
    persist::{self, AccessState, Persisted, WatchSpec},
};

#[derive(Debug, Clone)]
pub struct ToolSummary {
    pub server: String,
    pub name: String,
    pub description: String,
}

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
    /// optional LLM for natural-language agent answers
    pub llm: Option<Arc<Llm>>,
    /// bot handle (set at startup) so watches/agent can post to chats
    pub bot: Arc<Mutex<Option<teloxide::Bot>>>,
    /// durable server-push subscriptions (re-applied on MCP reconnect)
    pub push_subs: Arc<Mutex<Vec<crate::persist::PushSub>>>,
    /// Chats that passed the password gate.
    pub authorized_chat_ids: Arc<Mutex<HashSet<i64>>>,
    /// Owner/root chat id. The first successful authorization claims it.
    pub root_chat_id: Arc<Mutex<Option<i64>>>,
    /// Password used by /start <password>.
    bot_password: Arc<String>,
}

impl BotState {
    pub fn new(events: EventSender) -> Self {
        Self::with_llm(events, None)
    }

    pub fn with_llm(events: EventSender, llm: Option<Arc<Llm>>) -> Self {
        Self::with_llm_and_password(events, llm, "202020".to_string())
    }

    pub fn with_llm_and_password(
        events: EventSender,
        llm: Option<Arc<Llm>>,
        bot_password: String,
    ) -> Self {
        Self {
            mcps: Arc::new(Mutex::new(HashMap::new())),
            subscribers: Arc::new(Mutex::new(HashSet::new())),
            watches: Arc::new(Mutex::new(Vec::new())),
            watch_tasks: Arc::new(Mutex::new(HashMap::new())),
            next_watch_id: Arc::new(AtomicU64::new(1)),
            events,
            llm,
            bot: Arc::new(Mutex::new(None)),
            push_subs: Arc::new(Mutex::new(Vec::new())),
            authorized_chat_ids: Arc::new(Mutex::new(HashSet::new())),
            root_chat_id: Arc::new(Mutex::new(None)),
            bot_password: Arc::new(bot_password),
        }
    }

    pub async fn set_bot(&self, bot: teloxide::Bot) {
        *self.bot.lock().await = Some(bot);
    }

    pub async fn restore_access(&self, access: AccessState) {
        *self.root_chat_id.lock().await = access.root_chat_id;
        let mut authorized = self.authorized_chat_ids.lock().await;
        authorized.clear();
        authorized.extend(access.authorized_chat_ids);
    }

    pub async fn access_snapshot(&self) -> AccessState {
        let mut authorized_chat_ids: Vec<i64> = self
            .authorized_chat_ids
            .lock()
            .await
            .iter()
            .copied()
            .collect();
        authorized_chat_ids.sort();
        AccessState {
            root_chat_id: *self.root_chat_id.lock().await,
            authorized_chat_ids,
        }
    }

    pub async fn is_authorized(&self, chat_id: i64) -> bool {
        self.authorized_chat_ids.lock().await.contains(&chat_id)
    }

    pub async fn is_root(&self, chat_id: i64) -> bool {
        *self.root_chat_id.lock().await == Some(chat_id)
    }

    pub async fn grant_access(&self, chat_id: i64) {
        let changed = self.authorized_chat_ids.lock().await.insert(chat_id);
        if changed {
            self.persist().await;
        }
    }

    pub async fn revoke_access(&self, chat_id: i64) -> bool {
        if self.is_root(chat_id).await {
            return false;
        }
        let changed = self.authorized_chat_ids.lock().await.remove(&chat_id);
        if changed {
            self.persist().await;
        }
        changed
    }

    pub async fn set_root(&self, chat_id: i64) {
        {
            let mut root = self.root_chat_id.lock().await;
            *root = Some(chat_id);
        }
        self.authorized_chat_ids.lock().await.insert(chat_id);
        self.persist().await;
    }

    /// Try to unlock this chat. First successful chat becomes root/owner.
    pub async fn authorize(&self, chat_id: i64, password: &str) -> bool {
        if password.trim() != self.bot_password.as_str() {
            return false;
        }
        let mut changed = {
            let mut authorized = self.authorized_chat_ids.lock().await;
            authorized.insert(chat_id)
        };
        {
            let mut root = self.root_chat_id.lock().await;
            if root.is_none() {
                *root = Some(chat_id);
                changed = true;
            }
        }
        if changed {
            self.persist().await;
        }
        true
    }

    /// Spawn the periodic task for a watch and register its handle.
    /// Uses the stored bot handle to post results to the watch's chat.
    pub async fn start_watch(&self, spec: WatchSpec) {
        use std::time::Duration;
        use teloxide::prelude::*;

        let Some(bot) = self.bot.lock().await.clone() else {
            tracing::error!("start_watch: bot not set");
            return;
        };
        let id = spec.id;
        let state = self.clone();
        let handle = tokio::spawn(async move {
            let chat = teloxide::types::ChatId(spec.chat_id);
            let mut ticker =
                tokio::time::interval(Duration::from_secs(spec.interval_min.max(1) * 60));
            // `first` marks the immediate tick (fires now, also on every restart).
            // On it we only send a "collecting" heartbeat — never a DATA summary —
            // so data summaries always land on the interval grid and restarts can't
            // emit off-cadence posts.
            let mut first = true;
            let mut announced_waiting = false;
            loop {
                ticker.tick().await;
                let out = match state
                    .call_tool(&spec.server, &spec.tool, spec.args.clone())
                    .await
                {
                    Ok(out) => out,
                    Err(e) => {
                        if !first {
                            let _ = bot.send_message(chat, format!("⏱ watch failed: {e}")).await;
                        }
                        first = false;
                        continue;
                    }
                };

                let no_data = out.trim().is_empty() || looks_like_no_data(&out);

                if first {
                    // Immediate tick: heartbeat only (once), never an off-grid summary.
                    if no_data && !announced_waiting {
                        announced_waiting = true;
                        let _ = bot
                            .send_message(
                                chat,
                                format!(
                                    "⏳ Подписка активна: {}/{} каждые {} мин. Первая сводка — через {} мин.",
                                    spec.server, spec.tool, spec.interval_min, spec.interval_min
                                ),
                            )
                            .await;
                    }
                    first = false;
                    continue;
                }

                // No data yet: tell the user ONCE that it's collecting, then stay quiet.
                if no_data {
                    if !announced_waiting {
                        announced_waiting = true;
                        let _ = bot
                            .send_message(
                                chat,
                                format!(
                                    "⏳ Данные ещё собираются ({}). Пришлю сводку, как появятся замеры.",
                                    spec.server
                                ),
                            )
                            .await;
                    }
                    continue;
                }

                announced_waiting = false;
                // Humanize the raw tool output via the LLM when available.
                let body = state.humanize_watch(&spec.tool, &out).await;
                for chunk in chunk_text(&format!("📬 {body}"), 3900) {
                    let _ = bot.send_message(chat, chunk).await;
                }
            }
        });
        self.watch_tasks.lock().await.insert(id, handle);
    }

    /// Turn a raw tool result into a short human-readable summary via the LLM.
    /// Falls back to the raw text when no LLM is configured or the call fails.
    async fn humanize_watch(&self, tool: &str, raw: &str) -> String {
        let Some(llm) = &self.llm else {
            return raw.to_string();
        };
        let system = "You format MCP tool results for a Telegram user. \
            Summarize the data below in the user's language, short and human-readable \
            (use a few emoji, no JSON, no code blocks). If it is weather data, give \
            temperature/precip/wind highlights.";
        match llm
            .complete(system, &format!("Tool: {tool}\nResult:\n{raw}"))
            .await
        {
            Ok(s) if !s.trim().is_empty() => s,
            _ => raw.to_string(),
        }
    }

    /// Humanize a server-pushed summary payload (reuses the watch formatter).
    pub async fn humanize_summary(&self, data: &serde_json::Value) -> String {
        let raw = serde_json::to_string(data).unwrap_or_default();
        self.humanize_watch("get_weather_summary", &raw).await
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
        self.mcps.lock().await.insert(name.clone(), client);
        self.persist().await;
        // Re-apply durable push subscriptions for this server (survives restarts).
        self.reapply_push_subs(&name).await;
        Ok(count)
    }

    /// Re-call subscribe_summaries for every persisted push-sub on `server`.
    /// The MCP keeps subscriptions in memory keyed by the (new) session, so this
    /// restores push delivery after a reconnect.
    async fn reapply_push_subs(&self, server: &str) {
        let subs: Vec<_> = self
            .push_subs
            .lock()
            .await
            .iter()
            .filter(|s| s.server == server)
            .cloned()
            .collect();
        for sub in subs {
            let mut args = serde_json::Map::new();
            args.insert("session_id".into(), sub.chat_id.to_string().into());
            args.insert("period".into(), sub.period.clone().into());
            if let Err(e) = self
                .call_tool(server, "subscribe_summaries", Some(args))
                .await
            {
                tracing::warn!("reapply push-sub chat {} on {server}: {e}", sub.chat_id);
            }
        }
    }

    /// Record a durable push subscription (idempotent per chat+server).
    pub async fn add_push_sub(&self, chat_id: i64, server: String, period: String) {
        let mut guard = self.push_subs.lock().await;
        if let Some(s) = guard
            .iter_mut()
            .find(|s| s.chat_id == chat_id && s.server == server)
        {
            s.period = period;
        } else {
            guard.push(crate::persist::PushSub {
                chat_id,
                server,
                period,
            });
        }
        drop(guard);
        self.persist().await;
    }

    /// Remove durable push subscriptions for a chat (optionally a single server).
    pub async fn remove_push_subs(&self, chat_id: i64, server: Option<&str>) -> usize {
        let mut guard = self.push_subs.lock().await;
        let before = guard.len();
        guard
            .retain(|s| !(s.chat_id == chat_id && server.map(|sv| sv == s.server).unwrap_or(true)));
        let removed = before - guard.len();
        drop(guard);
        if removed > 0 {
            self.persist().await;
        }
        removed
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

    /// Safe capability inventory for planner agents. Contains only public MCP
    /// metadata (server, tool name, description), never saved env/secrets.
    pub async fn tool_inventory(&self) -> Vec<ToolSummary> {
        let guard = self.mcps.lock().await;
        let mut out = Vec::new();
        for (server, client) in guard.iter() {
            for tool in client.tools().await {
                out.push(ToolSummary {
                    server: server.clone(),
                    name: tool.name.to_string(),
                    description: tool.description.unwrap_or_default().to_string(),
                });
            }
        }
        out.sort_by(|a, b| a.server.cmp(&b.server).then_with(|| a.name.cmp(&b.name)));
        out
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

    /// Register AND start a periodic summary in one call (used by the agent's
    /// `schedule_summary` meta-tool and by /watch). Returns the watch id.
    /// Deduplicates: an identical watch (same chat/server/tool/args) is reused
    /// rather than creating garbage duplicates.
    pub async fn schedule_summary(
        &self,
        chat_id: i64,
        server: String,
        tool: String,
        args: Option<rmcp::model::JsonObject>,
        interval_min: u64,
        cleanup: Option<crate::persist::Cleanup>,
    ) -> u64 {
        if let Some(existing) = self.watches.lock().await.iter().find(|w| {
            w.chat_id == chat_id && w.server == server && w.tool == tool && w.args == args
        }) {
            return existing.id;
        }
        let id = self.alloc_watch_id();
        let spec = WatchSpec {
            id,
            chat_id,
            server,
            tool,
            args,
            interval_min: interval_min.max(1),
            cleanup,
        };
        self.add_watch(spec.clone()).await;
        self.start_watch(spec).await;
        id
    }

    /// Stop and remove every watch (bot side), tearing down each linked MCP
    /// resource. Returns how many were removed.
    pub async fn remove_all_watches(&self) -> usize {
        let ids: Vec<u64> = self.watches.lock().await.iter().map(|w| w.id).collect();
        let n = ids.len();
        for id in ids {
            self.remove_watch(id).await;
        }
        n
    }

    /// Remove all watches belonging to one chat (with linked teardown). Scoped
    /// to the chat so one user can't cancel another's subscriptions.
    pub async fn remove_watches_for_chat(&self, chat_id: i64) -> usize {
        let ids: Vec<u64> = self
            .watches
            .lock()
            .await
            .iter()
            .filter(|w| w.chat_id == chat_id)
            .map(|w| w.id)
            .collect();
        let n = ids.len();
        for id in ids {
            self.remove_watch(id).await;
        }
        n
    }

    pub async fn remove_watch_for_chat(&self, chat_id: i64, id: u64) -> bool {
        let belongs_to_chat = self
            .watches
            .lock()
            .await
            .iter()
            .any(|w| w.id == id && w.chat_id == chat_id);
        if !belongs_to_chat {
            return false;
        }
        self.remove_watch(id).await
    }

    pub async fn list_watches_for_chat(&self, chat_id: i64) -> Vec<WatchSpec> {
        self.watches
            .lock()
            .await
            .iter()
            .filter(|w| w.chat_id == chat_id)
            .cloned()
            .collect()
    }

    /// Remove a watch and tear down its linked MCP-side resource (e.g. cancel
    /// the collection cron job) so nothing is orphaned.
    pub async fn remove_watch(&self, id: u64) -> bool {
        // Capture and remove the spec under the lock, then act without it held.
        let removed_spec = {
            let mut guard = self.watches.lock().await;
            if let Some(pos) = guard.iter().position(|w| w.id == id) {
                Some(guard.remove(pos))
            } else {
                None
            }
        };
        if let Some(h) = self.watch_tasks.lock().await.remove(&id) {
            h.abort();
        }
        let Some(spec) = removed_spec else {
            return false;
        };
        // Best-effort teardown of the MCP-side resource this watch owned.
        if let Some(cleanup) = &spec.cleanup {
            match self
                .call_tool(&spec.server, &cleanup.tool, cleanup.args.clone())
                .await
            {
                Ok(_) => tracing::info!("watch #{id}: cleaned up {}/{}", spec.server, cleanup.tool),
                Err(e) => tracing::warn!("watch #{id} cleanup failed: {e}"),
            }
        }
        self.persist().await;
        true
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
        let push_subs = self.push_subs.lock().await.clone();
        Persisted {
            servers,
            subscribers,
            watches,
            next_watch_id: self.next_watch_id.load(Ordering::SeqCst),
            push_subs,
            access: self.access_snapshot().await,
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

/// Heuristic: a tool result that carries no collected data yet (don't spam it).
fn looks_like_no_data(s: &str) -> bool {
    let low = s.to_ascii_lowercase();
    low.contains("no data")
        || low.contains("\"readings\":0")
        || low.contains("\"readings\": 0")
        || low.contains("nothing collected")
}

/// Split text on line boundaries into chunks under `limit` bytes.
fn chunk_text(text: &str, limit: usize) -> Vec<String> {
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
    if out.is_empty() {
        out.push(String::new());
    }
    out
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

    #[tokio::test]
    async fn schedule_summary_dedups_identical() {
        let s = state();
        let id1 = s
            .schedule_summary(
                7,
                "weather".into(),
                "get_weather_summary".into(),
                None,
                10,
                None,
            )
            .await;
        let id2 = s
            .schedule_summary(
                7,
                "weather".into(),
                "get_weather_summary".into(),
                None,
                10,
                None,
            )
            .await;
        assert_eq!(id1, id2, "identical watch should be reused");
        assert_eq!(s.list_watches().await.len(), 1);
        // different chat → separate watch
        let id3 = s
            .schedule_summary(
                8,
                "weather".into(),
                "get_weather_summary".into(),
                None,
                10,
                None,
            )
            .await;
        assert_ne!(id1, id3);
        assert_eq!(s.list_watches().await.len(), 2);
    }

    #[tokio::test]
    async fn push_subs_add_dedup_remove() {
        let s = state();
        s.add_push_sub(5, "weather".into(), "1h".into()).await;
        s.add_push_sub(5, "weather".into(), "6h".into()).await; // same chat+server → update
        assert_eq!(s.push_subs.lock().await.len(), 1);
        assert_eq!(s.push_subs.lock().await[0].period, "6h");
        s.add_push_sub(6, "weather".into(), "1h".into()).await;
        assert_eq!(s.push_subs.lock().await.len(), 2);
        // scoped removal
        assert_eq!(s.remove_push_subs(5, None).await, 1);
        assert_eq!(s.push_subs.lock().await.len(), 1);
        assert_eq!(s.remove_push_subs(6, Some("other")).await, 0);
        assert_eq!(s.remove_push_subs(6, Some("weather")).await, 1);
        assert!(s.push_subs.lock().await.is_empty());
    }

    #[tokio::test]
    async fn authorization_first_user_becomes_root() {
        let s = state();
        assert!(!s.is_authorized(1).await);
        assert!(!s.authorize(1, "wrong").await);
        assert!(!s.is_authorized(1).await);

        assert!(s.authorize(1, "202020").await);
        assert!(s.is_authorized(1).await);
        assert!(s.is_root(1).await);

        assert!(s.authorize(2, "202020").await);
        assert!(s.is_authorized(2).await);
        assert!(!s.is_root(2).await);
    }

    #[tokio::test]
    async fn remove_watches_for_chat_is_scoped() {
        let s = state();
        s.schedule_summary(7, "a".into(), "t".into(), None, 5, None)
            .await;
        s.schedule_summary(7, "b".into(), "t".into(), None, 5, None)
            .await;
        s.schedule_summary(8, "c".into(), "t".into(), None, 5, None)
            .await;
        // cancelling chat 7 leaves chat 8's watch intact
        assert_eq!(s.remove_watches_for_chat(7).await, 2);
        let left = s.list_watches().await;
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].chat_id, 8);
        // cancelling again removes nothing
        assert_eq!(s.remove_watches_for_chat(7).await, 0);
    }

    #[tokio::test]
    async fn remove_all_watches_clears() {
        let s = state();
        s.schedule_summary(7, "a".into(), "t".into(), None, 5, None)
            .await;
        s.schedule_summary(7, "b".into(), "t".into(), None, 5, None)
            .await;
        assert_eq!(s.remove_all_watches().await, 2);
        assert!(s.list_watches().await.is_empty());
    }

    #[tokio::test]
    async fn remove_watch_with_cleanup_succeeds_even_if_server_absent() {
        let s = state();
        let cleanup = crate::persist::Cleanup {
            tool: "cancel_job".into(),
            args: None,
        };
        let id = s
            .schedule_summary(
                7,
                "weather".into(),
                "get_weather_summary".into(),
                None,
                10,
                Some(cleanup),
            )
            .await;
        // cleanup tool call fails (no MCP connected) but removal still succeeds
        assert!(s.remove_watch(id).await);
        assert!(s.list_watches().await.is_empty());
    }

    #[test]
    fn no_data_detection() {
        assert!(looks_like_no_data(
            r#"{"readings":0,"message":"No data collected yet"}"#
        ));
        assert!(looks_like_no_data(r#"{"readings": 0}"#));
        assert!(!looks_like_no_data(r#"{"readings":3,"avg_temp":21.5}"#));
    }
}
