use anyhow::{bail, Result};
use rmcp::{
    model::{LoggingMessageNotificationParam, Tool},
    service::{NotificationContext, RoleClient},
    ClientHandler, Peer, ServiceExt,
};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use tracing::{error, info};

use crate::config::McpConfig;

#[derive(Debug, Clone)]
pub enum McpEvent {
    ToolsChanged,
    LogMessage { level: String, data: String },
}

pub type EventSender = broadcast::Sender<McpEvent>;
pub type EventReceiver = broadcast::Receiver<McpEvent>;

/// Active MCP connection: holds peer handle + cached tool list
pub struct McpClient {
    #[allow(dead_code)]
    peer: Peer<RoleClient>,
    tools: Arc<Mutex<Vec<Tool>>>,
    /// Keep the JoinHandle alive so the connection doesn't drop
    _task: tokio::task::JoinHandle<()>,
}

impl McpClient {
    pub async fn connect(cfg: &McpConfig) -> Result<(Self, EventReceiver)> {
        let (tx, rx) = broadcast::channel::<McpEvent>(64);
        match cfg {
            McpConfig::Http { url } if !url.is_empty() => {
                Self::connect_http(url, tx, rx).await
            }
            McpConfig::Stdio { command, args } => {
                Self::connect_stdio(command, args, tx, rx).await
            }
            McpConfig::Http { .. } => bail!("MCP_HTTP_URL is empty"),
        }
    }

    async fn connect_http(
        url: &str,
        tx: EventSender,
        rx: EventReceiver,
    ) -> Result<(Self, EventReceiver)> {
        use rmcp::transport::StreamableHttpClientTransport;

        info!("Connecting MCP via HTTP: {url}");
        let transport = StreamableHttpClientTransport::from_uri(url);
        let svc = NotificationHandler { tx: tx.clone() }.serve(transport).await?;
        let peer: Peer<RoleClient> = svc.peer().clone();

        let tools = Arc::new(Mutex::new(list_all_tools(&peer).await?));
        info!("MCP ready, {} tools", tools.lock().await.len());

        let task = spawn_refresh_task(peer.clone(), tools.clone(), tx);

        Ok((McpClient { peer, tools, _task: task }, rx))
    }

    async fn connect_stdio(
        command: &str,
        args: &[String],
        tx: EventSender,
        rx: EventReceiver,
    ) -> Result<(Self, EventReceiver)> {
        use rmcp::transport::TokioChildProcess;
        use tokio::process::Command;

        info!("Connecting MCP via stdio: {command} {args:?}");
        let mut cmd = Command::new(command);
        cmd.args(args);
        let transport = TokioChildProcess::new(cmd)?;
        let svc = NotificationHandler { tx: tx.clone() }.serve(transport).await?;
        let peer: Peer<RoleClient> = svc.peer().clone();

        let tools = Arc::new(Mutex::new(list_all_tools(&peer).await?));
        info!("MCP stdio ready, {} tools", tools.lock().await.len());

        let task = spawn_refresh_task(peer.clone(), tools.clone(), tx);

        Ok((McpClient { peer, tools, _task: task }, rx))
    }

    pub async fn tools(&self) -> Vec<Tool> {
        self.tools.lock().await.clone()
    }
}

/// Fetch all tools (paginated)
async fn list_all_tools(peer: &Peer<RoleClient>) -> Result<Vec<Tool>> {
    let result = peer.list_all_tools().await?;
    Ok(result)
}

/// Background task: listens to broadcast and refreshes tool list on ToolsChanged
fn spawn_refresh_task(
    peer: Peer<RoleClient>,
    tools: Arc<Mutex<Vec<Tool>>>,
    tx: EventSender,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut rx = tx.subscribe();
        loop {
            match rx.recv().await {
                Ok(McpEvent::ToolsChanged) => {
                    match list_all_tools(&peer).await {
                        Ok(t) => {
                            let n = t.len();
                            *tools.lock().await = t;
                            info!("Tools refreshed: {n}");
                        }
                        Err(e) => error!("Refresh tools failed: {e}"),
                    }
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("Event channel lagged: missed {n}");
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

/// rmcp handler that forwards server notifications into our broadcast channel
struct NotificationHandler {
    tx: EventSender,
}

impl ClientHandler for NotificationHandler {
    fn on_tool_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async move {
            let _ = self.tx.send(McpEvent::ToolsChanged);
        }
    }

    fn on_logging_message(
        &self,
        params: LoggingMessageNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async move {
            let level = format!("{:?}", params.level);
            let data = params.data.as_str().unwrap_or("").to_string();
            let _ = self.tx.send(McpEvent::LogMessage { level, data });
        }
    }
}
