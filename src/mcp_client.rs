use anyhow::{Context, Result};
use rmcp::{
    model::{CallToolRequestParams, JsonObject, LoggingMessageNotificationParam, Tool},
    service::{NotificationContext, RoleClient},
    transport::{
        streamable_http_client::StreamableHttpClientTransportConfig, StreamableHttpClientTransport,
    },
    ClientHandler, Peer, ServiceExt,
};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{broadcast, Mutex};
use tracing::{error, info};

#[derive(Debug, Clone)]
pub enum McpEvent {
    /// Which server's tools changed
    ToolsChanged { server: String },
    LogMessage {
        server: String,
        level: String,
        data: String,
    },
}

pub type EventSender = broadcast::Sender<McpEvent>;

/// User-supplied connection parameters for one MCP server
#[derive(Debug, Clone)]
pub struct ConnectParams {
    pub name: String,
    pub url: String,
    /// Bearer token (sent as `Authorization: Bearer <auth>`)
    pub auth: Option<String>,
    /// Extra HTTP headers (e.g. X-Tracker-Token: ...)
    pub headers: Vec<(String, String)>,
}

/// One live MCP connection
pub struct McpClient {
    pub params: ConnectParams,
    peer: Peer<RoleClient>,
    tools: Arc<Mutex<Vec<Tool>>>,
    _task: tokio::task::JoinHandle<()>,
}

impl McpClient {
    /// Connect to an MCP server over Streamable HTTP.
    /// Shares one broadcast channel across all servers (event carries server name).
    pub async fn connect(params: ConnectParams, tx: EventSender) -> Result<Self> {
        info!("Connecting MCP '{}' -> {}", params.name, params.url);

        let mut cfg = StreamableHttpClientTransportConfig::with_uri(params.url.clone());
        if let Some(a) = &params.auth {
            cfg = cfg.auth_header(a.clone());
        }
        if !params.headers.is_empty() {
            let mut map = HashMap::new();
            for (k, v) in &params.headers {
                let name = http::HeaderName::from_bytes(k.as_bytes())
                    .with_context(|| format!("invalid header name: {k}"))?;
                let val = http::HeaderValue::from_str(v)
                    .with_context(|| format!("invalid header value for {k}"))?;
                map.insert(name, val);
            }
            cfg = cfg.custom_headers(map);
        }

        let transport = StreamableHttpClientTransport::from_config(cfg);
        let handler = NotificationHandler {
            server: params.name.clone(),
            tx: tx.clone(),
        };
        let svc = handler
            .serve(transport)
            .await
            .context("MCP handshake failed (check URL / auth / headers)")?;
        let peer: Peer<RoleClient> = svc.peer().clone();

        let tools = peer.list_all_tools().await.context("list_tools failed")?;
        let n = tools.len();
        let tools = Arc::new(Mutex::new(tools));
        info!("MCP '{}' ready, {n} tools", params.name);

        // Keep the service alive + refresh tools on change
        let task = {
            let peer = peer.clone();
            let tools = tools.clone();
            let server = params.name.clone();
            let mut rx = tx.subscribe();
            tokio::spawn(async move {
                // Hold svc so the connection stays open
                let _svc = svc;
                loop {
                    match rx.recv().await {
                        Ok(McpEvent::ToolsChanged { server: s }) if s == server => {
                            match peer.list_all_tools().await {
                                Ok(t) => {
                                    info!("'{server}' tools refreshed: {}", t.len());
                                    *tools.lock().await = t;
                                }
                                Err(e) => error!("'{server}' refresh failed: {e}"),
                            }
                        }
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(_)) => {}
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
            })
        };

        Ok(McpClient {
            params,
            peer,
            tools,
            _task: task,
        })
    }

    pub async fn tools(&self) -> Vec<Tool> {
        self.tools.lock().await.clone()
    }

    /// Call a tool by name with optional JSON-object arguments.
    /// Returns the result rendered as plain text (text content + structured JSON).
    pub async fn call_tool(&self, tool: &str, arguments: Option<JsonObject>) -> Result<String> {
        let mut params = CallToolRequestParams::new(tool.to_string());
        if let Some(args) = arguments {
            params = params.with_arguments(args);
        }
        let result = self
            .peer
            .call_tool(params)
            .await
            .with_context(|| format!("call_tool '{tool}' failed"))?;

        let mut out = String::new();
        for c in &result.content {
            if let Some(t) = c.as_text() {
                out.push_str(&t.text);
                out.push('\n');
            }
        }
        if let Some(sc) = &result.structured_content {
            if out.trim().is_empty() {
                out = serde_json::to_string_pretty(sc).unwrap_or_else(|_| sc.to_string());
            }
        }
        if result.is_error.unwrap_or(false) {
            anyhow::bail!("tool reported an error:\n{}", out.trim());
        }
        Ok(out.trim().to_string())
    }
}

struct NotificationHandler {
    server: String,
    tx: EventSender,
}

impl ClientHandler for NotificationHandler {
    fn on_tool_list_changed(
        &self,
        _ctx: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async move {
            let _ = self.tx.send(McpEvent::ToolsChanged {
                server: self.server.clone(),
            });
        }
    }

    fn on_logging_message(
        &self,
        params: LoggingMessageNotificationParam,
        _ctx: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async move {
            let level = format!("{:?}", params.level);
            let data = params.data.as_str().unwrap_or("").to_string();
            let _ = self.tx.send(McpEvent::LogMessage {
                server: self.server.clone(),
                level,
                data,
            });
        }
    }
}
