use anyhow::{Context, Result};
use rmcp::{
    model::{CallToolRequestParams, JsonObject, LoggingMessageNotificationParam, Tool},
    service::{NotificationContext, RoleClient, RunningService},
    transport::{
        streamable_http_client::StreamableHttpClientTransportConfig, ConfigureCommandExt,
        StreamableHttpClientTransport, TokioChildProcess,
    },
    ClientHandler, Peer, ServiceExt,
};
use std::{collections::HashMap, sync::Arc, time::Duration};
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
    /// Server-pushed summary (logging notification, logger="weather_summary").
    /// `data` is the structured summary payload (carries session_id, stats, …).
    PushSummary {
        server: String,
        data: serde_json::Value,
    },
}

pub type EventSender = broadcast::Sender<McpEvent>;

/// Per-tool-call ceiling. A geocode/forecast that stalls past this errors out
/// (surfaced to the model as a tool error) instead of hanging the chat turn.
const TOOL_TIMEOUT: Duration = Duration::from_secs(45);

/// User-supplied connection parameters for one MCP server.
///
/// Two transports are supported, chosen by which fields are set:
/// - **HTTP** (default): `url` is an `http(s)://` Streamable-HTTP endpoint.
/// - **stdio**: `command` is non-empty (`[program, arg, ...]`); the client
///   spawns it as a child process and talks over stdin/stdout. `url` is then
///   unused. Most npm/uvx MCP servers ship stdio-only — this lets the bot
///   connect to them directly without an HTTP bridge.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConnectParams {
    pub name: String,
    /// HTTP endpoint. Empty for stdio servers.
    #[serde(default)]
    pub url: String,
    /// Bearer token (sent as `Authorization: Bearer <auth>`). HTTP only.
    #[serde(default)]
    pub auth: Option<String>,
    /// Extra HTTP headers (e.g. X-Tracker-Token: ...). HTTP only.
    #[serde(default)]
    pub headers: Vec<(String, String)>,
    /// stdio transport: `[program, arg, ...]`. Non-empty ⇒ stdio server.
    #[serde(default)]
    pub command: Vec<String>,
    /// Extra environment variables for the stdio child (`KEY`, `VALUE`).
    #[serde(default)]
    pub env: Vec<(String, String)>,
}

impl ConnectParams {
    /// True when this server is launched as a stdio child process.
    pub fn is_stdio(&self) -> bool {
        !self.command.is_empty()
    }

    /// Short human-readable connection target (URL or the spawn command).
    pub fn target(&self) -> String {
        if self.is_stdio() {
            self.command.join(" ")
        } else {
            self.url.clone()
        }
    }
}

/// One live MCP connection
pub struct McpClient {
    pub params: ConnectParams,
    peer: Peer<RoleClient>,
    tools: Arc<Mutex<Vec<Tool>>>,
    _task: tokio::task::JoinHandle<()>,
}

impl McpClient {
    /// Connect to an MCP server. Transport is picked from `params`:
    /// stdio child process when `command` is set, otherwise Streamable HTTP.
    /// Shares one broadcast channel across all servers (event carries server name).
    pub async fn connect(params: ConnectParams, tx: EventSender) -> Result<Self> {
        info!("Connecting MCP '{}' -> {}", params.name, params.target());

        // The transport type differs per branch, but `serve()` erases it —
        // both arms yield the same `RunningService<RoleClient, _>`.
        let svc = if params.is_stdio() {
            Self::spawn_stdio(&params, &tx).await?
        } else {
            Self::connect_http(&params, &tx).await?
        };
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

    /// Build a Streamable-HTTP transport and complete the MCP handshake.
    async fn connect_http(
        params: &ConnectParams,
        tx: &EventSender,
    ) -> Result<RunningService<RoleClient, NotificationHandler>> {
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
        NotificationHandler {
            server: params.name.clone(),
            tx: tx.clone(),
        }
        .serve(transport)
        .await
        .context("MCP handshake failed (check URL / auth / headers)")
    }

    /// Spawn a stdio MCP server as a child process and complete the handshake.
    async fn spawn_stdio(
        params: &ConnectParams,
        tx: &EventSender,
    ) -> Result<RunningService<RoleClient, NotificationHandler>> {
        let (prog, args) = params
            .command
            .split_first()
            .context("stdio server has an empty command")?;
        // `which_command` resolves shims (npx.cmd, uvx) via PATH so spawning
        // works regardless of the host OS; fall back to a bare name.
        let cmd = rmcp::transport::which_command(prog)
            .unwrap_or_else(|_| tokio::process::Command::new(prog));
        let env = params.env.clone();
        let args = args.to_vec();
        let cmd = cmd.configure(|c| {
            c.args(&args);
            for (k, v) in &env {
                c.env(k, v);
            }
        });
        let transport = TokioChildProcess::new(cmd)
            .with_context(|| format!("failed to spawn stdio server: {prog}"))?;
        NotificationHandler {
            server: params.name.clone(),
            tx: tx.clone(),
        }
        .serve(transport)
        .await
        .context("MCP handshake failed (stdio server exited or spoke bad protocol)")
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
        // Bound every tool call — a stalled MCP server must not hang the agent
        // (and therefore the whole chat turn) indefinitely.
        let result = tokio::time::timeout(TOOL_TIMEOUT, self.peer.call_tool(params))
            .await
            .with_context(|| format!("call_tool '{tool}' timed out after {TOOL_TIMEOUT:?}"))?
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
            // Structured summary push (server-driven periodic delivery).
            if params.logger.as_deref() == Some("weather_summary") {
                let _ = self.tx.send(McpEvent::PushSummary {
                    server: self.server.clone(),
                    data: params.data.clone(),
                });
                return;
            }
            let level = format!("{:?}", params.level);
            let data = params
                .data
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| params.data.to_string());
            let _ = self.tx.send(McpEvent::LogMessage {
                server: self.server.clone(),
                level,
                data,
            });
        }
    }
}
