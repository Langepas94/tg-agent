use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub telegram_token: String,
    /// Chat ID that receives periodic digests (can be own user ID)
    pub digest_chat_id: i64,
    /// MCP server configuration
    pub mcp: McpConfig,
    /// How often to send digest, in minutes
    pub digest_interval_minutes: u64,
}

#[derive(Debug, Clone)]
pub enum McpConfig {
    /// Connect via HTTP (Streamable HTTP / SSE)
    Http { url: String },
    /// Spawn local process and talk via stdio
    Stdio { command: String, args: Vec<String> },
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let telegram_token = std::env::var("TELEGRAM_BOT_TOKEN")
            .context("TELEGRAM_BOT_TOKEN not set")?;

        let digest_chat_id = std::env::var("DIGEST_CHAT_ID")
            .context("DIGEST_CHAT_ID not set")?
            .parse::<i64>()
            .context("DIGEST_CHAT_ID must be an integer")?;

        let digest_interval_minutes = std::env::var("DIGEST_INTERVAL_MINUTES")
            .unwrap_or_else(|_| "360".into()) // 6 hours default
            .parse::<u64>()
            .unwrap_or(360);

        let mcp = if let Ok(url) = std::env::var("MCP_HTTP_URL") {
            McpConfig::Http { url }
        } else if let Ok(cmd) = std::env::var("MCP_COMMAND") {
            let args = std::env::var("MCP_ARGS")
                .unwrap_or_default()
                .split_whitespace()
                .map(String::from)
                .collect();
            McpConfig::Stdio { command: cmd, args }
        } else {
            // No MCP configured — bot runs without it
            McpConfig::Http { url: String::new() }
        };

        Ok(Config {
            telegram_token,
            digest_chat_id,
            mcp,
            digest_interval_minutes,
        })
    }
}
