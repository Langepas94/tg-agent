use std::sync::Arc;
use tokio::sync::Mutex;

use crate::mcp_client::McpClient;

/// Shared bot state — cloned into each handler
#[derive(Clone)]
pub struct BotState {
    pub mcp: Arc<Mutex<Option<McpClient>>>,
}

impl BotState {
    pub fn new() -> Self {
        Self {
            mcp: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_mcp(client: McpClient) -> Self {
        Self {
            mcp: Arc::new(Mutex::new(Some(client))),
        }
    }
}
