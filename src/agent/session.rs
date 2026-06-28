//! Per-chat agent session: layered memory + user profile + invariants,
//! persisted to disk (one JSON file per chat under the sessions dir).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{invariants::Invariant, memory::AgentMemory, notes::UserNotes, profile::UserProfile};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub chat_id: i64,
    #[serde(default)]
    pub memory: AgentMemory,
    #[serde(default)]
    pub profile: UserProfile,
    /// Free-form labelled preferences ("доп инфа"), conditionally injected.
    #[serde(default)]
    pub notes: UserNotes,
    /// Custom invariants; empty → defaults are used at prompt-build time.
    #[serde(default)]
    pub invariants: Vec<Invariant>,
    /// Active stateful trip-planning flow, if any (Clarify suspends here across
    /// turns). `None` outside a planning conversation.
    #[serde(default)]
    pub trip: Option<super::flow::TripFlowState>,
}

impl ChatSession {
    pub fn new(chat_id: i64) -> Self {
        Self {
            chat_id,
            memory: AgentMemory::default(),
            profile: UserProfile::default(),
            notes: UserNotes::default(),
            invariants: Vec::new(),
            trip: None,
        }
    }

    /// Effective invariants: custom if set, else the travel-weather defaults.
    pub fn effective_invariants(&self) -> Vec<Invariant> {
        if self.invariants.is_empty() {
            super::invariants::travel_weather_defaults()
        } else {
            self.invariants.clone()
        }
    }

    /// Drop legacy auto-extracted noise before it reaches the prompt again.
    pub fn sanitize(&mut self) -> bool {
        self.memory.sanitize()
    }
}

/// Directory holding per-chat session files: `$SESSIONS_DIR` or `./sessions`.
pub fn sessions_dir() -> PathBuf {
    std::env::var("SESSIONS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("sessions"))
}

fn session_path(chat_id: i64) -> PathBuf {
    sessions_dir().join(format!("{chat_id}.json"))
}

pub fn load(chat_id: i64) -> ChatSession {
    let path = session_path(chat_id);
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let mut session = serde_json::from_str(&s).unwrap_or_else(|e| {
                tracing::warn!("session {chat_id} corrupt ({e}); starting fresh");
                ChatSession::new(chat_id)
            });
            if session.sanitize() {
                let _ = save(&session);
            }
            session
        }
        Err(_) => ChatSession::new(chat_id),
    }
}

pub fn save(session: &ChatSession) -> anyhow::Result<()> {
    let dir = sessions_dir();
    std::fs::create_dir_all(&dir)?;
    let path = session_path(session.chat_id);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(session)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}
