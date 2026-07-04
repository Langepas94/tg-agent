use std::{path::PathBuf, process::Stdio, time::Duration};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use tokio::{process::Command, time::timeout};

use crate::config::RagConfig;

/// Per-turn budget: cold start on the 2GB VPS loads the embedding model from
/// disk (~30-60s) before the two chat API calls. RAG_TIMEOUT_SECS overrides.
fn rag_timeout() -> Duration {
    let secs = std::env::var("RAG_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(180);
    Duration::from_secs(secs)
}

#[derive(Clone, Debug)]
pub struct RagClient {
    cfg: RagConfig,
}

/// Parsed reply of `rag-indexer answer --mode rag --json`.
#[derive(Debug, Deserialize)]
pub struct RagReply {
    pub answer: String,
    #[serde(default)]
    pub sources: Vec<RagSource>,
    /// `false` = the agent refused: nothing in the index cleared the
    /// relevance floor, `answer` holds the fixed "не знаю" message.
    #[serde(default = "default_true")]
    pub relevant: bool,
    /// Standalone search query produced by the rewrite step, if it ran.
    #[serde(default)]
    pub rewritten_query: Option<String>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct RagSource {
    pub rank: usize,
    pub source: Option<String>,
    pub section: Option<String>,
    pub chunk_id: Option<String>,
    pub score: Option<f64>,
    /// Verbatim fragment from the retrieved chunk (citation).
    pub quote: Option<String>,
}

impl RagClient {
    pub fn new(cfg: RagConfig) -> Self {
        Self { cfg }
    }

    pub fn describe(&self) -> String {
        format!(
            "index={}, chat={}@{}, embed_model={}, mode={}, top_k={}, rewrite={}",
            self.cfg.index.display(),
            self.cfg.chat_model,
            self.cfg.chat_provider,
            self.cfg.embed_model,
            self.cfg.search_mode,
            self.cfg.top_k,
            self.cfg.rewrite
        )
    }

    pub fn is_ready(&self) -> bool {
        index_exists(&self.cfg.index)
    }

    /// One RAG turn. `history` is prior dialog turns (role, text), oldest
    /// first, WITHOUT the current question. `task_state` is a prose snapshot
    /// of the dialog goal / fixed constraints for the system prompt.
    pub async fn answer(
        &self,
        question: &str,
        history: &[(String, String)],
        task_state: Option<&str>,
    ) -> Result<RagReply> {
        let mut cmd = Command::new(&self.cfg.bin);
        cmd.arg("answer")
            .arg("--mode")
            .arg("rag")
            .arg("--json")
            .arg("--index")
            .arg(&self.cfg.index)
            .arg("--query")
            .arg(question)
            .arg("--model")
            .arg(&self.cfg.embed_model)
            .arg("--chat-model")
            .arg(&self.cfg.chat_model)
            .arg("--chat-url")
            .arg(&self.cfg.chat_url)
            .arg("--chat-provider")
            .arg(&self.cfg.chat_provider)
            .arg("--ollama-url")
            .arg(&self.cfg.ollama_url)
            .arg("--search-mode")
            .arg(&self.cfg.search_mode)
            .arg("--top-k")
            .arg(self.cfg.top_k.to_string());
        if let Some(min_score) = self.cfg.min_score {
            cmd.arg("--min-dense-score").arg(min_score.to_string());
        }

        if !history.is_empty() {
            let payload: Vec<serde_json::Value> = history
                .iter()
                .map(|(role, text)| serde_json::json!({ "role": role, "content": text }))
                .collect();
            cmd.arg("--history").arg(serde_json::to_string(&payload)?);
            // Rewrite only helps when there is history to resolve references from.
            if self.cfg.rewrite {
                cmd.arg("--rewrite");
            }
        }
        if let Some(state) = task_state.filter(|s| !s.trim().is_empty()) {
            cmd.arg("--task-state").arg(state);
        }
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let output = timeout(rag_timeout(), cmd.output())
            .await
            .context("RAG client timed out")?
            .context("failed to run RAG client")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("RAG client failed: {}", stderr.trim()));
        }

        serde_json::from_slice(&output.stdout).context("RAG client returned invalid JSON")
    }
}

impl RagReply {
    /// Telegram-ready text: answer + mandatory sources with citations.
    pub fn render(&self) -> String {
        let mut out = self.answer.trim().to_string();
        if self.sources.is_empty() {
            return out;
        }

        out.push_str("\n\nИсточники:");
        for source in self.sources.iter().take(5) {
            let path = source.source.as_deref().unwrap_or("unknown");
            let section = source.section.as_deref().unwrap_or("");
            let chunk = source
                .chunk_id
                .as_deref()
                .map(|id| format!(" #{id}"))
                .unwrap_or_default();
            let score = source
                .score
                .map(|value| format!(" score={value:.3}"))
                .unwrap_or_default();
            if section.is_empty() {
                out.push_str(&format!("\n[{}] {}{}{}", source.rank, path, chunk, score));
            } else {
                out.push_str(&format!(
                    "\n[{}] {} / {}{}{}",
                    source.rank, path, section, chunk, score
                ));
            }
            if let Some(quote) = source.quote.as_deref().filter(|q| !q.trim().is_empty()) {
                out.push_str(&format!("\n    «{}»", quote.trim()));
            }
        }
        out
    }
}

pub fn index_exists(path: &PathBuf) -> bool {
    path.join("manifest.json").exists() && path.join("chunks.jsonl").exists()
}
