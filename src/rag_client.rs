use std::{path::PathBuf, process::Stdio, time::Duration};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use tokio::{process::Command, time::timeout};

use crate::config::RagConfig;

const RAG_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Clone, Debug)]
pub struct RagClient {
    cfg: RagConfig,
}

#[derive(Debug, Deserialize)]
struct RagAnswer {
    answer: String,
    #[serde(default)]
    sources: Vec<RagSource>,
}

#[derive(Debug, Deserialize)]
struct RagSource {
    rank: usize,
    source: Option<String>,
    section: Option<String>,
    score: Option<f64>,
}

impl RagClient {
    pub fn new(cfg: RagConfig) -> Self {
        Self { cfg }
    }

    pub fn describe(&self) -> String {
        format!(
            "index={}, chat_model={}, embed_model={}, mode={}, top_k={}",
            self.cfg.index.display(),
            self.cfg.chat_model,
            self.cfg.embed_model,
            self.cfg.search_mode,
            self.cfg.top_k
        )
    }

    pub fn is_ready(&self) -> bool {
        index_exists(&self.cfg.index)
    }

    pub async fn answer(&self, question: &str) -> Result<String> {
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
            .arg("--ollama-url")
            .arg(&self.cfg.ollama_url)
            .arg("--search-mode")
            .arg(&self.cfg.search_mode)
            .arg("--top-k")
            .arg(self.cfg.top_k.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = timeout(RAG_TIMEOUT, cmd.output())
            .await
            .context("RAG client timed out")?
            .context("failed to run RAG client")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("RAG client failed: {}", stderr.trim()));
        }

        let payload: RagAnswer =
            serde_json::from_slice(&output.stdout).context("RAG client returned invalid JSON")?;
        Ok(render_answer(payload))
    }
}

fn render_answer(payload: RagAnswer) -> String {
    let mut out = payload.answer.trim().to_string();
    if payload.sources.is_empty() {
        return out;
    }

    out.push_str("\n\nИсточники:");
    for source in payload.sources.iter().take(5) {
        let path = source.source.as_deref().unwrap_or("unknown");
        let section = source.section.as_deref().unwrap_or("");
        let score = source
            .score
            .map(|value| format!(" score={value:.3}"))
            .unwrap_or_default();
        if section.is_empty() {
            out.push_str(&format!("\n[{}] {}{}", source.rank, path, score));
        } else {
            out.push_str(&format!(
                "\n[{}] {} / {}{}",
                source.rank, path, section, score
            ));
        }
    }
    out
}

pub fn index_exists(path: &PathBuf) -> bool {
    path.join("manifest.json").exists() && path.join("chunks.jsonl").exists()
}
