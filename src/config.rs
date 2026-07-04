use anyhow::{bail, Context, Result};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub telegram_token: String,
    /// Password required to unlock the bot for a Telegram chat/user.
    pub bot_password: String,
    /// Web admin UI bind address, e.g. 127.0.0.1:8080 behind nginx.
    pub admin_addr: Option<String>,
    pub admin_username: String,
    pub admin_password: Option<String>,
    /// How often to send digest, in minutes
    pub digest_interval_minutes: u64,
    /// LLM config; None means the agent answers only via explicit commands.
    pub llm: Option<LlmConfig>,
    /// Optional local RAG client config.
    pub rag: Option<RagConfig>,
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

#[derive(Debug, Clone)]
pub struct RagConfig {
    pub bin: String,
    pub index: PathBuf,
    pub embed_model: String,
    pub chat_model: String,
    pub chat_url: String,
    pub ollama_url: String,
    pub search_mode: String,
    pub top_k: usize,
    /// History-aware LLM query rewrite before retrieval (RAG_REWRITE, default on).
    pub rewrite: bool,
    /// Absolute dense-cosine relevance floor (RAG_MIN_SCORE). None → the
    /// indexer's default (0.35). Calibrate per corpus: for the real
    /// tg-agent+open-meteo corpus 0.5 separates legit questions (top chunks
    /// ≥0.63) from off-topic near-misses (≤0.47).
    pub min_score: Option<f64>,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let telegram_token =
            std::env::var("TELEGRAM_BOT_TOKEN").context("TELEGRAM_BOT_TOKEN not set")?;

        let digest_interval_minutes = std::env::var("DIGEST_INTERVAL_MINUTES")
            .unwrap_or_else(|_| "360".into()) // 6 hours default
            .parse::<u64>()
            .unwrap_or(360);
        let bot_password = std::env::var("BOT_PASSWORD").unwrap_or_else(|_| "202020".into());
        let admin_addr = std::env::var("ADMIN_ADDR")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| Some("127.0.0.1:8080".into()));
        let admin_username = std::env::var("ADMIN_USERNAME").unwrap_or_else(|_| "admin".into());
        let admin_password = std::env::var("ADMIN_PASSWORD")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        if admin_password.as_deref() == Some(bot_password.as_str()) {
            bail!("ADMIN_PASSWORD must be different from BOT_PASSWORD");
        }

        // Accept DEEPSEEK_API_KEY or generic LLM_API_KEY.
        let api_key = std::env::var("LLM_API_KEY")
            .ok()
            .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok())
            .filter(|s| !s.trim().is_empty());

        let llm = api_key.map(|api_key| LlmConfig {
            api_key,
            base_url: std::env::var("LLM_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
            model: std::env::var("LLM_MODEL").unwrap_or_else(|_| "deepseek-chat".into()),
        });

        let rag = std::env::var("RAG_INDEX")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(|index| RagConfig {
                bin: std::env::var("RAG_INDEXER_BIN").unwrap_or_else(|_| "rag-indexer".into()),
                index: PathBuf::from(index),
                embed_model: std::env::var("RAG_EMBED_MODEL")
                    .unwrap_or_else(|_| "qwen3-embedding".into()),
                chat_model: std::env::var("RAG_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:7b".into()),
                chat_url: std::env::var("RAG_CHAT_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".into()),
                ollama_url: std::env::var("RAG_OLLAMA_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".into()),
                search_mode: std::env::var("RAG_SEARCH_MODE").unwrap_or_else(|_| "hybrid".into()),
                top_k: std::env::var("RAG_TOP_K")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(5),
                rewrite: std::env::var("RAG_REWRITE")
                    .map(|s| !matches!(s.trim(), "0" | "false" | "off" | "no"))
                    .unwrap_or(true),
                min_score: std::env::var("RAG_MIN_SCORE")
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok()),
            });

        Ok(Config {
            telegram_token,
            bot_password,
            admin_addr,
            admin_username,
            admin_password,
            digest_interval_minutes,
            llm,
            rag,
        })
    }
}
