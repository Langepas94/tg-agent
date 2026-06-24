use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub telegram_token: String,
    /// How often to send digest, in minutes
    pub digest_interval_minutes: u64,
    /// LLM config; None means the agent answers only via explicit commands.
    pub llm: Option<LlmConfig>,
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let telegram_token =
            std::env::var("TELEGRAM_BOT_TOKEN").context("TELEGRAM_BOT_TOKEN not set")?;

        let digest_interval_minutes = std::env::var("DIGEST_INTERVAL_MINUTES")
            .unwrap_or_else(|_| "360".into()) // 6 hours default
            .parse::<u64>()
            .unwrap_or(360);

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

        Ok(Config {
            telegram_token,
            digest_interval_minutes,
            llm,
        })
    }
}
