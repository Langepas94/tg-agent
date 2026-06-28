use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub telegram_token: String,
    /// Password required to unlock the bot for a Telegram chat/user.
    pub bot_password: String,
    /// Web admin UI bind address, e.g. 127.0.0.1:8080 behind nginx.
    pub admin_addr: Option<String>,
    pub admin_username: String,
    pub admin_password: String,
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
        let bot_password = std::env::var("BOT_PASSWORD").unwrap_or_else(|_| "202020".into());
        let admin_addr = std::env::var("ADMIN_ADDR")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| Some("127.0.0.1:8080".into()));
        let admin_username = std::env::var("ADMIN_USERNAME").unwrap_or_else(|_| "admin".into());
        let admin_password = std::env::var("ADMIN_PASSWORD")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| bot_password.clone());

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
            bot_password,
            admin_addr,
            admin_username,
            admin_password,
            digest_interval_minutes,
            llm,
        })
    }
}
