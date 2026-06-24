use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub telegram_token: String,
    /// How often to send digest, in minutes
    pub digest_interval_minutes: u64,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let telegram_token =
            std::env::var("TELEGRAM_BOT_TOKEN").context("TELEGRAM_BOT_TOKEN not set")?;

        let digest_interval_minutes = std::env::var("DIGEST_INTERVAL_MINUTES")
            .unwrap_or_else(|_| "360".into()) // 6 hours default
            .parse::<u64>()
            .unwrap_or(360);

        Ok(Config {
            telegram_token,
            digest_interval_minutes,
        })
    }
}
