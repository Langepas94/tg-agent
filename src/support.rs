use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{sync::OnceLock, time::Duration};

const DEFAULT_SUPPORT_URL: &str = "https://support.5-129-234-9.sslip.io";

#[derive(Serialize)]
struct SupportRequest {
    command: String,
    telegram_user_id: u64,
}

#[derive(Deserialize)]
struct SupportResponse {
    answer: String,
}

pub async fn answer(question: &str, telegram_user_id: u64) -> Result<String> {
    let base_url =
        std::env::var("SUPPORT_SERVICE_URL").unwrap_or_else(|_| DEFAULT_SUPPORT_URL.to_string());
    let access_key =
        std::env::var("SUPPORT_ACCESS_KEY").context("support service access is not configured")?;
    let endpoint = format!("{}/api/support", base_url.trim_end_matches('/'));
    let response = client()
        .post(endpoint)
        .header("x-support-key", access_key)
        .json(&SupportRequest {
            command: format!("/support {}", question.trim()),
            telegram_user_id,
        })
        .send()
        .await
        .context("support request failed")?
        .error_for_status()
        .context("support service rejected request")?
        .json::<SupportResponse>()
        .await
        .context("support service returned invalid response")?;
    let answer = response.answer.trim();
    if answer.is_empty() {
        anyhow::bail!("support service returned an empty answer");
    }
    Ok(plain_telegram(answer))
}

fn plain_telegram(value: &str) -> String {
    value.replace("```", "").replace("**", "").replace('`', "")
}

fn client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(75))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_support_endpoint() {
        let base = DEFAULT_SUPPORT_URL.trim_end_matches('/');
        assert_eq!(
            format!("{base}/api/support"),
            "https://support.5-129-234-9.sslip.io/api/support"
        );
    }

    #[test]
    fn removes_markdown_from_support_answer() {
        assert_eq!(
            plain_telegram("**Шаг:** нажмите `Войти`"),
            "Шаг: нажмите Войти"
        );
    }

    #[test]
    fn request_contains_telegram_user_id() {
        let request = SupportRequest {
            command: "/support Не работает вход".into(),
            telegram_user_id: 101,
        };
        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["telegram_user_id"], 101);
    }
}
