use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{sync::OnceLock, time::Duration};

const DEFAULT_SUPPORT_URL: &str = "https://support.5-129-234-9.sslip.io";

#[derive(Serialize)]
struct SupportRequest {
    command: String,
}

#[derive(Deserialize)]
struct SupportResponse {
    answer: String,
}

pub async fn answer(question: &str) -> Result<String> {
    let base_url =
        std::env::var("SUPPORT_SERVICE_URL").unwrap_or_else(|_| DEFAULT_SUPPORT_URL.to_string());
    let endpoint = format!("{}/api/support", base_url.trim_end_matches('/'));
    let response = client()
        .post(endpoint)
        .json(&SupportRequest {
            command: format!("/support {}", question.trim()),
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
}
