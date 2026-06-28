//! Semantic intent router.
//!
//! Replaces the old substring keyword gates — trip-trigger words, "go" phrases,
//! and the on/off-topic vocabulary lists — with a single cheap LLM call that
//! decides, BY MEANING, what to do with a message: start the trip-planning
//! swarm, answer as a normal weather/travel chat, or refuse as off-topic.
//!
//! There are deliberately NO keyword lists here. If the model call fails we
//! default to [`Route::Chat`] (never silently block a user), which also matches
//! the old gate's "when unsure, allow" bias.

use serde::Deserialize;

use crate::llm::Llm;

/// What to do with an incoming message that is NOT already inside an active flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// The user wants to plan an outdoor / overnight trip → enter the swarm.
    Trip,
    /// A normal in-scope weather / travel / places message → answer it.
    Chat,
    /// Clearly unrelated to weather / travel / outdoors → refuse early.
    OffTopic,
}

const ROUTER_PROMPT: &str = "You are the intent router of a weather + travel/outdoor-trip \
planning assistant. Read the user's message and decide ONE route. Return ONLY JSON \
{\"route\":\"trip|chat|offtopic\",\"reason\":\"<short>\"}.\n\
- trip: the user wants to PLAN an outdoor/nature recreation activity, overnight stay, \
weekend getaway, multi-stop route, campsite/BBQ outing, field visit, or similar outdoor plan, \
or asks to plan/organize such a trip. Choose this even if the request is long and detailed.\n\
- chat: any other on-topic message — a weather question or forecast, a city/country/travel \
question, a place name, a short follow-up ('а завтра?', 'да', 'Москва'), or a greeting.\n\
- offtopic: clearly unrelated to weather, travel, or the outdoors — e.g. writing code, recipes, \
math, crypto/stocks, medicine, law, politics, homework, poems.\n\
When unsure between chat and offtopic, choose chat (never wrongly refuse a real request). \
Judge by meaning, in any language.";

#[derive(Debug, Deserialize)]
struct RouterJson {
    #[serde(default)]
    route: String,
}

/// Classify one message. On any LLM/parse failure, returns [`Route::Chat`] so a
/// transient model error never blocks the user or misroutes them out of chat.
pub async fn classify(llm: &Llm, user_text: &str) -> Route {
    let raw = match llm.complete(ROUTER_PROMPT, user_text).await {
        Ok(s) => s,
        Err(_) => return Route::Chat,
    };
    parse_route(&raw)
}

/// Parse the router JSON leniently (tolerates fences / surrounding prose).
fn parse_route(raw: &str) -> Route {
    let parsed: Option<RouterJson> = serde_json::from_str(&extract_json(raw)).ok();
    match parsed.map(|p| p.route.trim().to_ascii_lowercase()) {
        Some(r) if r == "trip" => Route::Trip,
        Some(r) if r == "offtopic" || r == "off_topic" || r == "off-topic" => Route::OffTopic,
        _ => Route::Chat,
    }
}

/// First `{...}` block from a possibly-fenced LLM reply.
fn extract_json(s: &str) -> String {
    let s = s.trim();
    if let (Some(start), Some(end)) = (s.find('{'), s.rfind('}')) {
        if end >= start {
            return s[start..=end].to_string();
        }
    }
    "{}".to_string()
}

/// The fixed refusal sent for off-topic messages (RU — primary user language).
pub const OFF_TOPIC_REPLY: &str = "Я ассистент по погоде и планированию путешествий — \
помогаю с прогнозами, выбором времени поездки и условиями в городах. \
По другим темам не подскажу. Спросите, например: «какая погода в Сочи на выходных?»";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_route_reads_each_variant() {
        assert_eq!(parse_route(r#"{"route":"trip","reason":"x"}"#), Route::Trip);
        assert_eq!(parse_route(r#"{"route":"chat"}"#), Route::Chat);
        assert_eq!(parse_route(r#"{"route":"offtopic"}"#), Route::OffTopic);
        assert_eq!(parse_route(r#"{"route":"off_topic"}"#), Route::OffTopic);
    }

    #[test]
    fn parse_route_tolerates_fences_and_prose() {
        let r = parse_route("sure ```json\n{\"route\":\"trip\"}\n``` done");
        assert_eq!(r, Route::Trip);
    }

    #[test]
    fn parse_route_defaults_to_chat_on_junk() {
        assert_eq!(parse_route("not json at all"), Route::Chat);
        assert_eq!(parse_route(r#"{"route":"weird"}"#), Route::Chat);
    }
}
