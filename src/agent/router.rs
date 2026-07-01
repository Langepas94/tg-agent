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

/// What to do when a trip flow is already open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveFlowRoute {
    /// The message answers/changes/continues the active trip flow.
    ContinueTrip,
    /// A side question that should be answered normally while preserving the flow.
    SideChat,
    /// Clearly unrelated to the assistant scope.
    OffTopic,
}

const ROUTER_PROMPT: &str = "You are the intent router of a weather + travel/outdoor-trip \
planning assistant. Read the user's message and decide ONE route. Return ONLY JSON \
{\"route\":\"trip|chat|offtopic\",\"reason\":\"<short>\"}.\n\
- trip: ONLY when the user explicitly wants to PLAN an outdoor/nature recreation activity, \
overnight stay, weekend getaway, multi-stop route, campsite/BBQ outing, field visit, or similar \
outdoor plan, or asks to plan/organize such a trip. Choose this even if the trip request is long \
and detailed. Do NOT choose trip for questions ABOUT software, MCP, this project, code files, \
implementation, architecture, or how tools work internally.\n\
- chat: any other on-topic message — a weather question or forecast, a city/country/travel \
question, a question about this assistant/project/MCP/tools, a place name, a short follow-up \
('а завтра?', 'да', 'Москва'), or a greeting.\n\
- offtopic: clearly unrelated to weather, travel, or the outdoors — e.g. writing code, recipes, \
math, crypto/stocks, medicine, law, politics, homework, poems.\n\
When unsure between chat and offtopic, choose chat (never wrongly refuse a real request). \
Judge by meaning, in any language.";

const ACTIVE_FLOW_ROUTER_PROMPT: &str = "You are the active-flow router of a Telegram assistant. \
There is an unfinished outdoor-trip planning flow in this chat. Decide whether the user's latest \
message continues that trip flow or is a side question. Return ONLY JSON \
{\"route\":\"continue_trip|side_chat|offtopic\",\"reason\":\"<short>\"}.\n\
- continue_trip: the user answers a clarification, chooses an option, says continue/go/yes, asks \
to change date/place/constraints/artifacts for the current trip, or otherwise talks about the \
active trip plan.\n\
- side_chat: the user asks a separate question that should be answered normally while the trip \
flow remains paused. This includes questions ABOUT MCP, connected tools, this project, code files, \
implementation, architecture, or how the bot works internally.\n\
- offtopic: clearly unrelated to weather, travel, outdoors, MCP/tools, or this assistant.\n\
When unsure between continue_trip and side_chat, choose continue_trip for short ambiguous replies \
like 'да', 'ок', 'первый', 'продолжай', place names, dates, or constraints. Judge by meaning, in \
any language.";

#[derive(Debug, Deserialize)]
struct RouterJson {
    #[serde(default)]
    route: String,
}

/// Classify one message. On any LLM/parse failure, returns [`Route::Chat`] so a
/// transient model error never blocks the user or misroutes them out of chat.
pub async fn classify(llm: &Llm, user_text: &str) -> Route {
    if is_project_or_mcp_question(user_text) {
        return Route::Chat;
    }
    let raw = match llm.complete(ROUTER_PROMPT, user_text).await {
        Ok(s) => s,
        Err(_) => return Route::Chat,
    };
    parse_route(&raw)
}

/// Classify a message while a trip flow is active.
///
/// On model failure we keep the old safe behavior for active flows and continue
/// the trip, except for explicit project/MCP implementation questions. Those
/// must never trigger route/artifact work.
pub async fn classify_active_flow(llm: &Llm, user_text: &str) -> ActiveFlowRoute {
    if is_project_or_mcp_question(user_text) {
        return ActiveFlowRoute::SideChat;
    }
    let raw = match llm.complete(ACTIVE_FLOW_ROUTER_PROMPT, user_text).await {
        Ok(s) => s,
        Err(_) => return ActiveFlowRoute::ContinueTrip,
    };
    parse_active_flow_route(&raw)
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

fn parse_active_flow_route(raw: &str) -> ActiveFlowRoute {
    let parsed: Option<RouterJson> = serde_json::from_str(&extract_json(raw)).ok();
    match parsed.map(|p| p.route.trim().to_ascii_lowercase()) {
        Some(r) if r == "side_chat" || r == "side-chat" || r == "chat" => ActiveFlowRoute::SideChat,
        Some(r) if r == "offtopic" || r == "off_topic" || r == "off-topic" => {
            ActiveFlowRoute::OffTopic
        }
        _ => ActiveFlowRoute::ContinueTrip,
    }
}

/// Deterministic guard for meta/project implementation questions. The LLM
/// router is still the main intent detector, but active trip flows must not run
/// side effects when the user asks how MCP/the bot/project works internally.
fn is_project_or_mcp_question(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let has_mcp = lower.contains("mcp") || lower.contains("мцп");
    let has_project = lower.contains("проект")
        || lower.contains("project")
        || lower.contains("код")
        || lower.contains("code")
        || lower.contains("файл")
        || lower.contains("file")
        || lower.contains("реализац")
        || lower.contains("implement")
        || lower.contains("архитектур")
        || lower.contains("architecture");
    let asks_how = lower.contains("как ")
        || lower.contains("how ")
        || lower.contains("в каких")
        || lower.contains("where")
        || lower.contains("inside")
        || lower.contains("изнутри");
    has_mcp || (has_project && asks_how)
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

    #[test]
    fn active_flow_parse_defaults_to_continue() {
        assert_eq!(
            parse_active_flow_route(r#"{"route":"continue_trip"}"#),
            ActiveFlowRoute::ContinueTrip
        );
        assert_eq!(
            parse_active_flow_route(r#"{"route":"side_chat"}"#),
            ActiveFlowRoute::SideChat
        );
        assert_eq!(
            parse_active_flow_route("not json"),
            ActiveFlowRoute::ContinueTrip
        );
    }

    #[test]
    fn project_mcp_questions_never_start_or_continue_trip() {
        let q = "как работают mcp в этом проекте изнутри? В каких файлах реализация";
        assert!(is_project_or_mcp_question(q));
    }
}
