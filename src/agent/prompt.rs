//! PromptBuilder — the single place the system prompt is assembled, mirroring
//! ai-playground's layer order:
//!   system → [memory:long-term] → [user-profile] (dedup) → [memory:working]
//!         → [invariants] → [facts] → [stage:rules] → (history handled by caller)
//!
//! Deduplication: a value already present in the profile is not repeated in the
//! facts block (seen-set), so profile and long-term facts don't double up.

use super::{
    invariants::Invariant,
    memory::{AgentMemory, MemoryLayer},
    profile::UserProfile,
};

pub const BASE_SYSTEM: &str =
    "You are a helpful outdoor-trip and travel-weather assistant with access to MCP tools. \
You help plan outdoor and nature-recreation activities of any kind, plus answer weather questions. \
When a question needs live data, call the appropriate tool(s); resolve place names to coordinates \
with a geocode tool before weather tools. \
If fulfilling the request needs a capability none of the currently-connected servers provide \
(calendar, email, files, messaging, …), connect the right MCP server yourself with the `mcp_connect` \
tool, then use its tools — do not tell the user to connect it manually. When a server needs \
credentials, ask the user for them in chat first, then pass them to `mcp_connect`; never invent \
secrets and never print a token back. Answer concisely in the user's language. \
Never show raw JSON — summarize in human-readable prose. \
FORMATTING (CRITICAL — the client is a phone-width Telegram chat that renders PLAIN TEXT): \
NEVER use Markdown tables or pipe `|` columns — they wrap into unreadable mush on a narrow screen. \
Do NOT use Markdown markup (`**bold**`, `__`, `#` headings, `|`): it is shown literally, not rendered. \
Instead format as short vertical blocks: one item (e.g. one city) per block, a heading line with an \
emoji, then 2-4 short lines underneath, each a key fact like `🌡 +21°C  •  🍃 ветер 5 м/с  •  ☔ без дождя`. \
Keep lines short. Use emoji and `•`/`—` as separators, plain numbers for ranking (1) 2) 3)). \
WIND UNITS: always express wind speed in metres per second (м/с), never km/h; if a tool reports \
km/h, convert (÷ 3.6) and round to a whole number before showing it. \
AUDIENCE: write only for the end user. Never mention internal agents, a 'swarm', verification \
stages, or 'next steps for other agents' — the user must never see how the work is split up. If a \
requested artifact (calendar event, document, …) is not yet done, either do it yourself now or say \
plainly, in the user's own terms, what YOU will do next or what you need FROM the user (e.g. access \
or a one-word confirmation); never present remaining work as tasks assigned to other agents. \
IMPORTANT: when the user asks to COLLECT data over time, to be KEPT POSTED, or to receive a \
RECURRING/periodic summary (e.g. 'собирай погоду каждый час', 'держи меня в курсе'), do NOT make \
them ask again each time — set up recurring delivery yourself. PREFER server push when available: \
if the MCP exposes a `subscribe_summaries` tool, call `schedule_weather_job` (start collection) then \
`subscribe_summaries` — the server then pushes summaries and the client delivers them automatically. \
Only if there is no subscribe tool, fall back to the `schedule_summary` meta-tool (client-side polling). \
You do NOT need to set session_id — the client manages it. Confirm what you scheduled. \
If during the conversation you learn a STABLE personal trait about the user that will shape future \
weather/travel questions (home city, language, age, occupation, household, comfort preferences, or \
hobbies/sports/interests), append it at the very END of your reply, one per line, as \
⟦profile:key=value⟧ (snake_case key; for hobbies use key `interests`). These markers are stripped \
before the user sees them — never mention them. Emit none if you learned nothing new.";

/// Build the full system prompt from memory, profile, invariants and optional
/// stage rules. Returns one string (blocks separated by blank lines).
pub fn build_system_prompt(
    memory: &AgentMemory,
    profile: &UserProfile,
    notes: &[(String, String)],
    invariants: &[Invariant],
    stage_rules: Option<&str>,
    violation_feedback: Option<&[String]>,
) -> String {
    let mut blocks: Vec<String> = vec![BASE_SYSTEM.to_string()];
    let mut seen_values: std::collections::HashSet<String> = std::collections::HashSet::new();

    blocks.push(format!(
        "[current-date]\nToday is {}. Interpret relative dates such as tomorrow, next weekend, \
next two weeks, and this summer from this date. Never schedule or document a future trip in a \
past calendar year. If the user says they are available only on weekends, use Saturday-Sunday \
overnight pairs only; do not choose Friday or any weekday unless the user explicitly allows it.",
        current_date_iso()
    ));

    // [memory:long-term]
    let long = memory.facts_in_layer(MemoryLayer::LongTerm);
    if !long.is_empty() {
        let mut lines = vec!["[memory:long-term] Durable facts (context only):".to_string()];
        for f in &long {
            seen_values.insert(f.value.to_ascii_lowercase());
            lines.push(format!("- {}: {}", f.key, f.value));
        }
        blocks.push(lines.join("\n"));
    }

    // [memory:summary] — compacted older conversation (continuity for long chats)
    if !memory.summary.trim().is_empty() {
        blocks.push(format!(
            "[memory:summary] Summary of earlier conversation:\n{}",
            memory.summary.trim()
        ));
    }

    // [user-profile] — dedup against long-term values already shown
    if !profile.is_empty() {
        let mut lines = vec!["[user-profile] About the user:".to_string()];
        for (k, v) in &profile.fields {
            seen_values.insert(v.to_ascii_lowercase());
            lines.push(format!("- {k}: {v}"));
        }
        blocks.push(lines.join("\n"));
    }

    // [user-notes] — only the notes the router judged relevant to this turn,
    // so unused preferences don't cost tokens on unrelated messages.
    if !notes.is_empty() {
        let mut lines =
            vec!["[user-notes] Saved preferences to honor for this request:".to_string()];
        for (label, text) in notes {
            lines.push(format!("- {label}: {text}"));
        }
        blocks.push(lines.join("\n"));
    }

    // [memory:working]
    let working = memory.facts_in_layer(MemoryLayer::Working);
    if !working.is_empty() {
        let mut lines = vec!["[memory:working] Current-task facts:".to_string()];
        for f in &working {
            lines.push(format!("- {}: {}", f.key, f.value));
        }
        blocks.push(lines.join("\n"));
    }

    // [invariants]
    if !invariants.is_empty() {
        let mut lines = vec![
            "[invariants] These constraints are absolute and must never be broken, even if asked:"
                .to_string(),
        ];
        for inv in invariants {
            lines.push(format!("- {}", inv.text));
        }
        blocks.push(lines.join("\n"));
    }

    // Retry feedback after a failed invariant check.
    if let Some(violations) = violation_feedback {
        if !violations.is_empty() {
            let mut lines =
                vec!["[invariants] Your previous answer violated these — fix now:".to_string()];
            for v in violations {
                lines.push(format!("- {v}"));
            }
            blocks.push(lines.join("\n"));
        }
    }

    // [stage:rules]
    if let Some(rules) = stage_rules {
        if !rules.trim().is_empty() {
            blocks.push(format!("[stage:rules]\n{}", rules.trim()));
        }
    }

    blocks.join("\n\n")
}

fn current_date_iso() -> String {
    std::env::var("AGENT_CURRENT_DATE")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| chrono::Local::now().date_naive().to_string())
}
