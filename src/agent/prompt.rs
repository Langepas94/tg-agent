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
    "You are a helpful travel-weather assistant with access to MCP tools. \
When a question needs live data, call the appropriate tool(s); resolve place names to coordinates \
with a geocode tool before weather tools. Answer concisely in the user's language. \
Never show raw JSON — summarize in human-readable prose. \
FORMATTING (CRITICAL — the client is a phone-width Telegram chat that renders PLAIN TEXT): \
NEVER use Markdown tables or pipe `|` columns — they wrap into unreadable mush on a narrow screen. \
Do NOT use Markdown markup (`**bold**`, `__`, `#` headings, `|`): it is shown literally, not rendered. \
Instead format as short vertical blocks: one item (e.g. one city) per block, a heading line with an \
emoji, then 2-4 short lines underneath, each a key fact like `🌡 +21°C  •  🍃 ветер 20 км/ч  •  ☔ без дождя`. \
Keep lines short. Use emoji and `•`/`—` as separators, plain numbers for ranking (1) 2) 3)). \
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

/// Always-on capability block: how to run a multi-server "plan an outdoor trip
/// and let me share it" flow. Describes the step ORDER and which server does
/// what; the user states a goal in plain language and the agent orchestrates
/// the tools itself. File formats are NOT hardcoded here — they come from the
/// user's saved notes (conditionally injected) with a sensible default.
pub const TRIP_FLOW: &str =
    "[capability:trip-planning] When the user wants to plan an outdoor activity/trip and pick the \
best option by weather (e.g. 'where should we kayak/hike this weekend', 'plan it and let me share \
with friends'), drive the whole flow yourself — the user will NOT name tools, fields, dates or file \
formats; infer them. Steps, in this order: \
(1) Resolve each candidate place to coordinates and compare weather over the relevant dates using \
the weather server; pick the best place and day for the activity (consider temperature, wind, rain, \
daylight). State your pick and why, briefly. \
(2) Write the trip plan as a file via the filesystem server: place, date, weather summary, daylight \
window, a short gear checklist. Use the file format/naming from the user's saved preferences if one \
is given (in the saved-preferences block, when present); otherwise default to Markdown. The file is \
a saved artifact — \
the plain-text rule above applies ONLY to chat messages, not to files you write. \
(3) Offer a reminder: create a calendar event for the chosen day and place via the calendar server, \
and give the user the event link. \
(4) Produce a shareable invite as a calendar file (`.ics` by default, or the user's preferred format) \
so they can forward it to friends, and deliver it as a document via the messaging server. \
Adapt to whichever servers are actually connected; if one (filesystem/calendar/messaging) is missing, \
do the steps you can and tell the user which server to connect for the rest. Keep the chat reply a \
short plain-text summary; the files are the shareable output.";

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
    let mut blocks: Vec<String> = vec![BASE_SYSTEM.to_string(), TRIP_FLOW.to_string()];
    let mut seen_values: std::collections::HashSet<String> = std::collections::HashSet::new();

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
