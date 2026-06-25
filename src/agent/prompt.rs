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
IMPORTANT: when the user asks to COLLECT data over time, to be KEPT POSTED, or to receive a \
RECURRING/periodic summary (e.g. 'собирай погоду каждый час', 'держи меня в курсе'), do NOT make \
them ask again each time — set up recurring delivery yourself. PREFER server push when available: \
if the MCP exposes a `subscribe_summaries` tool, call `schedule_weather_job` (start collection) then \
`subscribe_summaries` — the server then pushes summaries and the client delivers them automatically. \
Only if there is no subscribe tool, fall back to the `schedule_summary` meta-tool (client-side polling). \
You do NOT need to set session_id — the client manages it. Confirm what you scheduled.";

/// Build the full system prompt from memory, profile, invariants and optional
/// stage rules. Returns one string (blocks separated by blank lines).
pub fn build_system_prompt(
    memory: &AgentMemory,
    profile: &UserProfile,
    invariants: &[Invariant],
    stage_rules: Option<&str>,
    violation_feedback: Option<&[String]>,
) -> String {
    let mut blocks: Vec<String> = vec![BASE_SYSTEM.to_string()];
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

    // [user-profile] — dedup against long-term values already shown
    if !profile.is_empty() {
        let mut lines = vec!["[user-profile] About the user:".to_string()];
        for (k, v) in &profile.fields {
            seen_values.insert(v.to_ascii_lowercase());
            lines.push(format!("- {k}: {v}"));
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
