//! Stateful multi-agent trip-planning flow (the "swarm").
//!
//! Unlike the old one-shot weather pipeline, this flow SUSPENDS across user
//! turns: it interrogates the user first (Clarify), accumulating a `TripBrief`
//! that is persisted in the chat session. Only when the brief is complete does
//! it run the execution pipeline — Planning → Routing → Camp → Schedule → Doc —
//! where **each stage receives the prior stages' outputs** (real subagent
//! hand-off). The orchestrator (this code) owns transitions; each stage is an
//! LLM call, the execution stages using the connected MCP tools.
//!
//! State machine (only Clarify waits for the user):
//!   Clarify ⇄ Clarify … → Planning → Routing → Camp → Schedule → Doc → Done

use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{llm::Llm, state::BotState};

use super::session::ChatSession;

/// Hard cap on clarify rounds — after this we plan with whatever we have, so the
/// bot never interrogates forever.
const MAX_CLARIFY_ROUNDS: u8 = 3;

/// Per-stage output is clipped to this many bytes when handed to the next stage,
/// keeping the cumulative prompt bounded.
const HANDOFF_CLIP: usize = 700;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    Clarify,
    Planning,
    Routing,
    Camp,
    Schedule,
    Doc,
    Done,
}

/// Open-schema trip brief: the Clarify agent fills arbitrary keys
/// (area, date_window, nights, party, priorities, constraints, transport, …).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TripBrief {
    #[serde(default)]
    pub fields: BTreeMap<String, String>,
}

impl TripBrief {
    /// Merge non-empty values from a freshly-extracted map (new wins).
    fn merge(&mut self, extracted: BTreeMap<String, String>) {
        for (k, v) in extracted {
            let v = v.trim();
            if !v.is_empty() {
                self.fields.insert(k.trim().to_string(), v.to_string());
            }
        }
    }

    /// Render as `key: value` lines for downstream stage prompts.
    pub fn render(&self) -> String {
        if self.fields.is_empty() {
            return "(empty)".to_string();
        }
        self.fields
            .iter()
            .map(|(k, v)| format!("- {k}: {v}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// One stage's artifact, passed forward to later stages and shown as a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRecord {
    pub stage: String,
    pub output: String,
}

/// Persisted flow state, carried in the chat session across turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripFlowState {
    pub stage: Stage,
    #[serde(default)]
    pub brief: TripBrief,
    #[serde(default)]
    pub records: Vec<StageRecord>,
    #[serde(default)]
    pub clarify_rounds: u8,
}

impl TripFlowState {
    pub fn start() -> Self {
        Self {
            stage: Stage::Clarify,
            brief: TripBrief::default(),
            records: Vec::new(),
            clarify_rounds: 0,
        }
    }
}

/// Result of advancing the flow by one user turn.
pub struct FlowTurn {
    /// Message to send the user (clarifying questions, or the final plan).
    pub reply: String,
    /// Compact pipeline trace (one line per completed stage); empty while still
    /// clarifying.
    pub trace: Vec<String>,
    /// True once the flow reached Done (caller should clear `session.trip`).
    pub done: bool,
}

/// Keyword trigger for auto-detecting a trip-planning request in normal chat.
/// Token-frugal: a substring hit starts the flow, no extra LLM intent call.
const TRIP_TRIGGERS: &[&str] = &[
    // RU outdoor / overnight
    "поход",
    "сплав",
    "байдар",
    "каяк",
    "каноэ",
    "рафтинг",
    "кемпинг",
    "палатк",
    "ночёвк",
    "ночевк",
    "ночлег",
    "стоянк",
    "привал",
    "костёр",
    "костер",
    "шашлык",
    "барбекю",
    "сплан",
    "вылазк",
    "турпоход",
    // EN
    "kayak",
    "canoe",
    "rafting",
    "hike",
    "hiking",
    "camp",
    "camping",
    "bbq",
    "getaway",
    "paddle",
];

/// Cheap keyword pre-filter: does this message look like a trip-planning request?
pub fn looks_like_trip(text: &str) -> bool {
    let low = text.to_lowercase();
    TRIP_TRIGGERS.iter().any(|t| low.contains(t))
}

/// User phrases that force planning to start even if the brief is thin.
const GO_PHRASES: &[&str] = &[
    "поехали",
    "планируй",
    "хватит",
    "достаточно",
    "погнали",
    "давай уже",
    "just plan",
    "go ahead",
    "let's go",
];

fn user_forces_go(text: &str) -> bool {
    let low = text.to_lowercase();
    GO_PHRASES.iter().any(|p| low.contains(p))
}

// ---------------------------------------------------------------------------
// Clarify agent
// ---------------------------------------------------------------------------

const CLARIFY_PROMPT: &str = "You are the CLARIFY agent of an outdoor-trip planner \
(hikes, kayak/canoe trips, camping, weekend getaways). You receive the brief gathered \
so far and the user's newest message. \
MERGE any new facts into the brief. Critical slots to plan a trip: \
`area` (start region / where), `date_window` (when, even a rough range), and for overnight \
trips `nights`. Useful extras: `party` (size + how prepared/fit), `priorities` (what they \
care about, e.g. relaxing/BBQ over distance), `constraints` (e.g. campsite far from \
civilization, water nearby), `transport`, `gear`. \
Decide `ready=true` only when at least `area` and `date_window` are known. \
If not ready, ask up to 3 SHORT friendly questions for the MOST important missing facts only — \
never re-ask something already in the brief. \
Reply in the user's language. \
Return ONLY JSON: {\"brief\":{\"key\":\"value\",...},\"ready\":bool,\"questions\":[\"...\"],\
\"recap\":\"one short line summarizing the trip so far\"}.";

#[derive(Debug, Deserialize)]
struct ClarifyOut {
    #[serde(default)]
    brief: BTreeMap<String, String>,
    #[serde(default)]
    ready: bool,
    #[serde(default)]
    questions: Vec<String>,
    #[serde(default)]
    recap: String,
}

/// Parse the clarify agent's JSON (lenient: tolerates fences / surrounding prose).
fn parse_clarify(raw: &str) -> ClarifyOut {
    serde_json::from_str(&extract_json(raw)).unwrap_or(ClarifyOut {
        brief: BTreeMap::new(),
        ready: false,
        questions: vec![],
        recap: String::new(),
    })
}

/// Format the clarify reply (recap + numbered questions) for the chat.
fn render_clarify_reply(recap: &str, questions: &[String]) -> String {
    let mut out = String::new();
    if !recap.trim().is_empty() {
        out.push_str(recap.trim());
        out.push_str("\n\n");
    }
    if questions.is_empty() {
        out.push_str("Расскажите чуть подробнее о поездке?");
    } else {
        for (i, q) in questions.iter().enumerate() {
            out.push_str(&format!("{}) {}\n", i + 1, q.trim()));
        }
    }
    out.trim_end().to_string()
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Advance the flow by one user turn. The caller guarantees `session.trip` is
/// `Some`; this function mutates it and returns what to tell the user.
pub async fn advance(
    llm: &Llm,
    state: &BotState,
    session: &mut ChatSession,
    user_text: &str,
) -> Result<FlowTurn> {
    // Clone the parts we need so we can borrow the session immutably for the
    // execution pipeline (which reads memory/profile/invariants).
    let stage = session
        .trip
        .as_ref()
        .map(|t| t.stage.clone())
        .unwrap_or(Stage::Clarify);

    if stage == Stage::Clarify {
        // ---- Clarify: merge answers, decide whether to plan ----
        let brief_so_far = session
            .trip
            .as_ref()
            .map(|t| t.brief.render())
            .unwrap_or_default();
        let input = format!("Brief so far:\n{brief_so_far}\n\nUser message:\n{user_text}");
        let raw = llm
            .complete(CLARIFY_PROMPT, &input)
            .await
            .unwrap_or_default();
        let parsed = parse_clarify(&raw);

        let trip = session.trip.as_mut().expect("caller ensured Some");
        trip.brief.merge(parsed.brief);
        trip.clarify_rounds = trip.clarify_rounds.saturating_add(1);

        let forced = user_forces_go(user_text) || trip.clarify_rounds >= MAX_CLARIFY_ROUNDS;
        if !parsed.ready && !forced {
            // Stay in Clarify; ask the user for more.
            return Ok(FlowTurn {
                reply: render_clarify_reply(&parsed.recap, &parsed.questions),
                trace: vec![],
                done: false,
            });
        }
        // Brief is good enough → fall through to the execution pipeline.
        trip.stage = Stage::Planning;
    }

    run_pipeline(llm, state, session).await
}

/// Run Planning → Routing → Camp → Schedule → Doc sequentially, threading each
/// stage's output into the next, then compose the final user-facing plan.
async fn run_pipeline(llm: &Llm, state: &BotState, session: &mut ChatSession) -> Result<FlowTurn> {
    let brief = session
        .trip
        .as_ref()
        .map(|t| t.brief.clone())
        .unwrap_or_default();
    let brief_text = brief.render();
    let mut records: Vec<StageRecord> = Vec::new();

    let stages: &[(Stage, &str, &str)] = &[
        (
            Stage::Planning,
            "PLANNING",
            "Pick the single BEST day within the user's date window and the best concrete \
             river/area near their start region for this kind of trip. Use the weather tools \
             (geocode then forecast) to compare candidate days — favor low rain, light wind, \
             comfortable temperature. State the chosen DATE with a one-line weather rationale \
             (include numbers) and name the specific waterway/area.",
        ),
        (
            Stage::Routing,
            "ROUTING",
            "Design the actual on-water route for the chosen day and place. Give a concrete \
             put-in and take-out, 2-4 named stops with rough distances/times, and a relaxed pace \
             that matches how prepared the party is and their priorities (e.g. more rest/BBQ than \
             paddling). Keep total distance modest for an unprepared group.",
        ),
        (
            Stage::Camp,
            "CAMP",
            "Choose ONE overnight campsite on the route that satisfies the user's constraints — \
             notably any minimum distance from turbazy/villages/roads and maximum distance to \
             water. If a maps/geo tool is connected, use it to verify; otherwise propose a \
             plausible spot and clearly flag that the distances must be verified on the map. \
             State approximate coordinates or a clear landmark, distance to water, and distance \
             to the nearest civilization.",
        ),
        (
            Stage::Schedule,
            "SCHEDULE",
            "Create a calendar event for this trip (start = chosen date, overnight duration). \
             If no connected server can create calendar events, connect a calendar MCP yourself \
             with `mcp_connect` (ask the user for any credentials in chat first). Confirm the \
             event you created (title, date, time).",
        ),
        (
            Stage::Doc,
            "DOC",
            "Produce a shareable Google Doc containing the full plan (date, route with stops, \
             campsite, gear/BBQ notes) so the user can share it with friends. If no connected \
             server can create docs, connect a Google Docs MCP yourself with `mcp_connect` \
             (ask for credentials in chat first). Return the share link.",
        ),
    ];

    for (stage, name, instruction) in stages {
        let system = stage_system(session, name);
        let query = format!(
            "[trip-brief]\n{brief_text}\n\n[prior-stages]\n{}\n\n[your-task as the {name} agent]\n{instruction}",
            render_records(&records),
        );
        let output = match llm.answer_with_system(state, &system, &query).await {
            Ok(o) if !o.trim().is_empty() => o,
            Ok(_) => "(no output)".to_string(),
            Err(e) => format!("(stage failed: {e})"),
        };
        records.push(StageRecord {
            stage: name.to_string(),
            output,
        });
        // Persist progress after each stage so a crash mid-pipeline isn't lost.
        if let Some(trip) = session.trip.as_mut() {
            trip.stage = stage.clone();
            trip.records = records.clone();
        }
    }

    // ---- Compose the final user-facing plan from all stage artifacts ----
    let compose_input = format!(
        "[trip-brief]\n{brief_text}\n\n[stage-outputs]\n{}",
        render_records(&records),
    );
    let final_answer = llm
        .complete(COMPOSE_PROMPT, &compose_input)
        .await
        .unwrap_or_else(|_| fallback_compose(&records));

    let trace: Vec<String> = records
        .iter()
        .map(|r| format!("• {}: {}", r.stage, clip(&r.output, 90)))
        .collect();

    // Flow complete — caller clears session.trip.
    if let Some(trip) = session.trip.as_mut() {
        trip.stage = Stage::Done;
    }

    Ok(FlowTurn {
        reply: final_answer,
        trace,
        done: true,
    })
}

const COMPOSE_PROMPT: &str = "You assemble the FINAL trip plan a user will share with friends, \
from the stage outputs of a planning swarm. Write it in the user's language as a clean, \
phone-friendly PLAIN-TEXT message (no Markdown tables, no `|`, no `**`). Use short vertical \
blocks with emoji headings: chosen day + weather, the route with concrete stops, the overnight \
campsite (with distances), gear/BBQ notes, and — if created — the calendar event and the \
shareable doc link. Be concrete; do not invent a doc link or coordinates that the stages did \
not produce. Keep it tight.";

/// Build a stage's system prompt: the layered base prompt + a stage role line.
fn stage_system(session: &ChatSession, role: &str) -> String {
    let invariants = session.effective_invariants();
    super::prompt::build_system_prompt(
        &session.memory,
        &session.profile,
        &[],
        &invariants,
        Some(&format!(
            "You are the {role} agent in a multi-stage trip-planning swarm. Do ONLY your stage's \
             task, building on the prior stages' outputs. Use the connected MCP tools when you \
             need live data or to act."
        )),
        None,
    )
}

fn render_records(records: &[StageRecord]) -> String {
    if records.is_empty() {
        return "(none yet)".to_string();
    }
    records
        .iter()
        .map(|r| format!("[{}]\n{}", r.stage, clip(&r.output, HANDOFF_CLIP)))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// LLM-free fallback if the compose call fails: concatenate stage outputs.
fn fallback_compose(records: &[StageRecord]) -> String {
    records
        .iter()
        .map(|r| format!("{}:\n{}", r.stage, r.output))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Extract the first JSON object from a possibly fenced LLM reply.
fn extract_json(s: &str) -> String {
    let s = s.trim();
    if let (Some(start), Some(end)) = (s.find('{'), s.rfind('}')) {
        if end >= start {
            return s[start..=end].to_string();
        }
    }
    "{}".to_string()
}

/// Clip a string to `max` bytes on a char boundary, adding an ellipsis.
fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_trip_intent() {
        assert!(looks_like_trip("хочу в поход на байдарках на выходных"));
        assert!(looks_like_trip("plan a kayak trip with camping"));
        assert!(!looks_like_trip("какая погода в Москве завтра"));
    }

    #[test]
    fn go_phrase_forces_plan() {
        assert!(user_forces_go("да поехали уже планируй"));
        assert!(!user_forces_go("а можно поближе к воде?"));
    }

    #[test]
    fn brief_merge_keeps_nonempty_new_wins() {
        let mut b = TripBrief::default();
        let mut e = BTreeMap::new();
        e.insert("area".to_string(), "Карелия".to_string());
        e.insert("nights".to_string(), "  ".to_string()); // blank ignored
        b.merge(e);
        assert_eq!(b.fields.get("area").unwrap(), "Карелия");
        assert!(!b.fields.contains_key("nights"));

        let mut e2 = BTreeMap::new();
        e2.insert("area".to_string(), "Мещёра".to_string());
        b.merge(e2);
        assert_eq!(b.fields.get("area").unwrap(), "Мещёра");
    }

    #[test]
    fn parse_clarify_reads_fields_and_ready() {
        let out = parse_clarify(
            r#"prose ```json
            {"brief":{"area":"Мещёра","date_window":"next 2 weeks"},"ready":true,"questions":[],"recap":"kayak + camp"}
            ``` trailing"#,
        );
        assert!(out.ready);
        assert_eq!(out.brief.get("area").unwrap(), "Мещёра");
        assert_eq!(out.recap, "kayak + camp");
    }

    #[test]
    fn clip_respects_char_boundary() {
        let s = "Карелия";
        let c = clip(s, 5);
        assert!(c.ends_with('…'));
    }
}
