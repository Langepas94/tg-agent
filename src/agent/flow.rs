//! Stateful multi-agent trip-planning flow (the "swarm").
//!
//! Each stage is its own agent. An **LLM ORCHESTRATOR agent** decides every
//! transition — it is not a hardcoded sequential FSM: it reads the brief, which
//! stages already have output, and the user's latest message, and returns the
//! next stage + whether to run it or ask the user. This lets the user **step
//! back** at any point ("change the date / river / campsite"): the orchestrator
//! routes to that earlier stage and its (and all later stages') outputs are
//! recomputed.
//!
//! The flow SUSPENDS across user turns: Clarify interrogates first, building a
//! `TripBrief` persisted in the chat session; each execution stage
//! (Planning → Routing → Camp → Schedule → Doc) receives the prior stages'
//! outputs (real subagent hand-off) and uses the connected MCP tools. Code only
//! executes the chosen stage and guards against loops; the routing decisions are
//! the orchestrator agent's.
//!
//! Stages (the orchestrator may move forward, stay, or step back):
//!   Clarify → Planning → Routing → Camp → Schedule → Doc → Done

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

    /// Minimum to start planning: a start area/region and a date window. Used
    /// only by the orchestrator's deterministic fallback (key-name heuristic).
    pub fn has_minimum(&self) -> bool {
        let has = |needles: &[&str]| {
            self.fields
                .iter()
                .any(|(k, v)| !v.trim().is_empty() && needles.iter().any(|n| k.contains(n)))
        };
        has(&[
            "area",
            "region",
            "where",
            "start",
            "location",
            "место",
            "регион",
            "старт",
        ]) && has(&["date", "when", "window", "дат", "когда", "срок"])
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
// Stage ordering + execution instructions
// ---------------------------------------------------------------------------

impl Stage {
    /// Linear position; used to compare "earlier/later" for back-steps.
    pub fn order(&self) -> u8 {
        match self {
            Stage::Clarify => 0,
            Stage::Planning => 1,
            Stage::Routing => 2,
            Stage::Camp => 3,
            Stage::Schedule => 4,
            Stage::Doc => 5,
            Stage::Done => 6,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Stage::Clarify => "Clarify",
            Stage::Planning => "Planning",
            Stage::Routing => "Routing",
            Stage::Camp => "Camp",
            Stage::Schedule => "Schedule",
            Stage::Doc => "Doc",
            Stage::Done => "Done",
        }
    }

    /// Parse a stage name (case-insensitive) from the orchestrator's JSON.
    pub fn parse(s: &str) -> Option<Stage> {
        match s.trim().to_ascii_lowercase().as_str() {
            "clarify" => Some(Stage::Clarify),
            "planning" | "plan" => Some(Stage::Planning),
            "routing" | "route" => Some(Stage::Routing),
            "camp" | "campsite" => Some(Stage::Camp),
            "schedule" | "calendar" => Some(Stage::Schedule),
            "doc" | "document" => Some(Stage::Doc),
            "done" | "finish" => Some(Stage::Done),
            _ => None,
        }
    }

    /// Execution stages (have their own worker agent + MCP tools).
    pub fn is_exec(&self) -> bool {
        matches!(
            self,
            Stage::Planning | Stage::Routing | Stage::Camp | Stage::Schedule | Stage::Doc
        )
    }

    /// The worker-agent task for an execution stage (empty for Clarify/Done).
    fn instruction(&self) -> &'static str {
        match self {
            Stage::Planning => {
                "Pick the single BEST day within the user's date window and the best concrete \
                 river/area near their start region for this kind of trip. Use the weather tools \
                 (geocode then forecast) to compare candidate days — favor low rain, light wind, \
                 comfortable temperature. State the chosen DATE with a one-line weather rationale \
                 (include numbers) and name the specific waterway/area."
            }
            Stage::Routing => {
                "Design the actual on-water route for the chosen day and place. Give a concrete \
                 put-in and take-out, 2-4 named stops with rough distances/times, and a relaxed \
                 pace that matches how prepared the party is and their priorities (e.g. more \
                 rest/BBQ than paddling). Keep total distance modest for an unprepared group."
            }
            Stage::Camp => {
                "Choose ONE overnight campsite on the route that satisfies the user's constraints \
                 — notably any minimum distance from turbazy/villages/roads and maximum distance \
                 to water. If a maps/geo tool is connected, use it to verify; otherwise propose a \
                 plausible spot and clearly flag the distances must be verified on the map. State \
                 approximate coordinates or a clear landmark, distance to water, and distance to \
                 the nearest civilization."
            }
            Stage::Schedule => {
                "Create a calendar event for this trip (start = chosen date, overnight duration). \
                 If a connected tool can create calendar events, use it and confirm the event \
                 (title, date, time). If NONE can: do NOT loop or keep retrying — make at most ONE \
                 `mcp_connect` attempt for a calendar MCP, and if that is not possible right now, \
                 output a single short line that a calendar MCP must be connected to create the \
                 event, then STOP."
            }
            Stage::Doc => {
                "Produce the shareable plan as a Google Doc. If a connected tool can create docs, \
                 use it and return the share link. If NONE can: do NOT loop or keep retrying — make \
                 at most ONE `mcp_connect` attempt for a docs MCP, and if that is not possible right \
                 now, output the full plan as plain text the user can copy, plus one short line that \
                 a docs MCP must be connected for an auto-shared link, then STOP."
            }
            Stage::Clarify | Stage::Done => "",
        }
    }
}

// ---------------------------------------------------------------------------
// Orchestrator agent — an LLM decides every stage transition (incl. back-steps)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Execute the chosen stage's worker agent now.
    Run,
    /// Stop and send `message` to the user; wait for their reply.
    Ask,
}

/// The orchestrator's decision for one step.
#[derive(Debug, Clone)]
pub struct Decision {
    pub next: Stage,
    pub mode: Mode,
    pub message: String,
}

const ORCH_PROMPT: &str = "You are the ORCHESTRATOR of a trip-planning swarm. The stages, in \
order, are: Clarify, Planning, Routing, Camp, Schedule, Doc, Done. Each stage is run by its own \
worker agent; YOU decide which stage runs next. \
You are given: the trip brief, which stages already have output, and the user's latest message \
(which may be empty when you are auto-advancing within a turn). \
Decide the SINGLE next step and return ONLY JSON \
{\"next_stage\":\"<stage>\",\"mode\":\"run|ask\",\"message\":\"<text if ask>\",\"reason\":\"<short>\"}. \
Rules: \
- Use Clarify (mode=run) only while the brief lacks at least an area/start and a date window, AND \
only when the user just sent a message; never pick Clarify when the user's message is empty. \
- Otherwise advance one stage at a time in order, mode=run, picking the earliest stage that has \
no output yet. \
- If the user's latest message asks to CHANGE or REDO an earlier decision (different date, \
different river/route, move the campsite, change the event), pick that EARLIER stage (a step \
back), mode=run — its and all later stages' outputs will be recomputed. \
- Use mode=ask only when you genuinely need the user to decide something you cannot; put a short \
question in message. \
- When every stage through Doc has output and the user is not asking for changes, pick \
next_stage=Done, mode=run.";

#[derive(Debug, Deserialize)]
struct DecisionJson {
    #[serde(default)]
    next_stage: String,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    message: String,
}

/// Parse the orchestrator JSON, falling back to a deterministic decision if the
/// model returns junk (so the flow never stalls).
fn parse_decision(
    raw: &str,
    brief: &TripBrief,
    records: &[StageRecord],
    user_empty: bool,
) -> Decision {
    let parsed: Option<DecisionJson> = serde_json::from_str(&extract_json(raw)).ok();
    if let Some(d) = parsed {
        if let Some(stage) = Stage::parse(&d.next_stage) {
            let mode = if d.mode.trim().eq_ignore_ascii_case("ask") {
                Mode::Ask
            } else {
                Mode::Run
            };
            // Guard: never Clarify on an empty (auto-advance) message.
            if !(stage == Stage::Clarify && user_empty) {
                return Decision {
                    next: stage,
                    mode,
                    message: d.message,
                };
            }
        }
    }
    fallback_decision(brief, records, user_empty)
}

/// Deterministic safety net: clarify if the brief is thin and the user spoke,
/// else run the earliest stage without output, else finish.
fn fallback_decision(brief: &TripBrief, records: &[StageRecord], user_empty: bool) -> Decision {
    if !brief.has_minimum() && !user_empty {
        return Decision {
            next: Stage::Clarify,
            mode: Mode::Run,
            message: String::new(),
        };
    }
    Decision {
        next: next_exec_after(records),
        mode: Mode::Run,
        message: String::new(),
    }
}

/// Earliest execution stage that has no record yet; `Done` when all are present.
fn next_exec_after(records: &[StageRecord]) -> Stage {
    for stage in [
        Stage::Planning,
        Stage::Routing,
        Stage::Camp,
        Stage::Schedule,
        Stage::Doc,
    ] {
        if record_index(records, &stage).is_none() {
            return stage;
        }
    }
    Stage::Done
}

// ---------------------------------------------------------------------------
// Record helpers (replace-by-stage + downstream invalidation for back-steps)
// ---------------------------------------------------------------------------

fn record_index(records: &[StageRecord], stage: &Stage) -> Option<usize> {
    records.iter().position(|r| r.stage == stage.name())
}

/// Insert or replace a stage's output, keeping records ordered by stage order.
fn set_record(records: &mut Vec<StageRecord>, stage: &Stage, output: String) {
    if let Some(i) = record_index(records, stage) {
        records[i].output = output;
    } else {
        records.push(StageRecord {
            stage: stage.name().to_string(),
            output,
        });
        records.sort_by_key(|r| Stage::parse(&r.stage).map(|s| s.order()).unwrap_or(255));
    }
}

/// Drop the output of `stage` and every later stage — used when the user steps
/// back so stale downstream artifacts don't leak into the recomputed plan.
fn drop_from(records: &mut Vec<StageRecord>, stage: &Stage) {
    records.retain(|r| match Stage::parse(&r.stage) {
        Some(s) => s.order() < stage.order(),
        None => false,
    });
}

// ---------------------------------------------------------------------------
// Orchestrated turn
// ---------------------------------------------------------------------------

/// Maximum orchestrator steps per user turn (auto-advance bound), so one turn
/// can walk Planning→…→Doc→Done but never loops forever.
const MAX_ORCH_STEPS: usize = 12;

/// Advance the flow by one user turn. The caller guarantees `session.trip` is
/// `Some`. The ORCHESTRATOR agent decides every transition; the user can step
/// back at any point by asking to change an earlier decision.
pub async fn advance(
    llm: &Llm,
    state: &BotState,
    session: &mut ChatSession,
    user_text: &str,
) -> Result<FlowTurn> {
    let mut brief = session
        .trip
        .as_ref()
        .map(|t| t.brief.clone())
        .unwrap_or_default();
    let mut records = session
        .trip
        .as_ref()
        .map(|t| t.records.clone())
        .unwrap_or_default();
    let mut trace: Vec<String> = Vec::new();
    let forced = user_forces_go(user_text);
    // First contact with a fresh flow: ALWAYS clarify first (never let the
    // orchestrator dive into the 5-stage pipeline on a raw multi-part request).
    // Guarantees a fast first reply that asks questions, and avoids the old
    // single-shot "burn 12 tool calls then give up" failure on a megaprompt.
    let first_contact =
        records.is_empty() && session.trip.as_ref().map(|t| t.clarify_rounds).unwrap_or(0) == 0;

    for step in 0..MAX_ORCH_STEPS {
        let user_empty = step > 0; // only the first step carries the user's message
        let umsg = if user_empty { "" } else { user_text };

        // ---- Orchestrator agent picks the next transition ----
        let decision = if step == 0 && first_contact {
            // Skip the orchestrator call entirely on turn 1: clarify is mandatory.
            Decision {
                next: Stage::Clarify,
                mode: Mode::Run,
                message: String::new(),
            }
        } else {
            let orch_input = format!(
                "[trip-brief]\n{}\n\n[completed-stages]\n{}\n\n[user-message]\n{}{}",
                brief.render(),
                completed_list(&records),
                if umsg.trim().is_empty() {
                    "(none)"
                } else {
                    umsg
                },
                if forced && !user_empty {
                    "\n[hint] The user signaled they want to proceed with what's known."
                } else {
                    ""
                },
            );
            let raw = llm
                .complete(ORCH_PROMPT, &orch_input)
                .await
                .unwrap_or_default();
            parse_decision(&raw, &brief, &records, user_empty)
        };

        match decision.next {
            // ---- Clarify: dedicated agent extracts slots / asks questions ----
            Stage::Clarify => {
                let input = format!("Brief so far:\n{}\n\nUser message:\n{umsg}", brief.render());
                let parsed = parse_clarify(
                    &llm.complete(CLARIFY_PROMPT, &input)
                        .await
                        .unwrap_or_default(),
                );
                brief.merge(parsed.brief);
                let rounds = session
                    .trip
                    .as_mut()
                    .map(|t| {
                        t.clarify_rounds = t.clarify_rounds.saturating_add(1);
                        t.brief = brief.clone();
                        t.stage = Stage::Clarify;
                        t.clarify_rounds
                    })
                    .unwrap_or(0);
                let enough = parsed.ready || forced || rounds >= MAX_CLARIFY_ROUNDS;
                if !enough {
                    return Ok(FlowTurn {
                        reply: render_clarify_reply(&parsed.recap, &parsed.questions),
                        trace,
                        done: false,
                    });
                }
                // Brief good enough → let the orchestrator advance next step.
                continue;
            }

            // ---- Done: compose the shareable plan and finish ----
            Stage::Done => {
                let final_answer = compose_final(llm, &brief, &records).await;
                if let Some(t) = session.trip.as_mut() {
                    t.stage = Stage::Done;
                    t.records = records.clone();
                }
                return Ok(FlowTurn {
                    reply: final_answer,
                    trace,
                    done: true,
                });
            }

            // ---- Execution stage ----
            ref stage if stage.is_exec() => {
                if decision.mode == Mode::Ask {
                    if let Some(t) = session.trip.as_mut() {
                        t.stage = stage.clone();
                        t.brief = brief.clone();
                        t.records = records.clone();
                    }
                    return Ok(FlowTurn {
                        reply: if decision.message.trim().is_empty() {
                            "Уточните, пожалуйста?".into()
                        } else {
                            decision.message
                        },
                        trace,
                        done: false,
                    });
                }
                // Back-step: redoing a stage invalidates it and everything after.
                drop_from(&mut records, stage);
                let output = run_exec_stage(llm, state, session, &brief, &records, stage).await;
                set_record(&mut records, stage, output.clone());
                trace.push(format!("• {}: {}", stage.name(), clip(&output, 90)));
                if let Some(t) = session.trip.as_mut() {
                    t.stage = stage.clone();
                    t.records = records.clone();
                }
                continue;
            }

            _ => continue,
        }
    }

    // Safety: hit the step cap — compose whatever we have.
    let final_answer = compose_final(llm, &brief, &records).await;
    if let Some(t) = session.trip.as_mut() {
        t.stage = Stage::Done;
        t.records = records.clone();
    }
    Ok(FlowTurn {
        reply: final_answer,
        trace,
        done: true,
    })
}

/// Run one execution stage's worker agent (uses the connected MCP tools).
async fn run_exec_stage(
    llm: &Llm,
    state: &BotState,
    session: &ChatSession,
    brief: &TripBrief,
    records: &[StageRecord],
    stage: &Stage,
) -> String {
    let system = stage_system(session, stage.name());
    let query = format!(
        "[trip-brief]\n{}\n\n[prior-stages]\n{}\n\n[your-task as the {} agent]\n{}",
        brief.render(),
        render_records(records),
        stage.name(),
        stage.instruction(),
    );
    match llm.answer_with_system(state, &system, &query).await {
        Ok(o) if !o.trim().is_empty() => o,
        Ok(_) => "(no output)".to_string(),
        Err(e) => format!("(stage failed: {e})"),
    }
}

/// Compose the final user-facing plan from all stage artifacts.
async fn compose_final(llm: &Llm, brief: &TripBrief, records: &[StageRecord]) -> String {
    let compose_input = format!(
        "[trip-brief]\n{}\n\n[stage-outputs]\n{}",
        brief.render(),
        render_records(records),
    );
    llm.complete(COMPOSE_PROMPT, &compose_input)
        .await
        .unwrap_or_else(|_| fallback_compose(records))
}

/// One-line-per-completed-stage listing for the orchestrator prompt.
fn completed_list(records: &[StageRecord]) -> String {
    if records.is_empty() {
        return "(none)".to_string();
    }
    records
        .iter()
        .map(|r| format!("- {}: {}", r.stage, clip(&r.output, 120)))
        .collect::<Vec<_>>()
        .join("\n")
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

    // ---- Stage ordering + parsing ----

    #[test]
    fn stage_order_is_linear_and_parse_roundtrips() {
        let seq = [
            Stage::Clarify,
            Stage::Planning,
            Stage::Routing,
            Stage::Camp,
            Stage::Schedule,
            Stage::Doc,
            Stage::Done,
        ];
        for w in seq.windows(2) {
            assert!(w[0].order() < w[1].order(), "{:?} !< {:?}", w[0], w[1]);
        }
        for s in seq {
            assert_eq!(Stage::parse(s.name()), Some(s.clone()));
        }
        // aliases + case-insensitivity
        assert_eq!(Stage::parse("ROUTE"), Some(Stage::Routing));
        assert_eq!(Stage::parse("calendar"), Some(Stage::Schedule));
        assert_eq!(Stage::parse("garbage"), None);
    }

    // ---- brief.has_minimum heuristic ----

    #[test]
    fn has_minimum_needs_area_and_date() {
        let mut b = TripBrief::default();
        assert!(!b.has_minimum());
        b.fields.insert("area".into(), "Москва".into());
        assert!(!b.has_minimum(), "area alone is not enough");
        b.fields.insert("date_window".into(), "2 недели".into());
        assert!(b.has_minimum());
        // blank value doesn't count
        let mut b2 = TripBrief::default();
        b2.fields.insert("start_location".into(), "Питер".into());
        b2.fields.insert("when".into(), "   ".into());
        assert!(!b2.has_minimum());
    }

    // ---- orchestrator decision parsing + fallback ----

    #[test]
    fn parse_decision_reads_stage_and_mode() {
        let d = parse_decision(
            r#"{"next_stage":"Routing","mode":"ask","message":"какой реки?","reason":"x"}"#,
            &TripBrief::default(),
            &[],
            false,
        );
        assert_eq!(d.next, Stage::Routing);
        assert_eq!(d.mode, Mode::Ask);
        assert_eq!(d.message, "какой реки?");
    }

    #[test]
    fn parse_decision_never_clarifies_on_empty_user_message() {
        // Even if the model says Clarify, an auto-advance (empty msg) must not.
        let mut brief = TripBrief::default();
        brief.fields.insert("area".into(), "Москва".into());
        brief.fields.insert("date".into(), "июль".into());
        let d = parse_decision(
            r#"{"next_stage":"Clarify","mode":"run"}"#,
            &brief,
            &[],
            true, // user_empty
        );
        assert_ne!(d.next, Stage::Clarify);
        assert_eq!(d.next, Stage::Planning); // earliest stage without output
    }

    #[test]
    fn fallback_picks_earliest_stage_without_output() {
        let mut brief = TripBrief::default();
        brief.fields.insert("area".into(), "Москва".into());
        brief.fields.insert("date".into(), "июль".into());
        let mut records = Vec::new();
        set_record(&mut records, &Stage::Planning, "day chosen".into());
        let d = parse_decision("not json", &brief, &records, true);
        assert_eq!(d.next, Stage::Routing);
    }

    #[test]
    fn fallback_clarifies_when_brief_thin_and_user_spoke() {
        let d = parse_decision("junk", &TripBrief::default(), &[], false);
        assert_eq!(d.next, Stage::Clarify);
        assert_eq!(d.mode, Mode::Run);
    }

    #[test]
    fn next_exec_after_returns_done_when_all_present() {
        let mut r = Vec::new();
        for s in [
            Stage::Planning,
            Stage::Routing,
            Stage::Camp,
            Stage::Schedule,
            Stage::Doc,
        ] {
            set_record(&mut r, &s, "x".into());
        }
        assert_eq!(next_exec_after(&r), Stage::Done);
    }

    // ---- record helpers: replace + back-step invalidation ----

    #[test]
    fn set_record_replaces_and_keeps_order() {
        let mut r = Vec::new();
        set_record(&mut r, &Stage::Camp, "camp v1".into());
        set_record(&mut r, &Stage::Planning, "plan".into());
        set_record(&mut r, &Stage::Routing, "route".into());
        // ordered by stage order, not insertion order
        let names: Vec<&str> = r.iter().map(|x| x.stage.as_str()).collect();
        assert_eq!(names, vec!["Planning", "Routing", "Camp"]);
        // replace in place, no duplicate
        set_record(&mut r, &Stage::Camp, "camp v2".into());
        assert_eq!(r.iter().filter(|x| x.stage == "Camp").count(), 1);
        assert_eq!(
            record_index(&r, &Stage::Camp).map(|i| &r[i].output[..]),
            Some("camp v2")
        );
    }

    #[test]
    fn drop_from_invalidates_stage_and_downstream() {
        let mut r = Vec::new();
        for s in [
            Stage::Planning,
            Stage::Routing,
            Stage::Camp,
            Stage::Schedule,
            Stage::Doc,
        ] {
            set_record(&mut r, &s, "x".into());
        }
        // user steps back to Routing → Routing..Doc dropped, Planning kept
        drop_from(&mut r, &Stage::Routing);
        let names: Vec<&str> = r.iter().map(|x| x.stage.as_str()).collect();
        assert_eq!(names, vec!["Planning"]);
    }
}
