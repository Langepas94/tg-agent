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

use std::{collections::BTreeMap, time::Duration};

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
    /// When a checkpoint stage (Planning / Camp) has presented options and is
    /// waiting for the user to choose, this holds that stage. The user's next
    /// message is their choice: we re-run that stage to FINALIZE it, then
    /// auto-advance. `None` whenever we are not paused on a checkpoint.
    #[serde(default)]
    pub awaiting_choice: Option<Stage>,
}

impl TripFlowState {
    pub fn start() -> Self {
        Self {
            stage: Stage::Clarify,
            brief: TripBrief::default(),
            records: Vec::new(),
            clarify_rounds: 0,
            awaiting_choice: None,
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

// Trip-intent detection and "user wants to start now" judgement are NOT done
// here with keyword lists — the semantic router (`super::router`) decides entry
// into the flow, and the Clarify agent decides, by meaning, when the brief is
// ready to plan. See `router.rs` and `CLARIFY_PROMPT`.

// ---------------------------------------------------------------------------
// Clarify agent
// ---------------------------------------------------------------------------

const CLARIFY_PROMPT: &str = "You are the CLARIFY agent of an outdoor-trip planner \
(hikes, kayak/canoe trips, camping, weekend getaways). You receive the user's known profile, \
the brief gathered so far, and the user's newest message. MERGE any new facts into the brief. \
\
CORE PRINCIPLE: the planning agents decide WHERE to go (which river/route), WHICH day, and the \
campsite — that is the whole point of the assistant. NEVER ask the user to choose the river, the \
exact route, the specific day, or the campsite; do NOT ask for things already stated in the \
message or present in the profile/brief. \
\
The ONLY facts to clarify are ones that only the user can know AND are still missing: \
1) their home city / start region — but if the profile has a home city, use it as `area` and do \
NOT ask; 2) the date window, if the message gives none; 3) group size / experience level, if not \
implied; 4) any hard must-haves the user cares about (e.g. campsite far from civilization, water \
nearby). If the message already conveys these (e.g. 'команда неподготовленная, хочет шашлык', \
'одна ночёвка', 'в ближайшие 2 недели', 'вода в 30 м'), mark them filled — do NOT re-ask. \
\
Set `ready=true` as soon as you have a start region (from profile or message) and a rough date \
window; everything else the planners infer. Also set `ready=true` if the user signals they want \
to start now / stop being asked (judge this by meaning, e.g. 'поехали', 'хватит вопросов', \
'just plan it'). Only when something essential is genuinely missing, ask up to 2 SHORT questions \
for it. Reply in the user's language. \
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

/// Render the user's profile as context for the Clarify agent, so it never asks
/// for facts the profile already holds (home city, language, interests, …).
fn profile_context(profile: &super::profile::UserProfile) -> String {
    if profile.fields.is_empty() {
        return "(none known)".to_string();
    }
    profile
        .fields
        .iter()
        .map(|(k, v)| format!("- {k}: {v}"))
        .collect::<Vec<_>>()
        .join("\n")
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

    /// Checkpoint stages stop and present the user a choice before the flow
    /// continues. Planning proposes candidate day+place combos; Camp proposes
    /// the overnight site(s). On first run they present options and pause; on
    /// the user's reply they finalize the chosen option and the flow advances.
    pub fn is_checkpoint(&self) -> bool {
        matches!(self, Stage::Planning | Stage::Camp)
    }

    /// The worker-agent task for an execution stage (empty for Clarify/Done).
    fn instruction(&self) -> &'static str {
        match self {
            Stage::Planning => {
                "You are a CHECKPOINT stage: you offer the user a CHOICE with explicit trade-offs, \
                 you do NOT decide alone. Use the weather tools (geocode then forecast) AND geocode \
                 the user's start region so you can give the APPROXIMATE travel distance/time from \
                 their start to each candidate area.\n\
                 - If [user-choice] below is empty (first run): PROPOSE 2-3 genuinely DIFFERENT \
                 candidate options that SPREAD ACROSS THE TRADE-OFF, not just the single best \
                 weather. Deliberately include at least one CLOSER/nearer area even if its weather \
                 is slightly worse, and one with the best weather even if farther. Each option = a \
                 specific day + a specific waterway/area + weather numbers (rain, wind, temp) + the \
                 approx distance/drive time from the user's start. Make the trade-off explicit \
                 (e.g. 'на 1-2°C прохладнее, зато в 2 раза ближе'). NEVER collapse to one option or \
                 silently pick the farthest 'best weather' spot. End by asking the user to pick. Do \
                 NOT commit yet.\n\
                 - If [user-choice] below names the option the user picked (or asks to adjust): \
                 COMMIT to a single final choice — the chosen DATE and the specific waterway/area — \
                 with the confirming weather numbers and the distance. Output just that final pick."
            }
            Stage::Routing => {
                "Design the actual on-water route. You MUST call the maps/OSM tools to get REAL \
                 geographic data — geocode the area, find the waterway, and resolve concrete \
                 put-in and take-out points. NEVER invent or approximate coordinates: every \
                 coordinate you state must come from a tool result. Give the put-in and take-out \
                 with real coordinates, 2-4 named intermediate stops (real places from the map) \
                 with distances/times, and a relaxed pace matching an unprepared, BBQ-focused \
                 party. Keep total distance modest.\n\
                 HONESTY GATE: if the map service keeps failing (Overpass 429/504) and you CANNOT \
                 resolve real put-in/take-out coordinates and at least one real intermediate stop, \
                 do NOT write vague prose like 'маршрут уточняется' or 'точки не определены'. \
                 Instead end your reply with a line exactly: 'STAGE_INCOMPLETE: <short reason>'. \
                 Never present a route as ready without real coordinates."
            }
            Stage::Camp => {
                "You are a CHECKPOINT stage: you propose the overnight site and let the user \
                 confirm, you do NOT silently decide. You MUST use the maps/OSM tools to VERIFY \
                 constraints with real data: query nearby settlements/turbazy/roads to confirm the \
                 minimum distance from civilization, and confirm the site is within the required \
                 distance to water. NEVER guess coordinates — derive them from tool results.\n\
                 - If [user-choice] below is empty (first run): propose 1-2 candidate campsites on \
                 the route, each with real coordinates, the measured distance to water, and the \
                 measured distance to the nearest village/turbaza/road. End by asking the user to \
                 confirm one (or ask for another). Do NOT finalize yet.\n\
                 - If [user-choice] below confirms a site (or asks to move it): COMMIT to that one \
                 site and output its final verified coordinates and distances.\n\
                 HONESTY GATE: if the map service keeps failing and you CANNOT produce a real \
                 campsite with verified coordinates and the measured distances (to water and to the \
                 nearest village/turbaza), do NOT write 'место уточняется' or fabricate numbers. \
                 Instead end your reply with a line exactly: 'STAGE_INCOMPLETE: <short reason>'."
            }
            Stage::Schedule => {
                "Create a real calendar event for this trip via the connected Google/calendar \
                 tools (start = chosen date + time, end = next day; title, location, description \
                 with the plan). Use the user's email from their profile. NEVER ask the user for a \
                 token or credentials. Actually CALL the create-event tool, then confirm the event \
                 from the tool result. If a tool reports the user is NOT authenticated or returns \
                 an authorization URL (start_google_auth / auth flow), give the user that EXACT URL \
                 verbatim as a clickable link with a one-line instruction to open it and approve \
                 access — do NOT paraphrase it into 'нужен токен'. Then STOP. Never invent success."
            }
            Stage::Doc => {
                "Create a real shareable Google Doc with the full plan (date, route with real \
                 coordinates/stops, campsite with verified distances, gear/BBQ notes) via the \
                 connected Google/docs tools, then return the actual share link from the tool \
                 result. NEVER ask the user for a token. If a tool reports the user is NOT \
                 authenticated or returns an authorization URL (start_google_auth / auth flow), \
                 give the user that EXACT URL verbatim as a clickable link with a one-line \
                 instruction to open it and approve access — do NOT paraphrase it into 'нужен \
                 доступ'. Also output the full plan as plain text so it is never lost. Then STOP. \
                 Never fabricate a link."
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
- Always use mode=run for execution stages. The Planning and Camp stages are CHECKPOINTS: they \
present their own candidate options to the user and pause for a choice, so you do NOT need to ask \
the user yourself before running them. \
- Use mode=ask only in the rare case you need a fact no stage can obtain; put a short question in \
message. \
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
/// Safety net for ONE stage attempt that is genuinely stuck (a stalled external
/// tool connection), not a planning budget. A normal OSM/weather stage finishes
/// well under this; we also retry once and never dead-end on a hit, so a high
/// value just prevents a hang — it does not make the user wait this long in the
/// common case.
const STAGE_TIMEOUT: Duration = Duration::from_secs(300);

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
    // If we paused on a checkpoint last turn, the user's message now is their
    // choice for that stage: re-run it to finalize, then auto-advance.
    let awaiting = session
        .trip
        .as_ref()
        .and_then(|t| t.awaiting_choice.clone());
    // First contact with a fresh flow: ALWAYS clarify first (never let the
    // orchestrator dive into the pipeline on a raw multi-part request). After
    // clarifying, the flow advances to the Planning checkpoint, which presents
    // candidate days/places instead of silently planning everything at once.
    let first_contact = records.is_empty()
        && awaiting.is_none()
        && session.trip.as_ref().map(|t| t.clarify_rounds).unwrap_or(0) == 0;

    for step in 0..MAX_ORCH_STEPS {
        let user_empty = step > 0; // only the first step carries the user's message
        let umsg = if user_empty { "" } else { user_text };
        // True when this exec run finalizes the checkpoint the user just answered.
        let finalizing = |stage: &Stage| !user_empty && awaiting.as_ref() == Some(stage);

        // ---- Pick the next transition ----
        let decision = if step == 0 && first_contact {
            // Turn 1: clarify is mandatory; skip the orchestrator call.
            Decision {
                next: Stage::Clarify,
                mode: Mode::Run,
                message: String::new(),
            }
        } else if step == 0 && awaiting.is_some() {
            // The user is replying to a checkpoint → re-run that stage to finalize.
            Decision {
                next: awaiting.clone().unwrap(),
                mode: Mode::Run,
                message: String::new(),
            }
        } else {
            let orch_input = format!(
                "[trip-brief]\n{}\n\n[completed-stages]\n{}\n\n[user-message]\n{}",
                brief.render(),
                completed_list(&records),
                if umsg.trim().is_empty() {
                    "(none)"
                } else {
                    umsg
                },
            );
            let raw = llm
                .complete(ORCH_PROMPT, &orch_input)
                .await
                .unwrap_or_default();
            prevent_auto_repeat(
                parse_decision(&raw, &brief, &records, user_empty),
                &records,
                user_empty,
            )
        };

        match decision.next {
            // ---- Clarify: dedicated agent extracts slots / asks questions ----
            Stage::Clarify => {
                // Seed the start region from the user's known home city so we
                // never ask for something the profile already holds.
                if !brief.has_minimum() {
                    if let Some(home) = session.profile.fields.get("home_city") {
                        if !home.trim().is_empty()
                            && !brief.fields.keys().any(|k| k.contains("area"))
                        {
                            brief.fields.insert("area".into(), format!("около {home}"));
                        }
                    }
                }
                let input = format!(
                    "Known user profile:\n{}\n\nBrief so far:\n{}\n\nUser message:\n{umsg}",
                    profile_context(&session.profile),
                    brief.render(),
                );
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
                let enough = parsed.ready || rounds >= MAX_CLARIFY_ROUNDS;
                if !enough {
                    return Ok(FlowTurn {
                        reply: render_clarify_reply(&parsed.recap, &parsed.questions),
                        trace,
                        done: false,
                    });
                }
                // Brief good enough → advance to the Planning checkpoint, which
                // will present candidate days/places for the user to choose.
                continue;
            }

            // ---- Done: compose the shareable plan and finish ----
            Stage::Done => {
                // GATE: never declare the trip done while an essential geo stage
                // (day/place, route points, campsite) is missing or unresolved.
                if let Some(stage) = first_unresolved_essential(&records) {
                    if let Some(t) = session.trip.as_mut() {
                        t.stage = stage.clone();
                        t.brief = brief.clone();
                        t.records = records.clone();
                        t.awaiting_choice = Some(stage.clone());
                    }
                    return Ok(FlowTurn {
                        reply: render_incomplete_block(&stage),
                        trace,
                        done: false,
                    });
                }
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
                // GATE: the deliverable stages (calendar event, Google Doc) must
                // not run on a hollow plan. Block Schedule/Doc until day/place,
                // route points and campsite are all real and resolved.
                if matches!(stage, Stage::Schedule | Stage::Doc) {
                    if let Some(blocker) = first_unresolved_essential(&records) {
                        if let Some(t) = session.trip.as_mut() {
                            t.stage = blocker.clone();
                            t.brief = brief.clone();
                            t.records = records.clone();
                            t.awaiting_choice = Some(blocker.clone());
                        }
                        return Ok(FlowTurn {
                            reply: render_incomplete_block(&blocker),
                            trace,
                            done: false,
                        });
                    }
                }
                if decision.mode == Mode::Ask {
                    if let Some(t) = session.trip.as_mut() {
                        t.stage = stage.clone();
                        t.brief = brief.clone();
                        t.records = records.clone();
                        t.awaiting_choice = None;
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
                let is_final = finalizing(stage);
                // Back-step / re-run invalidates this stage and everything after.
                drop_from(&mut records, stage);
                // Pass the user's choice only when finalizing a checkpoint, so the
                // stage commits to the option they picked; otherwise it proposes.
                let choice = if is_final { user_text } else { "" };
                let output =
                    run_exec_stage(llm, state, session, &brief, &records, stage, choice).await;
                set_record(&mut records, stage, output.clone());
                trace.push(format!("• {}: {}", stage.name(), clip(&output, 90)));
                // A failed/timed-out stage is NOT a dead-end. Only a CHECKPOINT
                // failure pauses (it has nothing to show), with a short honest
                // note. A non-checkpoint failure is carried forward as
                // best-effort; the final compose flags whatever stayed
                // unverified. We never scold the user or ask them to "narrow the
                // river" — picking the river is the planner's job.
                if is_stage_unresolved(&output) && stage.is_checkpoint() {
                    if let Some(t) = session.trip.as_mut() {
                        t.stage = stage.clone();
                        t.brief = brief.clone();
                        t.records = records.clone();
                        // Keep awaiting on the checkpoint so the user can retry it.
                        t.awaiting_choice = Some(stage.clone());
                    }
                    return Ok(FlowTurn {
                        reply: render_checkpoint_stall(stage),
                        trace,
                        done: false,
                    });
                }
                // Checkpoint: on its FIRST run (not a finalize) present the
                // options and pause for the user's choice.
                if stage.is_checkpoint() && !is_final {
                    if let Some(t) = session.trip.as_mut() {
                        t.stage = stage.clone();
                        t.brief = brief.clone();
                        t.records = records.clone();
                        t.awaiting_choice = Some(stage.clone());
                    }
                    return Ok(FlowTurn {
                        reply: output,
                        trace,
                        done: false,
                    });
                }
                // Non-checkpoint, or a finalized checkpoint → keep advancing.
                if let Some(t) = session.trip.as_mut() {
                    t.stage = stage.clone();
                    t.records = records.clone();
                    t.awaiting_choice = None;
                }
                continue;
            }

            _ => continue,
        }
    }

    // Safety: hit the step cap. Still don't ship a hollow plan — if an essential
    // geo stage never resolved, stop honestly instead of composing a fake "done".
    if let Some(stage) = first_unresolved_essential(&records) {
        if let Some(t) = session.trip.as_mut() {
            t.stage = stage.clone();
            t.brief = brief.clone();
            t.records = records.clone();
            t.awaiting_choice = Some(stage.clone());
        }
        return Ok(FlowTurn {
            reply: render_incomplete_block(&stage),
            trace,
            done: false,
        });
    }
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
    user_choice: &str,
) -> String {
    let system = stage_system(session, stage.name());
    let choice = user_choice.trim();
    let query = format!(
        "[trip-brief]\n{}\n\n[prior-stages]\n{}\n\n[user-choice]\n{}\n\n\
         [your-task as the {} agent]\n{}",
        brief.render(),
        render_records(records),
        if choice.is_empty() {
            "(none yet)"
        } else {
            choice
        },
        stage.name(),
        stage.instruction(),
    );
    // One automatic retry: external map/weather tools (Overpass especially)
    // return transient 429/504s, so a single retry usually succeeds where the
    // first attempt timed out or errored. Failures are NOT surfaced to the user
    // as a dead-end here — the caller carries a best-effort result forward.
    let mut last = run_stage_once(llm, state, &system, &query).await;
    if is_stage_failure(&last) {
        last = run_stage_once(llm, state, &system, &query).await;
    }
    last
}

/// One bounded attempt at a stage's worker agent.
async fn run_stage_once(llm: &Llm, state: &BotState, system: &str, query: &str) -> String {
    match tokio::time::timeout(STAGE_TIMEOUT, llm.answer_with_system(state, system, query)).await {
        Err(_) => "(stage timed out)".to_string(),
        Ok(Ok(o)) if !o.trim().is_empty() => o,
        Ok(Ok(_)) => "(no output)".to_string(),
        Ok(Err(e)) => format!("(stage failed: {e})"),
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

/// On auto-advance there is no new user intent, so the orchestrator must not
/// re-run a stage that already has output. In prod this retried Routing several
/// times after tool-budget failures, turning one Telegram message into a very
/// long silent turn.
fn prevent_auto_repeat(decision: Decision, records: &[StageRecord], user_empty: bool) -> Decision {
    if !user_empty || !decision.next.is_exec() || record_index(records, &decision.next).is_none() {
        return decision;
    }
    Decision {
        next: next_exec_after(records),
        mode: Mode::Run,
        message: String::new(),
    }
}

fn is_stage_failure(output: &str) -> bool {
    let text = output.trim();
    text.starts_with("Stopped after too many tool calls")
        || text.starts_with("(stage failed:")
        || text.starts_with("(stage timed out")
}

/// A stage output is UNRESOLVED when it failed/timed out, OR the worker agent
/// itself signalled it could not obtain the real geo data (the `STAGE_INCOMPLETE`
/// marker we ask it to emit instead of writing vague "уточняется" prose).
fn is_stage_unresolved(output: &str) -> bool {
    is_stage_failure(output) || output.contains("STAGE_INCOMPLETE")
}

/// The essential geo stages that MUST carry real data before we create the
/// calendar event / Google Doc or declare the plan done. Returns the FIRST one
/// that is missing or unresolved, so we re-run exactly that step rather than
/// shipping a hollow "готово" with a real calendar event and doc attached.
fn first_unresolved_essential(records: &[StageRecord]) -> Option<Stage> {
    for stage in [Stage::Planning, Stage::Routing, Stage::Camp] {
        match record_index(records, &stage) {
            None => return Some(stage),
            Some(i) if is_stage_unresolved(&records[i].output) => return Some(stage),
            _ => {}
        }
    }
    None
}

/// Honest "not done yet" note when an essential geo stage has no real data.
/// We refuse to fabricate a finished plan (calendar + doc) on top of it.
fn render_incomplete_block(stage: &Stage) -> String {
    let what = match stage {
        Stage::Planning => "день и место",
        Stage::Routing => "точки маршрута (заезд, остановки, выход)",
        Stage::Camp => "место ночёвки с проверкой расстояний",
        _ => "данные с карт",
    };
    format!(
        "Финал ещё не готов: не удалось получить {what} — картографический сервис \
(OSM/Overpass) сейчас перегружен и отдаёт ошибки (504/429). Календарь и документ на \
неполном плане не создаю. Напишите «ещё раз» — повторю именно этот шаг.",
    )
}

/// Short, honest stall note for a CHECKPOINT stage whose map/weather tools did
/// not respond in time (even after a retry). No scolding, and never asks the
/// user to pick a river/area — that is the planner's job.
fn render_checkpoint_stall(stage: &Stage) -> String {
    let what = match stage {
        Stage::Planning => "подобрать день и место",
        Stage::Camp => "проверить стоянку по карте",
        _ => "собрать данные",
    };
    format!(
        "Картографические/погодные сервисы сейчас отвечают медленно — не успел {what}. \
Напишите «ещё раз» — повторю запрос.",
    )
}

const COMPOSE_PROMPT: &str = "You assemble the FINAL trip plan a user will share with friends, \
from the stage outputs of a planning swarm. Write it in the user's language as a clean, \
phone-friendly PLAIN-TEXT message (no Markdown tables, no `|`, no `**`). Use short vertical \
blocks with emoji headings: chosen day + weather, the route with concrete stops, the overnight \
campsite (with distances), gear/BBQ notes, and — if created — the calendar event and the \
shareable doc link. Be concrete; do not invent a doc link or coordinates that the stages did \
not produce. If a stage returned an AUTHORIZATION URL (Google sign-in / start_google_auth), \
keep that URL VERBATIM in the final message as a clickable link with a one-line 'open and approve \
access' instruction — never drop it and never turn it into a vague 'нужен токен/доступ'. \
If a stage output shows it stalled or failed (e.g. '(stage timed out)', '(stage failed: …)'), \
do NOT fabricate that section's data: present the rest of the plan normally and add one short \
honest line that that specific part (e.g. the campsite distances) is approximate / still to be \
verified — without scolding the user and without asking them to choose a river or narrow the \
area. Keep it tight.";

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
             need live data or to act. For Russian geography with OSM tools, pass region=\"RU\" \
             and include \"Россия\" in free-text place queries. For maps__osm_query_bbox, the \
             tags argument must be a JSON object/map such as {{\"waterway\":\"river\"}} or \
             {{\"tourism\":\"camp_site\"}}; never pass an array or a plain string. Avoid raw \
             name=<Cyrillic> tag selectors; geocode names first or query broad tags and filter the result. If \
             Overpass returns 400/429/504, do not keep retrying alternatives in a loop: state \
             what remains unverified and finish your stage."
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

    // ---- checkpoint stages ----

    #[test]
    fn only_planning_and_camp_are_checkpoints() {
        assert!(Stage::Planning.is_checkpoint());
        assert!(Stage::Camp.is_checkpoint());
        assert!(!Stage::Routing.is_checkpoint());
        assert!(!Stage::Schedule.is_checkpoint());
        assert!(!Stage::Doc.is_checkpoint());
        assert!(!Stage::Clarify.is_checkpoint());
        assert!(!Stage::Done.is_checkpoint());
    }

    #[test]
    fn awaiting_choice_survives_serde_roundtrip() {
        let mut st = TripFlowState::start();
        assert!(st.awaiting_choice.is_none());
        st.awaiting_choice = Some(Stage::Planning);
        let json = serde_json::to_string(&st).unwrap();
        let back: TripFlowState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.awaiting_choice, Some(Stage::Planning));
        // legacy state without the field deserializes to None (serde default)
        let legacy: TripFlowState = serde_json::from_str(r#"{"stage":"Clarify"}"#).unwrap();
        assert!(legacy.awaiting_choice.is_none());
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
    fn auto_advance_does_not_repeat_completed_stage() {
        let mut records = Vec::new();
        set_record(&mut records, &Stage::Planning, "plan".into());
        let d = prevent_auto_repeat(
            Decision {
                next: Stage::Planning,
                mode: Mode::Run,
                message: String::new(),
            },
            &records,
            true,
        );
        assert_eq!(d.next, Stage::Routing);
    }

    #[test]
    fn detects_stage_failure_outputs() {
        assert!(is_stage_failure(
            "Stopped after too many tool calls. Try rephrasing."
        ));
        assert!(is_stage_failure("(stage failed: LLM HTTP 500)"));
        assert!(is_stage_failure("(stage timed out after 150s)"));
        assert!(!is_stage_failure("Маршрут построен."));
    }

    #[test]
    fn unresolved_covers_failure_and_incomplete_marker() {
        assert!(is_stage_unresolved("(stage timed out)"));
        assert!(is_stage_unresolved(
            "Не смог получить точки.\nSTAGE_INCOMPLETE: Overpass 504"
        ));
        assert!(!is_stage_unresolved(
            "Заезд 48.63,43.55; выход 48.70,43.75; стоянка 48.66,43.60."
        ));
    }

    #[test]
    fn gate_blocks_until_essential_geo_stages_resolved() {
        let mut r = Vec::new();
        // nothing yet → Planning is the first blocker
        assert_eq!(first_unresolved_essential(&r), Some(Stage::Planning));
        set_record(&mut r, &Stage::Planning, "Пятница, Дон".into());
        // Routing missing → blocks on Routing
        assert_eq!(first_unresolved_essential(&r), Some(Stage::Routing));
        set_record(
            &mut r,
            &Stage::Routing,
            "STAGE_INCOMPLETE: Overpass 504".into(),
        );
        // Routing present but unresolved → still blocks on Routing
        assert_eq!(first_unresolved_essential(&r), Some(Stage::Routing));
        set_record(
            &mut r,
            &Stage::Routing,
            "put-in 48.6,43.5 stop 48.6,43.6".into(),
        );
        set_record(
            &mut r,
            &Stage::Camp,
            "site 48.66,43.60, 20m to water".into(),
        );
        // all three resolved → no blocker, deliverables may run
        assert_eq!(first_unresolved_essential(&r), None);
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
    fn profile_context_lists_known_facts() {
        let mut p = super::super::profile::UserProfile::default();
        assert_eq!(profile_context(&p), "(none known)");
        p.set("home_city", "Москва");
        p.set("interests", "байдарки");
        let ctx = profile_context(&p);
        assert!(ctx.contains("home_city: Москва"));
        assert!(ctx.contains("interests: байдарки"));
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
