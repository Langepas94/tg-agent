//! Stateful multi-agent trip-planning flow (the "swarm").
//!
//! The public runner is a dynamic swarm: BriefAgent extracts the request,
//! OptionsAgent offers concise alternatives, SwarmPlanner builds worker tasks
//! from the live MCP tool inventory, WorkerAgents execute isolated tasks,
//! VerifierAgent gates side effects, and FinalAgent composes the answer.
//!
//! The flow SUSPENDS across user turns: Clarify interrogates first, building a
//! `TripBrief` persisted in the chat session; each execution stage
//! The old fixed stage labels are retained only as serialized-state
//! compatibility shells; they no longer drive the public flow.

use std::{collections::BTreeMap, time::Duration};

#[cfg(test)]
use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{llm::Llm, state::BotState};

use super::{memory::MemoryLayer, profile, session::ChatSession};

/// Hard cap on clarify rounds — after this we plan with whatever we have, so the
/// bot never interrogates forever.
const MAX_CLARIFY_ROUNDS: u8 = 3;

/// Per-stage output is clipped to this many bytes when handed to the next stage,
/// keeping the cumulative prompt bounded. A committed stage result (chosen day,
/// route coords, campsite) fits well under this; the verbose user-facing prose
/// is for the chat, not the hand-off.
#[cfg(test)]
const HANDOFF_CLIP: usize = 400;
/// A checkpoint finalization must see the options it previously showed the
/// user. Clipping the Planning/Camp record to the normal hand-off size made
/// "first option" ambiguous and let the model bind it to the wrong candidate.
#[cfg(test)]
const CHECKPOINT_CHOICE_CLIP: usize = 3200;

#[derive(Debug, Clone, PartialEq, Eq, std::hash::Hash, Serialize, Deserialize)]
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

#[cfg(test)]
const CLARIFY_PROMPT: &str = "You are the CLARIFY agent of a general outdoor-trip / recreation \
planner (any outdoor or nature-recreation activity the user names; \
make NO assumption about the activity or terrain). You receive the user's known profile, the \
brief gathered so far, and the user's newest message. MERGE any new facts into the brief. \
\
CORE PRINCIPLE: the planning agents decide WHERE to go (the place/route), WHICH day, and the \
overnight spot — that is the whole point of the assistant. NEVER ask the user to choose the \
place, the exact route, the specific day, or the overnight spot; do NOT ask for things already \
stated in the message or present in the profile/brief. \
\
The ONLY facts to clarify are ones that only the user can know AND are still missing: \
1) their home city / start region — but if the profile has a home city, use it as `area` and do \
NOT ask; 2) the date window, if the message gives none; 3) group size / experience level, if not \
implied; 4) any hard must-haves the user cares about (whatever constraints they state). If the \
message already conveys these (e.g. 'команда неподготовленная, хочет шашлык', 'одна ночёвка', \
'в ближайшие 2 недели', 'вода в 30 м'), mark them filled — do NOT re-ask. \
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

    /// Short user-facing "what I'm doing now" line, shown before a (possibly
    /// multi-minute) stage runs so the chat is never silent. RU — primary user.
    #[cfg(test)]
    fn progress_label(&self) -> &'static str {
        match self {
            Stage::Planning => "📅 Подбираю даты и место по погоде…",
            Stage::Routing => "🗺 Прорабатываю маршрут по карте… это может занять пару минут",
            Stage::Camp => "🏕 Подбираю место для ночёвки… пару минут",
            Stage::Schedule => "📅 Создаю событие в календаре…",
            Stage::Doc => "📄 Готовлю документ с планом…",
            Stage::Clarify | Stage::Done => "",
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
    #[cfg(test)]
    fn instruction(&self) -> &'static str {
        match self {
            Stage::Planning => {
                "You are a CHECKPOINT stage: you offer the user a CHOICE with explicit trade-offs, \
                 you do NOT decide alone. Use the weather tools (geocode then forecast) AND geocode \
                 the user's start region so you can give the APPROXIMATE travel distance/time from \
                 their start to each candidate area.\n\
                 - If [user-choice] below is empty (first run): PROPOSE the genuinely distinct, \
                 worthwhile candidate options that actually exist for this trip — as many as truly \
                 make sense, no fixed number. Do not pad to a count, and do not drop a clearly good \
                 option to hit one; if only one area realistically fits, present just that and say \
                 why. SPREAD them across the trade-off (don't offer only the single best weather): \
                 if the brief says the user can only go on weekends, every option MUST be a \
                 Saturday→Sunday overnight pair in the current/future year; never offer Friday or \
                 any weekday just because the weather is better, and include weekday + day + month \
                 + year for every option. The place type, route type, and evidence must follow the \
                 user's actual activity and constraints; do not use fixed activity templates. \
                 where it applies, include a nearer area even if slightly worse weather and a \
                 best-weather one even if farther. Each option = a specific day + a specific \
                 place/area that fits THE ACTIVITY DESCRIBED IN THE BRIEF (infer what kind of place \
                 that activity needs; make no assumptions about terrain or whether water is \
                 involved) + weather numbers (rain, wind, temp) + the approx distance/travel time \
                 from the user's start. Make the trade-off explicit (e.g. 'на 1-2°C прохладнее, \
                 зато в 2 раза ближе'). End by asking the user to pick. Do NOT commit yet.\n\
                 - If [user-choice] below names the option the user picked: your earlier options \
                 are in [prior-stages]; just COMMIT to the chosen one — its DATE, place/area, \
                 weather numbers and distance — no new tool calls needed. Output just that pick."
            }
            Stage::Routing => {
                "Design the actual route for the trip. INFER from the brief what kind of route the \
                 stated activity needs and which map features matter — do not assume any particular \
                 mode or terrain. You MUST call the maps/OSM tools to get REAL geographic data: \
                 geocode the area, find the features the route follows, and resolve concrete start \
                 and end points. NEVER invent or approximate coordinates — every coordinate you \
                 state must come from a tool result. Give the start and end with real coordinates, \
                 2-4 named intermediate stops (real places from the map) with distances/times, and \
                 a pace matching the party's stated level and priorities from the brief. Keep total \
                 distance sensible for that level. The route mechanics must be inferred from the \
                 brief and the available tools; never substitute a different activity mode.\n\
                 TIGHT BUDGET: you have only a few map queries. Geocode once, run a SMALL number of \
                 lookups, then COMMIT — output the route with the concrete coordinates you have. Do \
                 NOT keep re-querying to perfect it. Better a good route committed in 4 queries \
                 than an endless refinement that times out.\n\
                 Only if you got NO usable coordinates at all end your reply with exactly: \
                 'STAGE_INCOMPLETE: <short reason>'. Never write vague prose like 'маршрут \
                 уточняется' instead of committing."
            }
            Stage::Camp => {
                "You are a CHECKPOINT stage: you propose the overnight spot and let the user \
                 confirm. Honour ONLY the CONSTRAINTS THE USER ACTUALLY STATED in the brief — do \
                 not invent requirements they never mentioned. Keys named `constraint_*` in \
                 [trip-brief] are HARD constraints. Read each stated constraint from the brief and \
                 verify it.\n\
                 BE ECONOMICAL — you have a limited tool budget; too many queries fail the stage:\n\
                 1. REUSE the coordinates the Routing stage already produced (see [prior-stages]). \
                 Do NOT re-discover the route — it is already known. Pick ONE candidate point along \
                 it as your starting guess.\n\
                 2. The spot does NOT need to be a tagged OSM feature — any suitable real location \
                 works. Do NOT enumerate many feature tags one by one.\n\
                 3. Verify the stated constraints with the FEWEST queries possible (aim 2-3), small \
                 bboxes only — NEVER sweep broad single-key tags over a big box (slow, wastes the \
                 budget). For a 'far from civilization' constraint, one small-bbox settlements query \
                 (e.g. {{\"place\":\"village\"}}) near the point is enough.\n\
                 CRITICAL — the spot must be on solid ground you can ACTUALLY use for an overnight \
                 stay: never return the centroid (`out center`) of a body of water or any other \
                 polygon you cannot stand on. If the user stated a proximity constraint to some \
                 feature, the spot is the nearby usable ground whose measured distance to that \
                 feature's EDGE meets the limit, not a point inside the feature.\n\
                 COMMIT within ~5 queries: after your few lookups, output a concrete spot with the \
                 real coordinates you have. HARD CONSTRAINTS stated by the user are not optional: \
                 if you cannot verify one of them, do not present the campsite as confirmable; end \
                 with 'STAGE_INCOMPLETE: <short reason>' instead of saying the constraint is \
                 unchecked. Do NOT keep querying.\n\
                 - If [user-choice] below is empty (first run): propose the candidate spot(s) that \
                 genuinely fit — present what actually works, not a fixed count (often one or two) \
                 — each with real coordinates and the measured distances for each constraint you \
                 COULD check. End by asking the user to confirm one. Do NOT finalize yet.\n\
                 - If [user-choice] below confirms a site: your earlier candidates are already in \
                 [prior-stages]; just pick the one the user chose and restate its coordinates and \
                 distances. Do NOT run new map queries (only re-query if they asked to MOVE it).\n\
                 Only if you got NO usable coordinates at all, end with exactly: \
                 'STAGE_INCOMPLETE: <short reason>'. Never write 'место уточняется' or fabricate \
                 numbers instead of committing real coordinates."
            }
            Stage::Schedule => {
                "Legacy compatibility placeholder. The dynamic ArtifactsAgent is responsible for \
                 requested external artifacts, based on actual runtime tool inventory and verifier \
                 approval; this fixed stage must not assume any calendar capability exists."
            }
            Stage::Doc => {
                "Legacy compatibility placeholder. The dynamic ArtifactsAgent is responsible for \
                 requested shareable documents, based on actual runtime tool inventory and verifier \
                 approval; this fixed stage must not assume any document capability exists."
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

#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
fn next_exec_after(records: &[StageRecord]) -> Stage {
    for stage in [
        Stage::Planning,
        Stage::Routing,
        Stage::Camp,
        Stage::Schedule,
        Stage::Doc,
    ] {
        match record_index(records, &stage) {
            None => return stage,
            Some(i) if is_stage_unresolved(&records[i].output) => return stage,
            _ => {}
        }
    }
    Stage::Done
}

// ---------------------------------------------------------------------------
// Record helpers (replace-by-stage + downstream invalidation for back-steps)
// ---------------------------------------------------------------------------

#[cfg(test)]
fn record_index(records: &[StageRecord], stage: &Stage) -> Option<usize> {
    records.iter().position(|r| r.stage == stage.name())
}

/// Insert or replace a stage's output, keeping records ordered by stage order.
#[cfg(test)]
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
#[cfg(test)]
fn drop_from(records: &mut Vec<StageRecord>, stage: &Stage) {
    records.retain(|r| match Stage::parse(&r.stage) {
        Some(s) => s.order() < stage.order(),
        None => false,
    });
}

/// Drop only the stages strictly AFTER `stage`, keeping `stage`'s own record —
/// used when finalizing a checkpoint so the worker can see the candidates it
/// already proposed and just commit to the user's pick instead of re-querying.
#[cfg(test)]
fn drop_after(records: &mut Vec<StageRecord>, stage: &Stage) {
    records.retain(|r| match Stage::parse(&r.stage) {
        Some(s) => s.order() <= stage.order(),
        None => false,
    });
}

#[cfg(test)]
fn seed_obvious_trip_facts(brief: &mut TripBrief, user_text: &str) {
    let _ = (brief, user_text);
}

#[cfg(test)]
fn validate_stage_output(brief: &TripBrief, stage: &Stage, output: &mut String) {
    let _ = brief;
    if matches!(stage, Stage::Routing | Stage::Camp) && output_admits_unresolved_core_data(output) {
        output.push_str("\nSTAGE_INCOMPLETE: core route/campsite data is not concrete");
    }
}

fn output_admits_unresolved_core_data(output: &str) -> bool {
    let low = output.to_lowercase();
    let uncertainty = [
        "не зафиксирован",
        "не зафиксирована",
        "не завершил",
        "не завершила",
        "не удалось построить",
        "не удалось получить",
        "требует уточнения",
        "нужно уточнить",
        "уточнить на местности",
        "still to be verified",
        "not fixed",
        "not finalized",
        "could not build",
        "could not get",
    ];
    let core = [
        "маршрут",
        "трек",
        "точк",
        "стоянк",
        "лагер",
        "route",
        "track",
        "point",
        "camp",
        "campsite",
    ];
    uncertainty.iter().any(|needle| low.contains(needle))
        && core.iter().any(|needle| low.contains(needle))
}

// ---------------------------------------------------------------------------
// Orchestrated turn
// ---------------------------------------------------------------------------

/// Maximum orchestrator steps per user turn (auto-advance bound), so one turn
/// can walk Planning→…→Doc→Done but never loops forever.
#[cfg(test)]
const MAX_ORCH_STEPS: usize = 12;
/// Extra recovery attempts after a stage explicitly failed or marked itself
/// incomplete. `run_exec_stage` already retries low-level HTTP/LLM failures
/// once; this handles semantic stage failures such as `STAGE_INCOMPLETE`.
#[cfg(test)]
const MAX_STAGE_RECOVERY_RETRIES: usize = 1;
/// Safety net for ONE stage attempt that is genuinely stuck (a stalled external
/// tool connection), not a planning budget. A normal OSM/weather stage finishes
/// well under this; we also retry once and never dead-end on a hit, so a high
/// value just prevents a hang — it does not make the user wait this long in the
/// common case.
const STAGE_TIMEOUT: Duration = Duration::from_secs(200);

/// Advance the flow by one user turn. The caller guarantees `session.trip` is
/// `Some`. The ORCHESTRATOR agent decides every transition; the user can step
/// back at any point by asking to change an earlier decision.
pub async fn advance(
    llm: &Llm,
    state: &BotState,
    session: &mut ChatSession,
    user_text: &str,
    progress: Option<&super::ProgressSender>,
) -> Result<FlowTurn> {
    advance_swarm(llm, state, session, user_text, progress).await
}

#[derive(Debug, Deserialize)]
struct SwarmPlan {
    #[serde(default)]
    tasks: Vec<SwarmTask>,
}

#[derive(Debug, Clone, Deserialize)]
struct SwarmTask {
    #[serde(default)]
    id: String,
    #[serde(default)]
    agent: String,
    #[serde(default)]
    task: String,
    #[serde(default)]
    tools: bool,
    #[serde(default)]
    side_effects: bool,
    #[serde(default)]
    checkpoint: bool,
}

#[derive(Debug, Deserialize)]
struct SwarmVerdict {
    #[serde(default)]
    ready: bool,
    #[serde(default)]
    missing: Vec<String>,
}

const SWARM_BRIEF_PROMPT: &str = "You are BriefAgent in an outdoor-recreation planning swarm. \
Extract the user's request into an open, activity-agnostic brief. Do not use fixed activity \
categories. Preserve the user's actual activity, constraints, preferences, requested artifacts, \
dates, group capability, start area, and uncertainty. Ask only for facts that only the user can \
know and that block planning. Return ONLY JSON: {\"brief\":{\"key\":\"value\",...},\
\"ready\":bool,\"questions\":[\"...\"],\"recap\":\"short\"}.";

const SWARM_OPTIONS_PROMPT: &str = "You are OptionsAgent. Give a SHORT menu of genuinely distinct \
options for the outdoor/recreation request. Do not deep-dive into one route yet. Use available \
tools only as much as needed to compare options at a high level. The options must match the user's \
actual activity and constraints, whatever they are. End by asking the user to choose one option. \
Plain text, no Markdown table.";

const SWARM_PLANNER_PROMPT: &str = "You are SwarmPlanner. Build a minimal swarm plan from the \
brief, the user's chosen option, prior records, and the ACTUAL connected MCP tool inventory. Do \
not assume calendar/docs/maps/weather tools exist; inspect the inventory. Do not use fixed \
activity templates. Create separate worker tasks with narrow responsibilities and isolated context. \
Include a side_effects=true task only for external artifacts explicitly requested by the user AND \
only if plausible tools are present or connectable; otherwise include a non-side-effect task that \
explains the missing capability. Return ONLY JSON: {\"tasks\":[{\"id\":\"short_snake\",\
\"agent\":\"AgentName\",\"task\":\"specific task\",\"tools\":true,\"side_effects\":false,\
\"checkpoint\":false}]}. The plan must include a verification task before any side-effect task.";

const SWARM_VERIFIER_PROMPT: &str = "You are VerifierAgent. Decide whether the gathered evidence is \
concrete enough to perform external side effects or present the final plan. Check the user's actual \
brief and constraints, not a fixed schema. If a route/place/date/constraint/artifact needed by the \
request is vague, missing, contradicted, or merely 'to be checked later', ready=false. Return ONLY \
JSON: {\"ready\":bool,\"missing\":[\"short missing item\",...]}";

const SWARM_FINAL_PROMPT: &str =
    "You are FinalAgent. Compose the user-facing answer from the swarm \
records. Preserve the selected option, concrete evidence, tool results, and artifact links. Do not \
claim an external artifact was created unless a tool result says so. If a requested capability was \
missing, say that plainly. Plain Telegram-friendly text, no Markdown tables.";

async fn advance_swarm(
    llm: &Llm,
    state: &BotState,
    session: &mut ChatSession,
    user_text: &str,
    progress: Option<&super::ProgressSender>,
) -> Result<FlowTurn> {
    let tools = state.tool_inventory().await;
    let mut tool_context = render_tool_inventory(&tools);
    let mut flow = session.trip.clone().unwrap_or_else(TripFlowState::start);
    let mut trace = Vec::new();

    if flow.records.is_empty() {
        let input = format!(
            "Known profile:\n{}\n\nUser request:\n{}",
            profile_context(&session.profile),
            user_text
        );
        let parsed = parse_clarify(&llm.complete(SWARM_BRIEF_PROMPT, &input).await?);
        flow.brief.merge(parsed.brief);
        flow.clarify_rounds = flow.clarify_rounds.saturating_add(1);
        let ready =
            parsed.ready || flow.brief.has_minimum() || flow.clarify_rounds >= MAX_CLARIFY_ROUNDS;
        if !ready {
            flow.stage = Stage::Clarify;
            session.trip = Some(flow);
            return Ok(FlowTurn {
                reply: render_clarify_reply(&parsed.recap, &parsed.questions),
                trace,
                done: false,
            });
        }

        if let Some(p) = progress {
            let _ = p.send("🧭 Подбираю короткие варианты…".to_string());
        }
        let options = run_swarm_worker(
            llm,
            state,
            session,
            "OptionsAgent",
            SWARM_OPTIONS_PROMPT,
            &format!(
                "[brief]\n{}\n\n[available-tools]\n{}\n\n[user-request]\n{}",
                flow.brief.render(),
                tool_context,
                user_text
            ),
            true,
        )
        .await;
        let options = clean_user_text(&options);
        flow.stage = Stage::Planning;
        set_named_record(&mut flow.records, "OptionsAgent", options.clone());
        flow.awaiting_choice = Some(Stage::Planning);
        session.trip = Some(flow);
        return Ok(FlowTurn {
            reply: options,
            trace,
            done: false,
        });
    }

    let user_choice = user_text.trim();
    if !user_choice.is_empty() {
        set_named_record(
            &mut flow.records,
            "UserChoice",
            clean_user_text(user_choice),
        );
    }

    let plan = make_swarm_plan(llm, &flow.brief, &flow.records, user_choice, &tool_context).await;
    if plan.tasks.is_empty() {
        return Ok(FlowTurn {
            reply: "Не смог собрать план агентов из текущего запроса. Состояние сохранено; попробуйте ещё раз коротко подтвердить выбранный вариант.".into(),
            trace,
            done: false,
        });
    }

    for task in plan.tasks {
        let id = sanitize_record_id(&task.id, &task.agent);
        if let Some(existing) = flow.records.iter().find(|r| r.stage == id) {
            if !is_stage_unresolved(&existing.output)
                && !output_admits_unresolved_core_data(&existing.output)
            {
                continue;
            }
            flow.records.retain(|r| r.stage != id);
        }
        if task.side_effects {
            let verdict = verify_swarm_ready(llm, &flow.brief, &flow.records, &tool_context).await;
            if !verdict.ready {
                flow.awaiting_choice = None;
                session.trip = Some(flow);
                return Ok(FlowTurn {
                    reply: render_swarm_incomplete(&verdict.missing),
                    trace,
                    done: false,
                });
            }
        }
        if let Some(p) = progress {
            let _ = p.send(format!("• {}: работаю", task.agent_or_default()));
        }
        let system = build_swarm_worker_system(session, &task, &tool_context);
        let query = format!(
            "[brief]\n{}\n\n[selected-option]\n{}\n\n[prior-agent-records]\n{}\n\n[task]\n{}",
            flow.brief.render(),
            user_choice,
            render_records_clipped(&flow.records, |_| 1200),
            task.task
        );
        let output = run_swarm_worker(
            llm,
            state,
            session,
            &task.agent_or_default(),
            &system,
            &query,
            task.tools,
        )
        .await;
        if task.tools {
            tool_context = render_tool_inventory(&state.tool_inventory().await);
        }
        let output = clean_user_text(&output);
        if output_admits_unresolved_core_data(&output) || output.contains("STAGE_INCOMPLETE") {
            set_named_record(&mut flow.records, &id, output.clone());
            flow.awaiting_choice = None;
            session.trip = Some(flow);
            trace.push(format!(
                "• {}: {}",
                task.agent_or_default(),
                clip(&output, 90)
            ));
            return Ok(FlowTurn {
                reply: render_swarm_incomplete(&[clip(&output, 160)]),
                trace,
                done: false,
            });
        }
        trace.push(format!(
            "• {}: {}",
            task.agent_or_default(),
            clip(&output, 90)
        ));
        set_named_record(&mut flow.records, &id, output.clone());
        if task.checkpoint {
            session.trip = Some(flow);
            return Ok(FlowTurn {
                reply: output,
                trace,
                done: false,
            });
        }
    }

    let verdict = verify_swarm_ready(llm, &flow.brief, &flow.records, &tool_context).await;
    if !verdict.ready {
        flow.awaiting_choice = None;
        session.trip = Some(flow);
        return Ok(FlowTurn {
            reply: render_swarm_incomplete(&verdict.missing),
            trace,
            done: false,
        });
    }

    let final_answer = llm
        .complete(
            SWARM_FINAL_PROMPT,
            &format!(
                "[brief]\n{}\n\n[available-tools]\n{}\n\n[agent-records]\n{}",
                flow.brief.render(),
                tool_context,
                render_records_clipped(&flow.records, |_| 1800)
            ),
        )
        .await
        .map(|s| clean_user_text(&s))
        .unwrap_or_else(|_| fallback_compose(&flow.records));
    session.trip = None;
    Ok(FlowTurn {
        reply: final_answer,
        trace,
        done: true,
    })
}

async fn make_swarm_plan(
    llm: &Llm,
    brief: &TripBrief,
    records: &[StageRecord],
    user_choice: &str,
    tool_context: &str,
) -> SwarmPlan {
    let input = format!(
        "[brief]\n{}\n\n[user-choice]\n{}\n\n[available-tools]\n{}\n\n[records]\n{}",
        brief.render(),
        user_choice,
        tool_context,
        render_records_clipped(records, |_| 1600)
    );
    let raw = llm
        .complete(SWARM_PLANNER_PROMPT, &input)
        .await
        .unwrap_or_default();
    serde_json::from_str(&extract_json(&raw)).unwrap_or_else(|_| SwarmPlan {
        tasks: vec![
            SwarmTask {
                id: "research".into(),
                agent: "ResearchAgent".into(),
                task: "Research the selected option with the available tools and produce a concrete plan with evidence for the user's actual request.".into(),
                tools: true,
                side_effects: false,
                checkpoint: false,
            },
            SwarmTask {
                id: "verify".into(),
                agent: "VerifierAgent".into(),
                task: "Verify that the plan satisfies the user's request and list any missing evidence.".into(),
                tools: false,
                side_effects: false,
                checkpoint: false,
            },
            SwarmTask {
                id: "artifacts".into(),
                agent: "ArtifactsAgent".into(),
                task: "If the user requested external artifacts and suitable tools are available, create them. If not, report the missing capability without claiming success.".into(),
                tools: true,
                side_effects: true,
                checkpoint: false,
            },
        ],
    })
}

async fn verify_swarm_ready(
    llm: &Llm,
    brief: &TripBrief,
    records: &[StageRecord],
    tool_context: &str,
) -> SwarmVerdict {
    let input = format!(
        "[brief]\n{}\n\n[available-tools]\n{}\n\n[records]\n{}",
        brief.render(),
        tool_context,
        render_records_clipped(records, |_| 1800)
    );
    let raw = llm
        .complete(SWARM_VERIFIER_PROMPT, &input)
        .await
        .unwrap_or_default();
    serde_json::from_str(&extract_json(&raw)).unwrap_or(SwarmVerdict {
        ready: false,
        missing: vec!["верификатор не подтвердил готовность плана".into()],
    })
}

async fn run_swarm_worker(
    llm: &Llm,
    state: &BotState,
    session: &ChatSession,
    agent: &str,
    system: &str,
    query: &str,
    allow_tools: bool,
) -> String {
    let full_system = format!(
        "{system}\n\nYou are {agent}. You are one worker in a real swarm: do only your task, \
         use only the context passed to you, and return a compact handoff artifact for the next agent."
    );
    let run = async {
        if allow_tools {
            llm.answer_in_chat(
                state,
                &full_system,
                query,
                &[],
                Some(session.chat_id),
                crate::llm::STAGE_MAX_STEPS,
            )
            .await
        } else {
            llm.complete(&full_system, query).await
        }
    };
    match tokio::time::timeout(STAGE_TIMEOUT, run).await {
        Err(_) => "(stage timed out)".to_string(),
        Ok(Ok(o)) if !o.trim().is_empty() => o,
        Ok(Ok(_)) => "(no output)".to_string(),
        Ok(Err(e)) => format!("(stage failed: {e})"),
    }
}

fn build_swarm_worker_system(
    session: &ChatSession,
    task: &SwarmTask,
    tool_context: &str,
) -> String {
    let invariants = session.effective_invariants();
    let mut memory = session.memory.clone();
    memory.facts.retain(|f| f.layer != MemoryLayer::Working);
    let role = format!(
        "Worker role: {}\nTask: {}\nSide effects allowed for this worker: {}\n\n\
         Available MCP tools, discovered at runtime:\n{}\n\n\
         Rules:\n- Do not assume tools outside this inventory exist.\n\
         - If a needed capability is missing, report it instead of pretending success.\n\
         - Do not create external artifacts unless this task explicitly allows side effects.\n\
         - Do not rely on fixed activity templates; infer the method from the user's brief.",
        task.agent_or_default(),
        task.task,
        task.side_effects,
        tool_context,
    );
    super::prompt::build_system_prompt(
        &memory,
        &session.profile,
        &[],
        &invariants,
        Some(&role),
        None,
    )
}

fn render_tool_inventory(tools: &[crate::state::ToolSummary]) -> String {
    if tools.is_empty() {
        return "(no MCP tools connected)".into();
    }
    tools
        .iter()
        .map(|t| {
            let desc = if t.description.trim().is_empty() {
                ""
            } else {
                t.description.trim()
            };
            format!("- {}__{}: {}", t.server, t.name, clip(desc, 160))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn set_named_record(records: &mut Vec<StageRecord>, stage: &str, output: String) {
    if let Some(r) = records.iter_mut().find(|r| r.stage == stage) {
        r.output = output;
    } else {
        records.push(StageRecord {
            stage: stage.to_string(),
            output,
        });
    }
}

fn sanitize_record_id(id: &str, agent: &str) -> String {
    let raw = if id.trim().is_empty() { agent } else { id };
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn render_swarm_incomplete(missing: &[String]) -> String {
    let items = if missing.is_empty() {
        "• не хватает подтверждённых данных".to_string()
    } else {
        missing
            .iter()
            .take(5)
            .map(|m| format!("• {m}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "Финал ещё не готов: верификатор не подтвердил план, поэтому внешние артефакты не создаю.\n{items}\n\nМожно написать «продолжай» — рой продолжит с недостающих проверок."
    )
}

impl SwarmTask {
    fn agent_or_default(&self) -> String {
        if self.agent.trim().is_empty() {
            "WorkerAgent".into()
        } else {
            self.agent.clone()
        }
    }
}

/// Legacy fixed pipeline kept for old unit coverage and persisted states while
/// the public flow above uses the dynamic swarm runner.
#[cfg(test)]
async fn advance_legacy(
    llm: &Llm,
    state: &BotState,
    session: &mut ChatSession,
    user_text: &str,
    progress: Option<&super::ProgressSender>,
) -> Result<FlowTurn> {
    let mut brief = session
        .trip
        .as_ref()
        .map(|t| t.brief.clone())
        .unwrap_or_default();
    seed_obvious_trip_facts(&mut brief, user_text);
    let mut records = session
        .trip
        .as_ref()
        .map(|t| t.records.clone())
        .unwrap_or_default();
    let mut trace: Vec<String> = Vec::new();
    let mut recovery_retries: HashMap<Stage, usize> = HashMap::new();
    // If we paused on a checkpoint last turn, the user's message now is their
    // choice for that stage: re-run it to finalize, then auto-advance.
    let raw_awaiting = session
        .trip
        .as_ref()
        .and_then(|t| t.awaiting_choice.clone());
    let awaiting = raw_awaiting
        .clone()
        .filter(|stage| valid_awaiting_choice(stage, &records));
    if awaiting != raw_awaiting {
        if let Some(t) = session.trip.as_mut() {
            t.awaiting_choice = awaiting.clone();
        }
    }
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
        let finalizing = |stage: &Stage| {
            stage.is_checkpoint() && !user_empty && awaiting.as_ref() == Some(stage)
        };

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
        } else if user_empty {
            // Auto-advance within a turn: the pipeline is linear, so pick the
            // next unfilled stage DETERMINISTICALLY. No orchestrator LLM call —
            // it would only re-derive the same order while re-sending the brief
            // and every stage summary, burning tokens on each of the ~5 steps.
            Decision {
                next: next_exec_after(&records),
                mode: Mode::Run,
                message: String::new(),
            }
        } else {
            // The user sent a message mid-flow (not first contact, not a pending
            // checkpoint): consult the orchestrator ONCE — it may route a
            // back-step like "change the date" / "move the camp".
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
        let decision = prevent_unresolved_skip(decision, &records);

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
                let enough = parsed.ready || brief.has_minimum() || rounds >= MAX_CLARIFY_ROUNDS;
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
                        t.awaiting_choice = awaiting_if_choice_available(&stage, &records);
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
                            t.awaiting_choice = awaiting_if_choice_available(&blocker, &records);
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
                // Tell the user which step is starting BEFORE the (possibly
                // multi-minute) stage runs, so the chat is never silent.
                if let Some(p) = progress {
                    let label = stage.progress_label();
                    if !label.is_empty() {
                        let _ = p.send(label.to_string());
                    }
                }
                // On a checkpoint FINALIZE, keep this stage's own prior output
                // (the candidates the user is choosing among) as context, and
                // drop only the now-stale LATER stages. Otherwise (fresh run or
                // back-step) invalidate this stage and everything after.
                if is_final {
                    drop_after(&mut records, stage);
                } else {
                    drop_from(&mut records, stage);
                }
                // Pass the user's choice only when finalizing a checkpoint, so the
                // stage commits to the option they picked; otherwise it proposes.
                let choice = if is_final { user_text } else { "" };
                let allow_tools = !(is_final && stage.is_checkpoint());
                let mut output = run_exec_stage(
                    llm,
                    state,
                    session,
                    &brief,
                    &records,
                    stage,
                    choice,
                    allow_tools,
                )
                .await;
                let _ = session.profile.apply_inline_markers(&output);
                output = clean_user_text(&output);
                validate_stage_output(&brief, stage, &mut output);
                set_record(&mut records, stage, output.clone());
                // A failed/timed-out/incomplete essential stage is not a valid
                // hand-off to downstream agents. Retry once in this turn, then
                // pause with the bad record removed so the next turn resumes at
                // the same stage instead of asking the user to "choose" it.
                if is_stage_unresolved(&output) {
                    let attempts = recovery_retries.entry(stage.clone()).or_insert(0);
                    if *attempts < MAX_STAGE_RECOVERY_RETRIES {
                        *attempts += 1;
                        drop_from(&mut records, stage);
                        if let Some(p) = progress {
                            let _ = p.send(format!(
                                "↻ Повторяю этап {} — предыдущая попытка не дала надёжных данных",
                                stage.name()
                            ));
                        }
                        continue;
                    }
                    trace.push(format!("• {}: {}", stage.name(), clip(&output, 90)));
                    drop_from(&mut records, stage);
                    if let Some(t) = session.trip.as_mut() {
                        t.stage = stage.clone();
                        t.brief = brief.clone();
                        t.records = records.clone();
                        t.awaiting_choice = None;
                    }
                    return Ok(FlowTurn {
                        reply: if stage.is_checkpoint() {
                            render_checkpoint_stall(stage)
                        } else {
                            render_incomplete_block(stage)
                        },
                        trace,
                        done: false,
                    });
                }
                trace.push(format!("• {}: {}", stage.name(), clip(&output, 90)));
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
            t.awaiting_choice = awaiting_if_choice_available(&stage, &records);
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
#[cfg(test)]
async fn run_exec_stage(
    llm: &Llm,
    state: &BotState,
    session: &ChatSession,
    brief: &TripBrief,
    records: &[StageRecord],
    stage: &Stage,
    user_choice: &str,
    allow_tools: bool,
) -> String {
    let system = stage_system(session, stage);
    let choice = user_choice.trim();
    let query = format!(
        "[trip-brief]\n{}\n\n[prior-stages]\n{}\n\n[user-choice]\n{}\n\n\
         [your-task as the {} agent]\n{}",
        brief.render(),
        render_records_for_stage(records, stage, !choice.is_empty()),
        if choice.is_empty() {
            "(none yet)"
        } else {
            choice
        },
        stage.name(),
        stage.instruction(),
    );
    // One automatic retry ONLY for a transient LLM/HTTP error. Do NOT retry a
    // timeout or a "too many tool calls" — those just repeat the slow grind and
    // double the user's wait; the caller handles them as best-effort/incomplete.
    let mut last = run_stage_once(llm, state, &system, &query, allow_tools).await;
    if last.starts_with("(stage failed:") {
        last = run_stage_once(llm, state, &system, &query, allow_tools).await;
    }
    last
}

/// One bounded attempt at a stage's worker agent. Uses a larger tool-loop budget
/// than a normal chat turn — OSM verification (settlements, water, land) needs
/// several queries, and 12 steps died with "too many tool calls".
#[cfg(test)]
async fn run_stage_once(
    llm: &Llm,
    state: &BotState,
    system: &str,
    query: &str,
    allow_tools: bool,
) -> String {
    let run = async {
        if allow_tools {
            llm.answer_with_system(state, system, query, crate::llm::STAGE_MAX_STEPS)
                .await
        } else {
            llm.complete(system, query).await
        }
    };
    match tokio::time::timeout(STAGE_TIMEOUT, run).await {
        Err(_) => "(stage timed out)".to_string(),
        Ok(Ok(o)) if !o.trim().is_empty() => o,
        Ok(Ok(_)) => "(no output)".to_string(),
        Ok(Err(e)) => format!("(stage failed: {e})"),
    }
}

/// Compose the final user-facing plan from all stage artifacts.
#[cfg(test)]
async fn compose_final(llm: &Llm, brief: &TripBrief, records: &[StageRecord]) -> String {
    let compose_input = format!(
        "[trip-brief]\n{}\n\n[stage-outputs]\n{}",
        brief.render(),
        render_records(records),
    );
    llm.complete(COMPOSE_PROMPT, &compose_input)
        .await
        .map(|s| ensure_final_artifacts(clean_user_text(&s), brief, records))
        .unwrap_or_else(|_| {
            ensure_final_artifacts(clean_user_text(&fallback_compose(records)), brief, records)
        })
}

/// One-line-per-completed-stage listing for the orchestrator prompt.
#[cfg(test)]
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
#[cfg(test)]
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

/// Never skip over an unresolved essential stage. This protects both automatic
/// advancement and old persisted sessions where an earlier build saved
/// `STAGE_INCOMPLETE` as if the stage were done.
#[cfg(test)]
fn prevent_unresolved_skip(decision: Decision, records: &[StageRecord]) -> Decision {
    let Some(blocker) = first_unresolved_essential(records) else {
        return decision;
    };
    if decision.next.order() <= blocker.order() {
        return decision;
    }
    Decision {
        next: blocker,
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
#[cfg(test)]
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

#[cfg(test)]
fn valid_awaiting_choice(stage: &Stage, records: &[StageRecord]) -> bool {
    stage.is_checkpoint()
        && record_index(records, stage)
            .map(|i| !is_stage_unresolved(&records[i].output))
            .unwrap_or(false)
}

#[cfg(test)]
fn awaiting_if_choice_available(stage: &Stage, records: &[StageRecord]) -> Option<Stage> {
    if valid_awaiting_choice(stage, records) {
        Some(stage.clone())
    } else {
        None
    }
}

/// Honest "not done yet" note when an essential geo stage has no real data.
/// We refuse to fabricate a finished plan (calendar + doc) on top of it.
#[cfg(test)]
fn render_incomplete_block(stage: &Stage) -> String {
    let what = match stage {
        Stage::Planning => "день и место",
        Stage::Routing => "точки маршрута (заезд, остановки, выход)",
        Stage::Camp => "место ночёвки с проверкой расстояний",
        _ => "данные с карт",
    };
    format!(
        "Финал ещё не готов: не удалось получить {what} — картографический сервис \
(OSM/Overpass) сейчас перегружен и отдаёт ошибки (504/429). Я уже повторил этот шаг, \
календарь и документ на неполном плане не создаю. Состояние сохранено; следующий запрос \
продолжит с этого этапа.",
    )
}

/// Short, honest stall note for a CHECKPOINT stage whose map/weather tools did
/// not respond in time (even after a retry). No scolding, and never asks the
/// user to pick a river/area — that is the planner's job.
#[cfg(test)]
fn render_checkpoint_stall(stage: &Stage) -> String {
    let what = match stage {
        Stage::Planning => "подобрать день и место",
        Stage::Camp => "проверить стоянку по карте",
        _ => "собрать данные",
    };
    format!(
        "Картографические/погодные сервисы сейчас отвечают медленно — не успел {what}. \
Я уже повторил этот шаг; состояние сохранено, следующий запрос продолжит с этого этапа.",
    )
}

#[cfg(test)]
const COMPOSE_PROMPT: &str = "You assemble the FINAL trip plan a user will share with friends, \
from the stage outputs of a planning swarm. Write it in the user's language as a clean, \
phone-friendly PLAIN-TEXT message (no Markdown tables, no `|`, no `**`). Use short vertical \
blocks with emoji headings: chosen day + weather, the route with concrete stops, the overnight \
site (with distances), gear notes suited to the activity, and — if created — the calendar event and the \
shareable doc link. Be concrete; do not invent a doc link or coordinates that the stages did \
not produce. Preserve the user's requested activity and constraints exactly; never convert it into \
a different kind of outing. Include the full year in the chosen date. \
If a stage produced a Google Doc URL, include that exact docs.google.com URL in the final. \
If a stage returned an AUTHORIZATION URL (Google sign-in / start_google_auth), \
keep that URL VERBATIM in the final message as a clickable link with a one-line 'open and approve \
access' instruction — never drop it and never turn it into a vague 'нужен токен/доступ'. \
If a stage output shows it stalled or failed (e.g. '(stage timed out)', '(stage failed: …)'), \
do NOT fabricate that section's data: present the rest of the plan normally and add one short \
honest line that that specific part (e.g. the overnight-spot distances) is approximate / still to \
be verified — without scolding the user and without asking them to choose a specific place or \
narrow the area. Keep it tight.";

/// The OSM query rules — large, but only the map-using stages (Routing, Camp)
/// need them. Including them in Planning/Schedule/Doc just re-sends ~500 tokens
/// of irrelevant instructions on every call, so they're attached per stage.
#[cfg(test)]
const OSM_QUERY_RULES: &str = "\n\
             OSM QUERY RULES (follow EXACTLY — violating them causes Overpass HTTP 400 and wastes \
             minutes):\n\
             1. To locate a NAMED feature (a place, road, or natural feature), call \
             geocode_address FIRST to get coordinates. NEVER put a name in osm_query_bbox tags — \
             `{\"name\":\"Медведица\"}` produces an unquoted-Cyrillic selector that Overpass \
             rejects with 400. After geocoding, query a SMALL bbox around those coordinates with a \
             generic tag (e.g. {\"tourism\":\"camp_site\"}), no name filter.\n\
             2. osm_query_bbox `tags` must be a JSON object whose VALUES are single strings: \
             {\"tourism\":\"camp_site\"} or {\"place\":\"village\"}. NEVER an array value \
             (`{\"place\":[\"village\",\"hamlet\"]}` is INVALID → 400) and never a plain string \
             or array at top level. Need several values? Run separate small queries or query just \
             the key.\n\
             3. Keep every bbox small (tenths of a degree). Big bboxes time out (504).\n\
             4. HARD STOP: if the SAME goal fails twice on Overpass (400/429/504), do NOT keep \
             trying new variations — stop and finish your stage (emit STAGE_INCOMPLETE if you have \
             no real coordinates). Looping on a failing query is the main cause of multi-minute \
             silent stalls and is forbidden.\n\
             5. BE ECONOMICAL — your tool budget is limited; too many calls kill the stage. REUSE \
             coordinates already produced by earlier stages (see [prior-stages]) instead of \
             re-querying them. Pick a specific tag + SMALL bbox; NEVER sweep a broad single-key tag \
             ([tourism], [leisure], [highway], [place]) over a large bbox — those are slow and \
             waste the whole budget. Target the minimum number of queries that answers your task.";

/// Build a stage's system prompt: the layered base prompt + a stage role line.
/// Only the map stages (Routing, Camp) get the bulky OSM rules appended.
#[cfg(test)]
fn stage_system(session: &ChatSession, stage: &Stage) -> String {
    let invariants = session.effective_invariants();
    let mut memory = session.memory.clone();
    memory.facts.retain(|f| f.layer != MemoryLayer::Working);
    let mut role = format!(
        "You are the {} agent in a multi-stage trip-planning swarm. Do ONLY your stage's task, \
         building on the prior stages' outputs. Use the connected MCP tools when you need live \
         data or to act. For Russian geography with OSM tools, pass region=\"RU\" and include \
         \"Россия\" in free-text place queries. The trip brief is binding: preserve the user's \
         activity, transport mode, date availability, and hard constraints exactly. Never convert \
         the outing into a different activity mode.",
        stage.name(),
    );
    if matches!(stage, Stage::Routing | Stage::Camp) {
        role.push_str(OSM_QUERY_RULES);
    }
    super::prompt::build_system_prompt(
        &memory,
        &session.profile,
        &[],
        &invariants,
        Some(&role),
        None,
    )
}

#[cfg(test)]
fn render_records(records: &[StageRecord]) -> String {
    render_records_clipped(records, |_| HANDOFF_CLIP)
}

#[cfg(test)]
fn render_records_for_stage(records: &[StageRecord], stage: &Stage, choosing: bool) -> String {
    if choosing && stage.is_checkpoint() {
        let active = stage.name();
        return render_records_clipped(records, |record_stage| {
            if record_stage == active {
                CHECKPOINT_CHOICE_CLIP
            } else {
                HANDOFF_CLIP
            }
        });
    }
    render_records(records)
}

fn render_records_clipped(
    records: &[StageRecord],
    clip_for_stage: impl Fn(&str) -> usize,
) -> String {
    if records.is_empty() {
        return "(none yet)".to_string();
    }
    records
        .iter()
        .map(|r| {
            format!(
                "[{}]\n{}",
                r.stage,
                clip(&r.output, clip_for_stage(&r.stage))
            )
        })
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

fn clean_user_text(text: &str) -> String {
    let text = profile::strip_inline_markers(text);
    let mut out = Vec::new();
    for raw in text.lines() {
        let mut line = raw.trim_end().to_string();
        let trimmed = line.trim();
        if trimmed.is_empty() {
            out.push(String::new());
            continue;
        }
        let table_rule = trimmed.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '));
        if table_rule && trimmed.contains('|') {
            continue;
        }
        while line.trim_start().starts_with('#') {
            let without = line.trim_start().trim_start_matches('#').trim_start();
            line = without.to_string();
        }
        line = line.replace("**", "").replace("__", "").replace('`', "");
        if line.contains('|') {
            let parts = line
                .split('|')
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .collect::<Vec<_>>();
            if !parts.is_empty() {
                line = parts.join(" • ");
            }
        }
        out.push(line);
    }

    let mut cleaned = Vec::new();
    let mut blank = false;
    for line in out {
        if line.trim().is_empty() {
            if !blank {
                cleaned.push(line);
            }
            blank = true;
        } else {
            blank = false;
            cleaned.push(line);
        }
    }
    cleaned.join("\n").trim().to_string()
}

#[cfg(test)]
fn ensure_trip_year(answer: String, records: &[StageRecord]) -> String {
    let year = current_year();
    if answer.contains(&year) || !records.iter().any(|r| r.output.contains(&year)) {
        return answer;
    }
    format!("{answer}\n\n📌 Даты указаны для {year} года.")
}

#[cfg(test)]
fn ensure_final_artifacts(answer: String, _brief: &TripBrief, records: &[StageRecord]) -> String {
    let answer = ensure_trip_year(answer, records);
    if answer.contains("docs.google.com") {
        return answer;
    }
    let Some(url) = first_url_containing(records, "docs.google.com") else {
        return answer;
    };
    format!("{answer}\n\n📄 Google Doc: {url}")
}

#[cfg(test)]
fn first_url_containing(records: &[StageRecord], needle: &str) -> Option<String> {
    records
        .iter()
        .flat_map(|r| r.output.split_whitespace())
        .map(|s| {
            s.trim_matches(|c: char| {
                matches!(
                    c,
                    '"' | '\'' | ')' | '(' | ',' | '.' | ']' | '[' | '<' | '>'
                )
            })
        })
        .find(|s| s.starts_with("http") && s.contains(needle))
        .map(str::to_string)
}

#[cfg(test)]
fn current_year() -> String {
    std::env::var("AGENT_CURRENT_DATE")
        .ok()
        .and_then(|d| d.get(0..4).map(str::to_string))
        .unwrap_or_else(|| chrono::Local::now().format("%Y").to_string())
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
            {"brief":{"area":"Мещёра","date_window":"next 2 weeks"},"ready":true,"questions":[],"recap":"outdoor trip"}
            ``` trailing"#,
        );
        assert!(out.ready);
        assert_eq!(out.brief.get("area").unwrap(), "Мещёра");
        assert_eq!(out.recap, "outdoor trip");
    }

    #[test]
    fn clip_respects_char_boundary() {
        let s = "Карелия";
        let c = clip(s, 5);
        assert!(c.ends_with('…'));
    }

    #[test]
    fn clean_user_text_strips_markdown_tables_and_markers() {
        let raw =
            "## Заголовок\n| A | B |\n|---|---|\n| 1 | 2 |\n**жирно**\n⟦profile:interests=походы⟧";
        let clean = clean_user_text(raw);

        assert!(clean.contains("Заголовок"));
        assert!(clean.contains("A • B"));
        assert!(clean.contains("1 • 2"));
        assert!(clean.contains("жирно"));
        assert!(!clean.contains("|---"));
        assert!(!clean.contains("**"));
        assert!(!clean.contains("profile:"));
    }

    #[test]
    fn ensure_trip_year_restores_year_from_records() {
        let year = current_year();
        let records = vec![StageRecord {
            stage: "Schedule".into(),
            output: format!("Событие: 11 июля {year}"),
        }];

        let answer = ensure_trip_year("🗓️ 11–12 июля".into(), &records);

        assert!(answer.contains(&year));
    }

    #[test]
    fn ensure_final_artifacts_restores_doc_link_from_records() {
        let records = vec![StageRecord {
            stage: "Doc".into(),
            output: "Документ: https://docs.google.com/document/d/abc/edit".into(),
        }];

        let answer =
            ensure_final_artifacts("Документ создан.".into(), &TripBrief::default(), &records);

        assert!(answer.contains("https://docs.google.com/document/d/abc/edit"));
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

    #[test]
    fn routing_rejects_vague_unfinalized_track() {
        let b = TripBrief::default();
        let mut output =
            "Маршрут в целом понятен, но детализированный трек не зафиксирован".to_string();

        validate_stage_output(&b, &Stage::Routing, &mut output);

        assert!(output.contains("STAGE_INCOMPLETE: core route/campsite data is not concrete"));
    }

    #[test]
    fn checkpoint_choice_handoff_preserves_options() {
        let long_options = format!(
            "Вариант A — Волга\n{}\nВариант C — Ахтуба",
            "подробности ".repeat(80)
        );
        let records = vec![StageRecord {
            stage: "Planning".into(),
            output: long_options,
        }];

        let choosing = render_records_for_stage(&records, &Stage::Planning, true);
        let normal = render_records(&records);

        assert!(choosing.contains("Вариант C"));
        assert!(!normal.contains("Вариант C"));
    }

    #[test]
    fn swarm_plan_json_is_dynamic_tasks() {
        let raw = r#"{"tasks":[
            {"id":"map_research","agent":"MapAgent","task":"find places","tools":true},
            {"id":"share","agent":"ArtifactsAgent","task":"create requested artifacts","tools":true,"side_effects":true}
        ]}"#;

        let plan: SwarmPlan = serde_json::from_str(raw).unwrap();

        assert_eq!(plan.tasks.len(), 2);
        assert_eq!(plan.tasks[0].agent, "MapAgent");
        assert!(plan.tasks[1].side_effects);
    }

    #[test]
    fn tool_inventory_render_is_safe_metadata_only() {
        let tools = vec![crate::state::ToolSummary {
            server: "maps".into(),
            name: "geocode_address".into(),
            description: "Find coordinates for a place".into(),
        }];

        let rendered = render_tool_inventory(&tools);

        assert!(rendered.contains("maps__geocode_address"));
        assert!(rendered.contains("Find coordinates"));
        assert!(!rendered.contains("TOKEN"));
    }

    #[test]
    fn swarm_worker_system_uses_runtime_inventory_not_activity_templates() {
        let session = ChatSession::new(1);
        let task = SwarmTask {
            id: "research".into(),
            agent: "ResearchAgent".into(),
            task: "Investigate the selected outdoor option".into(),
            tools: true,
            side_effects: false,
            checkpoint: false,
        };

        let system = build_swarm_worker_system(&session, &task, "- maps__geocode_address: x");

        assert!(system.contains("Available MCP tools, discovered at runtime"));
        assert!(system.contains("maps__geocode_address"));
        assert!(!system.contains("kayak"));
        assert!(!system.contains("cycling route"));
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
            "start 48.6,43.5 stop 48.6,43.6".into(),
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
    fn next_exec_retries_unresolved_routing_before_camp() {
        let mut r = Vec::new();
        set_record(&mut r, &Stage::Planning, "Пятница, Дон".into());
        set_record(
            &mut r,
            &Stage::Routing,
            "Не смог получить точки.\nSTAGE_INCOMPLETE: Overpass 504".into(),
        );

        assert_eq!(next_exec_after(&r), Stage::Routing);
    }

    #[test]
    fn guard_prevents_skipping_unresolved_routing() {
        let mut r = Vec::new();
        set_record(&mut r, &Stage::Planning, "Пятница, Дон".into());
        set_record(
            &mut r,
            &Stage::Routing,
            "STAGE_INCOMPLETE: Overpass 504".into(),
        );
        let d = prevent_unresolved_skip(
            Decision {
                next: Stage::Camp,
                mode: Mode::Run,
                message: String::new(),
            },
            &r,
        );

        assert_eq!(d.next, Stage::Routing);
        assert_eq!(d.mode, Mode::Run);
    }

    #[test]
    fn non_checkpoint_stage_is_never_awaiting_choice() {
        let mut r = Vec::new();
        set_record(&mut r, &Stage::Planning, "option 1".into());
        set_record(&mut r, &Stage::Routing, "route 48.6,43.5".into());

        assert!(valid_awaiting_choice(&Stage::Planning, &r));
        assert!(!valid_awaiting_choice(&Stage::Routing, &r));
        assert_eq!(awaiting_if_choice_available(&Stage::Routing, &r), None);
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

    #[test]
    fn stage_system_does_not_inject_working_memory() {
        let mut session = ChatSession::new(1);
        session
            .memory
            .upsert_fact("home_city", "Волгоград", MemoryLayer::LongTerm);
        session
            .memory
            .upsert_fact("trip_focus", "bbq", MemoryLayer::Working);

        let system = stage_system(&session, &Stage::Routing);

        assert!(system.contains("[memory:long-term]"));
        assert!(system.contains("home_city: Волгоград"));
        assert!(!system.contains("[memory:working]"));
        assert!(!system.contains("trip_focus"));
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
        p.set("interests", "походы");
        let ctx = profile_context(&p);
        assert!(ctx.contains("home_city: Москва"));
        assert!(ctx.contains("interests: походы"));
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

    #[test]
    fn drop_after_keeps_finalizing_stage_drops_only_later() {
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
        // finalizing Camp keeps its own candidates, drops only Schedule/Doc
        drop_after(&mut r, &Stage::Camp);
        let names: Vec<&str> = r.iter().map(|x| x.stage.as_str()).collect();
        assert_eq!(names, vec!["Planning", "Routing", "Camp"]);
    }
}
