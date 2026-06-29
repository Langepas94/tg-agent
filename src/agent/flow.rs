//! Stateful multi-agent trip-planning flow (the "swarm").
//!
//! The public runner is a dynamic swarm: BriefAgent extracts the request,
//! OptionsAgent offers concise alternatives, SwarmPlanner builds worker tasks
//! from the live MCP tool inventory, WorkerAgents execute isolated tasks,
//! VerifierAgent gates side effects, and FinalAgent composes the answer.
//!
//! The flow SUSPENDS across user turns: BriefAgent interrogates first, building
//! a `TripBrief` persisted in the chat session, then the planner-driven swarm
//! runs to a verified final answer. Each agent is a separate `SwarmAgentSpec`
//! (own role, permissions, and model — overridable per agent via
//! `SWARM_MODEL_<AGENT>` or per task). The old fixed `Stage` labels are retained
//! only as serialized-state compatibility shells; they no longer drive the flow.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{llm::Llm, state::BotState};

use super::{memory::MemoryLayer, profile, session::ChatSession};

/// Hard cap on clarify rounds — after this we plan with whatever we have, so the
/// bot never interrogates forever.
const MAX_CLARIFY_ROUNDS: u8 = 3;

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
    /// by the swarm's readiness fallback when BriefAgent does not set `ready`
    /// (key-name heuristic).
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
// into the flow, and BriefAgent decides, by meaning, when the brief is ready to
// plan. See `router.rs` and `SWARM_BRIEF_PROMPT`.

// ---------------------------------------------------------------------------
// Clarify agent
// ---------------------------------------------------------------------------

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

/// Recent user turns joined oldest→newest, so BriefAgent always sees the full
/// request even across clarify rounds (the structured brief may capture it only
/// partially). Assistant turns are excluded to keep the brief input focused.
fn recent_user_requests(session: &ChatSession) -> String {
    let joined = session
        .memory
        .recent
        .iter()
        .filter(|(role, _)| role == "user")
        .map(|(_, text)| text.trim())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n---\n");
    if joined.is_empty() {
        "(none)".to_string()
    } else {
        joined
    }
}

/// Seed the start area from the known profile (home city) when the brief has no
/// location yet, so BriefAgent never asks for a place the profile already holds
/// and downstream agents always have a start point. Location-only — makes no
/// assumption about the activity.
fn seed_area_from_profile(brief: &mut TripBrief, profile: &super::profile::UserProfile) {
    let area_needles = [
        "area",
        "region",
        "start",
        "location",
        "место",
        "регион",
        "старт",
        "город",
    ];
    let has_area = brief
        .fields
        .iter()
        .any(|(k, v)| !v.trim().is_empty() && area_needles.iter().any(|n| k.contains(n)));
    if has_area {
        return;
    }
    let home_needles = ["home_city", "home", "city", "город", "родной"];
    if let Some((_, city)) = profile
        .fields
        .iter()
        .find(|(k, v)| !v.trim().is_empty() && home_needles.iter().any(|n| k.contains(n)))
    {
        brief
            .fields
            .insert("start_area".into(), city.trim().to_string());
    }
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
// Record helpers (replace-by-stage + downstream invalidation for back-steps)
// ---------------------------------------------------------------------------

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

fn should_restart_flow_from_message(flow: &TripFlowState, text: &str) -> bool {
    if flow.records.is_empty() {
        return false;
    }
    if message_requests_continuation(text) {
        return false;
    }
    message_requests_restart(text) || looks_like_fresh_full_request(text)
}

fn message_requests_restart(text: &str) -> bool {
    let low = text.to_lowercase();
    contains_any(
        &low,
        &[
            "начнем заново",
            "начнём заново",
            "давай заново",
            "сначала",
            "новый запрос",
            "перезапусти",
            "restart",
            "start over",
        ],
    )
}

fn message_requests_continuation(text: &str) -> bool {
    let low = text.trim().to_lowercase();
    if low.chars().count() > 120 {
        return false;
    }
    contains_any(
        &low,
        &[
            "продолж",
            "дальше",
            "подтверждаю",
            "выбираю",
            "первый",
            "второй",
            "третий",
            "вариант",
            "создай",
            "создавай",
            "давай",
            "ок",
            "ok",
            "continue",
            "go on",
        ],
    )
}

fn looks_like_fresh_full_request(text: &str) -> bool {
    let low = text.trim().to_lowercase();
    let chars = low.chars().count();
    if chars < 140 {
        return false;
    }
    let sentence_marks = low
        .chars()
        .filter(|c| matches!(c, '.' | '?' | '!' | '\n'))
        .count();
    let request_hits = count_contains(
        &low,
        &[
            "хотим",
            "хочу",
            "сходить",
            "поход",
            "сплав",
            "байдар",
            "маршрут",
            "план",
            "точк",
            "мест",
            "trip",
            "plan",
            "route",
        ],
    );
    let constraint_hits = count_contains(
        &low,
        &[
            "выходн",
            "ноч",
            "палат",
            "шашлык",
            "календар",
            "док",
            "радиус",
            "км",
            "метр",
            "вода",
            "weekend",
            "calendar",
            "doc",
        ],
    );
    sentence_marks >= 2 && request_hits >= 2 && constraint_hits >= 2
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn count_contains(text: &str, needles: &[&str]) -> usize {
    needles
        .iter()
        .filter(|needle| text.contains(**needle))
        .count()
}

fn is_data_checkpoint_task(task: &SwarmTask) -> bool {
    task.tools && !task.side_effects
}

fn progress_message_for_task(task: &SwarmTask) -> String {
    let text = format!("{} {} {}", task.id, task.agent, task.task).to_lowercase();
    let msg = if task.side_effects
        || contains_any(
            &text,
            &["calendar", "календар", "doc", "document", "google", "док"],
        ) {
        "• Готовлю календарь или документ…"
    } else if contains_any(
        &text,
        &[
            "weather",
            "forecast",
            "погод",
            "ветер",
            "осад",
            "дат",
            "weekend",
        ],
    ) {
        "• Сравниваю ближайшие выходные по погоде…"
    } else if contains_any(
        &text,
        &[
            "camp",
            "campsite",
            "стоян",
            "лагер",
            "палат",
            "ноч",
            "water",
            "вода",
            "сел",
            "турбаз",
        ],
    ) {
        "• Проверяю стоянку: вода рядом и нет жилья поблизости…"
    } else if contains_any(
        &text,
        &[
            "route",
            "track",
            "point",
            "map",
            "osm",
            "маршрут",
            "точк",
            "карт",
        ],
    ) {
        "• Подбираю точки маршрута и подъезды…"
    } else {
        "• Проверяю один блок плана…"
    };
    msg.to_string()
}

// ---------------------------------------------------------------------------
// Orchestrated turn
// ---------------------------------------------------------------------------

/// Safety net for ONE stage attempt that is genuinely stuck (a stalled external
/// tool connection), not a planning budget. A normal OSM/weather stage finishes
/// well under this, but a constraint-heavy stage (e.g. a campsite search that
/// must check water proximity AND settlement isolation across several Overpass
/// queries) on a slow host needs more headroom before we call it stuck.
const STAGE_TIMEOUT: Duration = Duration::from_secs(300);
/// Keep Telegram turns interactive: one heavy data-gathering worker per user
/// message, then return a useful checkpoint and wait for "продолжай".
const DATA_TASKS_PER_TURN: usize = 1;

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
    model: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct SwarmAgentSpec {
    name: String,
    role: String,
    model: String,
    tools_allowed: bool,
    side_effects_allowed: bool,
}

#[derive(Debug, Clone)]
struct SwarmAgentRegistry {
    agents: BTreeMap<String, SwarmAgentSpec>,
}

impl SwarmAgentRegistry {
    fn from_env(default_model: &str) -> Self {
        let mut registry = Self::with_default_model(default_model);
        for spec in registry.agents.values_mut() {
            if let Ok(model) = std::env::var(agent_model_env_key(&spec.name)) {
                if !model.trim().is_empty() {
                    spec.model = model;
                }
            }
        }
        registry
    }

    fn with_default_model(default_model: &str) -> Self {
        let mut agents = BTreeMap::new();
        for spec in [
            SwarmAgentSpec {
                name: "BriefAgent".into(),
                role: "Extracts the request into an activity-agnostic brief and asks only blocking questions.".into(),
                model: default_model.into(),
                tools_allowed: false,
                side_effects_allowed: false,
            },
            SwarmAgentSpec {
                name: "OptionsAgent".into(),
                role: "Compares concise candidate options before any deep research.".into(),
                model: default_model.into(),
                tools_allowed: true,
                side_effects_allowed: false,
            },
            SwarmAgentSpec {
                name: "SwarmPlanner".into(),
                role: "Creates a task graph from the brief, selected option, records, and runtime tool inventory.".into(),
                model: default_model.into(),
                tools_allowed: false,
                side_effects_allowed: false,
            },
            SwarmAgentSpec {
                name: "VerifierAgent".into(),
                role: "Checks whether gathered evidence satisfies the user's actual request before side effects or final delivery.".into(),
                model: default_model.into(),
                tools_allowed: false,
                side_effects_allowed: false,
            },
            SwarmAgentSpec {
                name: "ArtifactsAgent".into(),
                role: "Creates only explicitly requested external artifacts, only after verifier approval and only with real tools.".into(),
                model: default_model.into(),
                tools_allowed: true,
                side_effects_allowed: true,
            },
            SwarmAgentSpec {
                name: "FinalAgent".into(),
                role: "Composes the final user-facing answer from verified records and tool results.".into(),
                model: default_model.into(),
                tools_allowed: false,
                side_effects_allowed: false,
            },
            SwarmAgentSpec {
                name: "WorkerAgent".into(),
                role: "Executes one narrow planner-assigned research or reasoning task with isolated context.".into(),
                model: default_model.into(),
                tools_allowed: true,
                side_effects_allowed: false,
            },
        ] {
            agents.insert(spec.name.clone(), spec);
        }
        Self { agents }
    }

    #[cfg(test)]
    fn with_model_overrides(default_model: &str, overrides: &[(&str, &str)]) -> Self {
        let mut registry = Self::with_default_model(default_model);
        for (agent, model) in overrides {
            if let Some(spec) = registry.agents.get_mut(*agent) {
                if !model.trim().is_empty() {
                    spec.model = (*model).to_string();
                }
            }
        }
        registry
    }

    fn get(&self, name: &str) -> SwarmAgentSpec {
        self.agents
            .get(name)
            .cloned()
            .unwrap_or_else(|| SwarmAgentSpec {
                name: if name.trim().is_empty() {
                    "WorkerAgent".into()
                } else {
                    name.to_string()
                },
                role: "Executes one planner-assigned swarm task with isolated context.".into(),
                model: self
                    .agents
                    .get("WorkerAgent")
                    .map(|s| s.model.clone())
                    .unwrap_or_else(|| "default".into()),
                tools_allowed: true,
                side_effects_allowed: false,
            })
    }

    fn for_task(&self, task: &SwarmTask) -> SwarmAgentSpec {
        let name = task.agent_or_default();
        let mut spec = self.get(&name);
        if let Some(model) = task.model.as_deref().filter(|m| !m.trim().is_empty()) {
            spec.model = model.to_string();
        }
        spec.tools_allowed = spec.tools_allowed && task.tools;
        spec.side_effects_allowed = spec.side_effects_allowed && task.side_effects;
        spec
    }

    fn task_requires_verifier_gate(&self, task: &SwarmTask) -> bool {
        task.side_effects || self.get(&task.agent_or_default()).side_effects_allowed
    }
}

fn agent_model_env_key(agent: &str) -> String {
    let suffix = agent
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("SWARM_MODEL_{suffix}")
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
actual activity and constraints, whatever they are. If the request has a date window/weekend \
constraint, include the concrete candidate weekend date(s) and a short weather/daylight rationale \
when tools allow it. \
If the user has NOT fixed a specific spot/route and gave only a broad area or region, the options \
MUST be genuinely DIFFERENT places (e.g. different rivers, lakes, trailheads), never date or style \
variants of a single spot — use the geo tools to discover real candidates within the user's \
travel-time/area limit and present every genuinely distinct, suitable one that exists. Let the \
count follow the terrain: offer as few or as many as are truly distinct and worthwhile; never pad \
to a number and never collapse to one when alternatives exist. If the user DID name a specific \
spot, offer distinct variants of that spot instead. Give each candidate's approximate coordinates \
as `lat, lon`. End by asking the user to choose one option. Plain text, no Markdown table.";

const SWARM_PLANNER_PROMPT: &str = "You are SwarmPlanner. Build a minimal swarm plan from the \
brief, the user's chosen option, prior records, and the ACTUAL connected MCP tool inventory. Do \
not assume calendar/docs/maps/weather tools exist; inspect the inventory. Do not use fixed \
activity templates. Create separate worker tasks with narrow responsibilities and isolated context. \
Include a side_effects=true task only for external artifacts explicitly requested by the user AND \
only if plausible tools are present or connectable; otherwise include a non-side-effect task that \
explains the missing capability. Return ONLY JSON: {\"tasks\":[{\"id\":\"short_snake\",\
\"agent\":\"AgentName\",\"task\":\"specific task\",\"tools\":true,\"side_effects\":false,\
\"checkpoint\":false,\"model\":\"optional-model-override\"}]}. The plan must include a \
verification task before any side-effect task.";

const SWARM_VERIFIER_PROMPT: &str = "You are VerifierAgent. Decide whether the gathered evidence is \
concrete enough to perform external side effects or present the final plan. Judge ONLY against the \
user's OWN stated deliverables and constraints — do NOT invent extra requirements or demand a \
precision the user never asked for. ready=true when each deliverable the user actually requested is \
present as concrete data (a named place plus coordinates where the user asked for points, a chosen \
date, and every explicit constraint addressed), EVEN IF further optional detail could still be \
added. Set ready=false only when something the user explicitly asked for is still vague, missing, \
contradicted, or merely 'to be checked later'. Do not block on nice-to-have sub-details \
(return-leg minutiae, ferry crossings, parking) the user did not request. Return ONLY JSON: \
{\"ready\":bool,\"missing\":[\"short missing item the USER asked for\",...]}";

const SWARM_FINAL_PROMPT: &str =
    "You are FinalAgent. Compose the user-facing answer from the swarm \
records. Preserve the selected option, concrete evidence, tool results, and artifact links. Do not \
claim an external artifact was created unless a tool result says so. \
Write ONLY for the end user: never mention agents, a 'swarm', verification stages, or 'next steps \
for other agents' — the user must not see how the work is split up. If a requested artifact \
(calendar event, document, …) was created, give its link; if it still needs doing, say plainly what \
YOU will do next or what you need FROM the user (access or a one-word confirmation), as a normal \
request — not as a task list for other agents. \
Always show wind speed in metres per second (м/с), never km/h (convert ÷ 3.6, round). \
If a requested capability was missing, say that plainly. Plain Telegram-friendly text, no Markdown \
tables.";

async fn advance_swarm(
    llm: &Llm,
    state: &BotState,
    session: &mut ChatSession,
    user_text: &str,
    progress: Option<&super::ProgressSender>,
) -> Result<FlowTurn> {
    let registry = SwarmAgentRegistry::from_env(llm.model());
    let tools = state.tool_inventory().await;
    let mut tool_context = render_tool_inventory(&tools);
    let mut flow = session.trip.clone().unwrap_or_else(TripFlowState::start);
    let mut trace = Vec::new();

    if should_restart_flow_from_message(&flow, user_text) {
        trace.push("active flow restarted from a fresh full request".to_string());
        flow = TripFlowState::start();
    }

    // The first agent call (BriefAgent) can itself take many seconds on a slow
    // model, so announce the swarm is working BEFORE it — otherwise the chat is
    // silent until the first stage boundary.
    if let Some(p) = progress {
        let _ = p.send("🔎 Разбираю запрос…".to_string());
    }

    if flow.records.is_empty() {
        // Preserve the user's ORIGINAL request verbatim so no later agent ever
        // loses it across clarify rounds — the structured brief may capture it
        // only partially, but the raw request always flows to Options/Planner.
        if flow
            .brief
            .fields
            .get("request")
            .map(|s| s.trim().is_empty())
            .unwrap_or(true)
            && !user_text.trim().is_empty()
        {
            flow.brief
                .fields
                .insert("request".into(), user_text.trim().to_string());
        }
        // Seed the start area from the known profile (home city) so BriefAgent
        // never interrogates for a location the profile already holds.
        seed_area_from_profile(&mut flow.brief, &session.profile);

        let input = format!(
            "Known profile:\n{}\n\nConversation so far (user turns, oldest first):\n{}\n\n\
             Brief gathered so far:\n{}\n\nLatest message:\n{}",
            profile_context(&session.profile),
            recent_user_requests(session),
            flow.brief.render(),
            user_text
        );
        let brief_agent = registry.get("BriefAgent");
        let parsed = parse_clarify(
            &complete_swarm_agent(llm, &brief_agent, SWARM_BRIEF_PROMPT, &input).await?,
        );
        flow.brief.merge(parsed.brief);
        flow.clarify_rounds = flow.clarify_rounds.saturating_add(1);
        // Proceed when the model says ready, when the heuristic minimum is met,
        // after the round cap, OR when we already hold a real request and the
        // model has nothing more to ask — never stall on a detailed opener.
        let have_request = flow
            .brief
            .fields
            .get("request")
            .map(|s| s.trim().chars().count() > 12)
            .unwrap_or(false);
        let ready = parsed.ready
            || flow.brief.has_minimum()
            || flow.clarify_rounds >= MAX_CLARIFY_ROUNDS
            || (have_request && parsed.questions.is_empty());
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
        let options_agent = registry.get("OptionsAgent");
        let options = run_swarm_worker(
            llm,
            state,
            session,
            &options_agent,
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

    if let Some(p) = progress {
        let _ = p.send("🧠 Планирую маршрут…".to_string());
    }
    let plan = make_swarm_plan(
        llm,
        &registry,
        &flow.brief,
        &flow.records,
        user_choice,
        &tool_context,
    )
    .await;
    if plan.tasks.is_empty() {
        return Ok(FlowTurn {
            reply: "Не смог собрать план агентов из текущего запроса. Состояние сохранено; попробуйте ещё раз коротко подтвердить выбранный вариант.".into(),
            trace,
            done: false,
        });
    }

    // True once we skip a side-effecting task because the verifier wasn't ready:
    // we keep planning/answering but must not mark the turn done (no artifacts yet).
    let mut artifacts_skipped = false;
    let tasks = plan.tasks;
    let task_count = tasks.len();
    let mut completed_data_tasks_this_turn = 0usize;
    for (task_index, task) in tasks.into_iter().enumerate() {
        let id = sanitize_record_id(&task.id, &task.agent);
        let task_agent = registry.for_task(&task);
        if let Some(existing) = flow.records.iter().find(|r| r.stage == id) {
            if !worker_output_needs_best_effort(&existing.output) {
                continue;
            }
            flow.records.retain(|r| r.stage != id);
        }
        if registry.task_requires_verifier_gate(&task) {
            let verdict =
                verify_swarm_ready(llm, &registry, &flow.brief, &flow.records, &tool_context).await;
            if !verdict.ready {
                // Verifier not satisfied: skip THIS side-effecting task (no gdoc /
                // calendar on unconfirmed data) but keep going — the user still
                // gets a draft from what was gathered, never a bare "not ready".
                trace.push(format!(
                    "verifier-gate not ready: {}",
                    verdict.missing.join("; ")
                ));
                artifacts_skipped = true;
                continue;
            }
        }
        if let Some(p) = progress {
            // Neutral liveness only — never leak internal agent names to the user.
            let _ = p.send(progress_message_for_task(&task));
        }
        let system = build_swarm_worker_system(session, &task, &task_agent, &tool_context);
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
            &task_agent,
            &system,
            &query,
            task.tools,
        )
        .await;
        if task.tools {
            tool_context = render_tool_inventory(&state.tool_inventory().await);
        }
        let output = clean_user_text(&output);
        if worker_output_needs_best_effort(&output) {
            set_named_record(&mut flow.records, &id, output.clone());
            flow.awaiting_choice = None;
            trace.push(format!("• {}: {}", task_agent.name, clip(&output, 90)));
            // A worker couldn't get every concrete value — hand back a draft of
            // what we DO have (places, dates, stops) instead of stranding the user.
            let draft =
                compose_best_effort(llm, &registry, &flow.brief, &flow.records, &tool_context)
                    .await;
            session.trip = Some(flow);
            return Ok(FlowTurn {
                reply: draft,
                trace,
                done: false,
            });
        }
        trace.push(format!("• {}: {}", task_agent.name, clip(&output, 90)));
        set_named_record(&mut flow.records, &id, output.clone());
        if is_data_checkpoint_task(&task) {
            completed_data_tasks_this_turn += 1;
        }
        if task.checkpoint {
            session.trip = Some(flow);
            return Ok(FlowTurn {
                reply: output,
                trace,
                done: false,
            });
        }
        let has_more_tasks = task_index + 1 < task_count;
        if completed_data_tasks_this_turn >= DATA_TASKS_PER_TURN && has_more_tasks {
            let checkpoint = compose_progress_checkpoint(
                llm,
                &registry,
                &flow.brief,
                &flow.records,
                &tool_context,
            )
            .await;
            session.trip = Some(flow);
            return Ok(FlowTurn {
                reply: checkpoint,
                trace,
                done: false,
            });
        }
    }

    let verdict =
        verify_swarm_ready(llm, &registry, &flow.brief, &flow.records, &tool_context).await;
    if !verdict.ready {
        // Don't strand the user: hand back a concrete draft of what was gathered
        // (places, dates, stops) flagged as unconfirmed, with no artifacts created.
        flow.awaiting_choice = None;
        trace.push(format!(
            "final verify not ready: {}",
            verdict.missing.join("; ")
        ));
        let draft =
            compose_best_effort(llm, &registry, &flow.brief, &flow.records, &tool_context).await;
        session.trip = Some(flow);
        return Ok(FlowTurn {
            reply: draft,
            trace,
            done: false,
        });
    }

    if let Some(p) = progress {
        let _ = p.send("📝 Собираю финальный план…".to_string());
    }
    let final_agent = registry.get("FinalAgent");
    let final_answer = complete_swarm_agent(
        llm,
        &final_agent,
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
    if artifacts_skipped {
        // Data is confirmed now, but the side-effect step was skipped this turn —
        // keep the flow open so «продолжай» creates the gdoc / calendar event.
        session.trip = Some(flow);
        return Ok(FlowTurn {
            reply: final_answer,
            trace,
            done: false,
        });
    }
    session.trip = None;
    Ok(FlowTurn {
        reply: final_answer,
        trace,
        done: true,
    })
}

async fn make_swarm_plan(
    llm: &Llm,
    registry: &SwarmAgentRegistry,
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
    let planner = registry.get("SwarmPlanner");
    let raw = complete_swarm_agent(llm, &planner, SWARM_PLANNER_PROMPT, &input)
        .await
        .unwrap_or_default();
    serde_json::from_str(&extract_json(&raw)).unwrap_or_else(|_| SwarmPlan {
        tasks: vec![
            SwarmTask {
                id: "research".into(),
                agent: "ResearchAgent".into(),
                model: None,
                task: "Research the selected option with the available tools and produce a concrete plan with evidence for the user's actual request.".into(),
                tools: true,
                side_effects: false,
                checkpoint: false,
            },
            SwarmTask {
                id: "verify".into(),
                agent: "VerifierAgent".into(),
                model: None,
                task: "Verify that the plan satisfies the user's request and list any missing evidence.".into(),
                tools: false,
                side_effects: false,
                checkpoint: false,
            },
            SwarmTask {
                id: "artifacts".into(),
                agent: "ArtifactsAgent".into(),
                model: None,
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
    registry: &SwarmAgentRegistry,
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
    let verifier = registry.get("VerifierAgent");
    let raw = complete_swarm_agent(llm, &verifier, SWARM_VERIFIER_PROMPT, &input)
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
    agent: &SwarmAgentSpec,
    system: &str,
    query: &str,
    allow_tools: bool,
) -> String {
    let full_system = format!(
        "{system}\n\nYou are {}. You are one worker in a real swarm: do only your task, \
         use only the context passed to you, and return a compact handoff artifact for the next \
         agent.\n\
         COMMIT, do not narrate: your output MUST be the finished artifact with concrete values \
         (names, coordinates, distances, dates), not a description of what you are about to do. \
         Never end with a promise like 'I will call…', 'сейчас вызову', 'уточню', 'собираю' — that \
         is not a valid handoff. Make the tool calls now and return the result. If after your tool \
         calls a required concrete value is still missing, output the marker \
         `STAGE_INCOMPLETE: <what is missing>` plus the concrete data you DID obtain — never a vague \
         plan.",
        agent.name
    );
    let run = async {
        if allow_tools && agent.tools_allowed {
            llm.answer_in_chat_with_model(
                state,
                &full_system,
                query,
                &[],
                Some(session.chat_id),
                crate::llm::STAGE_MAX_STEPS,
                Some(&agent.model),
            )
            .await
        } else {
            llm.complete_with_model(&full_system, query, Some(&agent.model))
                .await
        }
    };
    let started = Instant::now();
    let result = tokio::time::timeout(STAGE_TIMEOUT, run).await;
    let ms = started.elapsed().as_millis();
    // Per-stage timing so a `(stage timed out)` names WHICH worker was slow and
    // for how long — the only way to tell an LLM tool-loop from a hung MCP call.
    match &result {
        Err(_) => tracing::warn!(
            agent = %agent.name, model = %agent.model, tools = allow_tools && agent.tools_allowed,
            elapsed_ms = ms, "swarm worker TIMED OUT (limit {}s)", STAGE_TIMEOUT.as_secs()
        ),
        Ok(Err(e)) => tracing::warn!(
            agent = %agent.name, model = %agent.model, elapsed_ms = ms,
            "swarm worker failed: {e}"
        ),
        Ok(Ok(_)) => tracing::debug!(
            agent = %agent.name, model = %agent.model, tools = allow_tools && agent.tools_allowed,
            elapsed_ms = ms, "swarm worker done"
        ),
    }
    match result {
        Err(_) => "(stage timed out)".to_string(),
        Ok(Ok(o)) if !o.trim().is_empty() => o,
        Ok(Ok(_)) => "(no output)".to_string(),
        Ok(Err(e)) => format!("(stage failed: {e})"),
    }
}

fn build_swarm_worker_system(
    session: &ChatSession,
    task: &SwarmTask,
    agent: &SwarmAgentSpec,
    tool_context: &str,
) -> String {
    let invariants = session.effective_invariants();
    let mut memory = session.memory.clone();
    memory.facts.retain(|f| f.layer != MemoryLayer::Working);
    let role = format!(
        "Agent identity: {}\nConfigured model: {}\nAgent role: {}\nTask: {}\n\
         Tools allowed for this agent: {}\nSide effects allowed for this worker: {}\n\n\
         Available MCP tools, discovered at runtime:\n{}\n\n\
         Rules:\n- Do not assume tools outside this inventory exist.\n\
         - If a needed capability is missing, report it instead of pretending success.\n\
         - Do not create external artifacts unless this task explicitly allows side effects.\n\
         - Do not rely on fixed activity templates; infer the method from the user's brief.",
        agent.name,
        agent.model,
        agent.role,
        task.task,
        agent.tools_allowed,
        agent.side_effects_allowed,
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

/// Build the full system prompt for a tool-less swarm agent: its base role
/// prompt plus its own identity/model/role stamp. Pure (no I/O) so the
/// per-agent prompt boundary can be asserted in unit tests — each agent must
/// see only its own job, never another agent's identity or stage instructions.
fn swarm_agent_system(agent: &SwarmAgentSpec, base_prompt: &str) -> String {
    format!(
        "{base_prompt}\n\nAgent identity: {}\nConfigured model: {}\nAgent role: {}\n\
         This is a distinct swarm agent. Do only this agent's job and return a compact handoff.",
        agent.name, agent.model, agent.role
    )
}

async fn complete_swarm_agent(
    llm: &Llm,
    agent: &SwarmAgentSpec,
    system: &str,
    input: &str,
) -> Result<String> {
    let system = swarm_agent_system(agent, system);
    let started = Instant::now();
    let out = llm
        .complete_with_model(&system, input, Some(&agent.model))
        .await;
    let ms = started.elapsed().as_millis();
    match &out {
        Ok(_) => {
            tracing::debug!(agent = %agent.name, model = %agent.model, elapsed_ms = ms, "swarm agent done")
        }
        Err(e) => {
            tracing::warn!(agent = %agent.name, model = %agent.model, elapsed_ms = ms, "swarm agent failed: {e}")
        }
    }
    out
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

/// Compose a best-effort DRAFT when the verifier could not confirm the plan or
/// a worker stage came back incomplete. We never strand the user with a bare
/// "not ready" notice: surface the concrete places/dates/stops already gathered,
/// flag what is unconfirmed, and create NO external artifacts.
async fn compose_best_effort(
    llm: &Llm,
    registry: &SwarmAgentRegistry,
    brief: &TripBrief,
    records: &[StageRecord],
    tool_context: &str,
) -> String {
    let final_agent = registry.get("FinalAgent");
    let system = format!(
        "{SWARM_FINAL_PROMPT}\n\nThe plan is NOT fully verified yet. Produce a concrete, useful \
         DRAFT the user can act on right now: give the candidate place(s), date(s) and stops you \
         DO have with their real values (names, coordinates, distances), and mark anything \
         unconfirmed with «(не подтверждено)». Never reply that nothing is ready. Create NO \
         external artifacts and do not claim any were created. End with one short line inviting \
         the user to reply «продолжай» to finish the remaining checks."
    );
    complete_swarm_agent(
        llm,
        &final_agent,
        &system,
        &format!(
            "[brief]\n{}\n\n[available-tools]\n{}\n\n[agent-records]\n{}",
            brief.render(),
            tool_context,
            render_records_clipped(records, |_| 1800)
        ),
    )
    .await
    .map(|s| clean_user_text(&s))
    .unwrap_or_else(|_| fallback_compose(records))
}

async fn compose_progress_checkpoint(
    llm: &Llm,
    registry: &SwarmAgentRegistry,
    brief: &TripBrief,
    records: &[StageRecord],
    tool_context: &str,
) -> String {
    let final_agent = registry.get("FinalAgent");
    let system = format!(
        "{SWARM_FINAL_PROMPT}\n\nThis is an INTERMEDIATE checkpoint after one data-gathering \
         worker. Produce a useful partial result now, in the user's language. Include concrete \
         dates, places, coordinates, and constraints already found. Do not pretend the whole flow \
         is finished. Do not claim any calendar event or document was created. End with one short \
         line: ask the user to reply «продолжай» to run the next check/create artifacts."
    );
    complete_swarm_agent(
        llm,
        &final_agent,
        &system,
        &format!(
            "[brief]\n{}\n\n[available-tools]\n{}\n\n[agent-records]\n{}",
            brief.render(),
            tool_context,
            render_records_clipped(records, |_| 1800)
        ),
    )
    .await
    .map(|s| clean_user_text(&s))
    .unwrap_or_else(|_| fallback_compose(records))
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

fn is_stage_failure(output: &str) -> bool {
    let text = output.trim();
    text.starts_with("Stopped after too many tool calls")
        || text.starts_with("(stage failed:")
        || text.starts_with("(stage timed out")
}

/// Byte length of the UTF-8 char that begins with `first`.
fn utf8_len(first: u8) -> usize {
    match first {
        b if b < 0x80 => 1,
        b if b >> 5 == 0b110 => 2,
        b if b >> 4 == 0b1110 => 3,
        _ => 4,
    }
}

/// Parse a signed decimal like `50.1347` or `-43.5` starting exactly at `start`.
/// Returns the matched text and its end byte index. Requires a fractional part
/// and an integer part of 1–3 digits, so plain integers and version-like tokens
/// never match.
fn parse_decimal(s: &str, start: usize) -> Option<(String, usize)> {
    let b = s.as_bytes();
    let mut i = start;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let int_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let int_len = i - int_start;
    if int_len == 0 || int_len > 3 || i >= b.len() || b[i] != b'.' {
        return None;
    }
    i += 1;
    let frac_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == frac_start {
        return None;
    }
    Some((s[start..i].to_string(), i))
}

/// Parse a `lat, lon` pair at `start` and validate plausible geographic ranges,
/// so only real coordinates (not arbitrary number pairs) become map links.
fn parse_coord_pair(s: &str, start: usize) -> Option<(String, String, usize)> {
    let (lat, p1) = parse_decimal(s, start)?;
    let after = s[p1..].strip_prefix(',')?;
    let spaces = after.len() - after.trim_start_matches(' ').len();
    let (lon, end) = parse_decimal(s, p1 + 1 + spaces)?;
    let latf: f64 = lat.parse().ok()?;
    let lonf: f64 = lon.parse().ok()?;
    if (-90.0..=90.0).contains(&latf) && (-180.0..=180.0).contains(&lonf) {
        Some((lat, lon, end))
    } else {
        None
    }
}

/// Append a tappable map link after every `lat, lon` pair in user-facing text,
/// so a point opens in a map app instead of forcing the reader to copy raw
/// coordinates. Plain Yandex Maps URL — the chat uses no Markdown parse mode, so
/// Telegram auto-links bare URLs. Existing `http…` URLs are copied verbatim so we
/// never double-link or mangle them. No regex dependency.
pub fn linkify_geo(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len() + 64);
    let mut i = 0;
    let mut prev: u8 = b' ';
    while i < b.len() {
        if text[i..].starts_with("http") {
            let end = text[i..]
                .find(char::is_whitespace)
                .map(|d| i + d)
                .unwrap_or(b.len());
            out.push_str(&text[i..end]);
            prev = b' ';
            i = end;
            continue;
        }
        let at_num_start = (b[i].is_ascii_digit() || b[i] == b'+' || b[i] == b'-')
            && !(prev.is_ascii_digit() || prev == b'.');
        if at_num_start {
            if let Some((lat, lon, end)) = parse_coord_pair(text, i) {
                out.push_str(&text[i..end]);
                out.push_str(&format!(
                    " (https://yandex.ru/maps/?ll={lon}%2C{lat}&z=14&pt={lon}%2C{lat})"
                ));
                prev = b')';
                i = end;
                continue;
            }
        }
        let len = utf8_len(b[i]);
        out.push_str(&text[i..i + len]);
        prev = b[i];
        i += len;
    }
    out
}

/// A stage output is UNRESOLVED when it failed/timed out, OR the worker agent
/// itself signalled it could not obtain the real geo data (the `STAGE_INCOMPLETE`
/// marker we ask it to emit instead of writing vague "уточняется" prose).
fn is_stage_unresolved(output: &str) -> bool {
    is_stage_failure(output) || output.contains("STAGE_INCOMPLETE")
}

fn worker_output_needs_best_effort(output: &str) -> bool {
    is_stage_unresolved(output) || output_admits_unresolved_core_data(output)
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

/// LLM-free fallback if the compose call fails.
fn fallback_compose(records: &[StageRecord]) -> String {
    let useful = records
        .iter()
        .filter(|r| {
            let output = r.output.trim();
            !output.is_empty() && output != "(no output)" && !is_stage_failure(output)
        })
        .map(|r| format!("{}:\n{}", r.stage, r.output))
        .collect::<Vec<_>>()
        .join("\n\n");

    if useful.trim().is_empty() {
        return "Пока не смог завершить сбор данных: один из внешних запросов завис или вернул пустой ответ. Состояние сохранено; напишите «продолжай», и я повторю сбор без создания календаря или документа на непроверенных данных.".to_string();
    }

    format!(
        "Пока не удалось довести проверку до конца, но вот что уже собрано. Внешние артефакты ещё не создавал.\n\n{useful}\n\nНапишите «продолжай», и я повторю недостающий сбор данных."
    )
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
    fn linkify_geo_links_coords_lon_lat_order() {
        let out = linkify_geo("Лагерь — 50.160, 43.535 (излучина)");
        assert!(out.contains("50.160, 43.535"));
        // Yandex wants lon,lat — longitude first.
        assert!(out.contains("https://yandex.ru/maps/?ll=43.535%2C50.160"));
    }

    #[test]
    fn linkify_geo_ignores_non_coordinate_numbers() {
        // Sunshine hours, precipitation, wind, scores, dates — none are lat/lon
        // pairs, so none must get a map link.
        let raw = "Солнце 15.6 ч, осадки 3.4 мм (22%), ветер 15 км/ч, оценка 142/100, 4–5 июля";
        assert_eq!(linkify_geo(raw), raw);
    }

    #[test]
    fn linkify_geo_leaves_existing_urls_intact() {
        let raw = "точка https://yandex.ru/maps/?ll=43.5%2C50.1 рядом";
        // The coords live inside the URL → copied verbatim, no second link.
        assert_eq!(linkify_geo(raw), raw);
    }

    #[test]
    fn linkify_geo_rejects_out_of_range_pair() {
        // 200.0 is not a valid latitude — must stay plain.
        let raw = "версия 200.0, 43.5 тест";
        assert_eq!(linkify_geo(raw), raw);
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
    fn worker_failures_trigger_best_effort_path() {
        assert!(worker_output_needs_best_effort("(stage timed out)"));
        assert!(worker_output_needs_best_effort(
            "(stage failed: MCP request failed)"
        ));
        assert!(worker_output_needs_best_effort(
            "STAGE_INCOMPLETE: не найдена стоянка у воды"
        ));
        assert!(!worker_output_needs_best_effort(
            "Маршрут готов: старт 50.100, 43.200; лагерь 50.120, 43.230."
        ));
    }

    #[test]
    fn fallback_compose_hides_internal_stage_failures() {
        let records = vec![StageRecord {
            stage: "research".into(),
            output: "(stage timed out)".into(),
        }];

        let out = fallback_compose(&records);

        assert!(!out.contains("(stage timed out)"));
        assert!(!out.contains("research:"));
        assert!(out.contains("продолжай"));
    }

    #[test]
    fn fallback_compose_keeps_useful_records_and_skips_failures() {
        let records = vec![
            StageRecord {
                stage: "OptionsAgent".into(),
                output: "Вариант: Донская излучина, 50.100, 43.200.".into(),
            },
            StageRecord {
                stage: "research".into(),
                output: "(stage failed: timeout)".into(),
            },
        ];

        let out = fallback_compose(&records);

        assert!(out.contains("Донская излучина"));
        assert!(!out.contains("(stage failed:"));
        assert!(out.contains("Внешние артефакты ещё не создавал"));
    }

    #[test]
    fn long_fresh_request_restarts_active_flow() {
        let mut flow = TripFlowState::start();
        flow.records.push(StageRecord {
            stage: "OptionsAgent".into(),
            output: "1. Старый вариант".into(),
        });
        flow.awaiting_choice = Some(Stage::Planning);

        let text = "Хотим сходить в поход на байдарках, посмотри в какой день лучше это сделать в течение следующих двух недель (мы можем только на выходных). И в каких местах.\n\n\
            Сплав с одной ночевкой, команда больше хочет шашлыки, чем грести. Ночлег будет в палатках, чтобы в радиусе минимум 1км не было турбаз,сел и так далее. Вода должна быть рядом, максимум в 30 метрах от ночлега.\n\n\
            Составь конкретный план с точками остановки и ночлега, событие в календаре, потом из гугл док поделюсь с друзьями.";

        assert!(should_restart_flow_from_message(&flow, text));
    }

    #[test]
    fn short_choice_continues_active_flow() {
        let mut flow = TripFlowState::start();
        flow.records.push(StageRecord {
            stage: "OptionsAgent".into(),
            output: "1. Старый вариант".into(),
        });

        assert!(!should_restart_flow_from_message(
            &flow,
            "Давай первый вариант. Подтверждаю, двигайся дальше."
        ));
        assert!(!should_restart_flow_from_message(&flow, "продолжай"));
    }

    #[test]
    fn progress_messages_describe_task_kind() {
        let camp = SwarmTask {
            id: "camp".into(),
            agent: "WorkerAgent".into(),
            model: None,
            task: "find campsite with water nearby and no settlements".into(),
            tools: true,
            side_effects: false,
            checkpoint: false,
        };
        let route = SwarmTask {
            id: "route".into(),
            agent: "WorkerAgent".into(),
            model: None,
            task: "build route points".into(),
            tools: true,
            side_effects: false,
            checkpoint: false,
        };

        assert!(progress_message_for_task(&camp).contains("стоянку"));
        assert!(progress_message_for_task(&route).contains("маршрута"));
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
    fn swarm_plan_json_is_dynamic_tasks() {
        let raw = r#"{"tasks":[
            {"id":"map_research","agent":"MapAgent","model":"map-model","task":"find places","tools":true},
            {"id":"share","agent":"ArtifactsAgent","task":"create requested artifacts","tools":true,"side_effects":true}
        ]}"#;

        let plan: SwarmPlan = serde_json::from_str(raw).unwrap();

        assert_eq!(plan.tasks.len(), 2);
        assert_eq!(plan.tasks[0].agent, "MapAgent");
        assert_eq!(plan.tasks[0].model.as_deref(), Some("map-model"));
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
            model: None,
            task: "Investigate the selected outdoor option".into(),
            tools: true,
            side_effects: false,
            checkpoint: false,
        };
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        let agent = registry.for_task(&task);

        let system =
            build_swarm_worker_system(&session, &task, &agent, "- maps__geocode_address: x");

        assert!(system.contains("Available MCP tools, discovered at runtime"));
        assert!(system.contains("Agent identity: ResearchAgent"));
        assert!(system.contains("Configured model: base-model"));
        assert!(system.contains("Tools allowed for this agent: true"));
        assert!(system.contains("maps__geocode_address"));
        assert!(!system.contains("kayak"));
        assert!(!system.contains("cycling route"));
    }

    #[test]
    fn swarm_worker_system_uses_task_model_override() {
        let session = ChatSession::new(1);
        let task = SwarmTask {
            id: "research".into(),
            agent: "ResearchAgent".into(),
            model: Some("specialist-model".into()),
            task: "Investigate the selected outdoor option".into(),
            tools: true,
            side_effects: false,
            checkpoint: false,
        };
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        let agent = registry.for_task(&task);

        let system = build_swarm_worker_system(&session, &task, &agent, "(no tools)");

        assert!(system.contains("Agent identity: ResearchAgent"));
        assert!(system.contains("Configured model: specialist-model"));
    }

    #[test]
    fn swarm_registry_defines_distinct_configurable_agents() {
        let registry = SwarmAgentRegistry::with_default_model("base-model");

        let brief = registry.get("BriefAgent");
        let planner = registry.get("SwarmPlanner");
        let verifier = registry.get("VerifierAgent");
        let artifacts = registry.get("ArtifactsAgent");

        assert_ne!(brief.name, planner.name);
        assert_ne!(planner.role, verifier.role);
        assert!(!brief.tools_allowed);
        assert!(!verifier.side_effects_allowed);
        assert!(artifacts.tools_allowed);
        assert!(artifacts.side_effects_allowed);
        assert_eq!(brief.model, "base-model");
    }

    #[test]
    fn swarm_registry_supports_per_agent_model_overrides() {
        let registry = SwarmAgentRegistry::with_model_overrides(
            "base-model",
            &[
                ("BriefAgent", "cheap-brief-model"),
                ("VerifierAgent", "strict-verifier-model"),
                ("ArtifactsAgent", "reliable-writer-model"),
            ],
        );

        assert_eq!(registry.get("BriefAgent").model, "cheap-brief-model");
        assert_eq!(registry.get("VerifierAgent").model, "strict-verifier-model");
        assert_eq!(
            registry.get("ArtifactsAgent").model,
            "reliable-writer-model"
        );
        assert_eq!(registry.get("OptionsAgent").model, "base-model");
    }

    #[test]
    fn swarm_task_can_override_agent_model() {
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        let task = SwarmTask {
            id: "map".into(),
            agent: "WorkerAgent".into(),
            model: Some("fast-map-model".into()),
            task: "map research".into(),
            tools: true,
            side_effects: false,
            checkpoint: false,
        };

        let spec = registry.for_task(&task);

        assert_eq!(spec.name, "WorkerAgent");
        assert_eq!(spec.model, "fast-map-model");
        assert!(spec.tools_allowed);
        assert!(!spec.side_effects_allowed);
    }

    #[test]
    fn swarm_registry_env_key_is_per_agent() {
        assert_eq!(
            agent_model_env_key("VerifierAgent"),
            "SWARM_MODEL_VERIFIERAGENT"
        );
        assert_eq!(agent_model_env_key("map-agent"), "SWARM_MODEL_MAP_AGENT");
    }

    #[test]
    fn ordinary_worker_cannot_escalate_to_side_effects() {
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        let task = SwarmTask {
            id: "bad_write".into(),
            agent: "WorkerAgent".into(),
            model: None,
            task: "create external artifact".into(),
            tools: true,
            side_effects: true,
            checkpoint: false,
        };

        let spec = registry.for_task(&task);

        assert_eq!(spec.name, "WorkerAgent");
        assert!(!spec.side_effects_allowed);
    }

    #[test]
    fn artifacts_agent_is_the_only_default_side_effect_agent() {
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        for name in [
            "BriefAgent",
            "OptionsAgent",
            "SwarmPlanner",
            "VerifierAgent",
            "FinalAgent",
            "WorkerAgent",
        ] {
            assert!(
                !registry.get(name).side_effects_allowed,
                "{name} should not be allowed to write external artifacts"
            );
        }
        assert!(registry.get("ArtifactsAgent").side_effects_allowed);
    }

    #[test]
    fn side_effect_tasks_are_detectable_for_verifier_gate() {
        let plan: SwarmPlan = serde_json::from_str(
            r#"{"tasks":[
                {"id":"research","agent":"WorkerAgent","task":"research","tools":true},
                {"id":"verify","agent":"VerifierAgent","task":"verify","tools":false},
                {"id":"write","agent":"ArtifactsAgent","task":"create doc","tools":true,"side_effects":true}
            ]}"#,
        )
        .unwrap();

        let side_effect_index = plan.tasks.iter().position(|t| t.side_effects).unwrap();
        let verifier_index = plan
            .tasks
            .iter()
            .position(|t| t.agent == "VerifierAgent")
            .unwrap();

        assert!(verifier_index < side_effect_index);

        let registry = SwarmAgentRegistry::with_default_model("base-model");
        assert!(!registry.task_requires_verifier_gate(&plan.tasks[0]));
        assert!(!registry.task_requires_verifier_gate(&plan.tasks[1]));
        assert!(registry.task_requires_verifier_gate(&plan.tasks[2]));
    }

    #[test]
    fn artifacts_agent_requires_gate_even_if_task_flag_is_missing() {
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        let task = SwarmTask {
            id: "write".into(),
            agent: "ArtifactsAgent".into(),
            model: None,
            task: "create requested Google doc".into(),
            tools: true,
            side_effects: false,
            checkpoint: false,
        };

        assert!(registry.task_requires_verifier_gate(&task));

        let spec = registry.for_task(&task);
        assert_eq!(spec.name, "ArtifactsAgent");
        assert!(!spec.side_effects_allowed);
    }

    #[test]
    fn seed_area_from_profile_fills_start_area_from_home_city() {
        let mut profile = super::super::profile::UserProfile::default();
        profile.set("home_city", "Волгоград");
        let mut brief = TripBrief::default();

        seed_area_from_profile(&mut brief, &profile);

        assert_eq!(
            brief.fields.get("start_area").map(String::as_str),
            Some("Волгоград")
        );
    }

    #[test]
    fn seed_area_from_profile_is_noop_when_brief_has_area() {
        let mut profile = super::super::profile::UserProfile::default();
        profile.set("home_city", "Волгоград");
        let mut brief = TripBrief::default();
        brief.fields.insert("area".into(), "Карелия".into());

        seed_area_from_profile(&mut brief, &profile);

        // existing location is kept; profile home does not overwrite it
        assert_eq!(
            brief.fields.get("area").map(String::as_str),
            Some("Карелия")
        );
        assert!(!brief.fields.contains_key("start_area"));
    }

    #[test]
    fn recent_user_requests_keeps_only_user_turns_in_order() {
        let mut session = ChatSession::new(1);
        session
            .memory
            .push_message("user", "Хотим велопоход на выходных");
        session
            .memory
            .push_message("assistant", "Расскажите подробнее?");
        session
            .memory
            .push_message("user", "Команда любительская, одна ночёвка");

        let joined = recent_user_requests(&session);

        assert!(joined.contains("велопоход"));
        assert!(joined.contains("одна ночёвка"));
        assert!(!joined.contains("Расскажите подробнее"));
    }

    // ---- swarm orchestra: distinct, independently configured agents ----

    #[test]
    fn swarm_agent_system_carries_only_its_own_identity() {
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        let brief = registry.get("BriefAgent");

        let system = swarm_agent_system(&brief, SWARM_BRIEF_PROMPT);

        assert!(system.contains("Agent identity: BriefAgent"));
        assert!(system.contains("Configured model: base-model"));
        assert!(system.contains(&brief.role));
        // no other agent's identity may leak into this agent's prompt
        assert!(!system.contains("Agent identity: VerifierAgent"));
        assert!(!system.contains("Agent identity: ArtifactsAgent"));
        assert!(!system.contains("Agent identity: FinalAgent"));
    }

    #[test]
    fn each_swarm_agent_gets_a_distinct_system_prompt() {
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        let verifier = swarm_agent_system(&registry.get("VerifierAgent"), SWARM_VERIFIER_PROMPT);
        let final_a = swarm_agent_system(&registry.get("FinalAgent"), SWARM_FINAL_PROMPT);

        assert_ne!(verifier, final_a);
        assert!(verifier.contains("Agent identity: VerifierAgent"));
        assert!(final_a.contains("Agent identity: FinalAgent"));
        assert!(!verifier.contains("Agent identity: FinalAgent"));
        assert!(!final_a.contains("Agent identity: VerifierAgent"));
    }

    #[test]
    fn per_agent_model_override_reaches_the_agent_system_prompt() {
        let registry = SwarmAgentRegistry::with_model_overrides(
            "base-model",
            &[("VerifierAgent", "strict-verifier-model")],
        );
        let verifier = registry.get("VerifierAgent");
        let options = registry.get("OptionsAgent");

        assert_eq!(verifier.model, "strict-verifier-model");
        assert_eq!(options.model, "base-model");

        let vsys = swarm_agent_system(&verifier, SWARM_VERIFIER_PROMPT);
        assert!(vsys.contains("Configured model: strict-verifier-model"));
        // an independent agent keeps its own model — models are not shared
        let osys = swarm_agent_system(&options, SWARM_OPTIONS_PROMPT);
        assert!(osys.contains("Configured model: base-model"));
        assert!(!osys.contains("strict-verifier-model"));
    }

    #[test]
    fn worker_system_excludes_working_memory_but_keeps_long_term() {
        let mut session = ChatSession::new(1);
        session
            .memory
            .upsert_fact("home_city", "Волгоград", MemoryLayer::LongTerm);
        session
            .memory
            .upsert_fact("scratch", "draft note", MemoryLayer::Working);
        let task = SwarmTask {
            id: "research".into(),
            agent: "WorkerAgent".into(),
            model: None,
            task: "investigate option".into(),
            tools: true,
            side_effects: false,
            checkpoint: false,
        };
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        let agent = registry.for_task(&task);

        let system = build_swarm_worker_system(&session, &task, &agent, "- maps__geocode: x");

        assert!(system.contains("home_city: Волгоград"));
        assert!(!system.contains("scratch"));
        assert!(!system.contains("draft note"));
    }

    #[test]
    fn tool_less_worker_system_marks_tools_disallowed() {
        let session = ChatSession::new(1);
        let task = SwarmTask {
            id: "reason".into(),
            agent: "WorkerAgent".into(),
            model: None,
            task: "reason over prior records".into(),
            tools: false,
            side_effects: false,
            checkpoint: false,
        };
        let registry = SwarmAgentRegistry::with_default_model("base-model");
        let agent = registry.for_task(&task);

        let system = build_swarm_worker_system(&session, &task, &agent, "- maps__geocode: x");

        assert!(system.contains("Tools allowed for this agent: false"));
    }
}
