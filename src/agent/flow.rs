//! Multi-agent travel-weather flow — a code-routed FSM pipeline (mirrors the
//! ai-playground swarm: Planning → Execution → Validation → Done). Each stage
//! is a real LLM call; Execution uses the connected weather MCP tools. The
//! orchestrator (code) owns transitions; the model only fills stage artifacts.

use anyhow::Result;
use serde::Deserialize;

use crate::{llm::Llm, state::BotState};

use super::{
    invariants::{self, Invariant},
    session::ChatSession,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    Planning,
    Execution,
    Validation,
    Done,
}

/// Trip parsed by the Planning agent.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct TripPlan {
    #[serde(default)]
    pub cities: Vec<String>,
    #[serde(default)]
    pub dates: Vec<String>,
    #[serde(default)]
    pub note: String,
}

/// One stage's record for transparency / debugging.
#[derive(Debug, Clone)]
pub struct StageRecord {
    pub stage: Stage,
    pub output: String,
}

#[derive(Debug, Clone)]
pub struct TravelReport {
    pub plan: TripPlan,
    pub records: Vec<StageRecord>,
    pub answer: String,
    pub invariant_violations: Vec<String>,
}

const PLANNING_PROMPT: &str = "You are the PLANNING agent of a travel-weather pipeline. \
From the user's message, extract the trip. Return ONLY JSON \
{\"cities\":[\"...\"],\"dates\":[\"YYYY-MM-DD\" or \"this weekend\" etc],\"note\":\"short intent\"}. \
If no city is given, leave cities empty.";

/// Parse a trip plan from the user message (Planning agent).
pub async fn plan(llm: &Llm, user_text: &str) -> Result<TripPlan> {
    let raw = llm.complete(PLANNING_PROMPT, user_text).await?;
    let json = extract_json(&raw);
    Ok(serde_json::from_str::<TripPlan>(&json).unwrap_or_default())
}

/// Run the full pipeline. Execution calls the weather MCP via the tool loop.
pub async fn run(
    llm: &Llm,
    state: &BotState,
    session: &ChatSession,
    user_text: &str,
) -> Result<TravelReport> {
    let mut records = Vec::new();

    // ---- Planning ----
    let plan = plan(llm, user_text).await?;
    records.push(StageRecord {
        stage: Stage::Planning,
        output: format!("cities={:?} dates={:?}", plan.cities, plan.dates),
    });

    if plan.cities.is_empty() {
        // Nothing to execute — bounce back to the user for a city.
        return Ok(TravelReport {
            plan,
            records,
            answer: "Which city (or cities) and dates is the trip for?".into(),
            invariant_violations: vec![],
        });
    }

    // ---- Execution (uses MCP tools) ----
    let invariants = session.effective_invariants();
    let exec_system = build_execution_system(session, &invariants);
    let exec_query = format!(
        "Trip: cities={:?}, dates={:?}. For each city, use the weather tools \
         (geocode then forecast) and report current/relevant conditions with concrete numbers. \
         Note: {}",
        plan.cities, plan.dates, plan.note
    );
    let mut answer = llm
        .answer_with_system(state, &exec_system, &exec_query)
        .await?;
    records.push(StageRecord {
        stage: Stage::Execution,
        output: truncate(&answer, 300),
    });

    // ---- Validation (code invariant gate, one retry) ----
    let mut report = invariants::check(&invariants, &answer);
    if report.status() == invariants::InvariantStatus::Failed {
        let retry_system = format!(
            "{exec_system}\n\n[invariants] Your previous answer violated these — fix now:\n- {}",
            report.violations.join("\n- ")
        );
        answer = llm
            .answer_with_system(state, &retry_system, &exec_query)
            .await?;
        report = invariants::check(&invariants, &answer);
    }
    records.push(StageRecord {
        stage: Stage::Validation,
        output: format!("{:?}", report.status()),
    });

    // ---- Done ----
    records.push(StageRecord {
        stage: Stage::Done,
        output: "composed final recommendation".into(),
    });

    Ok(TravelReport {
        plan,
        records,
        answer,
        invariant_violations: report.violations,
    })
}

fn build_execution_system(session: &ChatSession, invariants: &[Invariant]) -> String {
    super::prompt::build_system_prompt(
        &session.memory,
        &session.profile,
        &[],
        invariants,
        Some("You are the EXECUTION agent. Use tools to gather real weather data, then recommend."),
        None,
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

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}
