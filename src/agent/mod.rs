//! Agent runtime ported from ai-playground (`Ai teach`): layered sticky-facts
//! memory, user profile, code-checked invariants, a layered PromptBuilder and a
//! dynamic multi-agent outdoor planning swarm. The swarm discovers available
//! MCP tools at runtime, delegates isolated worker tasks, verifies evidence
//! before side effects, and then composes the answer.

pub mod context_budget;
pub mod flow;
pub mod invariants;
pub mod memory;
pub mod notes;
pub mod profile;
pub mod prompt;
pub mod router;
pub mod session;

#[cfg(test)]
mod tests;

use crate::{llm::Llm, state::BotState};

use self::{invariants::InvariantStatus, session::ChatSession};

/// Channel for streaming live "step in progress" notes to the user mid-turn.
/// The trip swarm sends one line before each multi-minute stage so a long turn
/// is never silent (the bot relays each line to Telegram as it arrives).
pub type ProgressSender = tokio::sync::mpsc::UnboundedSender<String>;

/// Max invariant retries for a normal turn.
const MAX_INVARIANT_RETRIES: usize = 1;

/// Result of one orchestrated turn.
#[derive(Debug, Clone)]
pub struct TurnResult {
    pub answer: String,
    pub facts_learned: usize,
    pub profile_updated: usize,
    pub invariant_status: InvariantStatus,
    /// Pipeline trace when this turn ran the trip-planning swarm (one line per
    /// completed stage); empty for ordinary turns.
    pub trace: Vec<String>,
}

/// Run one conversational turn:
/// 1. record the user message in short-term memory
/// 2. sticky-facts extraction BEFORE answering (LLM, keyword fallback)
/// 3. profile interview extraction
/// 4. build the layered system prompt (memory + profile + invariants)
/// 5. answer via the MCP tool loop
/// 6. code invariant gate with one retry
/// 7. record the answer; caller persists the session
pub async fn run_turn(
    llm: &Llm,
    state: &BotState,
    session: &mut ChatSession,
    user_text: &str,
    progress: Option<ProgressSender>,
) -> anyhow::Result<TurnResult> {
    session.memory.push_message("user", user_text);

    // --- semantic intent routing ---
    // A message already inside an active flow always continues it. Otherwise a
    // single cheap LLM router decides — BY MEANING, not keyword lists — whether
    // this starts a trip-planning swarm, is a normal chat message, or is
    // off-topic and should be refused early.
    let in_flow = session.trip.is_some();
    let route = if in_flow {
        router::Route::Trip
    } else {
        router::classify(llm, user_text).await
    };

    // --- trip-planning swarm routing (stateful, suspends across turns) ---
    // The flow runs a dynamic agent swarm: brief → short options → planner-built
    // worker tasks → verifier-gated artifacts → final answer.
    if in_flow || route == router::Route::Trip {
        if !in_flow {
            session.trip = Some(flow::TripFlowState::start());
        }
        let turn = flow::advance(llm, state, session, user_text, progress.as_ref()).await?;
        if turn.done {
            session.trip = None;
        }
        session.memory.push_message("assistant", &turn.reply);
        return Ok(TurnResult {
            answer: turn.reply,
            facts_learned: 0,
            profile_updated: 0,
            invariant_status: InvariantStatus::Passed,
            trace: turn.trace,
        });
    }

    // --- off-topic refusal (before any extraction / answer loop) ---
    // Refused here so it costs ~0 tokens beyond the one router call: no fact /
    // profile extraction, no answer loop, no invariant retries.
    if route == router::Route::OffTopic {
        let answer = router::OFF_TOPIC_REPLY.to_string();
        session.memory.push_message("assistant", &answer);
        return Ok(TurnResult {
            answer,
            facts_learned: 0,
            profile_updated: 0,
            invariant_status: InvariantStatus::Passed,
            trace: vec![],
        });
    }

    // --- sticky-facts extraction (before answering) ---
    let facts_learned = extract_facts(llm, session, user_text).await;

    // --- profile interview extraction ---
    let profile_updated = extract_profile(llm, session, user_text).await;

    // --- context-budget compaction (summarize older turns at 80% of window) ---
    maybe_compact(llm, session).await;

    // --- context router: pick which saved notes are relevant to THIS turn ---
    // Token-frugal: skipped entirely when no notes exist; otherwise one small
    // selector call decides what (if anything) to inject — an unused preference
    // never bloats the answering prompt.
    let notes_ctx = select_notes(llm, session, user_text).await;

    // --- build layered prompt + answer ---
    let invariants = session.effective_invariants();
    let history = session.memory.history_for_answer();
    let mut violations: Vec<String> = Vec::new();
    let mut answer = String::new();
    let mut status = InvariantStatus::Passed;
    let mut profile_inline = 0;

    for attempt in 0..=MAX_INVARIANT_RETRIES {
        let feedback = if attempt == 0 {
            None
        } else {
            Some(violations.as_slice())
        };
        let system = prompt::build_system_prompt(
            &session.memory,
            &session.profile,
            &notes_ctx,
            &invariants,
            None,
            feedback,
        );
        answer = {
            use anyhow::Context;
            llm.answer_in_chat(
                state,
                &system,
                user_text,
                &history,
                Some(session.chat_id),
                crate::llm::MAX_STEPS,
            )
            .await
            .context("answering failed")?
        };

        // Agent self-extraction: pull any ⟦profile:k=v⟧ markers, then strip them
        // so the invariant check and the user both see clean prose.
        profile_inline = session.profile.apply_inline_markers(&answer);
        answer = profile::strip_inline_markers(&answer);

        let report = invariants::check(&invariants, &answer);
        status = report.status();
        violations = report.violations;
        if status != InvariantStatus::Failed {
            break;
        }
    }

    session.memory.push_message("assistant", &answer);

    Ok(TurnResult {
        answer,
        facts_learned,
        profile_updated: profile_updated + profile_inline,
        invariant_status: status,
        trace: vec![],
    })
}

async fn extract_facts(llm: &Llm, session: &mut ChatSession, user_text: &str) -> usize {
    match llm
        .complete(memory::FACTS_EXTRACTION_PROMPT, user_text)
        .await
    {
        Ok(json) => {
            let n = session.memory.merge_extracted_json(&strip_fence(&json));
            if n == 0 {
                session.memory.extract_keyword_fallback(user_text)
            } else {
                n
            }
        }
        // No LLM / error → keyword fallback keeps memory working offline.
        Err(_) => session.memory.extract_keyword_fallback(user_text),
    }
}

/// Context-router agent: choose which of the user's saved notes ("доп инфа")
/// are relevant to the current message. Returns the `(label, text)` pairs to
/// inject. Cheap by construction:
/// - no notes → returns empty, makes no LLM call;
/// - LLM picks only relevant labels (a file-format note is left out on a turn
///   that has nothing to do with files);
/// - on LLM failure, falls back to a keyword match so it still works offline.
async fn select_notes(llm: &Llm, session: &ChatSession, user_text: &str) -> Vec<(String, String)> {
    if session.notes.is_empty() {
        return Vec::new();
    }
    let listing = session.notes.render_lines().join("\n");
    let input = format!("User message:\n{user_text}\n\nSaved notes:\n{listing}");
    match llm.complete(notes::NOTES_SELECTOR_PROMPT, &input).await {
        Ok(json) => {
            let labels = notes::parse_selected_labels(&strip_fence(&json));
            session.notes.pick(&labels)
        }
        Err(_) => session.notes.keyword_candidates(user_text),
    }
}

async fn extract_profile(llm: &Llm, session: &mut ChatSession, user_text: &str) -> usize {
    match llm
        .complete(profile::PROFILE_EXTRACTION_PROMPT, user_text)
        .await
    {
        Ok(json) => session.profile.merge_extracted_json(&strip_fence(&json)),
        Err(_) => 0,
    }
}

const SUMMARY_PROMPT: &str = "You compress chat history for a travel-weather assistant. \
Summarize the following older messages into a few terse bullet points capturing \
durable context: the user's goal, cities/dates, decisions, and stated preferences. \
Omit pleasantries. Plain text, no JSON, under 600 characters.";

/// Keep the prompt under the model's context budget: summarize older turns into
/// the rolling summary instead of sending (or losing) them. Runs every turn;
/// cheap when there is nothing to compact.
async fn maybe_compact(llm: &Llm, session: &mut ChatSession) {
    // 1. Always fold turns beyond the verbatim window into the summary, so the
    //    history we send each turn stays small (token cost) without data loss.
    compact_overflow(llm, session, memory::RECENT_WINDOW).await;

    // 2. Budget-driven shrink. Only kicks in at the 80% trigger, but then
    //    compacts all the way down to the lower TARGET (~55%), so we don't
    //    re-summarize on every following turn while hovering near the limit.
    //    Only chat history (`recent`) is summarized here — profile and sticky
    //    facts live outside this budget and are never touched.
    if estimate_session_tokens(session) <= context_budget::compact_threshold(llm.model()) {
        return;
    }
    let target = context_budget::compact_target(llm.model());
    let mut guard = 0;
    while estimate_session_tokens(session) > target && session.memory.recent.len() > 2 && guard < 8
    {
        let keep = (session.memory.recent.len() / 2).max(2);
        compact_overflow(llm, session, keep).await;
        guard += 1;
    }
}

/// Drain everything older than `keep_tail`, summarize it, fold into the rolling
/// summary. Falls back to a mechanical digest if the LLM call fails, so the
/// budget always shrinks and continuity is never silently dropped.
async fn compact_overflow(llm: &Llm, session: &mut ChatSession, keep_tail: usize) {
    let drained = session.memory.drain_oldest(keep_tail);
    if drained.is_empty() {
        return;
    }
    let joined = drained
        .iter()
        .map(|(role, text)| format!("{role}: {text}"))
        .collect::<Vec<_>>()
        .join("\n");
    let summary = match llm.complete(SUMMARY_PROMPT, &joined).await {
        Ok(s) if !s.trim().is_empty() => s,
        _ => mechanical_digest(&drained),
    };
    session.memory.append_summary(&summary);
}

/// LLM-free fallback summary: first line of each drained message, clipped.
fn mechanical_digest(drained: &[(String, String)]) -> String {
    drained
        .iter()
        .map(|(role, text)| {
            let one = text.lines().next().unwrap_or("").trim();
            let clip: String = one.chars().take(80).collect();
            format!("- {role}: {clip}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Rough token estimate of the session-derived prompt content (summary + recent
/// window + fact values) plus a flat allowance for the static system/invariants.
fn estimate_session_tokens(session: &ChatSession) -> usize {
    const BASE_SYSTEM_TOKENS: usize = 600;
    let mut total = BASE_SYSTEM_TOKENS + context_budget::estimate_tokens(&session.memory.summary);
    for (_, text) in &session.memory.recent {
        total += context_budget::estimate_tokens(text);
    }
    for f in &session.memory.facts {
        total +=
            context_budget::estimate_tokens(&f.key) + context_budget::estimate_tokens(&f.value);
    }
    total
}

/// Strip ```json fences and surrounding prose, keeping the first JSON object.
fn strip_fence(s: &str) -> String {
    let s = s.trim();
    if let (Some(start), Some(end)) = (s.find('{'), s.rfind('}')) {
        if end >= start {
            return s[start..=end].to_string();
        }
    }
    s.to_string()
}
