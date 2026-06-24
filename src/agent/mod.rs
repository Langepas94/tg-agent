//! Agent runtime ported from ai-playground (`Ai teach`): layered sticky-facts
//! memory, user profile, code-checked invariants, a layered PromptBuilder and a
//! multi-agent travel-weather flow. The orchestrator is a deterministic code
//! router; the LLM fills artifacts and answers.

pub mod flow;
pub mod invariants;
pub mod memory;
pub mod profile;
pub mod prompt;
pub mod session;

#[cfg(test)]
mod tests;

use crate::{llm::Llm, state::BotState};

use self::{invariants::InvariantStatus, session::ChatSession};

/// Max invariant retries for a normal turn.
const MAX_INVARIANT_RETRIES: usize = 1;

/// Result of one orchestrated turn.
#[derive(Debug, Clone)]
pub struct TurnResult {
    pub answer: String,
    pub facts_learned: usize,
    pub profile_updated: usize,
    pub invariant_status: InvariantStatus,
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
) -> anyhow::Result<TurnResult> {
    session.memory.push_message("user", user_text);

    // --- sticky-facts extraction (before answering) ---
    let facts_learned = extract_facts(llm, session, user_text).await;

    // --- profile interview extraction ---
    let profile_updated = extract_profile(llm, session, user_text).await;

    // --- build layered prompt + answer ---
    let invariants = session.effective_invariants();
    let mut violations: Vec<String> = Vec::new();
    let mut answer = String::new();
    let mut status = InvariantStatus::Passed;

    for attempt in 0..=MAX_INVARIANT_RETRIES {
        let feedback = if attempt == 0 {
            None
        } else {
            Some(violations.as_slice())
        };
        let system = prompt::build_system_prompt(
            &session.memory,
            &session.profile,
            &invariants,
            None,
            feedback,
        );
        answer = llm.answer_with_system(state, &system, user_text).await?;

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
        profile_updated,
        invariant_status: status,
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

async fn extract_profile(llm: &Llm, session: &mut ChatSession, user_text: &str) -> usize {
    match llm
        .complete(profile::PROFILE_EXTRACTION_PROMPT, user_text)
        .await
    {
        Ok(json) => session.profile.merge_extracted_json(&strip_fence(&json)),
        Err(_) => 0,
    }
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
