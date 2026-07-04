//! Task state for RAG dialogs: what the user already clarified, fixed
//! constraints/terms, and the dialog goal. Filled by an LLM extraction agent
//! before each RAG answer (mirrors sticky-facts extraction in `mod.rs`);
//! degrades to seeding the goal from the first question when no LLM is
//! configured. Rendered into the RAG system prompt so the assistant keeps
//! the goal across a long dialog.

use serde::{Deserialize, Serialize};

use crate::llm::Llm;

const MAX_ITEMS: usize = 8;
const MAX_ITEM_CHARS: usize = 160;

pub const RAG_TASK_STATE_PROMPT: &str = "You maintain compact task state for a dialog with a RAG \
assistant over a technical knowledge base. Input: current state JSON and the user's new message. \
Return ONLY the UPDATED state as JSON {\"goal\":\"...\",\"clarified\":[...],\"constraints\":[...]}: \
goal = what the dialog is trying to achieve overall (keep the existing goal unless the user clearly \
switches topic); clarified = short facts the user has already clarified or confirmed; constraints = \
fixed constraints and term definitions the answers must respect. Short items, user's language, \
max 8 per list. Never store secrets or tokens. No commentary, JSON only.";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RagTaskState {
    /// What the dialog is trying to achieve.
    #[serde(default)]
    pub goal: String,
    /// What the user has already clarified.
    #[serde(default)]
    pub clarified: Vec<String>,
    /// Fixed constraints and term definitions.
    #[serde(default)]
    pub constraints: Vec<String>,
}

impl RagTaskState {
    pub fn is_empty(&self) -> bool {
        self.goal.trim().is_empty() && self.clarified.is_empty() && self.constraints.is_empty()
    }

    /// Merge an extractor result: goal is replaced only by a non-empty one,
    /// lists are replaced wholesale (the extractor sees the previous state and
    /// re-emits what still matters), then bounded.
    pub fn merge(&mut self, update: RagTaskState) {
        let goal = update.goal.trim();
        if !goal.is_empty() {
            self.goal = truncate(goal);
        }
        if !update.clarified.is_empty() {
            self.clarified = bound_items(update.clarified);
        }
        if !update.constraints.is_empty() {
            self.constraints = bound_items(update.constraints);
        }
    }

    /// Prose snapshot passed to the RAG agent's system prompt.
    pub fn to_prompt(&self) -> String {
        let mut lines = Vec::new();
        if !self.goal.trim().is_empty() {
            lines.push(format!("Цель диалога: {}", self.goal.trim()));
        }
        if !self.clarified.is_empty() {
            lines.push(format!("Уже уточнено: {}", self.clarified.join("; ")));
        }
        if !self.constraints.is_empty() {
            lines.push(format!(
                "Ограничения/термины: {}",
                self.constraints.join("; ")
            ));
        }
        lines.join("\n")
    }
}

fn truncate(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= MAX_ITEM_CHARS {
        return trimmed.to_string();
    }
    trimmed.chars().take(MAX_ITEM_CHARS).collect()
}

fn bound_items(items: Vec<String>) -> Vec<String> {
    items
        .into_iter()
        .map(|item| truncate(&item))
        .filter(|item| !item.is_empty())
        .take(MAX_ITEMS)
        .collect()
}

/// Update the task state from the new user message. LLM decides what the goal,
/// clarifications and constraints now are; without an LLM the goal is seeded
/// from the first question so the state is never empty.
pub async fn update_task_state(llm: Option<&Llm>, task: &mut RagTaskState, user_text: &str) {
    if let Some(llm) = llm {
        let input = serde_json::json!({
            "current_state": task,
            "user_message": user_text,
        });
        if let Ok(json) = llm
            .complete(RAG_TASK_STATE_PROMPT, &input.to_string())
            .await
        {
            if let Ok(update) = serde_json::from_str::<RagTaskState>(strip_fence(&json)) {
                task.merge(update);
                return;
            }
        }
    }
    if task.goal.trim().is_empty() {
        task.goal = truncate(user_text);
    }
}

fn strip_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    let trimmed = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    trimmed.strip_suffix("```").unwrap_or(trimmed).trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_keeps_goal_when_update_is_empty() {
        let mut state = RagTaskState {
            goal: "понять архитектуру tg-agent".into(),
            clarified: vec!["интересует MCP".into()],
            constraints: vec![],
        };
        state.merge(RagTaskState::default());
        assert_eq!(state.goal, "понять архитектуру tg-agent");
        assert_eq!(state.clarified, vec!["интересует MCP".to_string()]);
    }

    #[test]
    fn merge_replaces_lists_and_bounds_them() {
        let mut state = RagTaskState::default();
        state.merge(RagTaskState {
            goal: "goal".into(),
            clarified: (0..20).map(|i| format!("item {i}")).collect(),
            constraints: vec!["  only VPS  ".into(), "".into()],
        });
        assert_eq!(state.clarified.len(), MAX_ITEMS);
        assert_eq!(state.constraints, vec!["only VPS".to_string()]);
    }

    #[test]
    fn to_prompt_renders_all_sections() {
        let state = RagTaskState {
            goal: "спланировать деплой".into(),
            clarified: vec!["сервер 2GB".into()],
            constraints: vec!["без docker".into()],
        };
        let prompt = state.to_prompt();
        assert!(prompt.contains("Цель диалога: спланировать деплой"));
        assert!(prompt.contains("Уже уточнено: сервер 2GB"));
        assert!(prompt.contains("Ограничения/термины: без docker"));
    }

    #[test]
    fn empty_state_renders_empty_prompt() {
        assert!(RagTaskState::default().to_prompt().is_empty());
        assert!(RagTaskState::default().is_empty());
    }
}
