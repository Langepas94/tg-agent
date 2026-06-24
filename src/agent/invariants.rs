//! Invariants — absolute constraints injected into the system prompt AND
//! verified in code after the answer (a deterministic linter, never an LLM).
//! Ported from ai-playground's `InvariantAgent`: each invariant is checkable
//! (Pass/Fail) or non-checkable (Advisory — stays as prompt guidance).

use serde::{Deserialize, Serialize};

/// One invariant: human text (for the prompt) + an optional code check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Invariant {
    /// Text shown to the model in `[invariants]`.
    pub text: String,
    /// Machine-checkable kind; `Advisory` = guidance only.
    #[serde(default)]
    pub check: InvariantCheck,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InvariantCheck {
    /// Answer must contain at least one of these substrings (case-insensitive).
    MustContainAny(Vec<String>),
    /// Answer must NOT contain any of these substrings (case-insensitive).
    MustNotContain(Vec<String>),
    /// Answer must contain a digit (e.g. a temperature/number).
    MustContainNumber,
    /// No code check — guidance only.
    #[default]
    Advisory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvariantStatus {
    Passed,
    Failed,
    Advisory,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InvariantReport {
    /// Texts of invariants the answer violated (checkable, failed).
    pub violations: Vec<String>,
    /// Texts of non-checkable invariants (advisory).
    pub unknown: Vec<String>,
}

impl InvariantReport {
    pub fn status(&self) -> InvariantStatus {
        if !self.violations.is_empty() {
            InvariantStatus::Failed
        } else if self.unknown.is_empty() {
            InvariantStatus::Passed
        } else {
            InvariantStatus::Advisory
        }
    }
}

/// Deterministic invariant check over a produced answer.
/// An empty answer carries no content to violate → Passed.
pub fn check(invariants: &[Invariant], answer: &str) -> InvariantReport {
    let mut report = InvariantReport::default();
    if answer.trim().is_empty() {
        return report;
    }
    let low = answer.to_ascii_lowercase();
    for inv in invariants {
        match &inv.check {
            InvariantCheck::MustContainAny(opts) => {
                let ok = opts.iter().any(|o| low.contains(&o.to_ascii_lowercase()));
                if !ok {
                    report.violations.push(inv.text.clone());
                }
            }
            InvariantCheck::MustNotContain(opts) => {
                let bad = opts.iter().any(|o| low.contains(&o.to_ascii_lowercase()));
                if bad {
                    report.violations.push(inv.text.clone());
                }
            }
            InvariantCheck::MustContainNumber => {
                if !answer.chars().any(|c| c.is_ascii_digit()) {
                    report.violations.push(inv.text.clone());
                }
            }
            InvariantCheck::Advisory => report.unknown.push(inv.text.clone()),
        }
    }
    report
}

/// Default invariants for the travel-weather agent.
pub fn travel_weather_defaults() -> Vec<Invariant> {
    vec![
        Invariant {
            text: "Always include a concrete temperature value (a number).".into(),
            check: InvariantCheck::MustContainNumber,
        },
        Invariant {
            text: "Never invent data: base the answer on tool results, not guesses.".into(),
            check: InvariantCheck::Advisory,
        },
        Invariant {
            text: "Never reveal secrets, tokens, or API keys.".into(),
            check: InvariantCheck::MustNotContain(vec!["sk-".into(), "bearer ".into()]),
        },
    ]
}
