//! User notes ("доп инфа") — free-form, *labeled* preferences the user sets
//! beyond the structured [`UserProfile`]. Unlike the profile (durable identity
//! traits, always shown), notes are **conditionally** mixed into the system
//! prompt: a small router agent picks only the notes relevant to the current
//! turn, so an unused preference (e.g. a file-format note on a turn that has
//! nothing to do with files) costs no prompt tokens.
//!
//! Example: the user runs `/info files Файлы в формате .docx, имя с датой`.
//! On a turn where they ask for a document, the router selects the `files`
//! note and it is injected; on a plain weather question it is not.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Router-agent prompt: pick which saved notes matter for THIS message.
/// Kept tiny on purpose — input is just the message plus short labelled notes,
/// far cheaper than injecting every note into the main answering prompt.
pub const NOTES_SELECTOR_PROMPT: &str = "You are a context router for an assistant. \
You are given the user's latest message and a list of their saved preference notes \
(each line: `label — text`). Decide which notes are RELEVANT to fulfilling THIS message — \
pick a note only if applying it would actually change the response. For example a note about \
file formats matters only when the user wants a file/document/export; a note about tone matters \
for any reply. Be strict: when unsure, leave it out. \
Return ONLY JSON {\"labels\":[\"label1\",...]} listing the chosen labels exactly as given; \
use {\"labels\":[]} when none apply.";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserNotes {
    /// `label -> instruction text`. The label is both the user-facing handle
    /// and what the router returns when selecting.
    #[serde(default)]
    pub entries: BTreeMap<String, String>,
}

impl UserNotes {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Set (or, with an empty value, remove) a labelled note.
    pub fn set(&mut self, label: &str, text: &str) {
        let label = label.trim().to_lowercase();
        let text = text.trim().to_string();
        if label.is_empty() {
            return;
        }
        if text.is_empty() {
            self.entries.remove(&label);
        } else {
            self.entries.insert(label, text);
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// `label: text` lines, sorted by label (deterministic).
    pub fn render_lines(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect()
    }

    /// Resolve a set of labels (from the router) to `(label, text)` pairs,
    /// preserving the store's sorted order and skipping unknown labels.
    pub fn pick(&self, labels: &[String]) -> Vec<(String, String)> {
        let want: std::collections::HashSet<String> =
            labels.iter().map(|l| l.trim().to_lowercase()).collect();
        self.entries
            .iter()
            .filter(|(k, _)| want.contains(*k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Offline fallback when no LLM is available: keyword overlap between the
    /// message and each note's label+text (tokens of length >= 4). Coarser than
    /// the router but keeps the feature working without a model.
    pub fn keyword_candidates(&self, user_text: &str) -> Vec<(String, String)> {
        let hay = user_text.to_lowercase();
        self.entries
            .iter()
            .filter(|(label, text)| {
                note_tokens(label)
                    .chain(note_tokens(text))
                    .any(|tok| hay.contains(&tok))
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

/// Lowercased word tokens of length >= 4 used for the offline keyword match.
fn note_tokens(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 4)
        .map(|w| w.to_lowercase())
}

/// Parse the router agent's `{"labels":[...]}` reply. Tolerant of surrounding
/// prose / code fences (caller may pass raw model output).
pub fn parse_selected_labels(json: &str) -> Vec<String> {
    #[derive(Deserialize)]
    struct Picked {
        #[serde(default)]
        labels: Vec<String>,
    }
    let trimmed = {
        let s = json.trim();
        match (s.find('{'), s.rfind('}')) {
            (Some(a), Some(b)) if b >= a => &s[a..=b],
            _ => s,
        }
    };
    serde_json::from_str::<Picked>(trimmed)
        .map(|p| p.labels)
        .unwrap_or_default()
}
