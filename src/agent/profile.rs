//! User profile — durable per-chat identity the agent personalizes around.
//! Editable by the user (/profile) and auto-filled by the ProfileAgent
//! interview extraction. Injected into the system prompt as `[user-profile]`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const PROFILE_EXTRACTION_PROMPT: &str = "You maintain a compact user profile for a travel-weather assistant. \
From the latest user message, extract any STABLE personal trait that would shape how the user \
asks about weather or plans travel. The key set is OPEN (snake_case) — use the common keys when \
they fit (home_city, preferred_units, comfort_temp_min, comfort_temp_max, dislikes_rain, language, \
age, occupation, household), and for hobbies/activities/sports use the `interests` key with a \
comma-separated list (e.g. \"kayaking, basketball\"), MERGING with any you already know. \
Capture only durable facts, never one-off trip details and never secrets. \
Return ONLY JSON {\"fields\":[{\"key\":\"snake_case\",\"value\":\"short\"}]}; use {\"fields\":[]} if none.";

/// Known profile keys we surface in /profile help (free keys are allowed too).
pub const KNOWN_KEYS: &[&str] = &[
    "home_city",
    "preferred_units",
    "comfort_temp_min",
    "comfort_temp_max",
    "dislikes_rain",
    "language",
    "age",
    "occupation",
    "household",
    "interests",
];

/// Union two comma-separated lists, preserving order, dropping case-insensitive
/// duplicates and empties.
fn union_csv(existing: Option<&str>, addition: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for part in existing
        .into_iter()
        .flat_map(|s| s.split(','))
        .chain(addition.split(','))
    {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if seen.insert(p.to_lowercase()) {
            out.push(p.to_string());
        }
    }
    out.join(", ")
}

/// Opening / closing delimiters for inline profile markers.
const MARK_OPEN: &str = "⟦profile:";
const MARK_CLOSE: &str = "⟧";

/// Extract `(key, value)` pairs from `⟦profile:key=value⟧` markers in text.
fn scan_inline_markers(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(MARK_OPEN) {
        let after = &rest[start + MARK_OPEN.len()..];
        let Some(end) = after.find(MARK_CLOSE) else {
            break;
        };
        let body = &after[..end];
        if let Some((k, v)) = body.split_once('=') {
            let k = k.trim();
            let v = v.trim();
            if !k.is_empty() && !v.is_empty() {
                out.push((k.to_string(), v.to_string()));
            }
        }
        rest = &after[end + MARK_CLOSE.len()..];
    }
    out
}

/// Remove every `⟦profile:…⟧` marker from text and tidy whitespace, so the user
/// never sees the machine annotations the agent appended.
pub fn strip_inline_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(MARK_OPEN) {
        out.push_str(&rest[..start]);
        let after = &rest[start + MARK_OPEN.len()..];
        match after.find(MARK_CLOSE) {
            Some(end) => rest = &after[end + MARK_CLOSE.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    // Tidy trailing space and collapse blank-line runs the markers left behind,
    // but preserve intentional paragraph breaks.
    let mut cleaned = String::with_capacity(out.len());
    let mut blank_run = 0;
    for line in out.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                cleaned.push('\n');
            }
        } else {
            blank_run = 0;
            cleaned.push_str(line);
            cleaned.push('\n');
        }
    }
    cleaned.trim().to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserProfile {
    #[serde(default)]
    pub fields: BTreeMap<String, String>,
}

impl UserProfile {
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn set(&mut self, key: &str, value: &str) {
        let key = key.trim().to_string();
        let value = value.trim().to_string();
        if key.is_empty() {
            return;
        }
        if value.is_empty() {
            self.fields.remove(&key);
        } else if key == "interests" {
            // Interests accumulate: union the comma-separated lists, case-insensitive.
            let merged = union_csv(self.fields.get(&key).map(String::as_str), &value);
            self.fields.insert(key, merged);
        } else {
            self.fields.insert(key, value);
        }
    }

    pub fn clear(&mut self) {
        self.fields.clear();
    }

    /// Render as deterministic "key: value" lines (sorted by key).
    pub fn render_lines(&self) -> Vec<String> {
        self.fields
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect()
    }

    /// Apply inline `⟦profile:key=value⟧` markers the answering agent may emit
    /// when it notices a durable trait mid-conversation. Returns the count
    /// applied. (The markers themselves are stripped by [`strip_inline_markers`].)
    pub fn apply_inline_markers(&mut self, answer: &str) -> usize {
        let mut n = 0;
        for (key, value) in scan_inline_markers(answer) {
            if super::memory::looks_sensitive(&value) {
                continue;
            }
            self.set(&key, &value);
            n += 1;
        }
        n
    }

    /// Merge `{"fields":[{"key","value"}]}` from the profile extractor.
    pub fn merge_extracted_json(&mut self, json: &str) -> usize {
        #[derive(Deserialize)]
        struct Extracted {
            #[serde(default)]
            fields: Vec<Field>,
        }
        #[derive(Deserialize)]
        struct Field {
            key: String,
            value: String,
        }
        let parsed: Extracted = match serde_json::from_str(json.trim()) {
            Ok(p) => p,
            Err(_) => return 0,
        };
        let mut n = 0;
        for f in parsed.fields {
            if super::memory::looks_sensitive(&f.value) {
                continue;
            }
            self.set(&f.key, &f.value);
            n += 1;
        }
        n
    }
}
