//! User profile — durable per-chat identity the agent personalizes around.
//! Editable by the user (/profile) and auto-filled by the ProfileAgent
//! interview extraction. Injected into the system prompt as `[user-profile]`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const PROFILE_EXTRACTION_PROMPT: &str = "You maintain a compact user profile for a travel-weather assistant. \
From the latest user message, extract only STABLE profile fields (home_city, preferred_units, \
comfort_temp_min, comfort_temp_max, dislikes_rain, interests, language). \
Return ONLY JSON {\"fields\":[{\"key\":\"snake_case\",\"value\":\"short\"}]}; use {\"fields\":[]} if none. \
Never store secrets or the whole message.";

/// Known profile keys we surface in /profile help (free keys are allowed too).
pub const KNOWN_KEYS: &[&str] = &[
    "home_city",
    "preferred_units",
    "comfort_temp_min",
    "comfort_temp_max",
    "dislikes_rain",
    "interests",
    "language",
];

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
