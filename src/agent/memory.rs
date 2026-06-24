//! Layered sticky-facts memory, ported from the ai-playground (`Ai teach`)
//! chat runtime. Facts are atomic key→value pairs, each assigned a
//! [`MemoryLayer`]. A bounded recent-message window (default 10) gives
//! short-term context. Sticky-facts extraction runs BEFORE the answer so the
//! durable memory updates the moment the user states a new fact.

use serde::{Deserialize, Serialize};

/// Number of recent messages kept as short-term context.
pub const RECENT_WINDOW: usize = 10;

pub const FACTS_EXTRACTION_PROMPT: &str = "You maintain the layered local memory of a chat agent. \
After each user message, extract only DURABLE facts and choose a memory layer for each. \
Layers: \"working\" = data of the CURRENT task (active goal, trip cities, dates, constraints); \
\"long-term\" = stable knowledge surviving across sessions (home city, preferences, interests, \
person profile). Return ONLY JSON {\"facts\":[{\"key\":\"snake_case\",\"value\":\"short string\",\"layer\":\"working|long-term\"}]}. \
Use {\"facts\":[]} if nothing durable. Never store secrets, tokens, passwords, API keys, or the whole message.";

/// Memory layers, most volatile → most durable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryLayer {
    ShortTerm,
    Working,
    LongTerm,
}

impl MemoryLayer {
    pub const ORDERED: [MemoryLayer; 3] = [
        MemoryLayer::ShortTerm,
        MemoryLayer::Working,
        MemoryLayer::LongTerm,
    ];

    /// Whether facts in this layer survive into a brand-new session.
    pub fn persists_across_sessions(self) -> bool {
        matches!(self, MemoryLayer::LongTerm)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MemoryLayer::ShortTerm => "short-term",
            MemoryLayer::Working => "working",
            MemoryLayer::LongTerm => "long-term",
        }
    }
}

impl std::fmt::Display for MemoryLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for MemoryLayer {
    type Err = ();
    fn from_str(v: &str) -> Result<Self, ()> {
        match v.trim().to_ascii_lowercase().as_str() {
            "short-term" | "short" | "shortterm" => Ok(MemoryLayer::ShortTerm),
            "working" | "work" => Ok(MemoryLayer::Working),
            "long-term" | "long" | "longterm" => Ok(MemoryLayer::LongTerm),
            _ => Err(()),
        }
    }
}

// Serialize as plain string so it round-trips cleanly in JSON state files.
impl Serialize for MemoryLayer {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}
impl<'de> Deserialize<'de> for MemoryLayer {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        raw.parse()
            .map_err(|_| serde::de::Error::custom(format!("unknown memory layer: {raw}")))
    }
}

/// Default routing when the extractor doesn't pick a layer.
/// `goal`/`constraints`/trip data → Working; everything else → LongTerm.
/// ShortTerm is never a fact layer (it's the message window).
pub fn default_fact_layer(key: &str) -> MemoryLayer {
    match key {
        "goal" | "constraints" | "trip_cities" | "trip_dates" | "destination" => {
            MemoryLayer::Working
        }
        _ => MemoryLayer::LongTerm,
    }
}

/// Heuristic: text that looks like a secret must never enter durable memory.
pub fn looks_sensitive(text: &str) -> bool {
    let t = text.trim();
    if t.len() >= 20
        && t.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && t.chars().any(|c| c.is_ascii_digit())
        && t.chars().any(|c| c.is_ascii_alphabetic())
    {
        return true;
    }
    let low = t.to_ascii_lowercase();
    [
        "sk-", "bearer ", "token", "password", "api_key", "apikey", "secret",
    ]
    .iter()
    .any(|m| low.contains(m))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    pub key: String,
    pub value: String,
    pub layer: MemoryLayer,
}

/// Per-chat layered memory: KV facts + a bounded recent-message window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMemory {
    #[serde(default)]
    pub facts: Vec<Fact>,
    /// Recent messages as (role, text); capped at RECENT_WINDOW.
    #[serde(default)]
    pub recent: Vec<(String, String)>,
}

impl AgentMemory {
    /// Push a message into the short-term window, trimming to RECENT_WINDOW.
    pub fn push_message(&mut self, role: &str, text: &str) {
        self.recent.push((role.to_string(), text.to_string()));
        let overflow = self.recent.len().saturating_sub(RECENT_WINDOW);
        if overflow > 0 {
            self.recent.drain(0..overflow);
        }
    }

    /// Upsert a fact, rejecting sensitive values from durable layers.
    /// Returns false if the fact was rejected.
    pub fn upsert_fact(&mut self, key: &str, value: &str, layer: MemoryLayer) -> bool {
        if layer.persists_across_sessions() && looks_sensitive(value) {
            return false;
        }
        let key = key.trim().to_string();
        let value = value.trim().to_string();
        if key.is_empty() || value.is_empty() {
            return false;
        }
        match self.facts.iter_mut().find(|f| f.key == key) {
            Some(f) => {
                f.value = value;
                f.layer = layer;
            }
            None => self.facts.push(Fact { key, value, layer }),
        }
        true
    }

    pub fn facts_in_layer(&self, layer: MemoryLayer) -> Vec<&Fact> {
        self.facts.iter().filter(|f| f.layer == layer).collect()
    }

    /// Drop everything that does not survive a new session (keep long-term).
    pub fn reset_for_new_session(&mut self) {
        self.facts.retain(|f| f.layer.persists_across_sessions());
        self.recent.clear();
    }

    /// Merge facts extracted from an LLM JSON response of the form
    /// `{"facts":[{"key","value","layer"}]}`. Returns count accepted.
    pub fn merge_extracted_json(&mut self, json: &str) -> usize {
        #[derive(Deserialize)]
        struct Extracted {
            #[serde(default)]
            facts: Vec<RawFact>,
        }
        #[derive(Deserialize)]
        struct RawFact {
            key: String,
            value: String,
            #[serde(default)]
            layer: Option<String>,
        }
        let parsed: Extracted = match serde_json::from_str(json.trim()) {
            Ok(p) => p,
            Err(_) => return 0,
        };
        let mut n = 0;
        for rf in parsed.facts {
            let layer = rf
                .layer
                .as_deref()
                .and_then(|s| s.parse().ok())
                // Model-chosen ShortTerm is demoted to Working (facts are never short-term).
                .map(|l: MemoryLayer| {
                    if l == MemoryLayer::ShortTerm {
                        MemoryLayer::Working
                    } else {
                        l
                    }
                })
                .unwrap_or_else(|| default_fact_layer(&rf.key));
            if self.upsert_fact(&rf.key, &rf.value, layer) {
                n += 1;
            }
        }
        n
    }

    /// Keyword fallback extraction (no LLM): catches a few common patterns.
    /// Returns count accepted.
    pub fn extract_keyword_fallback(&mut self, user_msg: &str) -> usize {
        let low = user_msg.to_ascii_lowercase();
        let mut n = 0;
        // "я из <city>" / "живу в <city>" → home_city (long-term)
        for marker in ["я из ", "живу в ", "i live in ", "i'm from ", "im from "] {
            if let Some(idx) = low.find(marker) {
                let rest = &user_msg[idx + marker.len()..];
                let city: String = rest
                    .chars()
                    .take_while(|c| c.is_alphabetic() || c.is_whitespace() || *c == '-')
                    .collect();
                let city = city.trim().trim_end_matches(['.', ',']);
                if !city.is_empty()
                    && city.len() <= 40
                    && self.upsert_fact("home_city", city, MemoryLayer::LongTerm)
                {
                    n += 1;
                }
            }
        }
        n
    }
}
