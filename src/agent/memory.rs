//! Layered sticky-facts memory, ported from the ai-playground (`Ai teach`)
//! chat runtime. Facts are atomic key→value pairs, each assigned a
//! [`MemoryLayer`]. A bounded recent-message window (default 10) gives
//! short-term context. Sticky-facts extraction runs BEFORE the answer so the
//! durable memory updates the moment the user states a new fact.

use serde::{Deserialize, Serialize};

/// Target number of recent messages kept verbatim as short-term context and
/// sent with each answer. Older turns are summarized into `summary`, not lost.
pub const RECENT_WINDOW: usize = 10;

/// Hard safety ceiling on `recent` (compaction normally keeps it at
/// RECENT_WINDOW; this only bounds paths that skip compaction).
pub const MAX_RECENT: usize = 40;

pub const FACTS_EXTRACTION_PROMPT: &str = "You maintain a tiny local memory for a travel-weather assistant. \
Extract ONLY facts that will help future turns. Do NOT store ordinary one-off requests, weather query \
parameters, subscriptions, active_task/current_task labels, or internal workflow state. \
Layers: \"long-term\" = stable user context (home city, email, language, interests, durable \
preferences); \"working\" = compact trip-only context for an active trip, using ONLY keys prefixed \
with trip_ (trip_activity, trip_date_window, trip_duration, trip_skill_level, trip_focus, \
trip_shelter, trip_constraints, trip_start_area). Return ONLY JSON \
{\"facts\":[{\"key\":\"snake_case\",\"value\":\"short string\",\"layer\":\"working|long-term\"}]}. \
Use {\"facts\":[]} if nothing genuinely useful. Never store secrets, tokens, passwords, API keys, \
or the whole message.";

/// Memory layers, most volatile → most durable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryLayer {
    ShortTerm,
    Working,
    LongTerm,
}

const MAX_FACT_VALUE_CHARS: usize = 160;
const MAX_LONG_TERM_FACTS: usize = 24;
const MAX_WORKING_FACTS: usize = 10;

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
    match key.trim().to_ascii_lowercase().as_str() {
        "trip_activity" | "trip_date_window" | "trip_duration" | "trip_skill_level"
        | "trip_focus" | "trip_shelter" | "trip_constraints" | "trip_start_area"
        | "activity_type" | "date_window" | "timeframe" | "team_skill" | "team_experience"
        | "activity_focus" | "shelter" | "constraints" => MemoryLayer::Working,
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

fn truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "yes" | "y" | "1" | "да" | "ага" | "люблю" | "нравится"
    )
}

fn clean_value(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value.chars().count() > MAX_FACT_VALUE_CHARS {
        return None;
    }
    Some(value.to_string())
}

fn canonical_long_term(key: &str, value: &str) -> Option<(String, String)> {
    let key = key.trim().to_ascii_lowercase();
    let value = clean_value(value)?;
    let canonical = match key.as_str() {
        "home_city"
        | "email"
        | "google_email"
        | "preferred_units"
        | "comfort_temp_min"
        | "comfort_temp_max"
        | "dislikes_rain"
        | "language"
        | "age"
        | "occupation"
        | "household"
        | "interests"
        | "preferred_food"
        | "accommodation_preference"
        | "max_travel_time_hours" => (key, value),
        "interest_activity" | "activity_interest" | "hobby" | "hobbies" => {
            ("interests".to_string(), value)
        }
        _ if key.starts_with("likes_") => {
            if truthy(&value) {
                (
                    "interests".to_string(),
                    key.trim_start_matches("likes_").to_string(),
                )
            } else {
                return None;
            }
        }
        _ => return None,
    };
    Some(canonical)
}

fn canonical_working(key: &str, value: &str) -> Option<(String, String)> {
    let key = key.trim().to_ascii_lowercase();
    let value = clean_value(value)?;
    let canonical_key = match key.as_str() {
        "trip_activity" | "activity_type" => "trip_activity",
        "trip_date_window" | "date_window" | "timeframe" => "trip_date_window",
        "trip_duration" => "trip_duration",
        "trip_skill_level" | "team_skill" | "team_experience" => "trip_skill_level",
        "trip_focus" | "activity_focus" | "team_preference" => "trip_focus",
        "trip_shelter" | "shelter" => "trip_shelter",
        "trip_constraints" | "constraints" => "trip_constraints",
        "campsite_requirement_isolation" | "constraint_no_civilization_distance" => {
            return Some((
                "trip_constraints".to_string(),
                format!("isolation: {value}"),
            ));
        }
        "campsite_requirement_water_distance" | "constraint_water_distance" => {
            return Some((
                "trip_constraints".to_string(),
                format!("water distance: {value}"),
            ));
        }
        "trip_start_area" | "area" | "start_area" => "trip_start_area",
        _ => return None,
    };
    Some((canonical_key.to_string(), value))
}

fn normalize_fact(
    key: &str,
    value: &str,
    layer: MemoryLayer,
) -> Option<(String, String, MemoryLayer)> {
    let layer = if layer == MemoryLayer::ShortTerm {
        MemoryLayer::Working
    } else {
        layer
    };
    let (key, value) = match layer {
        MemoryLayer::LongTerm => canonical_long_term(key, value)?,
        MemoryLayer::Working => canonical_working(key, value)?,
        MemoryLayer::ShortTerm => unreachable!(),
    };
    Some((key, value, layer))
}

fn union_csv(existing: &str, addition: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for part in existing.split(',').chain(addition.split(',')) {
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

fn merge_semicolon(existing: &str, addition: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for part in existing.split(';').chain(addition.split(';')) {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if seen.insert(p.to_lowercase()) {
            out.push(p.to_string());
        }
    }
    out.join("; ")
}

/// Per-chat layered memory: KV facts + a bounded recent-message window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMemory {
    #[serde(default)]
    pub facts: Vec<Fact>,
    /// Recent messages as (role, text); capped at RECENT_WINDOW.
    #[serde(default)]
    pub recent: Vec<(String, String)>,
    /// Rolling prose summary of older turns that aged out of `recent` or were
    /// compacted to stay under the model's context budget. Injected into the
    /// prompt as `[memory:summary]` so long conversations keep continuity.
    #[serde(default)]
    pub summary: String,
}

impl AgentMemory {
    /// Push a message into the short-term window. Trims only at the hard
    /// safety ceiling (`MAX_RECENT`); normal trimming to `RECENT_WINDOW` is
    /// done by budget-aware compaction, which preserves dropped turns in the
    /// rolling summary instead of discarding them.
    pub fn push_message(&mut self, role: &str, text: &str) {
        self.recent.push((role.to_string(), text.to_string()));
        let overflow = self.recent.len().saturating_sub(MAX_RECENT);
        if overflow > 0 {
            self.recent.drain(0..overflow);
        }
    }

    /// Recent messages as OpenAI chat objects, excluding the trailing entry
    /// (the current user message, sent separately by the answering loop).
    pub fn history_for_answer(&self) -> Vec<(&str, &str)> {
        let n = self.recent.len();
        let upto = n.saturating_sub(1);
        self.recent[..upto]
            .iter()
            .map(|(r, t)| (r.as_str(), t.as_str()))
            .collect()
    }

    /// Fold an addition into the rolling summary (blank-line separated, kept
    /// to a sane length so the summary itself never grows unbounded).
    pub fn append_summary(&mut self, addition: &str) {
        let addition = addition.trim();
        if addition.is_empty() {
            return;
        }
        if self.summary.is_empty() {
            self.summary = addition.to_string();
        } else {
            self.summary.push_str("\n\n");
            self.summary.push_str(addition);
        }
        // Hard cap the rolling summary so it can't itself blow the budget.
        const MAX_SUMMARY_CHARS: usize = 4000;
        if self.summary.chars().count() > MAX_SUMMARY_CHARS {
            let start = self.summary.chars().count() - MAX_SUMMARY_CHARS;
            self.summary = self.summary.chars().skip(start).collect();
        }
    }

    /// Remove the oldest messages from `recent`, keeping the last `keep_tail`.
    /// Returns the drained messages (oldest→newest) for summarization.
    pub fn drain_oldest(&mut self, keep_tail: usize) -> Vec<(String, String)> {
        let n = self.recent.len();
        if n <= keep_tail {
            return Vec::new();
        }
        self.recent.drain(0..n - keep_tail).collect()
    }

    /// Upsert a fact, rejecting sensitive values and keys outside the schema.
    /// Returns false if the fact was rejected.
    pub fn upsert_fact(&mut self, key: &str, value: &str, layer: MemoryLayer) -> bool {
        let Some((key, value, layer)) = normalize_fact(key, value, layer) else {
            return false;
        };
        if looks_sensitive(&value) {
            return false;
        }
        match self.facts.iter_mut().find(|f| f.key == key) {
            Some(f) => {
                if f.key == "interests" {
                    f.value = union_csv(&f.value, &value);
                } else if f.key == "trip_constraints" {
                    f.value = merge_semicolon(&f.value, &value);
                } else {
                    f.value = value;
                }
                f.layer = layer;
            }
            None => self.facts.push(Fact { key, value, layer }),
        }
        self.prune_layer(MemoryLayer::LongTerm, MAX_LONG_TERM_FACTS);
        self.prune_layer(MemoryLayer::Working, MAX_WORKING_FACTS);
        true
    }

    fn prune_layer(&mut self, layer: MemoryLayer, max: usize) {
        let overflow = self
            .facts
            .iter()
            .filter(|f| f.layer == layer)
            .count()
            .saturating_sub(max);
        if overflow == 0 {
            return;
        }
        let mut remaining = overflow;
        self.facts.retain(|f| {
            if f.layer == layer && remaining > 0 {
                remaining -= 1;
                false
            } else {
                true
            }
        });
    }

    /// Re-apply the current schema to persisted facts. This drops legacy
    /// extractor noise such as weather_query_* and current_task while preserving
    /// useful durable context.
    pub fn sanitize(&mut self) -> bool {
        let before = self.facts.clone();
        let old = std::mem::take(&mut self.facts);
        for f in old {
            let _ = self.upsert_fact(&f.key, &f.value, f.layer);
        }
        before != self.facts
    }

    pub fn facts_in_layer(&self, layer: MemoryLayer) -> Vec<&Fact> {
        self.facts.iter().filter(|f| f.layer == layer).collect()
    }

    /// Drop everything that does not survive a new session (keep long-term).
    pub fn reset_for_new_session(&mut self) {
        self.facts.retain(|f| f.layer.persists_across_sessions());
        self.clear_short_context();
    }

    /// Clear ephemeral conversational context while preserving durable facts.
    pub fn clear_short_context(&mut self) {
        self.facts.retain(|f| f.layer.persists_across_sessions());
        self.recent.clear();
        self.summary.clear();
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
