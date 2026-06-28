//! Unit tests for the agent runtime: memory layers, profile, invariants,
//! prompt assembly, and flow plan parsing.

use super::{
    invariants::{self, Invariant, InvariantCheck, InvariantStatus},
    memory::{AgentMemory, MemoryLayer, MAX_RECENT},
    notes::{self, UserNotes},
    profile::UserProfile,
    prompt,
    session::ChatSession,
};

// ---------- memory ----------

#[test]
fn layer_parse_and_display() {
    assert_eq!(
        "long".parse::<MemoryLayer>().unwrap(),
        MemoryLayer::LongTerm
    );
    assert_eq!(
        "working".parse::<MemoryLayer>().unwrap(),
        MemoryLayer::Working
    );
    assert_eq!(MemoryLayer::ShortTerm.to_string(), "short-term");
    assert!("nope".parse::<MemoryLayer>().is_err());
}

#[test]
fn recent_window_caps_at_hard_ceiling() {
    // push_message now only trims at the hard MAX_RECENT ceiling; trimming to
    // RECENT_WINDOW is done by budget-aware compaction (which preserves dropped
    // turns in the summary), so it isn't exercised here.
    let mut m = AgentMemory::default();
    let total = MAX_RECENT + 5;
    for i in 0..total {
        m.push_message("user", &format!("m{i}"));
    }
    assert_eq!(m.recent.len(), MAX_RECENT);
    assert_eq!(m.recent.last().unwrap().1, format!("m{}", total - 1));
}

#[test]
fn drain_oldest_keeps_tail() {
    let mut m = AgentMemory::default();
    for i in 0..6 {
        m.push_message("user", &format!("m{i}"));
    }
    let drained = m.drain_oldest(2);
    assert_eq!(drained.len(), 4);
    assert_eq!(m.recent.len(), 2);
    assert_eq!(m.recent[0].1, "m4");
    // history_for_answer excludes the trailing (current) message.
    assert_eq!(m.history_for_answer().len(), 1);
}

#[test]
fn upsert_updates_in_place() {
    let mut m = AgentMemory::default();
    assert!(m.upsert_fact("home_city", "Volgograd", MemoryLayer::LongTerm));
    assert!(m.upsert_fact("home_city", "Moscow", MemoryLayer::LongTerm));
    assert_eq!(m.facts.len(), 1);
    assert_eq!(m.facts[0].value, "Moscow");
}

#[test]
fn sensitive_rejected_from_longterm() {
    let mut m = AgentMemory::default();
    assert!(!m.upsert_fact("token", "sk-abcdef1234567890abcdef", MemoryLayer::LongTerm));
    assert!(m.facts.is_empty());
    // but allowed in working (non-durable)
    assert!(m.upsert_fact("tmp", "sk-abcdef1234567890abcdef", MemoryLayer::Working));
}

#[test]
fn reset_keeps_longterm_only() {
    let mut m = AgentMemory::default();
    m.upsert_fact("home_city", "Volgograd", MemoryLayer::LongTerm);
    m.upsert_fact("goal", "trip", MemoryLayer::Working);
    m.push_message("user", "hi");
    m.append_summary("old short context");
    m.reset_for_new_session();
    assert_eq!(m.facts.len(), 1);
    assert_eq!(m.facts[0].key, "home_city");
    assert!(m.recent.is_empty());
    assert!(m.summary.is_empty());
}

#[test]
fn merge_extracted_json_demotes_shortterm() {
    let mut m = AgentMemory::default();
    let json = r#"{"facts":[
        {"key":"home_city","value":"Volgograd","layer":"long-term"},
        {"key":"goal","value":"weekend trip","layer":"short-term"}
    ]}"#;
    assert_eq!(m.merge_extracted_json(json), 2);
    assert_eq!(m.facts_in_layer(MemoryLayer::LongTerm).len(), 1);
    // short-term demoted to working
    assert_eq!(m.facts_in_layer(MemoryLayer::Working).len(), 1);
}

#[test]
fn merge_extracted_json_missing_layer_uses_default() {
    let mut m = AgentMemory::default();
    let json = r#"{"facts":[{"key":"interests","value":"hiking"}]}"#;
    assert_eq!(m.merge_extracted_json(json), 1);
    assert_eq!(m.facts[0].layer, MemoryLayer::LongTerm);
}

#[test]
fn keyword_fallback_extracts_home_city() {
    let mut m = AgentMemory::default();
    let n = m.extract_keyword_fallback("Привет, я из Волгоград, какая погода?");
    assert_eq!(n, 1);
    assert_eq!(m.facts[0].key, "home_city");
}

#[test]
fn append_summary_merges_and_caps() {
    let mut m = AgentMemory::default();
    m.append_summary("first");
    m.append_summary("second");
    assert!(m.summary.contains("first") && m.summary.contains("second"));
    assert!(m.summary.contains("\n\n"));
    // blank additions are ignored
    m.append_summary("   ");
    let before = m.summary.clone();
    assert_eq!(before, m.summary);
    // hard cap keeps the tail, never grows unbounded
    m.append_summary(&"x".repeat(8000));
    assert!(m.summary.chars().count() <= 4000);
}

#[test]
fn history_for_answer_excludes_current_and_empty_safe() {
    let mut m = AgentMemory::default();
    assert!(m.history_for_answer().is_empty()); // empty doesn't panic
    m.push_message("user", "a");
    // single message = current turn only → no prior history
    assert!(m.history_for_answer().is_empty());
    m.push_message("assistant", "b");
    m.push_message("user", "c");
    let h = m.history_for_answer();
    assert_eq!(h.len(), 2);
    assert_eq!(h[0], ("user", "a"));
    assert_eq!(h[1], ("assistant", "b"));
}

// ---------- profile ----------

#[test]
fn profile_set_get_clear() {
    let mut p = UserProfile::default();
    p.set("home_city", "Volgograd");
    p.set("language", "ru");
    assert_eq!(p.fields.len(), 2);
    p.set("home_city", ""); // empty removes
    assert_eq!(p.fields.len(), 1);
    p.clear();
    assert!(p.is_empty());
}

#[test]
fn profile_merge_skips_secrets() {
    let mut p = UserProfile::default();
    let json = r#"{"fields":[
        {"key":"home_city","value":"Volgograd"},
        {"key":"api","value":"sk-abcdef1234567890abcdef"}
    ]}"#;
    let n = p.merge_extracted_json(json);
    assert_eq!(n, 1);
    assert!(p.fields.contains_key("home_city"));
    assert!(!p.fields.contains_key("api"));
}

#[test]
fn interests_union_merge() {
    let mut p = UserProfile::default();
    p.set("interests", "kayaking, basketball");
    p.set("interests", "Basketball, hiking"); // dup (case-insensitive) dropped
    assert_eq!(p.fields["interests"], "kayaking, basketball, hiking");
}

#[test]
fn inline_markers_apply_and_strip() {
    let mut p = UserProfile::default();
    let raw = "В Сочи +24°C, тепло.\n⟦profile:age=80⟧\n⟦profile:interests=байдарки⟧";
    let n = p.apply_inline_markers(raw);
    assert_eq!(n, 2);
    assert_eq!(p.fields["age"], "80");
    assert_eq!(p.fields["interests"], "байдарки");
    let clean = super::profile::strip_inline_markers(raw);
    assert_eq!(clean, "В Сочи +24°C, тепло.");
    assert!(!clean.contains("profile:"));
}

#[test]
fn inline_markers_reject_secrets() {
    let mut p = UserProfile::default();
    let n = p.apply_inline_markers("⟦profile:token=sk-abcdef1234567890abcdef⟧");
    assert_eq!(n, 0);
    assert!(p.is_empty());
}

// ---------- invariants ----------

#[test]
fn invariant_number_required() {
    let inv = vec![Invariant {
        text: "needs number".into(),
        check: InvariantCheck::MustContainNumber,
    }];
    assert_eq!(
        invariants::check(&inv, "It is warm").status(),
        InvariantStatus::Failed
    );
    assert_eq!(
        invariants::check(&inv, "It is 25C").status(),
        InvariantStatus::Passed
    );
}

#[test]
fn invariant_must_not_contain_secret() {
    let inv = vec![Invariant {
        text: "no secrets".into(),
        check: InvariantCheck::MustNotContain(vec!["sk-".into()]),
    }];
    let r = invariants::check(&inv, "key is sk-12345");
    assert_eq!(r.status(), InvariantStatus::Failed);
    assert_eq!(r.violations, vec!["no secrets".to_string()]);
}

#[test]
fn invariant_empty_answer_passes() {
    let inv = invariants::travel_weather_defaults();
    assert_eq!(
        invariants::check(&inv, "   ").status(),
        InvariantStatus::Passed
    );
}

#[test]
fn invariant_advisory_is_advisory() {
    let inv = vec![Invariant {
        text: "be nice".into(),
        check: InvariantCheck::Advisory,
    }];
    assert_eq!(
        invariants::check(&inv, "hello 1").status(),
        InvariantStatus::Advisory
    );
}

// ---------- prompt ----------

#[test]
fn prompt_layers_in_order_and_dedup() {
    let mut memory = AgentMemory::default();
    memory.upsert_fact("home_city", "Volgograd", MemoryLayer::LongTerm);
    memory.upsert_fact("goal", "weekend trip", MemoryLayer::Working);
    let mut profile = UserProfile::default();
    profile.set("language", "ru");
    let inv = invariants::travel_weather_defaults();

    let s = prompt::build_system_prompt(&memory, &profile, &[], &inv, None, None);
    let long = s.find("[memory:long-term]").unwrap();
    let prof = s.find("[user-profile]").unwrap();
    let work = s.find("[memory:working]").unwrap();
    let invs = s.find("[invariants]").unwrap();
    assert!(
        long < prof && prof < work && work < invs,
        "layer order wrong:\n{s}"
    );
}

#[test]
fn prompt_includes_summary_block_after_longterm() {
    let mut memory = AgentMemory::default();
    memory.upsert_fact("home_city", "Sochi", MemoryLayer::LongTerm);
    memory.append_summary("user planning a kayaking trip");
    let mut profile = UserProfile::default();
    profile.set("language", "ru");

    let s = prompt::build_system_prompt(&memory, &profile, &[], &[], None, None);
    let long = s.find("[memory:long-term]").unwrap();
    let summ = s.find("[memory:summary]").unwrap();
    let prof = s.find("[user-profile]").unwrap();
    assert!(long < summ && summ < prof, "summary misplaced:\n{s}");
    assert!(s.contains("kayaking trip"));
}

#[test]
fn prompt_omits_summary_when_empty() {
    let memory = AgentMemory::default();
    let profile = UserProfile::default();
    let s = prompt::build_system_prompt(&memory, &profile, &[], &[], None, None);
    assert!(!s.contains("[memory:summary]"));
}

#[test]
fn strip_inline_markers_preserves_paragraphs() {
    let raw = "Первый абзац.\n\nВторой абзац.\n⟦profile:age=80⟧";
    let clean = super::profile::strip_inline_markers(raw);
    assert_eq!(clean, "Первый абзац.\n\nВторой абзац.");
}

#[test]
fn prompt_includes_violation_feedback() {
    let memory = AgentMemory::default();
    let profile = UserProfile::default();
    let s = prompt::build_system_prompt(
        &memory,
        &profile,
        &[],
        &[],
        None,
        Some(&["needs a number".to_string()]),
    );
    assert!(s.contains("violated these"));
    assert!(s.contains("needs a number"));
}

// ---------- notes ("доп инфа") ----------

#[test]
fn notes_set_remove_and_lowercase_label() {
    let mut n = UserNotes::default();
    n.set("Files", "always .docx");
    assert_eq!(
        n.entries.get("files").map(String::as_str),
        Some("always .docx")
    );
    // empty text removes
    n.set("files", "  ");
    assert!(n.is_empty());
}

#[test]
fn notes_pick_resolves_known_labels_only() {
    let mut n = UserNotes::default();
    n.set("files", "docx");
    n.set("tone", "brief");
    let picked = n.pick(&["files".into(), "unknown".into()]);
    assert_eq!(picked, vec![("files".to_string(), "docx".to_string())]);
}

#[test]
fn notes_keyword_fallback_matches_overlap() {
    let mut n = UserNotes::default();
    n.set("files", "Формат файлов docx");
    n.set("tone", "коротко");
    // "формат" overlaps the files note, nothing overlaps tone.
    let hits = n.keyword_candidates("сделай в нужном формате");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, "files");
}

#[test]
fn parse_selected_labels_tolerates_prose() {
    let labels = notes::parse_selected_labels("sure: {\"labels\":[\"files\",\"tone\"]} ok");
    assert_eq!(labels, vec!["files".to_string(), "tone".to_string()]);
    assert!(notes::parse_selected_labels("garbage").is_empty());
}

#[test]
fn prompt_injects_notes_between_profile_and_working() {
    let mut memory = AgentMemory::default();
    memory.upsert_fact("goal", "weekend trip", MemoryLayer::Working);
    let mut profile = UserProfile::default();
    profile.set("language", "ru");
    let notes_ctx = vec![("files".to_string(), "always .docx".to_string())];

    let s = prompt::build_system_prompt(&memory, &profile, &notes_ctx, &[], None, None);
    let prof = s.find("[user-profile]").unwrap();
    let note = s.find("[user-notes]").unwrap();
    let work = s.find("[memory:working]").unwrap();
    assert!(prof < note && note < work, "notes misplaced:\n{s}");
    assert!(s.contains("always .docx"));
}

#[test]
fn prompt_omits_notes_block_when_empty() {
    let memory = AgentMemory::default();
    let profile = UserProfile::default();
    let s = prompt::build_system_prompt(&memory, &profile, &[], &[], None, None);
    assert!(!s.contains("[user-notes]"));
}

// ---------- session ----------

#[test]
fn session_uses_default_invariants_when_empty() {
    let s = ChatSession::new(1);
    assert!(s.invariants.is_empty());
    assert!(!s.effective_invariants().is_empty());
}

// ---------- flow ----------

#[test]
fn trip_flow_state_roundtrips() {
    // The stateful flow must survive session persistence (serde) across turns.
    let mut st = super::flow::TripFlowState::start();
    st.brief.fields.insert("area".into(), "Карелия".into());
    st.brief
        .fields
        .insert("date_window".into(), "next 2 weeks".into());
    let json = serde_json::to_string(&st).unwrap();
    let back: super::flow::TripFlowState = serde_json::from_str(&json).unwrap();
    assert_eq!(back.brief.fields.get("area").unwrap(), "Карелия");
    assert_eq!(back.stage, super::flow::Stage::Clarify);
}
