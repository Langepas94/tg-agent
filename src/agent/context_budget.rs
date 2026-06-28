//! Context budget — model context-window awareness + a cheap token estimate.
//!
//! The agent must know how big its model's context window is, and start
//! compacting history *before* the provider truncates or rejects the request.
//! We don't have a real tokenizer, so we use a conservative chars→tokens
//! estimate that holds for mixed Latin/Cyrillic text.

/// Fraction of the window at which we begin summarizing older history.
pub const COMPACT_AT: f32 = 0.80;

/// Once compaction starts (at [`COMPACT_AT`]), shrink the session well below the
/// trigger — down to this fraction — so we don't re-summarize on every following
/// turn while hovering at the limit. Profile and sticky facts are never part of
/// this budget (they live outside chat history), so compacting to here never
/// touches them.
pub const COMPACT_TARGET: f32 = 0.55;

/// Conservative chars-per-token. English ≈ 4, Cyrillic ≈ 2–3 (multi-byte,
/// fewer chars per token); 3.0 keeps the estimate from *under*-counting.
const CHARS_PER_TOKEN: f32 = 3.0;

/// Estimated token count of a string. Deliberately rounds up.
pub fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count() as f32;
    (chars / CHARS_PER_TOKEN).ceil() as usize
}

/// Sum of [`estimate_tokens`] over many strings.
pub fn estimate_many<'a, I: IntoIterator<Item = &'a str>>(parts: I) -> usize {
    parts.into_iter().map(estimate_tokens).sum()
}

/// Context-window size (in tokens) for a model id.
///
/// Resolution order:
/// 1. `LLM_CONTEXT_TOKENS` env override (operator wins, any provider/model).
/// 2. Built-in table by model-id substring.
/// 3. Conservative default (`32_000`).
pub fn context_window(model: &str) -> usize {
    if let Ok(v) = std::env::var("LLM_CONTEXT_TOKENS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    let m = model.to_ascii_lowercase();
    // Substring match — model ids often carry suffixes/dates.
    const TABLE: &[(&str, usize)] = &[
        ("deepseek-reasoner", 65_536),
        ("deepseek-chat", 65_536),
        ("deepseek", 65_536),
        ("gpt-4o", 128_000),
        ("gpt-4.1", 1_000_000),
        ("gpt-4-turbo", 128_000),
        ("gpt-4", 8_192),
        ("gpt-3.5", 16_385),
        ("o1", 200_000),
        ("o3", 200_000),
        ("claude", 200_000),
        ("gemini-1.5", 1_000_000),
        ("gemini", 1_000_000),
        ("qwen", 32_768),
        ("llama", 128_000),
        ("mistral", 32_768),
    ];
    for (key, win) in TABLE {
        if m.contains(key) {
            return *win;
        }
    }
    32_000
}

/// The token budget at which compaction should kick in for a model.
pub fn compact_threshold(model: &str) -> usize {
    (context_window(model) as f32 * COMPACT_AT) as usize
}

/// The lower token budget compaction shrinks down to once it has been triggered.
pub fn compact_target(model: &str) -> usize {
    (context_window(model) as f32 * COMPACT_TARGET) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env-var read is process-global; keep all cases that touch
    // LLM_CONTEXT_TOKENS in ONE test so they can't race parallel tests.
    #[test]
    fn window_resolution_table_and_env() {
        // Ensure no stray override from the environment.
        std::env::remove_var("LLM_CONTEXT_TOKENS");
        assert_eq!(context_window("deepseek-chat"), 65_536);
        assert_eq!(context_window("gpt-4o-2024-08-06"), 128_000);
        assert_eq!(context_window("claude-opus-4-8"), 200_000);
        assert_eq!(context_window("totally-made-up-model"), 32_000);
        // 65_536 * 0.8 = 52_428.8 → 52_428
        assert_eq!(compact_threshold("deepseek-chat"), 52_428);

        // Env override wins for any model, then restore.
        std::env::set_var("LLM_CONTEXT_TOKENS", "12345");
        assert_eq!(context_window("deepseek-chat"), 12_345);
        std::env::remove_var("LLM_CONTEXT_TOKENS");
    }

    #[test]
    fn token_estimate_rounds_up() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 1); // 3/3
        assert_eq!(estimate_tokens("abcd"), 2); // 4/3 → 2
    }
}
