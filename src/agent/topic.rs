//! Topic-scope gate — a CODE invariant (not a prompt request) that keeps this
//! bot on weather + travel-planning. Off-topic questions are refused *before*
//! any LLM call (fact/profile extraction + answer loop), so they cost ~0 tokens.
//!
//! The classifier is intentionally lenient: it blocks only messages that are
//! clearly off-topic AND carry no weather/travel signal. Anything ambiguous or
//! conversational (greetings, follow-ups, "yes", a bare city name) is allowed,
//! so we never silently drop a legitimate request to save a few tokens.

/// Verdict for one user message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Weather/travel signal present, or too short/ambiguous to reject.
    InScope,
    /// Clear off-topic intent with no weather/travel signal — refuse early.
    OffTopic,
}

/// Weather/travel vocabulary (RU + EN). A single hit ⇒ in scope.
const ON_TOPIC: &[&str] = &[
    // weather
    "погод",
    "темпер",
    "градус",
    "дожд",
    "снег",
    "ветер",
    "ветр",
    "облач",
    "ясн",
    "солнеч",
    "влажн",
    "осадк",
    "прогноз",
    "клима",
    "жар",
    "холод",
    "мороз",
    "гроза",
    "weather",
    "temperature",
    "forecast",
    "rain",
    "snow",
    "wind",
    "cloud",
    "sunny",
    "humid",
    "storm",
    "climate",
    "degree",
    "celsius",
    "fahrenheit",
    "hot",
    "cold",
    "warm",
    "cool",
    "umbrella",
    "frost",
    // travel / places
    "путеш",
    "поездк",
    "поезд",
    "город",
    "стран",
    "лет",
    "рейс",
    "отел",
    "виза",
    "маршрут",
    "тур",
    "отпуск",
    "куда",
    "поеду",
    "поехать",
    "съезд",
    "командировк",
    "travel",
    "trip",
    "flight",
    "city",
    "country",
    "hotel",
    "visa",
    "route",
    "tour",
    "vacation",
    "holiday",
    "destination",
    "journey",
    "abroad",
    "pack",
];

/// Clear off-topic intent markers (RU + EN). Used only to *reject*, and only
/// when no ON_TOPIC term is present.
const OFF_TOPIC: &[&str] = &[
    "код",
    "програм",
    "функц",
    "python",
    "rust",
    "javascript",
    "sql",
    "регуляр",
    "рецепт",
    "приготов",
    "акци",
    "бирж",
    "крипт",
    "биткоин",
    "налог",
    "юрист",
    "медицин",
    "болезн",
    "диагноз",
    "лекарств",
    "стих",
    "сочини",
    "эссе",
    "перевед",
    "математ",
    "уравнен",
    "интеграл",
    "филосо",
    "политик",
    "выбор",
    "новост",
    "code",
    "function",
    "recipe",
    "cook",
    "stock",
    "crypto",
    "bitcoin",
    "tax",
    "lawyer",
    "medical",
    "disease",
    "diagnos",
    "poem",
    "essay",
    "translate",
    "math",
    "equation",
    "philosoph",
    "politic",
    "election",
    "news",
    "homework",
];

/// Below this character count we never reject — too little signal, likely a
/// follow-up ("да", "а завтра?", "Москва").
const MIN_LEN_TO_JUDGE: usize = 12;

/// Classify a user message. Default is [`Scope::InScope`]; we only return
/// [`Scope::OffTopic`] on a clear off-topic hit with no weather/travel signal.
pub fn classify(text: &str) -> Scope {
    let low = text.to_lowercase();

    // Any weather/travel signal ⇒ always allow.
    if ON_TOPIC.iter().any(|t| low.contains(t)) {
        return Scope::InScope;
    }

    // Too short / ambiguous ⇒ allow (probably a follow-up or place name).
    if low.chars().count() < MIN_LEN_TO_JUDGE {
        return Scope::InScope;
    }

    // No on-topic signal + a clear off-topic marker ⇒ refuse.
    if OFF_TOPIC.iter().any(|t| low.contains(t)) {
        return Scope::OffTopic;
    }

    Scope::InScope
}

/// The fixed refusal sent for off-topic messages (RU — primary user language).
pub const OFF_TOPIC_REPLY: &str = "Я ассистент по погоде и планированию путешествий — \
помогаю с прогнозами, выбором времени поездки и условиями в городах. \
По другим темам не подскажу. Спросите, например: «какая погода в Сочи на выходных?»";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weather_is_in_scope() {
        assert_eq!(classify("Какая погода в Москве завтра?"), Scope::InScope);
        assert_eq!(
            classify("will it rain in Berlin this weekend"),
            Scope::InScope
        );
    }

    #[test]
    fn travel_is_in_scope() {
        assert_eq!(
            classify("Хочу спланировать поездку в Сочи на следующей неделе"),
            Scope::InScope
        );
        assert_eq!(
            classify("planning a trip to Rome, need a hotel"),
            Scope::InScope
        );
    }

    #[test]
    fn clear_off_topic_is_rejected() {
        assert_eq!(
            classify("Напиши функцию на python для сортировки"),
            Scope::OffTopic
        );
        assert_eq!(
            classify("give me a recipe for lasagna please"),
            Scope::OffTopic
        );
        assert_eq!(
            classify("реши уравнение x^2 + 2x + 1 = 0 пожалуйста"),
            Scope::OffTopic
        );
    }

    #[test]
    fn short_or_ambiguous_is_allowed() {
        // Follow-ups and bare place names must not be dropped.
        assert_eq!(classify("да"), Scope::InScope);
        assert_eq!(classify("а завтра?"), Scope::InScope);
        assert_eq!(classify("Москва"), Scope::InScope);
    }

    #[test]
    fn off_topic_word_with_weather_signal_is_allowed() {
        // Mixed message: weather wins, never reject.
        assert_eq!(
            classify("я программист, какая погода в Питере?"),
            Scope::InScope
        );
    }
}
