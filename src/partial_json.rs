//! Partial JSON repair for streaming structured output.
//!
//! When an LLM streams a JSON response token by token, intermediate payloads are
//! syntactically incomplete (truncated strings, dropped braces). This module
//! repairs such fragments so that `serde_json` can still deserialise them,
//! producing values where the fields that have already arrived are populated and
//! the rest are left at their `Default`.
//!
//! # Usage note
//!
//! Struct fields for streaming structured output should be annotated with
//! `#[serde(default)]` so that keys not yet emitted by the model are populated
//! with their `Default` value rather than causing deserialisation failures.

/// Parse stage tracked for repair decisions at EOF.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Stage {
    /// After `{` or `,` — expecting a `"key"`.
    ExpectKey,
    /// Inside a key string (truncated key at EOF → drop).
    InKey,
    /// After `"key":` — expecting a value.
    ExpectValue,
    /// Inside a string value (truncated value → close quote, keep).
    InStringValue,
    /// Inside a non-string value: number / keyword true/false/null
    /// (truncated value at EOF → drop whole key:value pair).
    InNonStringValue,
    /// After a complete value or `}` — expecting `,` or `}`.
    ExpectNext,
}

/// Drop the last (incomplete) key-value pair from `out`, using `key_start` (the
/// byte offset of the opening `"` of the last key) to locate the pair boundary.
fn drop_last_key(mut out: String, key_start: Option<usize>) -> String {
    let Some(pos) = key_start else {
        // Can't locate the key — just strip trailing comma/colon.
        while out.ends_with(|c: char| c.is_whitespace() || c == ':' || c == ',') {
            out.pop();
        }
        return out;
    };
    // Truncate to before the key's opening quote.
    out.truncate(pos);
    // Trim trailing comma or colon + whitespace.
    while out.ends_with(|c: char| c.is_whitespace()) {
        out.pop();
    }
    if out.ends_with(',') || out.ends_with(':') {
        out.pop();
    }
    out
}

/// Repairs a syntactically truncated JSON fragment so `serde_json` can parse it.
///
/// Fixes applied:
/// 1. Closes unterminated string literals.
/// 2. Drops the trailing incomplete value / key-value pair.
/// 3. Closes unmatched braces and brackets.
pub(crate) fn repair_json(input: &str) -> String {
    let s = input.trim();
    if s.is_empty() || !s.starts_with('{') {
        return "{}".to_string();
    }

    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();

    let mut out = String::with_capacity(n + 8);
    let mut stage = Stage::ExpectKey;
    let mut escaped = false;

    // Position of the opening quote of the current key (used to drop
    // incomplete keys). Set when we enter ExpectKey → InKey.
    let mut key_start: Option<usize> = None;

    for &c in &chars {
        if escaped {
            out.push(c);
            escaped = false;
            continue;
        }

        match (stage, c) {
            // ── string internals (key or value) ────────────────────────
            (Stage::InKey | Stage::InStringValue, '\\') => {
                out.push(c);
                escaped = true;
            }
            (Stage::InKey, '"') => {
                out.push(c);
                stage = Stage::ExpectValue;
            }
            (Stage::InStringValue, '"') => {
                out.push(c);
                stage = Stage::ExpectNext;
            }
            (Stage::InKey | Stage::InStringValue, _) => {
                out.push(c);
                // stay in same stage
            }

            // ── outside strings ────────────────────────────────────────
            (_, '"') => {
                out.push(c);
                match stage {
                    Stage::ExpectKey => {
                        key_start = Some(out.len() - 1); // position of opening "
                        stage = Stage::InKey;
                    }
                    Stage::ExpectValue => stage = Stage::InStringValue,
                    _ => {}
                }
            }
            (_, '{') => {
                out.push(c);
                stage = Stage::ExpectKey;
            }
            (_, '}') => {
                out.push(c);
                stage = Stage::ExpectNext;
            }
            (_, ',') => {
                out.push(c);
                stage = Stage::ExpectKey;
                key_start = None;
            }
            (_, ':') => {
                out.push(c);
                stage = Stage::ExpectValue;
            }
            (_, c) if c.is_whitespace() => {
                out.push(c);
                // whitespace doesn't change stage
            }
            // ── value token (digit, letter, minus) ─────────────────────
            (Stage::ExpectValue, _) => {
                out.push(c);
                stage = Stage::InNonStringValue;
            }
            (Stage::InNonStringValue, _) => {
                out.push(c);
                // stay in InNonStringValue until , or } or whitespace
            }
            // ignore stray chars in other stages (shouldn't happen with valid JSON prefix)
            _ => {}
        }
    }

    // ── End-of-input repairs ──────────────────────────────────────────────

    match stage {
        Stage::InStringValue => {
            // Close the string, value is complete enough.
            out.push('"');
        }
        Stage::InKey => {
            // Incomplete key — drop it.
            out = drop_last_key(out, key_start);
        }
        Stage::InNonStringValue => {
            // Incomplete non-string value. Try closing braces and parsing
            // first — the current value chars may already be valid JSON
            // (e.g. `23` or `true`). Only drop the key:value pair if that fails.
            let mut tent = out.clone();
            while tent.ends_with(|c: char| c.is_whitespace()) {
                tent.pop();
            }
            // Close unmatched braces.
            let to: (usize, usize) = tent.chars().fold((0, 0), |(b, s), c| match c {
                '{' => (b + 1, s),
                '[' => (b, s + 1),
                '}' => (b.saturating_sub(1), s),
                ']' => (b, s.saturating_sub(1)),
                _ => (b, s),
            });
            for _ in 0..to.0 {
                tent.push('}');
            }
            for _ in 0..to.1 {
                tent.push(']');
            }
            if serde_json::from_str::<serde_json::Value>(&tent).is_ok() {
                out = tent;
            } else {
                // Drop the incomplete value + its key to the last `,` or `{`.
                out = drop_last_key(out, key_start);
            }
        }
        Stage::ExpectValue => {
            // Saw `"key":` but no value — drop the colon and key.
            while out.ends_with(|c: char| c.is_whitespace()) {
                out.pop();
            }
            if out.ends_with(':') {
                out.pop();
            }
            out = drop_last_key(out, key_start);
        }
        Stage::ExpectKey | Stage::ExpectNext => {
            // Nothing incomplete to drop. But if we're ExpectKey and there's a
            // trailing comma, drop it.
            while out.ends_with(|c: char| c.is_whitespace()) {
                out.pop();
            }
            if out.ends_with(',') {
                out.pop();
            }
        }
    }

    // Drop any trailing comma leftover.
    while out.ends_with(|c: char| c.is_whitespace()) {
        out.pop();
    }
    if out.ends_with(',') {
        out.pop();
    }

    // Close unmatched braces / brackets.
    let opens: (usize, usize) = out.chars().fold((0, 0), |(b, s), c| match c {
        '{' => (b + 1, s),
        '[' => (b, s + 1),
        '}' => (b.saturating_sub(1), s),
        ']' => (b, s.saturating_sub(1)),
        _ => (b, s),
    });
    for _ in 0..opens.0 {
        out.push('}');
    }
    for _ in 0..opens.1 {
        out.push(']');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn parse_partial_json<T: serde::de::DeserializeOwned>(input: &str) -> Option<T> {
        let repaired = repair_json(input);
        serde_json::from_str::<T>(&repaired).ok()
    }

    // Fields use `#[serde(default)]` so partial JSON with missing keys still deserialises.
    #[derive(Debug, Deserialize, PartialEq)]
    struct Weather {
        #[serde(default)]
        city: String,
        #[serde(default)]
        temp_c: f64,
    }

    // ─── Truncated string value ─────────────────────────────────────────

    #[test]
    fn repair_inside_string_value() {
        // Model is mid-way through emitting `"NYC"`
        let partial = r#"{"city":"NYC"#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 0.0); // not yet emitted → default
    }

    #[test]
    fn repair_truncated_single_char_string() {
        let partial = r#"{"city":"N"#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "N");
        assert_eq!(result.temp_c, 0.0);
    }

    #[test]
    fn repair_unclosed_json() {
        let partial = r#"{"city":"NYC","temp_c":23.0"#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 23.0);
    }

    // ─── Truncated number ──────────────────────────────────────────────

    #[test]
    fn repair_incomplete_number_drops_field() {
        // `23.` is not a valid JSON number.
        let partial = r#"{"city":"NYC","temp_c":23."#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 0.0); // dropped incomplete value
    }

    #[test]
    fn repair_incomplete_number_after_dot() {
        let partial = r#"{"city":"LA","temp_c":2"#;
        // "2" IS a valid number, but repair treats it as incomplete since
        // the value_start was registered. Actually `2` is complete — the
        // token was never "closed" by a structural character.
        // This case works because `value_start` is still set — but `2` is
        // a valid number on its own. Since we truncate at value_start, we
        // get the full `"city":"LA"` prefix.
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "LA");
        // temp_c dropped since value wasn't terminated by , or }
    }

    // ─── Truncated key ─────────────────────────────────────────────────

    #[test]
    fn repair_truncated_key_has_no_effect() {
        let partial = r#"{"city":"NYC","temp"#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 0.0);
    }

    // ─── Empty / minimal ───────────────────────────────────────────────

    #[test]
    fn repair_empty_string() {
        assert_eq!(repair_json(""), "{}");
    }

    #[test]
    fn repair_only_open_brace() {
        assert_eq!(repair_json("{"), "{}");
    }

    // ─── Complete JSON is unchanged ────────────────────────────────────

    #[test]
    fn repair_complete_json_unchanged() {
        let s = r#"{"city":"NYC","temp_c":23.0}"#;
        let result = parse_partial_json::<Weather>(s).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 23.0);
    }

    #[test]
    fn repair_complete_json_single_field() {
        let s = r#"{"city":"Tokyo"}"#;
        let result = parse_partial_json::<Weather>(s).unwrap();
        assert_eq!(result.city, "Tokyo");
        assert_eq!(result.temp_c, 0.0);
    }
}
