//! Partial JSON repair for streaming structured output.
//!
//! When an LLM streams a JSON response token by token, intermediate payloads are
//! syntactically incomplete (truncated strings, dropped braces). This module
//! repairs such fragments so that `serde_json` can still deserialise them,
//! producing values where the fields that have already arrived are populated and
//! the rest are left at their `Default`.
//!
//! The algorithm is a single-pass scanner backed by a state stack, adapted from
//! the Vercel AI SDK's `fixJson`. It tracks `last_valid_index` and slices
//! before closing open structures, so truncated literals/numbers/keys that are
//! not yet valid are dropped without complex backtracking.
//!
//! # Usage note
//!
//! Struct fields for streaming structured output should be annotated with
//! `#[serde(default)]` so that keys not yet emitted by the model are populated
//! with their `Default` value rather than causing deserialisation failures.

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Root,
    Finish,
    InsideString,
    InsideStringEscape,
    InsideLiteral,
    InsideNumber,
    InsideObjectStart,
    InsideObjectKey,
    InsideObjectAfterKey,
    InsideObjectBeforeValue,
    InsideObjectAfterValue,
    InsideObjectAfterComma,
    InsideArrayStart,
    InsideArrayAfterValue,
    InsideArrayAfterComma,
}

/// Repairs a syntactically truncated JSON fragment so `serde_json` can parse it.
///
/// The repair strategy:
/// 1. Scan the input byte-by-byte, tracking structural validity via a state stack.
/// 2. Record `last_valid_index` — the last position where the JSON was structurally complete.
/// 3. At EOF, slice to `last_valid_index + 1`, then walk the stack and close open
///    constructs (strings, objects, arrays).
/// 4. For `INSIDE_LITERAL`, complete truncated `true`/`false`/`null`.
pub(crate) fn repair_json(input: &str) -> String {
    let s = input.trim();
    if s.is_empty() || !s.starts_with('{') {
        return "{}".to_string();
    }

    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();

    let mut stack: Vec<State> = vec![State::Root];
    let mut last_valid_index: i64 = -1;
    let mut literal_start: Option<usize> = None;

    // Helper: push `swap` before the new value state.
    let process_value_start =
        |stack: &mut Vec<State>,
         c: char,
         swap: State,
         last_valid_index: &mut i64,
         literal_start: &mut Option<usize>,
         i: usize| {
            match c {
                '"' => {
                    *last_valid_index = i as i64;
                    stack.pop();
                    stack.push(swap);
                    stack.push(State::InsideString);
                }
                'f' | 't' | 'n' => {
                    *last_valid_index = i as i64;
                    *literal_start = Some(i);
                    stack.pop();
                    stack.push(swap);
                    stack.push(State::InsideLiteral);
                }
                '-' => {
                    stack.pop();
                    stack.push(swap);
                    stack.push(State::InsideNumber);
                }
                '0'..='9' => {
                    *last_valid_index = i as i64;
                    stack.pop();
                    stack.push(swap);
                    stack.push(State::InsideNumber);
                }
                '{' => {
                    *last_valid_index = i as i64;
                    stack.pop();
                    stack.push(swap);
                    stack.push(State::InsideObjectStart);
                }
                '[' => {
                    *last_valid_index = i as i64;
                    stack.pop();
                    stack.push(swap);
                    stack.push(State::InsideArrayStart);
                }
                _ => {}
            }
        };

    for i in 0..n {
        let c = chars[i];
        let current = stack[stack.len() - 1];

        match current {
            State::Root => {
                process_value_start(
                    &mut stack, c, State::Finish, &mut last_valid_index, &mut literal_start, i,
                );
            }

            State::InsideObjectStart => match c {
                '"' => {
                    stack.pop();
                    stack.push(State::InsideObjectKey);
                }
                '}' => {
                    last_valid_index = i as i64;
                    stack.pop();
                }
                _ => {}
            },

            State::InsideObjectAfterComma => match c {
                '"' => {
                    stack.pop();
                    stack.push(State::InsideObjectKey);
                }
                _ => {}
            },

            State::InsideObjectKey => {
                // We're inside a key string — just wait for the closing quote.
                // Escape sequences inside keys are rare but handled generically.
                match c {
                    '\\' => {
                        stack.push(State::InsideStringEscape);
                    }
                    '"' => {
                        stack.pop();
                        stack.push(State::InsideObjectAfterKey);
                    }
                    _ => {}
                }
            }

            State::InsideObjectAfterKey => match c {
                ':' => {
                    stack.pop();
                    stack.push(State::InsideObjectBeforeValue);
                }
                _ => {}
            },

            State::InsideObjectBeforeValue => {
                process_value_start(
                    &mut stack,
                    c,
                    State::InsideObjectAfterValue,
                    &mut last_valid_index,
                    &mut literal_start,
                    i,
                );
            }

            State::InsideObjectAfterValue => {
                process_after_object_value(c, i, &mut stack, &mut last_valid_index);
            }

            State::InsideString => match c {
                '"' => {
                    stack.pop();
                    last_valid_index = i as i64;
                }
                '\\' => {
                    stack.push(State::InsideStringEscape);
                }
                _ => {
                    last_valid_index = i as i64;
                }
            },

            State::InsideStringEscape => {
                stack.pop();
                last_valid_index = i as i64;
            }

            State::InsideArrayStart => match c {
                ']' => {
                    last_valid_index = i as i64;
                    stack.pop();
                }
                _ => {
                    last_valid_index = i as i64;
                    process_value_start(
                        &mut stack,
                        c,
                        State::InsideArrayAfterValue,
                        &mut last_valid_index,
                        &mut literal_start,
                        i,
                    );
                }
            },

            State::InsideArrayAfterValue => {
                process_after_array_value(c, i, &mut stack, &mut last_valid_index);
            }

            State::InsideArrayAfterComma => {
                process_value_start(
                    &mut stack,
                    c,
                    State::InsideArrayAfterValue,
                    &mut last_valid_index,
                    &mut literal_start,
                    i,
                );
            }

            State::InsideNumber => match c {
                '0'..='9' => {
                    last_valid_index = i as i64;
                }
                'e' | 'E' | '-' | '.' => {
                    // These are allowed inside numbers but aren't "complete" positions.
                }
                ',' => {
                    stack.pop();
                    match stack.last() {
                        Some(State::InsideArrayAfterValue) => {
                            process_after_array_value(c, i, &mut stack, &mut last_valid_index);
                        }
                        Some(State::InsideObjectAfterValue) => {
                            process_after_object_value(c, i, &mut stack, &mut last_valid_index);
                        }
                        _ => {}
                    }
                }
                '}' => {
                    stack.pop();
                    if stack.last() == Some(&State::InsideObjectAfterValue) {
                        process_after_object_value(c, i, &mut stack, &mut last_valid_index);
                    }
                }
                ']' => {
                    stack.pop();
                    if stack.last() == Some(&State::InsideArrayAfterValue) {
                        process_after_array_value(c, i, &mut stack, &mut last_valid_index);
                    }
                }
                _ => {
                    // Invalid char inside number — pop the number state.
                    stack.pop();
                }
            },

            State::InsideLiteral => {
                let Some(start) = literal_start else {
                    stack.pop();
                    continue;
                };
                let partial: String = chars[start..=i].iter().collect();
                if !"false".starts_with(&partial)
                    && !"true".starts_with(&partial)
                    && !"null".starts_with(&partial)
                {
                    stack.pop();
                    match stack.last() {
                        Some(State::InsideObjectAfterValue) => {
                            process_after_object_value(c, i, &mut stack, &mut last_valid_index);
                        }
                        Some(State::InsideArrayAfterValue) => {
                            process_after_array_value(c, i, &mut stack, &mut last_valid_index);
                        }
                        _ => {}
                    }
                } else {
                    last_valid_index = i as i64;
                }
            }

            State::Finish => {
                // Should not be reached during scan.
            }
        }
    }

    // ── Build the result ───────────────────────────────────────────────────

    let end = (last_valid_index + 1) as usize;
    // Clamp: if nothing was valid, keep the opening brace.
    let end = end.min(n).max(1);
    let mut result: String = chars[..end].iter().collect();

    // Walk the stack top-down and close open structures.
    for i in (0..stack.len()).rev() {
        let state = stack[i];
        match state {
            State::InsideString => {
                result.push('"');
            }
            State::InsideObjectKey
            | State::InsideObjectAfterKey
            | State::InsideObjectAfterComma
            | State::InsideObjectStart
            | State::InsideObjectBeforeValue
            | State::InsideObjectAfterValue => {
                result.push('}');
            }
            State::InsideArrayStart
            | State::InsideArrayAfterComma
            | State::InsideArrayAfterValue => {
                result.push(']');
            }
            State::InsideLiteral => {
                if let Some(start) = literal_start {
                    let partial: String = chars[start..].iter().collect();
                    if "true".starts_with(&partial) {
                        result.push_str(&"true"[partial.len()..]);
                    } else if "false".starts_with(&partial) {
                        result.push_str(&"false"[partial.len()..]);
                    } else if "null".starts_with(&partial) {
                        result.push_str(&"null"[partial.len()..]);
                    }
                }
            }
            // These states are transient and should not appear on the stack at EOF.
            State::InsideStringEscape | State::InsideNumber | State::Root | State::Finish => {}
        }
    }

    result
}

fn process_after_object_value(
    c: char,
    i: usize,
    stack: &mut Vec<State>,
    last_valid_index: &mut i64,
) {
    match c {
        ',' => {
            stack.pop();
            stack.push(State::InsideObjectAfterComma);
        }
        '}' => {
            *last_valid_index = i as i64;
            stack.pop();
        }
        _ => {}
    }
}

fn process_after_array_value(
    c: char,
    i: usize,
    stack: &mut Vec<State>,
    last_valid_index: &mut i64,
) {
    match c {
        ',' => {
            stack.pop();
            stack.push(State::InsideArrayAfterComma);
        }
        ']' => {
            *last_valid_index = i as i64;
            stack.pop();
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn parse_partial_json<T: serde::de::DeserializeOwned>(input: &str) -> Option<T> {
        let repaired = repair_json(input);
        serde_json::from_str::<T>(&repaired).ok()
    }

    fn parse_value(input: &str) -> Option<serde_json::Value> {
        let repaired = repair_json(input);
        serde_json::from_str::<serde_json::Value>(&repaired).ok()
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Weather {
        #[serde(default)]
        city: String,
        #[serde(default)]
        temp_c: f64,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Person {
        #[serde(default)]
        name: String,
        #[serde(default)]
        age: f64,
        #[serde(default)]
        active: bool,
        #[serde(default)]
        nickname: Option<String>,
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Empty / minimal
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn empty_string_returns_empty_object() {
        assert_eq!(repair_json(""), "{}");
    }

    #[test]
    fn whitespace_only_returns_empty_object() {
        assert_eq!(repair_json("   "), "{}");
    }

    #[test]
    fn non_json_garbage_returns_empty_object() {
        assert_eq!(repair_json("not json at all"), "{}");
    }

    #[test]
    fn only_open_brace() {
        assert_eq!(repair_json("{"), "{}");
    }

    #[test]
    fn empty_object_is_valid() {
        let result = parse_value("{}").unwrap();
        assert_eq!(result, serde_json::json!({}));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Complete JSON unchanged
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn complete_simple_object() {
        let s = r#"{"city":"NYC","temp_c":23.0}"#;
        let result = parse_partial_json::<Weather>(s).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 23.0);
    }

    #[test]
    fn complete_single_field() {
        let s = r#"{"city":"Tokyo"}"#;
        let result = parse_partial_json::<Weather>(s).unwrap();
        assert_eq!(result.city, "Tokyo");
        assert_eq!(result.temp_c, 0.0);
    }

    #[test]
    fn complete_with_boolean() {
        let s = r#"{"name":"Alice","age":30.0,"active":true}"#;
        let result = parse_partial_json::<Person>(s).unwrap();
        assert_eq!(result.name, "Alice");
        assert_eq!(result.age, 30.0);
        assert!(result.active);
    }

    #[test]
    fn complete_with_null() {
        let s = r#"{"name":"Bob","age":25.0,"active":false,"nickname":null}"#;
        let result = parse_partial_json::<Person>(s).unwrap();
        assert_eq!(result.name, "Bob");
        assert_eq!(result.nickname, None);
    }

    #[test]
    fn complete_nested_object() {
        let s = r#"{"a":{"b":{"c":1}}}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"a": {"b": {"c": 1}}}));
    }

    #[test]
    fn complete_array() {
        let s = r#"{"items":[1,2,3]}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"items": [1, 2, 3]}));
    }

    #[test]
    fn complete_array_of_strings() {
        let s = r#"{"names":["a","b","c"]}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"names": ["a", "b", "c"]}));
    }

    #[test]
    fn complete_array_of_objects() {
        let s = r#"{"people":[{"name":"A"},{"name":"B"}]}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"people": [{"name": "A"}, {"name": "B"}]}));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Truncated string values
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn inside_string_value() {
        let partial = r#"{"city":"NYC"#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 0.0);
    }

    #[test]
    fn truncated_single_char_string() {
        let partial = r#"{"city":"N"#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "N");
        assert_eq!(result.temp_c, 0.0);
    }

    #[test]
    fn unclosed_brace_but_values_complete() {
        let partial = r#"{"city":"NYC","temp_c":23.0"#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 23.0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // String escaping
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn escaped_quote_inside_string() {
        // "text":"He said \"hello\"" — fully valid
        let s = r#"{"text":"He said \"hello\""}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"text": r#"He said "hello""#}));
    }

    #[test]
    fn truncated_after_escaped_quote() {
        // Model stopped mid-escape: `\` then EOF
        let partial = r#"{"text":"Hello \"#;
        let repaired = repair_json(partial);
        // Should close string and object.
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(v["text"].as_str().unwrap(), "Hello ");
    }

    #[test]
    fn truncated_string_after_backslash() {
        // Stream stopped right after a backslash.
        let partial = r#"{"text":"Hello\"#;
        let repaired = repair_json(partial);
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(v["text"].as_str().unwrap(), "Hello");
    }

    #[test]
    fn escaped_backslash_in_string() {
        // "path":"C:\\Users" — backslash before another char
        let s = r#"{"path":"C:\\Users"}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"path": r#"C:\Users"#}));
    }

    #[test]
    fn truncated_mid_escape_in_string() {
        // Model stopped right after reading `\` and the escaped char is missing.
        // `{"path":"C:\` — backslash at EOF, no following char.
        let partial = r#"{"path":"C:\"#;
        let repaired = repair_json(partial);
        // Should survive — the escape state pops but we still close the string.
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Truncated numbers
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn truncated_float_before_fraction() {
        // `23.` — `23` before the `.` is a valid integer, so kept.
        let partial = r#"{"city":"NYC","temp_c":23."#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 23.0);
    }

    #[test]
    fn truncated_number_single_digit() {
        // `2` is a valid number but the stream cut before `,` or `}`.
        // The scanner treats it as still inside a number and drops it.
        let partial = r#"{"city":"LA","temp_c":2"#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "LA");
        // temp_c is dropped because this is EOF without structural close.
    }

    #[test]
    fn negative_number_complete() {
        let s = r#"{"temp":-5}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"temp": -5}));
    }

    #[test]
    fn negative_number_truncated() {
        // `{"temp":-` — minus sign with no digits.
        let partial = r#"{"temp":-"#;
        let repaired = repair_json(partial);
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        // The number state is on the stack, closed by `}` — no value.
        assert!(v.as_object().unwrap().is_empty());
    }

    #[test]
    fn negative_number_partial_digit() {
        // `-5` is a valid complete number — kept.
        let partial = r#"{"a":1,"temp":-5"#;
        let repaired = repair_json(partial);
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(v, serde_json::json!({"a": 1, "temp": -5}));
    }

    #[test]
    fn scientific_notation() {
        let s = r#"{"val":1.5e10}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"val": 1.5e10}));
    }

    #[test]
    fn scientific_notation_truncated() {
        // `{"val":1.5e` — `1.5` before the `e` is a valid number, so kept.
        let partial = r#"{"val":1.5e"#;
        let repaired = repair_json(partial);
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(v, serde_json::json!({"val": 1.5}));
    }

    #[test]
    fn zero_value() {
        let s = r#"{"count":0}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"count": 0}));
    }

    #[test]
    fn negative_zero() {
        let s = r#"{"val":-0}"#;
        let v = parse_value(s).unwrap();
        // -0.0 == 0.0 in IEEE 754.
        assert_eq!(v["val"].as_f64().unwrap(), 0.0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Booleans & null
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn complete_true() {
        let partial = r#"{"name":"Eve","active":true}"#;
        let result = parse_partial_json::<Person>(partial).unwrap();
        assert_eq!(result.name, "Eve");
        assert!(result.active);
    }

    #[test]
    fn complete_false() {
        let partial = r#"{"name":"Eve","active":false}"#;
        let result = parse_partial_json::<Person>(partial).unwrap();
        assert!(!result.active);
    }

    #[test]
    fn complete_null() {
        let partial = r#"{"name":"Eve","nickname":null}"#;
        let result = parse_partial_json::<Person>(partial).unwrap();
        assert_eq!(result.nickname, None);
    }

    #[test]
    fn truncated_true_tr() {
        // At EOF, `tru` → completes to `true`.
        let partial = r#"{"active":tru"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"active": true}));
    }

    #[test]
    fn truncated_true_t_only() {
        let partial = r#"{"active":t"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"active": true}));
    }

    #[test]
    fn truncated_true_in_object_with_multiple_fields() {
        let partial = r#"{"name":"Zoe","active":tr"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"name": "Zoe", "active": true}));
    }

    #[test]
    fn truncated_false_fals() {
        let partial = r#"{"active":fals"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"active": false}));
    }

    #[test]
    fn truncated_false_f_only() {
        let partial = r#"{"active":f"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"active": false}));
    }

    #[test]
    fn truncated_null_nul() {
        let partial = r#"{"nickname":nul"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"nickname": null}));
    }

    #[test]
    fn truncated_null_n_only() {
        let partial = r#"{"nickname":n"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"nickname": null}));
    }

    #[test]
    fn garbage_literal_dropped() {
        // `x` is not a valid start of any literal → dropped.
        let partial = r#"{"a":x"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({}));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Truncated keys
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn truncated_key_dropped() {
        let partial = r#"{"city":"NYC","temp"#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 0.0);
    }

    #[test]
    fn truncated_key_with_escape() {
        // Escape inside a key: `{"foo\"bar` — truncated mid-escape.
        let partial = r#"{"city":"NYC","foo\"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"city": "NYC"}));
    }

    #[test]
    fn truncated_key_after_opening_quote() {
        // `{"ci` — opened key quote, started key chars, EOF.
        let partial = r#"{"ci"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({}));
    }

    #[test]
    fn truncated_after_key_and_colon() {
        let partial = "{\"city\":\"";
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"city": ""}));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Arrays
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn array_truncated_mid_element() {
        let partial = r#"{"items":[1,2,3"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"items": [1, 2, 3]}));
    }

    #[test]
    fn array_truncated_after_comma() {
        let partial = r#"{"items":[1,"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"items": [1]}));
    }

    #[test]
    fn array_truncated_empty() {
        let partial = r#"{"items":["#;
        // Array opened but nothing valid inside. last_valid_index is after `[`.
        // `InsideArrayStart` on stack → closes with `]`.
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"items": []}));
    }

    #[test]
    fn empty_array() {
        let s = r#"{"items":[]}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"items": []}));
    }

    #[test]
    fn array_of_strings_truncated() {
        let partial = r#"{"names":["alice","bob"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"names": ["alice", "bob"]}));
    }

    #[test]
    fn array_of_strings_truncated_mid_comma() {
        let partial = r#"{"names":["alice","#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"names": ["alice"]}));
    }

    #[test]
    fn array_of_objects_truncated() {
        let partial = r#"{"people":[{"name":"A"},{"name":"B"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"people": [{"name": "A"}, {"name": "B"}]})
        );
    }

    #[test]
    fn array_of_objects_truncated_mid_key() {
        let partial = r#"{"people":[{"name":"A"},{"na"#;
        let v = parse_value(partial).unwrap();
        // Truncated key inside second object — `{` was valid so we get an empty `{}`.
        assert_eq!(
            v,
            serde_json::json!({"people": [{"name": "A"}, {}]})
        );
    }

    #[test]
    fn nested_array_truncated() {
        let partial = r#"{"matrix":[[1,2],[3,4"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"matrix": [[1, 2], [3, 4]]}));
    }

    #[test]
    fn boolean_in_array_truncated() {
        let partial = r#"{"flags":[true,fal"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"flags": [true, false]}));
    }

    #[test]
    fn null_in_array_truncated() {
        let partial = r#"{"vals":[1,nul"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"vals": [1, null]}));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Trailing comma (streaming midpoint between fields)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn trailing_comma_inside_object() {
        // Model emitted `"city":"NYC",` and then stopped.
        let partial = r#"{"city":"NYC","#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"city": "NYC"}));
    }

    #[test]
    fn trailing_comma_after_number() {
        let partial = r#"{"city":"NYC","temp_c":23.0,"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"city": "NYC", "temp_c": 23.0}));
    }

    #[test]
    fn trailing_comma_after_boolean() {
        let partial = r#"{"active":true,"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"active": true}));
    }

    #[test]
    fn trailing_comma_after_array() {
        let partial = r#"{"items":[1,2],"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"items": [1, 2]}));
    }

    #[test]
    fn trailing_comma_after_nested_object() {
        let partial = r#"{"inner":{"a":1},"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"inner": {"a": 1}}));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Nested objects
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn nested_object_truncated() {
        let partial = r#"{"a":{"b":1}"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"a": {"b": 1}}));
    }

    #[test]
    fn deeply_nested_truncated() {
        let partial = r#"{"a":{"b":{"c":{"d":42"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"a": {"b": {"c": {"d": 42}}}}));
    }

    #[test]
    fn nested_object_mid_value_truncated() {
        let partial = "{\"a\":{\"b\":\"";
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"a": {"b": ""}}));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Strings containing structural chars
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn braces_inside_string() {
        let s = r#"{"json":"{\"nested\":true}"}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"json": r#"{"nested":true}"#}));
    }

    #[test]
    fn comma_inside_string_does_not_switch_stage() {
        let s = r#"{"greeting":"hello, world"}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"greeting": "hello, world"})
        );
    }

    #[test]
    fn colon_inside_string() {
        let s = r#"{"time":"12:00"}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"time": "12:00"}));
    }

    #[test]
    fn brackets_inside_string() {
        let s = r#"{"arr_str":"[1,2,3]"}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"arr_str": "[1,2,3]"}));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Mixed / multi-field scenarios
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn three_fields_mixed_types() {
        let s = r#"{"name":"Ada","age":30.0,"active":true}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"name": "Ada", "age": 30.0, "active": true})
        );
    }

    #[test]
    fn three_fields_truncated_mid_second_value() {
        let partial = r#"{"name":"Ada","age":30."#;
        let v = parse_value(partial).unwrap();
        // `30.` — `30` before the `.` is a valid number, so kept.
        assert_eq!(v, serde_json::json!({"name": "Ada", "age": 30}));
    }

    #[test]
    fn deeply_mixed_nested_array_and_object() {
        let s = r#"{"user":{"name":"Ada","tags":["rust","llm"],"meta":{"active":true}}}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "user": {
                    "name": "Ada",
                    "tags": ["rust", "llm"],
                    "meta": {"active": true}
                }
            })
        );
    }

    #[test]
    fn deeply_mixed_truncated_mid_array() {
        let partial = r#"{"user":{"name":"Ada","tags":["rust","llm"#;
        let repaired = repair_json(partial);
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "user": {
                    "name": "Ada",
                    "tags": ["rust", "llm"]
                }
            })
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Streaming chunk simulation
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn streaming_chunks_simulated() {
        // Simulates what stream_object sees in production: progressively more JSON.
        let chunks = [
            r#"{"cit"#,
            r#"{"city":"NYC","#,
            r#"{"city":"NYC","temp_c""#,
            r#"{"city":"NYC","temp_c":23"#,
            r#"{"city":"NYC","temp_c":23.0"#,
            r#"{"city":"NYC","temp_c":23.0}"#,
        ];

        for &chunk in chunks.iter() {
            let repaired = repair_json(chunk);
            assert!(
                serde_json::from_str::<serde_json::Value>(&repaired).is_ok(),
                "chunk `{}` → `{}` should parse",
                chunk,
                repaired
            );
        }
    }

    /// Verify all intermediate streaming states parse successfully.
    #[test]
    fn every_streaming_intermediate_is_parseable() {
        let states = [
            // Opening
            r#"{"#,
            r#"{"n"#,
            r#"{"na"#,
            r#"{"nam"#,
            r#"{"name"#,
            r#"{"name""#,
            r#"{"name":"#,
            r#"{"name":""#,
            r#"{"name":"A"#,
            r#"{"name":"Ad"#,
            r#"{"name":"Ada"#,
            r#"{"name":"Ada""#,
            r#"{"name":"Ada","#,
            // Second field
            r#"{"name":"Ada","a"#,
            r#"{"name":"Ada","ag"#,
            r#"{"name":"Ada","age"#,
            r#"{"name":"Ada","age""#,
            r#"{"name":"Ada","age":"#,
            r#"{"name":"Ada","age":3"#,
            r#"{"name":"Ada","age":30"#,
            r#"{"name":"Ada","age":30."#,
            r#"{"name":"Ada","age":30.0"#,
            r#"{"name":"Ada","age":30.0,"#,
            // Third field
            r#"{"name":"Ada","age":30.0,"a"#,
            r#"{"name":"Ada","age":30.0,"ac"#,
            r#"{"name":"Ada","age":30.0,"act"#,
            r#"{"name":"Ada","age":30.0,"acti"#,
            r#"{"name":"Ada","age":30.0,"activ"#,
            r#"{"name":"Ada","age":30.0,"active"#,
            r#"{"name":"Ada","age":30.0,"active""#,
            r#"{"name":"Ada","age":30.0,"active":"#,
            r#"{"name":"Ada","age":30.0,"active":t"#,
            r#"{"name":"Ada","age":30.0,"active":tr"#,
            r#"{"name":"Ada","age":30.0,"active":tru"#,
            r#"{"name":"Ada","age":30.0,"active":true"#,
            // Closing
            r#"{"name":"Ada","age":30.0,"active":true}"#,
        ];

        for state in states.iter() {
            let repaired = repair_json(state);
            let parse_result = serde_json::from_str::<serde_json::Value>(&repaired);
            assert!(
                parse_result.is_ok(),
                "state `{}` → `{}` should parse: {:?}",
                state,
                repaired,
                parse_result.err()
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Whitespace handling
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn whitespace_around_colon_and_comma() {
        let s = r#"{ "city" : "NYC" , "temp_c" : 23.0 }"#;
        let result = parse_partial_json::<Weather>(s).unwrap();
        assert_eq!(result.city, "NYC");
        assert_eq!(result.temp_c, 23.0);
    }

    #[test]
    fn truncated_with_trailing_whitespace() {
        let partial = r#"{"city":"NYC"   "#;
        let result = parse_partial_json::<Weather>(partial).unwrap();
        assert_eq!(result.city, "NYC");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Unicode / CJK
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn unicode_string_value() {
        let s = r#"{"city":"北京","temp_c":25.0}"#;
        let result = parse_partial_json::<Weather>(s).unwrap();
        assert_eq!(result.city, "北京");
        assert_eq!(result.temp_c, 25.0);
    }

    #[test]
    fn unicode_truncated_mid_char() {
        // A multi-byte character truncated in the middle (e.g. "北" = 2 bytes in UTF-8).
        // The input is valid UTF-8 string, so we test truncation at a code-point boundary.
        let partial = r#"{"city":"北京"#;
        let v = parse_value(partial).unwrap();
        assert_eq!(v, serde_json::json!({"city": "北京"}));
    }

    #[test]
    fn emoji_in_string() {
        let s = r#"{"text":"hello 👋"}"#;
        let v = parse_value(s).unwrap();
        assert_eq!(v, serde_json::json!({"text": "hello 👋"}));
    }
}
