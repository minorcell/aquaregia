#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ThinkTagSegment {
    Text(String),
    Reasoning(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct ThinkTagSplitResult {
    pub(crate) text: String,
    pub(crate) reasoning: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ThinkTagStreamParser {
    in_reasoning: bool,
    carry: String,
    case_insensitive: bool,
}

impl ThinkTagStreamParser {
    pub(crate) fn new(case_insensitive: bool) -> Self {
        Self {
            in_reasoning: false,
            carry: String::new(),
            case_insensitive,
        }
    }

    pub(crate) fn feed(&mut self, chunk: &str) -> Vec<ThinkTagSegment> {
        if chunk.is_empty() {
            return Vec::new();
        }

        let input = if self.carry.is_empty() {
            chunk.to_string()
        } else {
            let mut merged = String::with_capacity(self.carry.len() + chunk.len());
            merged.push_str(&self.carry);
            merged.push_str(chunk);
            self.carry.clear();
            merged
        };

        self.consume(&input)
    }

    pub(crate) fn finish(&mut self) -> Vec<ThinkTagSegment> {
        if self.carry.is_empty() {
            return Vec::new();
        }
        let trailing = std::mem::take(&mut self.carry);
        let mut out = Vec::new();
        self.push_segment(&mut out, &trailing);
        out
    }

    fn consume(&mut self, input: &str) -> Vec<ThinkTagSegment> {
        let mut out = Vec::new();
        let mut cursor = 0usize;
        let mut plain_start = 0usize;

        while cursor < input.len() {
            let Some(relative_lt) = input[cursor..].find('<') else {
                break;
            };
            let tag_start = cursor + relative_lt;
            let remainder = &input[tag_start..];

            match classify_tag_prefix(remainder, self.case_insensitive) {
                TagMatch::Full(token, tag_len) => {
                    let can_toggle = match token {
                        TagToken::Open => !self.in_reasoning,
                        TagToken::Close => self.in_reasoning,
                    };

                    if can_toggle {
                        if tag_start > plain_start {
                            self.push_segment(&mut out, &input[plain_start..tag_start]);
                        }
                        self.in_reasoning = matches!(token, TagToken::Open);
                        let after_tag = tag_start + tag_len;
                        cursor = after_tag;
                        plain_start = after_tag;
                    } else {
                        cursor = tag_start + 1;
                    }
                }
                TagMatch::Partial => {
                    if tag_start > plain_start {
                        self.push_segment(&mut out, &input[plain_start..tag_start]);
                    }
                    self.carry.push_str(&input[tag_start..]);
                    return out;
                }
                TagMatch::None => {
                    cursor = tag_start + 1;
                }
            }
        }

        if plain_start < input.len() {
            self.push_segment(&mut out, &input[plain_start..]);
        }
        out
    }

    fn push_segment(&self, out: &mut Vec<ThinkTagSegment>, text: &str) {
        if text.is_empty() {
            return;
        }
        match (self.in_reasoning, out.last_mut()) {
            (false, Some(ThinkTagSegment::Text(existing))) => existing.push_str(text),
            (true, Some(ThinkTagSegment::Reasoning(existing))) => existing.push_str(text),
            (false, _) => out.push(ThinkTagSegment::Text(text.to_string())),
            (true, _) => out.push(ThinkTagSegment::Reasoning(text.to_string())),
        }
    }
}

pub(crate) fn split_think_tags(input: &str, case_insensitive: bool) -> ThinkTagSplitResult {
    let mut parser = ThinkTagStreamParser::new(case_insensitive);
    let mut result = ThinkTagSplitResult::default();
    for segment in parser
        .feed(input)
        .into_iter()
        .chain(parser.finish().into_iter())
    {
        match segment {
            ThinkTagSegment::Text(text) => result.text.push_str(&text),
            ThinkTagSegment::Reasoning(text) => result.reasoning.push_str(&text),
        }
    }
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagToken {
    Open,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagMatch {
    Full(TagToken, usize),
    Partial,
    None,
}

const THINK_OPEN: &str = "<think>";
const THINKING_OPEN: &str = "<thinking>";
const THINK_CLOSE: &str = "</think>";
const THINKING_CLOSE: &str = "</thinking>";
const KNOWN_TAGS: [(&str, TagToken); 4] = [
    (THINK_OPEN, TagToken::Open),
    (THINKING_OPEN, TagToken::Open),
    (THINK_CLOSE, TagToken::Close),
    (THINKING_CLOSE, TagToken::Close),
];

fn classify_tag_prefix(input: &str, case_insensitive: bool) -> TagMatch {
    if !input.starts_with('<') {
        return TagMatch::None;
    }

    for (literal, token) in KNOWN_TAGS {
        if has_prefix(input, literal, case_insensitive) {
            return TagMatch::Full(token, literal.len());
        }
    }

    for (literal, _) in KNOWN_TAGS {
        if is_prefix_of_known_tag(input, literal, case_insensitive) {
            return TagMatch::Partial;
        }
    }

    TagMatch::None
}

fn has_prefix(input: &str, prefix: &str, case_insensitive: bool) -> bool {
    if input.len() < prefix.len() {
        return false;
    }
    let input_prefix = &input[..prefix.len()];
    if case_insensitive {
        input_prefix.eq_ignore_ascii_case(prefix)
    } else {
        input_prefix == prefix
    }
}

fn is_prefix_of_known_tag(input: &str, full_tag: &str, case_insensitive: bool) -> bool {
    if input.len() >= full_tag.len() {
        return false;
    }
    let full_tag_prefix = &full_tag[..input.len()];
    if case_insensitive {
        input.eq_ignore_ascii_case(full_tag_prefix)
    } else {
        input == full_tag_prefix
    }
}

#[cfg(test)]
mod tests {
    use super::{ThinkTagSegment, ThinkTagStreamParser, split_think_tags};

    #[test]
    fn split_think_tags_extracts_reasoning_and_output() {
        let parsed = split_think_tags("<thinking>internal plan</thinking>final answer", true);
        assert_eq!(parsed.reasoning, "internal plan");
        assert_eq!(parsed.text, "final answer");
    }

    #[test]
    fn split_think_tags_does_not_misclassify_reason_text() {
        let parsed = split_think_tags("reasoning is hard; no xml tags here", true);
        assert_eq!(parsed.reasoning, "");
        assert_eq!(parsed.text, "reasoning is hard; no xml tags here");
    }

    #[test]
    fn stream_parser_handles_cross_chunk_tags() {
        let mut parser = ThinkTagStreamParser::new(true);
        let first = parser.feed("<thi");
        let second = parser.feed("nking>alpha");
        let third = parser.feed(" beta</thinking>ok");
        let fourth = parser.finish();
        let all = [first, second, third, fourth].concat();

        let mut reasoning = String::new();
        let mut text = String::new();
        for segment in all {
            match segment {
                ThinkTagSegment::Reasoning(value) => reasoning.push_str(&value),
                ThinkTagSegment::Text(value) => text.push_str(&value),
            }
        }

        assert_eq!(reasoning, "alpha beta");
        assert_eq!(text, "ok");
    }

    #[test]
    fn stream_parser_keeps_partial_unclosed_tail() {
        let mut parser = ThinkTagStreamParser::new(true);
        let chunks = parser.feed("hello <thi");
        assert_eq!(chunks, vec![ThinkTagSegment::Text("hello ".to_string())]);
        let tail = parser.finish();
        assert_eq!(tail, vec![ThinkTagSegment::Text("<thi".to_string())]);
    }

    #[test]
    fn stream_parser_treats_nested_open_tag_as_literal_text_inside_reasoning() {
        let parsed = split_think_tags("<think>outer<think>inner</think>tail", true);
        assert_eq!(parsed.reasoning, "outer<think>inner");
        assert_eq!(parsed.text, "tail");
    }
}
