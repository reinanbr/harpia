//! Tolerant JSON parsing: turns almost-JSON produced by LLMs into a [`serde_json::Value`].
//!
//! Handles: single-quoted strings, unquoted keys and bare words, missing or trailing commas,
//! missing colons, unclosed strings/objects/arrays (closed at the end of input), raw newlines
//! and control characters inside strings, unescaped inner quotes, `//` and `/* */` comments,
//! Python/JS literals (`True`, `False`, `None`, `undefined`), leading `+` and `.5` numbers.

use harpia_core::{HarpiaError, Result};
use serde_json::{Map, Number, Value};

/// Parse `text` leniently. The first JSON-looking value is parsed; the rest is ignored.
///
/// ```
/// use harpia_structured::repair::repair;
/// let v = repair("{name: 'Ana', tags: ['a', 'b',], ok: True, n: .5").unwrap();
/// assert_eq!(v, serde_json::json!({"name": "Ana", "tags": ["a", "b"], "ok": true, "n": 0.5}));
/// ```
pub fn repair(text: &str) -> Result<Value> {
    let body = crate::extract::strip_fences(text);
    let chars: Vec<char> = body.chars().collect();
    let Some(start) = chars.iter().position(|c| *c == '{' || *c == '[') else {
        // no container: accept only a strictly valid scalar (e.g. `42`, `"text"`, `true`)
        return serde_json::from_str(body.trim())
            .map_err(|_| HarpiaError::Validation("no JSON object or array found in text".into()));
    };
    let mut p = Parser { s: &chars, i: start };
    p.value().ok_or_else(|| HarpiaError::Validation("no JSON value found in text".into()))
}

/// [`repair`] and serialize back to compact, valid JSON.
pub fn repair_to_string(text: &str) -> Result<String> {
    Ok(serde_json::to_string(&repair(text)?)?)
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn eof(&self) -> bool {
        self.i >= self.s.len()
    }

    fn skip_ws(&mut self) {
        loop {
            while self.peek().is_some_and(char::is_whitespace) {
                self.i += 1;
            }
            match (self.peek(), self.s.get(self.i + 1)) {
                (Some('/'), Some('/')) | (Some('#'), _) => {
                    while self.peek().is_some_and(|c| c != '\n') {
                        self.i += 1;
                    }
                }
                (Some('/'), Some('*')) => {
                    self.i += 2;
                    while !self.eof() && !(self.peek() == Some('*') && self.s.get(self.i + 1) == Some(&'/')) {
                        self.i += 1;
                    }
                    self.i = (self.i + 2).min(self.s.len());
                }
                _ => return,
            }
        }
    }

    fn value(&mut self) -> Option<Value> {
        self.skip_ws();
        match self.peek()? {
            '{' => Some(self.object()),
            '[' => Some(self.array()),
            q @ ('"' | '\'' | '“' | '‘') => Some(Value::String(self.string(q))),
            c if c.is_ascii_digit() || matches!(c, '-' | '+' | '.') => Some(self.number()),
            c if c.is_alphabetic() || c == '_' || c == '$' => Some(self.word_value()),
            _ => None,
        }
    }

    fn object(&mut self) -> Value {
        self.i += 1; // '{'
        let mut map = Map::new();
        loop {
            self.skip_ws();
            let Some(c) = self.peek() else { break };
            match c {
                '}' => {
                    self.i += 1;
                    break;
                }
                ',' => {
                    self.i += 1;
                    continue;
                }
                ']' => {
                    // mismatched closer: treat as end of object
                    self.i += 1;
                    break;
                }
                _ => {}
            }
            let key = match c {
                '"' | '\'' | '“' | '‘' => self.string(c),
                c if c.is_alphanumeric() || c == '_' || c == '$' || c == '-' => self.word(),
                _ => {
                    self.i += 1;
                    continue;
                }
            };
            self.skip_ws();
            if matches!(self.peek(), Some(':') | Some('=')) {
                self.i += 1;
            }
            self.skip_ws();
            if matches!(self.peek(), Some(',') | Some('}')) || self.eof() {
                // key without value: drop it (common in truncated output)
                continue;
            }
            match self.value() {
                Some(v) => {
                    map.insert(key, v);
                }
                None => {
                    if !self.eof() {
                        self.i += 1;
                    }
                }
            }
        }
        Value::Object(map)
    }

    fn array(&mut self) -> Value {
        self.i += 1; // '['
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            let Some(c) = self.peek() else { break };
            match c {
                ']' => {
                    self.i += 1;
                    break;
                }
                ',' => {
                    self.i += 1;
                    continue;
                }
                '}' => {
                    self.i += 1;
                    break;
                }
                _ => {}
            }
            match self.value() {
                Some(v) => out.push(v),
                None => self.i += 1,
            }
        }
        Value::Array(out)
    }

    fn closing(open: char) -> char {
        match open {
            '“' => '”',
            '‘' => '’',
            c => c,
        }
    }

    /// A quote ends the string only if what follows looks like JSON structure.
    fn is_real_close(&self, at: usize) -> bool {
        let mut j = at + 1;
        while self.s.get(j).is_some_and(|c| *c == ' ' || *c == '\t') {
            j += 1;
        }
        match self.s.get(j) {
            None => true,
            Some(c) => matches!(c, ',' | '}' | ']' | ':' | '\n' | '\r'),
        }
    }

    fn string(&mut self, open: char) -> String {
        let close = Self::closing(open);
        self.i += 1;
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if c == '\\' {
                self.i += 1;
                let Some(e) = self.peek() else { break };
                self.i += 1;
                match e {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'u' => {
                        let hex: String = self.s[self.i..].iter().take(4).collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16)
                            .map_err(|_| ())
                            .and_then(|c| if hex.len() == 4 { Ok(c) } else { Err(()) })
                        {
                            self.i += 4;
                            if (0xD800..0xDC00).contains(&code)
                                && self.s.get(self.i) == Some(&'\\')
                                && self.s.get(self.i + 1) == Some(&'u')
                            {
                                let lo: String = self.s[self.i + 2..].iter().take(4).collect();
                                if let Ok(lo) = u32::from_str_radix(&lo, 16)
                                    && (0xDC00..0xE000).contains(&lo)
                                {
                                    self.i += 6;
                                    let cp = 0x10000 + ((code - 0xD800) << 10) + (lo - 0xDC00);
                                    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                    continue;
                                }
                            }
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                        } else {
                            out.push('u');
                        }
                    }
                    other => out.push(other),
                }
                continue;
            }
            if c == close && (open != '"' || self.is_real_close(self.i)) {
                self.i += 1;
                return out;
            }
            out.push(c);
            self.i += 1;
        }
        out
    }

    fn word(&mut self) -> String {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '$' | '-' | '.')) {
            self.i += 1;
        }
        self.s[start..self.i].iter().collect()
    }

    fn word_value(&mut self) -> Value {
        let start = self.i;
        let w = self.word();
        match w.as_str() {
            "true" | "True" | "TRUE" => Value::Bool(true),
            "false" | "False" | "FALSE" => Value::Bool(false),
            "null" | "None" | "NULL" | "undefined" | "nil" => Value::Null,
            _ => {
                // bare word: read until a structural char (allows spaces inside, e.g. `key: some text,`)
                let mut end = self.i;
                while self.s.get(end).is_some_and(|c| !matches!(c, ',' | '}' | ']' | '\n')) {
                    end += 1;
                }
                self.i = end;
                Value::String(self.s[start..end].iter().collect::<String>().trim_end().to_string())
            }
        }
    }

    fn number(&mut self) -> Value {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E')) {
            self.i += 1;
        }
        let mut raw: String = self.s[start..self.i].iter().collect();
        if raw.starts_with('+') {
            raw.remove(0);
        }
        if raw.starts_with('.') {
            raw.insert(0, '0');
        }
        if raw.starts_with("-.") {
            raw.insert(1, '0');
        }
        // drop trailing junk such as "1e", "2.", "-"
        while !raw.is_empty() && raw.parse::<f64>().is_err() {
            raw.pop();
        }
        if raw.is_empty() || raw == "-" {
            return Value::Null;
        }
        if let Ok(i) = raw.parse::<i64>() {
            return Value::Number(i.into());
        }
        if let Ok(u) = raw.parse::<u64>() {
            return Value::Number(u.into());
        }
        raw.parse::<f64>().ok().and_then(Number::from_f64).map(Value::Number).unwrap_or(Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn valid_json_is_unchanged() {
        let src = r#"{"a": [1, 2.5, -3e2, "x\né"], "b": {"c": null, "d": false}}"#;
        assert_eq!(repair(src).unwrap(), serde_json::from_str::<Value>(src).unwrap());
    }

    #[test]
    fn common_llm_mistakes() {
        assert_eq!(repair(r#"{"a": 1 "b": 2}"#).unwrap(), json!({"a": 1, "b": 2}));
        assert_eq!(repair(r#"{"a": [1, 2,],}"#).unwrap(), json!({"a": [1, 2]}));
        assert_eq!(
            repair(
                r#"{"a": "line1
line2"}"#
            )
            .unwrap(),
            json!({"a": "line1\nline2"})
        );
        assert_eq!(repair(r#"{"q": "he said "hi" ok", "n": 1}"#).unwrap(), json!({"q": "he said \"hi\" ok", "n": 1}));
        assert_eq!(repair("{'a': None, // c\n 'b': /* x */ +2}").unwrap(), json!({"a": null, "b": 2}));
        assert_eq!(
            repair("Sure! Here it is:\n```json\n{\"x\": [1, {\"y\": \"z").unwrap(),
            json!({"x": [1, {"y": "z"}]})
        );
        assert_eq!(repair(r#"{"a": 1, "b":"#).unwrap(), json!({"a": 1}));
        assert_eq!(repair("{status: in progress, n: 3}").unwrap(), json!({"status": "in progress", "n": 3}));
        assert_eq!(repair(r#"["😀"]"#).unwrap(), json!(["😀"]));
    }

    #[test]
    fn no_value_is_an_error() {
        assert!(repair("   ").is_err());
    }

    proptest::proptest! {
        #[test]
        fn never_panics(s in "\\PC{0,64}") {
            let _ = repair(&s);
        }

        #[test]
        fn roundtrips_valid_json(v in arb_json()) {
            let text = serde_json::to_string(&v).unwrap();
            proptest::prop_assert_eq!(repair(&text).unwrap(), v);
        }
    }

    fn arb_json() -> impl proptest::strategy::Strategy<Value = Value> {
        use proptest::prelude::*;
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            any::<i32>().prop_map(|i| json!(i)),
            "[a-zA-Z0-9 ]{0,8}".prop_map(Value::String),
        ];
        leaf.prop_recursive(3, 16, 4, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
                proptest::collection::btree_map("[a-z]{1,5}", inner, 0..4)
                    .prop_map(|m| Value::Object(m.into_iter().collect())),
            ]
        })
    }
}
