//! Constraints for constrained decoding.
//!
//! A [`Constraint`] decides whether a text prefix can still be extended into a valid output.
//! [`token_mask`] turns that into a per-token allow-list for the next decoding step, which
//! `harpia-local` applies to the logits.
//!
//! Supported constraints:
//! - [`Choice`]: one of a fixed list of strings;
//! - [`RegexConstraint`]: the whole output matches a regular expression;
//! - [`JsonSchemaConstraint`]: strict JSON matching a JSON Schema subset (`type`, `properties`,
//!   `required`, `additionalProperties`, `items`, `enum`, `const`, `minItems`, `maxItems`,
//!   `maxLength`). Unsupported keywords (`$ref`, `anyOf`, `oneOf`, `allOf`, `pattern`, ...) are
//!   rejected at construction so the constraint never claims guarantees it cannot give.

use harpia_core::{HarpiaError, Result};
use rayon::prelude::*;
use regex_automata::Anchored;
use regex_automata::dfa::{Automaton, dense};
use regex_automata::util::start::Config as StartConfig;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

/// Status of a prefix under a constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixStatus {
    /// No extension of this prefix can satisfy the constraint.
    Invalid,
    /// Valid so far but not yet a complete output.
    Partial,
    /// A complete valid output (it may or may not be extensible).
    Complete,
}

/// A constraint over generated text.
pub trait Constraint: Send + Sync {
    /// Classify `prefix`.
    fn check(&self, prefix: &str) -> PrefixStatus;
}

/// Allowed next tokens: `mask[i]` is `true` when `prefix + vocab[i]` is not
/// [`PrefixStatus::Invalid`]. Empty vocabulary entries are never allowed. Use the returned
/// [`PrefixStatus`] of the current prefix to decide whether end-of-sequence is allowed
/// (only when it is [`PrefixStatus::Complete`]).
pub fn token_mask<C: Constraint + ?Sized>(constraint: &C, prefix: &str, vocab: &[String]) -> (Vec<bool>, PrefixStatus) {
    let status = constraint.check(prefix);
    let mask = vocab
        .par_iter()
        .map(|tok| {
            if tok.is_empty() {
                return false;
            }
            let mut s = String::with_capacity(prefix.len() + tok.len());
            s.push_str(prefix);
            s.push_str(tok);
            constraint.check(&s) != PrefixStatus::Invalid
        })
        .collect();
    (mask, status)
}

// ------------------------------------------------------------------ choice

/// Output must be exactly one of the given strings.
///
/// ```
/// use harpia_structured::constraint::{Choice, Constraint, PrefixStatus};
/// let c = Choice::new(["yes", "no"]);
/// assert_eq!(c.check("y"), PrefixStatus::Partial);
/// assert_eq!(c.check("no"), PrefixStatus::Complete);
/// assert_eq!(c.check("maybe"), PrefixStatus::Invalid);
/// ```
#[derive(Debug, Clone)]
pub struct Choice {
    options: Vec<String>,
}

impl Choice {
    /// New choice constraint.
    pub fn new<I, S>(options: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self { options: options.into_iter().map(Into::into).collect() }
    }
}

impl Constraint for Choice {
    fn check(&self, prefix: &str) -> PrefixStatus {
        if self.options.iter().any(|o| o == prefix) {
            PrefixStatus::Complete
        } else if self.options.iter().any(|o| o.starts_with(prefix)) {
            PrefixStatus::Partial
        } else {
            PrefixStatus::Invalid
        }
    }
}

// ------------------------------------------------------------------ regex

/// The whole output must match a regular expression (anchored at both ends).
///
/// ```
/// use harpia_structured::constraint::{Constraint, PrefixStatus, RegexConstraint};
/// let c = RegexConstraint::new(r"\d{4}-\d{2}-\d{2}").unwrap();
/// assert_eq!(c.check("2026-1"), PrefixStatus::Partial);
/// assert_eq!(c.check("2026-11-02"), PrefixStatus::Complete);
/// assert_eq!(c.check("2026/"), PrefixStatus::Invalid);
/// ```
#[derive(Debug, Clone)]
pub struct RegexConstraint {
    dfa: dense::DFA<Vec<u32>>,
}

impl RegexConstraint {
    /// Compile `pattern`.
    pub fn new(pattern: &str) -> Result<Self> {
        let dfa = dense::Builder::new()
            .configure(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Anchored))
            .build(&format!("(?:{pattern})$"))
            .map_err(|e| HarpiaError::invalid(format!("invalid regex constraint: {e}")))?;
        Ok(Self { dfa })
    }
}

impl Constraint for RegexConstraint {
    fn check(&self, prefix: &str) -> PrefixStatus {
        let cfg = StartConfig::new().anchored(Anchored::Yes);
        let Ok(mut state) = self.dfa.start_state(&cfg) else { return PrefixStatus::Invalid };
        for &b in prefix.as_bytes() {
            state = self.dfa.next_state(state, b);
            if self.dfa.is_dead_state(state) {
                return PrefixStatus::Invalid;
            }
        }
        let end = self.dfa.next_eoi_state(state);
        if self.dfa.is_match_state(end) { PrefixStatus::Complete } else { PrefixStatus::Partial }
    }
}

// ------------------------------------------------------------------ JSON schema

#[derive(Debug, Clone, Default)]
struct Node {
    types: Option<HashSet<&'static str>>,
    properties: BTreeMap<String, Node>,
    required: Vec<String>,
    /// `None` = any extra property allowed; `Some(None)` = forbidden; `Some(Some(n))` = schema.
    additional: Option<Option<Box<Node>>>,
    items: Option<Box<Node>>,
    enum_values: Option<Vec<Value>>,
    min_items: Option<usize>,
    max_items: Option<usize>,
    max_length: Option<usize>,
}

const SUPPORTED: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "minItems",
    "maxItems",
    "maxLength",
    "title",
    "description",
    "default",
    "examples",
    "$schema",
    "$id",
    "format",
];

impl Node {
    fn parse(v: &Value) -> Result<Node> {
        let obj = match v {
            Value::Bool(true) => return Ok(Node::default()),
            Value::Object(o) => o,
            _ => return Err(HarpiaError::invalid("schema must be an object or `true`")),
        };
        if let Some(k) = obj.keys().find(|k| !SUPPORTED.contains(&k.as_str())) {
            return Err(HarpiaError::invalid(format!(
                "unsupported JSON Schema keyword for constrained decoding: `{k}`"
            )));
        }
        let mut n = Node::default();
        if let Some(t) = obj.get("type") {
            let names: Vec<&str> = match t {
                Value::String(s) => vec![s.as_str()],
                Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
                _ => return Err(HarpiaError::invalid("`type` must be a string or array")),
            };
            let mut set = HashSet::new();
            for name in names {
                set.insert(match name {
                    "object" => "object",
                    "array" => "array",
                    "string" => "string",
                    "number" => "number",
                    "integer" => "integer",
                    "boolean" => "boolean",
                    "null" => "null",
                    other => return Err(HarpiaError::invalid(format!("unknown type `{other}`"))),
                });
            }
            n.types = Some(set);
        }
        if let Some(Value::Object(props)) = obj.get("properties") {
            for (k, s) in props {
                n.properties.insert(k.clone(), Node::parse(s)?);
            }
        }
        if let Some(Value::Array(req)) = obj.get("required") {
            n.required = req.iter().filter_map(Value::as_str).map(String::from).collect();
        }
        match obj.get("additionalProperties") {
            Some(Value::Bool(false)) => n.additional = Some(None),
            Some(Value::Bool(true)) | None => {}
            Some(s) => n.additional = Some(Some(Box::new(Node::parse(s)?))),
        }
        if let Some(items) = obj.get("items") {
            n.items = Some(Box::new(Node::parse(items)?));
        }
        if let Some(Value::Array(e)) = obj.get("enum") {
            n.enum_values = Some(e.clone());
        }
        if let Some(c) = obj.get("const") {
            n.enum_values = Some(vec![c.clone()]);
        }
        let as_usize = |k: &str| obj.get(k).and_then(Value::as_u64).map(|x| x as usize);
        n.min_items = as_usize("minItems");
        n.max_items = as_usize("maxItems");
        n.max_length = as_usize("maxLength");
        Ok(n)
    }

    fn allows(&self, t: &str) -> bool {
        match &self.types {
            None => true,
            Some(s) => s.contains(t) || (t == "integer" && s.contains("number")),
        }
    }

    fn property(&self, key: &str) -> Option<Option<&Node>> {
        if let Some(n) = self.properties.get(key) {
            return Some(Some(n));
        }
        match &self.additional {
            None => Some(None),
            Some(None) => None,
            Some(Some(n)) => Some(Some(n)),
        }
    }
}

static ANY: Node = Node {
    types: None,
    properties: BTreeMap::new(),
    required: Vec::new(),
    additional: None,
    items: None,
    enum_values: None,
    min_items: None,
    max_items: None,
    max_length: None,
};

/// Outcome of validating a value prefix.
enum Step {
    /// Value finished at this char index.
    Done(usize),
    /// Input ended inside the value (still a valid prefix).
    Eof,
}

type V = std::result::Result<Step, ()>;

/// Strict JSON output matching a JSON Schema subset.
///
/// ```
/// use harpia_structured::constraint::{Constraint, JsonSchemaConstraint, PrefixStatus};
/// use serde_json::json;
/// let c = JsonSchemaConstraint::new(&json!({
///     "type": "object",
///     "properties": {"label": {"enum": ["supports", "contradicts"]}, "score": {"type": "number"}},
///     "required": ["label", "score"],
///     "additionalProperties": false
/// })).unwrap();
/// assert_eq!(c.check(r#"{"label": "sup"#), PrefixStatus::Partial);
/// assert_eq!(c.check(r#"{"label": "maybe"#), PrefixStatus::Invalid);
/// assert_eq!(c.check(r#"{"other""#), PrefixStatus::Invalid);
/// assert_eq!(c.check(r#"{"label": "supports"}"#), PrefixStatus::Invalid); // missing "score"
/// assert_eq!(c.check(r#"{"label": "supports", "score": 0.9}"#), PrefixStatus::Complete);
/// ```
#[derive(Debug, Clone)]
pub struct JsonSchemaConstraint {
    root: Node,
}

impl JsonSchemaConstraint {
    /// Build from a JSON Schema value.
    pub fn new(schema: &Value) -> Result<Self> {
        Ok(Self { root: Node::parse(schema)? })
    }
}

impl Constraint for JsonSchemaConstraint {
    fn check(&self, prefix: &str) -> PrefixStatus {
        let s: Vec<char> = prefix.chars().collect();
        match value(&s, 0, &self.root) {
            Err(()) => PrefixStatus::Invalid,
            Ok(Step::Eof) => PrefixStatus::Partial,
            Ok(Step::Done(i)) => {
                if s[i..].iter().all(|c| c.is_whitespace()) {
                    PrefixStatus::Complete
                } else {
                    PrefixStatus::Invalid
                }
            }
        }
    }
}

fn ws(s: &[char], mut i: usize) -> usize {
    while i < s.len() && matches!(s[i], ' ' | '\n' | '\r' | '\t') {
        i += 1;
    }
    i
}

fn value(s: &[char], i: usize, n: &Node) -> V {
    let i = ws(s, i);
    if i >= s.len() {
        return Ok(Step::Eof);
    }
    match s[i] {
        '{' if n.allows("object") => object(s, i, n),
        '[' if n.allows("array") => array(s, i, n),
        '"' if n.allows("string") => {
            let (step, content) = string(s, i, n.max_length)?;
            check_enum(n, &step, &Value::String(content.clone()), Some(&content))?;
            Ok(step)
        }
        c if (c == '-' || c.is_ascii_digit()) && (n.allows("number") || n.allows("integer")) => {
            let integer_only = !n.allows("number");
            let (step, text) = number(s, i, integer_only)?;
            if let (Step::Done(_), Some(vals)) = (&step, &n.enum_values) {
                let v: Value = serde_json::from_str(&text).map_err(|_| ())?;
                if !vals.iter().any(|e| e.as_f64() == v.as_f64()) {
                    return Err(());
                }
            }
            Ok(step)
        }
        't' | 'f' if n.allows("boolean") => literal(s, i, n, &["true", "false"]),
        'n' if n.allows("null") => literal(s, i, n, &["null"]),
        _ => Err(()),
    }
}

fn check_enum(n: &Node, step: &Step, full: &Value, partial_str: Option<&str>) -> std::result::Result<(), ()> {
    let Some(vals) = &n.enum_values else { return Ok(()) };
    let ok = match step {
        Step::Done(_) => vals.contains(full),
        Step::Eof => match partial_str {
            Some(p) => vals.iter().any(|v| v.as_str().is_some_and(|e| e.starts_with(p))),
            None => true,
        },
    };
    if ok { Ok(()) } else { Err(()) }
}

fn literal(s: &[char], i: usize, n: &Node, words: &[&str]) -> V {
    for w in words {
        let wc: Vec<char> = w.chars().collect();
        let avail = (s.len() - i).min(wc.len());
        if s[i..i + avail] == wc[..avail] {
            if avail < wc.len() {
                return Ok(Step::Eof);
            }
            let v: Value = serde_json::from_str(w).map_err(|_| ())?;
            let step = Step::Done(i + wc.len());
            check_enum(n, &step, &v, None)?;
            return Ok(step);
        }
    }
    Err(())
}

/// Returns the step and the decoded content so far.
fn string(s: &[char], i: usize, max_len: Option<usize>) -> std::result::Result<(Step, String), ()> {
    let mut j = i + 1;
    let mut out = String::new();
    while j < s.len() {
        let c = s[j];
        match c {
            '"' => return Ok((Step::Done(j + 1), out)),
            '\\' => {
                let Some(&e) = s.get(j + 1) else { return Ok((Step::Eof, out)) };
                match e {
                    '"' | '\\' | '/' => out.push(e),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let hex = &s[(j + 2).min(s.len())..(j + 6).min(s.len())];
                        if !hex.iter().all(char::is_ascii_hexdigit) {
                            return Err(());
                        }
                        if hex.len() < 4 {
                            return Ok((Step::Eof, out));
                        }
                        j += 4;
                        out.push('\u{FFFD}');
                    }
                    _ => return Err(()),
                }
                j += 2;
            }
            c if (c as u32) < 0x20 => return Err(()),
            c => {
                out.push(c);
                j += 1;
            }
        }
        if max_len.is_some_and(|m| out.chars().count() > m) {
            return Err(());
        }
    }
    Ok((Step::Eof, out))
}

fn number(s: &[char], i: usize, integer_only: bool) -> std::result::Result<(Step, String), ()> {
    let mut j = i;
    let at = |j: usize| s.get(j).copied();
    if at(j) == Some('-') {
        j += 1;
    }
    match at(j) {
        None => return Ok((Step::Eof, String::new())),
        Some('0') => j += 1,
        Some(c) if c.is_ascii_digit() => {
            while at(j).is_some_and(|c| c.is_ascii_digit()) {
                j += 1;
            }
        }
        _ => return Err(()),
    }
    if at(j) == Some('.') {
        if integer_only {
            return Err(());
        }
        j += 1;
        if at(j).is_none() {
            return Ok((Step::Eof, String::new()));
        }
        if !at(j).is_some_and(|c| c.is_ascii_digit()) {
            return Err(());
        }
        while at(j).is_some_and(|c| c.is_ascii_digit()) {
            j += 1;
        }
    }
    if matches!(at(j), Some('e' | 'E')) {
        if integer_only {
            return Err(());
        }
        j += 1;
        if matches!(at(j), Some('+' | '-')) {
            j += 1;
        }
        if at(j).is_none() {
            return Ok((Step::Eof, String::new()));
        }
        if !at(j).is_some_and(|c| c.is_ascii_digit()) {
            return Err(());
        }
        while at(j).is_some_and(|c| c.is_ascii_digit()) {
            j += 1;
        }
    }
    if j >= s.len() {
        // could still grow (more digits); a valid prefix
        return Ok((Step::Eof, String::new()));
    }
    Ok((Step::Done(j), s[i..j].iter().collect()))
}

fn object(s: &[char], i: usize, n: &Node) -> V {
    let mut j = ws(s, i + 1);
    let mut seen: HashSet<String> = HashSet::new();
    let mut first = true;
    loop {
        if j >= s.len() {
            return Ok(Step::Eof);
        }
        if s[j] == '}' && first {
            return close_object(n, &seen, j);
        }
        if s[j] != '"' {
            return Err(());
        }
        let (step, key) = string(s, j, None)?;
        let Step::Done(after_key) = step else {
            // partial key: must be a prefix of an allowed, unseen key (unless any key is allowed)
            if n.property("\u{0}").is_some() {
                return Ok(Step::Eof);
            }
            let ok = n.properties.keys().any(|k| k.starts_with(&key) && !seen.contains(k));
            return if ok { Ok(Step::Eof) } else { Err(()) };
        };
        if seen.contains(&key) {
            return Err(());
        }
        let child = match n.property(&key) {
            None => return Err(()),
            Some(c) => c.unwrap_or(&ANY),
        };
        j = ws(s, after_key);
        if j >= s.len() {
            return Ok(Step::Eof);
        }
        if s[j] != ':' {
            return Err(());
        }
        match value(s, j + 1, child)? {
            Step::Eof => return Ok(Step::Eof),
            Step::Done(e) => j = ws(s, e),
        }
        seen.insert(key);
        first = false;
        if j >= s.len() {
            return Ok(Step::Eof);
        }
        match s[j] {
            ',' => {
                j = ws(s, j + 1);
                // a trailing comma must be followed by a key; reject if nothing more may be added
                if matches!(n.additional, Some(None)) && n.properties.keys().all(|k| seen.contains(k)) {
                    return Err(());
                }
                if j < s.len() && s[j] != '"' {
                    return Err(());
                }
            }
            '}' => return close_object(n, &seen, j),
            _ => return Err(()),
        }
    }
}

fn close_object(n: &Node, seen: &HashSet<String>, j: usize) -> V {
    if n.required.iter().all(|r| seen.contains(r)) { Ok(Step::Done(j + 1)) } else { Err(()) }
}

fn array(s: &[char], i: usize, n: &Node) -> V {
    let item = n.items.as_deref().unwrap_or(&ANY);
    let mut j = ws(s, i + 1);
    let mut count = 0usize;
    if j >= s.len() {
        return Ok(Step::Eof);
    }
    if s[j] == ']' {
        return if n.min_items.unwrap_or(0) == 0 { Ok(Step::Done(j + 1)) } else { Err(()) };
    }
    loop {
        if n.max_items.is_some_and(|m| count >= m) {
            return Err(());
        }
        match value(s, j, item)? {
            Step::Eof => return Ok(Step::Eof),
            Step::Done(e) => j = ws(s, e),
        }
        count += 1;
        if j >= s.len() {
            return Ok(Step::Eof);
        }
        match s[j] {
            ',' => j += 1,
            ']' => {
                return if count >= n.min_items.unwrap_or(0) { Ok(Step::Done(j + 1)) } else { Err(()) };
            }
            _ => return Err(()),
        }
    }
}

/// Convenience: is `text` a complete output under a JSON Schema constraint?
pub fn json_matches(schema: &Value, text: &str) -> Result<bool> {
    Ok(JsonSchemaConstraint::new(schema)?.check(text) == PrefixStatus::Complete)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn numbers_and_types() {
        let c = JsonSchemaConstraint::new(&json!({"type": "integer"})).unwrap();
        assert_eq!(c.check("12"), PrefixStatus::Partial); // more digits may follow
        assert_eq!(c.check("12 "), PrefixStatus::Complete);
        assert_eq!(c.check("1."), PrefixStatus::Invalid);
        assert_eq!(c.check("\"a\""), PrefixStatus::Invalid);
        let b = JsonSchemaConstraint::new(&json!({"type": ["boolean", "null"]})).unwrap();
        assert_eq!(b.check("tr"), PrefixStatus::Partial);
        assert_eq!(b.check("null"), PrefixStatus::Complete);
        assert_eq!(b.check("nope"), PrefixStatus::Invalid);
    }

    #[test]
    fn arrays() {
        let c =
            JsonSchemaConstraint::new(&json!({"type": "array", "items": {"type": "string"}, "maxItems": 2})).unwrap();
        assert_eq!(c.check(r#"["a", "b"]"#), PrefixStatus::Complete);
        assert_eq!(c.check(r#"["a", 1"#), PrefixStatus::Invalid);
        assert_eq!(c.check(r#"["a", "b", "#), PrefixStatus::Invalid);
    }

    #[test]
    fn unsupported_keywords_rejected() {
        assert!(JsonSchemaConstraint::new(&json!({"anyOf": []})).is_err());
    }

    #[test]
    fn mask_over_vocab() {
        let c = Choice::new(["supports", "contradicts", "neutral"]);
        let vocab: Vec<String> = ["sup", "con", "x", "ports", ""].iter().map(|s| s.to_string()).collect();
        let (mask, st) = token_mask(&c, "", &vocab);
        assert_eq!(mask, [true, true, false, false, false]);
        assert_eq!(st, PrefixStatus::Partial);
        let (mask, _) = token_mask(&c, "sup", &vocab);
        assert_eq!(mask, [false, false, false, true, false]);
    }

    proptest::proptest! {
        #[test]
        fn every_prefix_of_valid_output_is_allowed(v in proptest::collection::btree_map("[a-z]{1,4}", -1000i64..1000, 0..4)) {
            let schema = json!({"type": "object", "additionalProperties": {"type": "integer"}});
            let c = JsonSchemaConstraint::new(&schema).unwrap();
            let text = serde_json::to_string(&v).unwrap();
            for (i, _) in text.char_indices() {
                proptest::prop_assert_ne!(c.check(&text[..i]), PrefixStatus::Invalid, "{}", &text[..i]);
            }
            proptest::prop_assert_eq!(c.check(&text), PrefixStatus::Complete);
        }
    }
}
