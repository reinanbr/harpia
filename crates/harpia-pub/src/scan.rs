//! Lightweight, fully tolerant scanner over (masked) LaTeX: commands with their arguments,
//! `\begin`/`\end` events and well-formed environment blocks. Byte offsets throughout.

use crate::source::backslashes_before;

/// One `{…}` or `[…]` argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arg {
    /// Content between the delimiters.
    pub text: String,
    /// Byte offset of the content start (after the opening delimiter).
    pub start: usize,
    /// Byte offset of the closing delimiter.
    pub end: usize,
    /// `true` for `[…]`.
    pub optional: bool,
}

/// A command occurrence such as `\cite[p.~2]{key}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    /// Name without the backslash.
    pub name: String,
    /// Followed by `*`.
    pub star: bool,
    /// Offset of the backslash.
    pub start: usize,
    /// Offset just after the last parsed argument.
    pub end: usize,
    /// Parsed arguments in order.
    pub args: Vec<Arg>,
}

impl Command {
    /// Mandatory (`{…}`) arguments.
    pub fn mandatory(&self) -> impl Iterator<Item = &Arg> {
        self.args.iter().filter(|a| !a.optional)
    }

    /// First mandatory argument.
    pub fn first(&self) -> Option<&Arg> {
        self.mandatory().next()
    }

    /// First optional argument.
    pub fn first_optional(&self) -> Option<&Arg> {
        self.args.iter().find(|a| a.optional)
    }
}

/// Offset of the delimiter closing the group opened at `text[i]` (`{` or `[`), if any.
pub fn match_group(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    let (open, close) = match b.get(i)? {
        b'{' => (b'{', b'}'),
        b'[' => (b'[', b']'),
        _ => return None,
    };
    let mut depth = 0usize;
    let mut j = i;
    while j < b.len() {
        let c = b[j];
        if c == b'\\' {
            j += 2;
            continue;
        }
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(j);
            }
        } else if open == b'[' && c == b'{' {
            // brackets may contain groups with `]` inside
            j = match_group(text, j)?;
        }
        j += 1;
    }
    None
}

/// Skip spaces, allowing at most one newline (a blank line ends an argument list).
pub fn skip_ws(text: &str, mut i: usize) -> usize {
    let b = text.as_bytes();
    let mut newlines = 0;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') {
        if b[i] == b'\n' {
            newlines += 1;
            if newlines > 1 {
                break;
            }
        }
        i += 1;
    }
    i
}

/// Read `[..]`/`{..}` arguments starting at `pos`: up to `max_mandatory` braces and
/// `max_optional` brackets (brackets only before the first brace). Returns the arguments
/// and the offset after the last one.
pub fn parse_args(text: &str, pos: usize, max_mandatory: usize, max_optional: usize) -> (Vec<Arg>, usize) {
    let mut args = Vec::new();
    let (mut mand, mut opt) = (0, 0);
    let mut i = pos;
    while mand < max_mandatory {
        let j = skip_ws(text, i);
        let Some(&c) = text.as_bytes().get(j) else { break };
        if c == b'[' && opt < max_optional && mand == 0 {
            let Some(k) = match_group(text, j) else { break };
            args.push(Arg { text: text[j + 1..k].to_string(), start: j + 1, end: k, optional: true });
            opt += 1;
            i = k + 1;
        } else if c == b'{' && mand < max_mandatory {
            let Some(k) = match_group(text, j) else { break };
            args.push(Arg { text: text[j + 1..k].to_string(), start: j + 1, end: k, optional: false });
            mand += 1;
            i = k + 1;
        } else {
            break;
        }
    }
    (args, i)
}

/// Read a control word starting at the backslash at `i`: `(name, offset after name)`.
pub fn control_word(text: &str, i: usize) -> Option<(&str, usize)> {
    let b = text.as_bytes();
    if b.get(i) != Some(&b'\\') {
        return None;
    }
    let mut j = i + 1;
    while j < b.len() && (b[j].is_ascii_alphabetic() || b[j] == b'@') {
        j += 1;
    }
    (j > i + 1).then(|| (&text[i + 1..j], j))
}

/// All occurrences of `\name` for the given names (exact names; `\\name` is skipped).
pub fn find_commands(text: &str, names: &[&str], mandatory: usize, optional: usize) -> Vec<Command> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'\\' {
            i += 1;
            continue;
        }
        if backslashes_before(b, i) % 2 == 1 {
            i += 1;
            continue;
        }
        let Some((name, mut j)) = control_word(text, i) else {
            i += 2;
            continue;
        };
        if names.contains(&name) {
            let star = b.get(j) == Some(&b'*');
            if star {
                j += 1;
            }
            let (args, end) = parse_args(text, j, mandatory, optional);
            out.push(Command { name: name.to_string(), star, start: i, end, args });
        }
        i = j;
    }
    out
}

/// `\begin{name}` or `\end{name}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvEvent {
    /// `true` for `\begin`.
    pub begin: bool,
    /// Environment name.
    pub name: String,
    /// Offset of the backslash.
    pub start: usize,
    /// Offset after the closing `}`.
    pub end: usize,
}

/// All `\begin{…}`/`\end{…}` events in order.
pub fn env_events(text: &str) -> Vec<EnvEvent> {
    let mut out = Vec::new();
    for cmd in find_commands(text, &["begin", "end"], 0, 0) {
        let b = text.as_bytes();
        let j = skip_ws(text, cmd.start + cmd.name.len() + 1);
        if b.get(j) != Some(&b'{') {
            continue;
        }
        let Some(k) = text[j + 1..].find(['{', '}']).map(|k| j + 1 + k) else { continue };
        if b[k] != b'}' {
            continue;
        }
        let name = text[j + 1..k].trim();
        if name.is_empty() {
            continue;
        }
        out.push(EnvEvent { begin: cmd.name == "begin", name: name.to_string(), start: cmd.start, end: k + 1 });
    }
    out
}

/// A well-formed environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvBlock {
    /// Name.
    pub name: String,
    /// Offset of `\begin`.
    pub start: usize,
    /// Offset after `\begin{name}`.
    pub body_start: usize,
    /// Offset of `\end`.
    pub body_end: usize,
    /// Offset after `\end{name}`.
    pub end: usize,
}

/// Environment blocks paired with a stack (unmatched events are ignored), sorted by start.
/// `names = None` keeps every environment.
pub fn env_blocks(text: &str, names: Option<&[&str]>) -> Vec<EnvBlock> {
    let mut stack: Vec<EnvEvent> = Vec::new();
    let mut out = Vec::new();
    for ev in env_events(text) {
        if ev.begin {
            stack.push(ev);
            continue;
        }
        if let Some(idx) = stack.iter().rposition(|b| b.name == ev.name) {
            let b = stack[idx].clone();
            stack.truncate(idx);
            if names.is_none_or(|n| n.contains(&b.name.as_str())) {
                out.push(EnvBlock { name: b.name, start: b.start, body_start: b.end, body_end: ev.start, end: ev.end });
            }
        }
    }
    out.sort_by_key(|b| b.start);
    out
}

/// Split a key list such as `{a, b,c}` into `(key, absolute offset)`.
pub fn split_keys(arg: &Arg) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut pos = 0;
    for part in arg.text.split(',') {
        let lead = part.len() - part.trim_start().len();
        let key = part.trim();
        if !key.is_empty() {
            out.push((key.to_string(), arg.start + pos + lead));
        }
        pos += part.len() + 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_commands_with_args() {
        let t = r"see \cite[p.~2]{a, b} and \\cite{no} \citeonline{c}";
        let cs = find_commands(t, &["cite", "citeonline"], 1, 2);
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0].args.len(), 2);
        assert_eq!(cs[0].first_optional().unwrap().text, "p.~2");
        let keys = split_keys(cs[0].first().unwrap());
        assert_eq!(keys.iter().map(|k| k.0.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(&t[keys[1].1..keys[1].1 + 1], "b");
        assert_eq!(cs[1].name, "citeonline");
    }

    #[test]
    fn does_not_match_prefixes() {
        assert!(find_commands(r"\citep{x}", &["cite"], 1, 1).is_empty());
    }

    #[test]
    fn env_pairing() {
        let t = r"\begin{figure}\begin{center}x\end{center}\end{figure}\end{table}";
        let bl = env_blocks(t, Some(&["figure"]));
        assert_eq!(bl.len(), 1);
        assert_eq!(&t[bl[0].body_start..bl[0].body_end], r"\begin{center}x\end{center}");
        assert_eq!(env_events(t).len(), 5);
    }
}
