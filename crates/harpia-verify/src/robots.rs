//! `robots.txt` (RFC 9309): groups by user agent, `Allow`/`Disallow` with `*` and `$`, longest
//! match wins, `Allow` wins ties.

/// Parsed rules that apply to one user agent.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Robots {
    rules: Vec<(bool, String)>,
}

impl Robots {
    /// Allow everything (e.g. `robots.txt` answered 4xx).
    pub fn allow_all() -> Self {
        Self::default()
    }

    /// Disallow everything (e.g. `robots.txt` unreachable with 5xx).
    pub fn disallow_all() -> Self {
        Self { rules: vec![(false, "/".into())] }
    }

    /// Parse `body` for `agent` (product token, case-insensitive). The most specific matching
    /// group is used; `*` is the fallback.
    pub fn parse(body: &str, agent: &str) -> Self {
        let agent = agent.to_ascii_lowercase();
        // (user agents, rules)
        type Group = (Vec<String>, Vec<(bool, String)>);
        let mut groups: Vec<Group> = Vec::new();
        let mut in_agents = false;
        for line in body.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let Some((k, v)) = line.split_once(':') else { continue };
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim().to_string());
            match k.as_str() {
                "user-agent" => {
                    if !in_agents || groups.is_empty() {
                        groups.push((Vec::new(), Vec::new()));
                    }
                    groups.last_mut().expect("group").0.push(v.to_ascii_lowercase());
                    in_agents = true;
                }
                "allow" | "disallow" => {
                    in_agents = false;
                    if let Some(g) = groups.last_mut()
                        && !(k == "disallow" && v.is_empty())
                    {
                        g.1.push((k == "allow", v));
                    }
                }
                _ => {}
            }
        }
        let specific = groups.iter().find(|(agents, _)| agents.iter().any(|a| a != "*" && agent.contains(a.as_str())));
        let chosen = specific.or_else(|| groups.iter().find(|(agents, _)| agents.iter().any(|a| a == "*")));
        Self { rules: chosen.map(|g| g.1.clone()).unwrap_or_default() }
    }

    /// Whether `path` (with query) may be fetched.
    ///
    /// ```
    /// use harpia_verify::robots::Robots;
    /// let r = Robots::parse("User-agent: *\nDisallow: /private\nAllow: /private/ok$\n", "harpia");
    /// assert!(!r.allowed("/private/x"));
    /// assert!(r.allowed("/private/ok"));
    /// assert!(r.allowed("/public"));
    /// ```
    pub fn allowed(&self, path: &str) -> bool {
        let mut best: Option<(usize, bool)> = None;
        for (allow, pattern) in &self.rules {
            if matches(pattern, path) {
                let len = pattern.len();
                if best.is_none_or(|(l, a)| len > l || (len == l && *allow && !a)) {
                    best = Some((len, *allow));
                }
            }
        }
        best.is_none_or(|(_, a)| a)
    }
}

fn matches(pattern: &str, path: &str) -> bool {
    let (pattern, anchored) = match pattern.strip_suffix('$') {
        Some(p) => (p, true),
        None => (pattern, false),
    };
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return if anchored { path == pattern } else { path.starts_with(pattern) };
    }
    if !path.starts_with(parts[0]) {
        return false;
    }
    let mut pos = parts[0].len();
    let last = parts[parts.len() - 1];
    for part in &parts[1..parts.len() - 1] {
        match path[pos..].find(part) {
            Some(p) => pos += p + part.len(),
            None => return false,
        }
    }
    if anchored { path.len() >= pos + last.len() && path.ends_with(last) } else { path[pos..].contains(last) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_and_wildcards() {
        let txt = "User-agent: Googlebot\nDisallow: /\n\nUser-agent: harpia\nUser-agent: other\nDisallow: /*.pdf$\nAllow: /\n\nUser-agent: *\nDisallow: /search\n";
        let h = Robots::parse(txt, "harpia/0.1");
        assert!(!h.allowed("/paper.pdf") && h.allowed("/paper.pdf?x=1") && h.allowed("/search"));
        assert!(!h.allowed("/a.pdf.pdf"), "anchored suffix after a wildcard");
        let star = Robots::parse(txt, "somebot");
        assert!(!star.allowed("/search?q=x") && star.allowed("/wiki"));
        assert!(Robots::parse("User-agent: *\nDisallow:\n", "x").allowed("/anything"));
        assert!(!Robots::disallow_all().allowed("/"));
        assert!(Robots::allow_all().allowed("/x"));
    }
}
