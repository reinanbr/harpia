//! HTML to text (feature `html`), pure Rust and dependency-free.
//!
//! [`html_to_text`] keeps the visible text with block elements turned into line breaks;
//! [`main_text`] additionally prefers `<article>`/`<main>` and drops navigation, headers,
//! footers, asides and forms (simple boilerplate removal for web pages used as evidence or RAG
//! sources). Scripts, styles and other non-content elements are never emitted. The parser is a
//! tolerant tokenizer, not a full HTML5 tree builder: it is meant for text extraction.

/// Extracted page text.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MainText {
    /// `<title>` content, if any.
    pub title: Option<String>,
    /// Main text, paragraphs separated by blank lines.
    pub text: String,
    /// `<html lang>` attribute, if any.
    pub lang: Option<String>,
}

const SKIP: &[&str] = &["script", "style", "noscript", "template", "svg", "math", "head", "iframe", "object", "canvas"];
const BOILERPLATE: &[&str] = &["nav", "header", "footer", "aside", "form", "menu", "dialog", "button"];
const BLOCK: &[&str] = &[
    "p",
    "div",
    "section",
    "article",
    "main",
    "br",
    "li",
    "ul",
    "ol",
    "table",
    "tr",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
    "hr",
    "dl",
    "dt",
    "dd",
    "figure",
    "figcaption",
    "address",
    "header",
    "footer",
    "nav",
    "aside",
    "caption",
    "title",
];
const VOID: &[&str] =
    &["br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"];

#[derive(Debug)]
enum Token<'a> {
    Text(&'a str),
    Open { name: String, attrs: &'a str, self_closing: bool },
    Close(String),
}

fn tokens(html: &str) -> Vec<Token<'_>> {
    let b = html.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let mut text_start = 0;
    while i < b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        let is_markup = html[i..].starts_with("<!")
            || html[i..].starts_with("<?")
            || b.get(i + 1).is_some_and(|c| c.is_ascii_alphabetic())
            || (b.get(i + 1) == Some(&b'/') && b.get(i + 2).is_some_and(|c| c.is_ascii_alphabetic()));
        if !is_markup {
            // A literal "<" in text.
            i += 1;
            continue;
        }
        if text_start < i {
            out.push(Token::Text(&html[text_start..i]));
        }
        if html[i..].starts_with("<!--") {
            i = html[i..].find("-->").map_or(b.len(), |e| i + e + 3);
            text_start = i;
            continue;
        }
        if html[i..].starts_with("<!") || html[i..].starts_with("<?") {
            i = html[i..].find('>').map_or(b.len(), |e| i + e + 1);
            text_start = i;
            continue;
        }
        let close = b.get(i + 1) == Some(&b'/');
        let name_start = i + 1 + usize::from(close);
        let mut j = name_start;
        while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'-') {
            j += 1;
        }
        let name = html[name_start..j].to_ascii_lowercase();
        // Find the end of the tag, skipping quoted attribute values.
        let mut k = j;
        let mut quote = None;
        while k < b.len() {
            match (quote, b[k]) {
                (None, b'"' | b'\'') => quote = Some(b[k]),
                (Some(q), c) if c == q => quote = None,
                (None, b'>') => break,
                _ => {}
            }
            k += 1;
        }
        let attrs = &html[j..k.min(b.len())];
        let self_closing = attrs.trim_end().ends_with('/');
        out.push(if close {
            Token::Close(name.clone())
        } else {
            Token::Open { name: name.clone(), attrs, self_closing }
        });
        i = (k + 1).min(b.len());
        // Raw-text elements: jump to their closing tag.
        if !close && matches!(name.as_str(), "script" | "style" | "textarea" | "title") && !self_closing {
            let end_tag = format!("</{name}");
            let lower = html[i..].to_ascii_lowercase();
            let end = lower.find(&end_tag).map_or(b.len(), |e| i + e);
            if name == "title" || name == "textarea" {
                out.push(Token::Text(&html[i..end]));
            }
            i = end;
        }
        text_start = i;
    }
    if text_start < b.len() {
        out.push(Token::Text(&html[text_start..]));
    }
    out
}

fn attr<'a>(attrs: &'a str, key: &str) -> Option<&'a str> {
    let lower = attrs.to_ascii_lowercase();
    let mut from = 0;
    while let Some(p) = lower[from..].find(key) {
        let at = from + p;
        let before_ok = at == 0 || !matches!(lower.as_bytes()[at - 1], b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b':');
        let rest = attrs[at + key.len()..].trim_start();
        if before_ok && let Some(v) = rest.strip_prefix('=') {
            let v = v.trim_start();
            return Some(match v.chars().next() {
                Some(q @ ('"' | '\'')) => v[1..].split(q).next().unwrap_or(""),
                _ => v.split(|c: char| c.is_whitespace() || c == '>').next().unwrap_or(""),
            });
        }
        from = at + key.len();
    }
    None
}

/// Decode HTML character references (named common ones and numeric).
///
/// ```
/// assert_eq!(harpia_data::html::decode_entities("caf&eacute; &amp; &#x1F985;&lt;"), "café & 🦅<");
/// ```
pub fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let end = rest[1..].find(|c: char| c == ';' || c == '&' || c.is_whitespace()).map(|e| e + 1);
        let decoded = end.filter(|&e| rest.as_bytes().get(e) == Some(&b';')).and_then(|e| {
            let name = &rest[1..e];
            let ch = if let Some(num) = name.strip_prefix('#') {
                let n = match num.strip_prefix(['x', 'X']) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok(),
                    None => num.parse().ok(),
                };
                n.and_then(char::from_u32).map(String::from)
            } else {
                named_entity(name).map(String::from)
            };
            ch.map(|c| (c, e + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push_str(&c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn named_entity(name: &str) -> Option<&'static str> {
    Some(match name {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => "\u{a0}",
        "ndash" => "–",
        "mdash" => "—",
        "hellip" => "…",
        "lsquo" => "‘",
        "rsquo" => "’",
        "ldquo" => "“",
        "rdquo" => "”",
        "laquo" => "«",
        "raquo" => "»",
        "copy" => "©",
        "reg" => "®",
        "deg" => "°",
        "middot" => "·",
        "eacute" => "é",
        "Eacute" => "É",
        "aacute" => "á",
        "Aacute" => "Á",
        "iacute" => "í",
        "oacute" => "ó",
        "uacute" => "ú",
        "atilde" => "ã",
        "otilde" => "õ",
        "ccedil" => "ç",
        "Ccedil" => "Ç",
        "acirc" => "â",
        "ecirc" => "ê",
        "ocirc" => "ô",
        "agrave" => "à",
        "uuml" => "ü",
        "ouml" => "ö",
        "auml" => "ä",
        _ => return None,
    })
}

fn render(tokens: &[Token<'_>], drop_boilerplate: bool) -> String {
    let mut out = String::new();
    let mut skip_depth = 0usize;
    let mut skip_stack: Vec<String> = Vec::new();
    let mut pre = 0usize;
    for t in tokens {
        match t {
            Token::Open { name, self_closing, .. } => {
                let n = name.as_str();
                let skipping = SKIP.contains(&n) || (drop_boilerplate && BOILERPLATE.contains(&n));
                if skipping && !self_closing && !VOID.contains(&n) {
                    skip_depth += 1;
                    skip_stack.push(name.clone());
                    continue;
                }
                if skip_depth > 0 {
                    continue;
                }
                if n == "pre" {
                    pre += 1;
                }
                if BLOCK.contains(&n) {
                    out.push('\n');
                }
            }
            Token::Close(name) => {
                if skip_depth > 0 {
                    if skip_stack.last() == Some(name) {
                        skip_stack.pop();
                        skip_depth -= 1;
                    }
                    continue;
                }
                if name == "pre" {
                    pre = pre.saturating_sub(1);
                }
                if BLOCK.contains(&name.as_str()) {
                    out.push('\n');
                }
            }
            Token::Text(s) if skip_depth == 0 => {
                let d = decode_entities(s);
                if pre > 0 {
                    out.push_str(&d);
                } else {
                    let mut last_space = out.ends_with([' ', '\n']) || out.is_empty();
                    for c in d.chars() {
                        if c.is_whitespace() {
                            if !last_space {
                                out.push(' ');
                                last_space = true;
                            }
                        } else {
                            out.push(c);
                            last_space = false;
                        }
                    }
                }
            }
            Token::Text(_) => {}
        }
    }
    // Normalize lines: trim, collapse runs of blank lines into one paragraph break.
    let mut text = String::new();
    let mut blank = 0;
    for line in out.lines().map(|l| l.trim_matches(|c: char| c == ' ' || c == '\u{a0}')) {
        if line.is_empty() {
            blank += 1;
            continue;
        }
        if !text.is_empty() {
            text.push_str(if blank > 0 { "\n\n" } else { "\n" });
        }
        text.push_str(line);
        blank = 0;
    }
    text
}

/// All visible text: paragraphs and other blocks separated by blank lines, `<br>` as a line
/// break, runs of whitespace (including non-breaking spaces) collapsed to one space.
///
/// ```
/// let t = harpia_data::html::html_to_text("<p>Harpy&nbsp;eagle</p><script>x()</script><p>Brazil<br>Panama</p>");
/// assert_eq!(t, "Harpy eagle\n\nBrazil\nPanama");
/// ```
pub fn html_to_text(html: &str) -> String {
    render(&tokens(html), false)
}

/// Main content of a page: inside `<article>` (or `<main>`) when present, otherwise the body
/// without navigation, header, footer, aside and form elements.
pub fn main_text(html: &str) -> MainText {
    let toks = tokens(html);
    let mut title = None;
    let mut lang = None;
    for (i, t) in toks.iter().enumerate() {
        match t {
            Token::Open { name, attrs, .. } if name == "html" && lang.is_none() => {
                lang = attr(attrs, "lang").map(|s| s.to_string()).filter(|s| !s.is_empty());
            }
            Token::Open { name, .. } if name == "title" && title.is_none() => {
                if let Some(Token::Text(s)) = toks.get(i + 1) {
                    let t = decode_entities(s).split_whitespace().collect::<Vec<_>>().join(" ");
                    title = Some(t).filter(|t| !t.is_empty());
                }
            }
            _ => {}
        }
    }
    let region = ["article", "main"].iter().find_map(|tag| {
        let start = toks.iter().position(|t| matches!(t, Token::Open { name, .. } if name == tag))?;
        let mut depth = 0;
        let end = toks[start..].iter().position(|t| {
            match t {
                Token::Open { name, .. } if name == tag => depth += 1,
                Token::Close(name) if name == tag => {
                    depth -= 1;
                    return depth == 0;
                }
                _ => {}
            }
            false
        });
        Some(&toks[start..end.map_or(toks.len(), |e| start + e + 1)])
    });
    let text = match region {
        Some(r) => render(r, true),
        None => render(&toks, true),
    };
    MainText { title, text, lang }
}

/// `href` targets of `<a>` elements, resolved against `base_url` (absolute `http(s)` URLs,
/// root-relative `/path` and path-relative links), without fragments, deduplicated in order.
///
/// ```
/// let l = harpia_data::html::links(r#"<a href="/a#x">A</a> <a href='b'>B</a> <a href="mailto:x">M</a>"#, "https://x.org/dir/page");
/// assert_eq!(l, ["https://x.org/a", "https://x.org/dir/b"]);
/// ```
pub fn links(html: &str, base_url: &str) -> Vec<String> {
    let (origin, dir) = match base_url.split_once("://") {
        Some((scheme, rest)) => {
            let host_end = rest.find('/').unwrap_or(rest.len());
            let origin = format!("{scheme}://{}", &rest[..host_end]);
            let path = &rest[host_end..];
            let path = path.split(['?', '#']).next().unwrap_or("");
            let dir = match path.rfind('/') {
                Some(i) => format!("{origin}{}", &path[..=i]),
                None => format!("{origin}/"),
            };
            (origin, dir)
        }
        None => (String::new(), String::new()),
    };
    let mut out: Vec<String> = Vec::new();
    for t in tokens(html) {
        let Token::Open { name, attrs, .. } = t else { continue };
        if name != "a" {
            continue;
        }
        let Some(href) = attr(attrs, "href") else { continue };
        let href = decode_entities(href.trim());
        let href = href.split('#').next().unwrap_or("");
        let abs = if href.starts_with("http://") || href.starts_with("https://") {
            href.to_string()
        } else if let Some(rest) = href.strip_prefix("//") {
            format!("{}://{rest}", origin.split("://").next().unwrap_or("https"))
        } else if href.starts_with('/') {
            format!("{origin}{href}")
        } else if href.is_empty() || href.contains(':') || origin.is_empty() {
            continue;
        } else {
            format!("{dir}{href}")
        };
        if !out.contains(&abs) {
            out.push(abs);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = r#"<!DOCTYPE html><html lang="pt-BR"><head><title>Harpia &ndash; Wiki</title>
<style>p{color:red}</style><script>var x = "<p>not text</p>";</script></head>
<body><nav><a href="/">Home</a> | <a href="/x">Menu</a></nav>
<header><h1>Site</h1></header>
<article><h1>Harpia harpyja</h1><p>The harpy eagle is a <b>large</b> raptor.</p>
<!-- comment --><p>It lives in<br>Brazil &amp; Panama.</p><aside>Ad</aside>
<pre>  keep   spacing </pre></article><footer>© 2026</footer></body></html>"#;

    #[test]
    fn main_text_prefers_article_and_drops_boilerplate() {
        let m = main_text(PAGE);
        assert_eq!(m.title.as_deref(), Some("Harpia – Wiki"));
        assert_eq!(m.lang.as_deref(), Some("pt-BR"));
        assert_eq!(
            m.text,
            "Harpia harpyja\n\nThe harpy eagle is a large raptor.\n\nIt lives in\nBrazil & Panama.\n\nkeep   spacing"
        );
        assert!(!m.text.contains("Menu") && !m.text.contains("Ad") && !m.text.contains("not text"));
    }

    #[test]
    fn full_text_and_edge_cases() {
        let all = html_to_text(PAGE);
        assert!(all.contains("Menu") && all.contains("© 2026") && !all.contains("color:red"));
        assert_eq!(html_to_text("a < b and c > d"), "a < b and c > d");
        assert_eq!(html_to_text("<p title='x > y'>ok</p>"), "ok");
        assert_eq!(decode_entities("&unknown; & &#xZZ;"), "&unknown; & &#xZZ;");
        let no_article = main_text("<body><nav>skip</nav><p>keep</p><footer>f</footer></body>");
        assert_eq!(no_article.text, "keep");
        assert_eq!(attr(r#" data-lang="x" lang='en'"#, "lang"), Some("en"));
    }
}
