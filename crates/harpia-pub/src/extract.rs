//! Clean-text extraction from LaTeX with a byte-level map back to the source.
//!
//! The walker is tolerant: malformed input never fails, it only degrades. Every byte of the
//! clean text maps to a source byte offset; synthetic characters (citation placeholders, `x`
//! for inline math, `1` for cross-references) map to the command that produced them.
//!
//! What is kept: body text, headings, captions, list items, abstracts and quotes, each tagged
//! with a [`ParaKind`] and a language. What is dropped: the preamble, math, tables, code,
//! footnotes, `\texttt`, labels, URLs and layout commands.

use crate::scan::{match_group, parse_args, skip_ws};
use crate::source::{Location, SourceFile, VERBATIM_ENVS};
use serde::Serialize;
use unicode_normalization::UnicodeNormalization;

/// Sectioning commands and their depth.
pub const SECTION_LEVELS: &[(&str, i32)] = &[
    ("part", -1),
    ("chapter", 0),
    ("section", 1),
    ("subsection", 2),
    ("subsubsection", 3),
    ("paragraph", 4),
    ("subparagraph", 5),
];
/// Parenthetical citation commands.
pub const CITE_PAREN: &[&str] =
    &["cite", "citep", "parencite", "Parencite", "autocite", "Autocite", "footcite", "citealp", "citeyearpar", "Cite"];
/// Narrative (in-text) citation commands.
pub const CITE_TEXT: &[&str] = &["citeonline", "citet", "textcite", "Textcite", "citealt", "Citet", "citeauthoronline"];
/// Author-only citation commands.
pub const CITE_AUTHOR: &[&str] = &["citeauthor", "Citeauthor"];
/// Year-only citation commands.
pub const CITE_YEAR: &[&str] = &["citeyear"];
/// abnTeX2 secondary-citation commands.
pub const CITE_APUD: &[&str] = &["apud", "apudonline"];
/// Cross-reference commands.
pub const REF_MACROS: &[&str] = &["ref", "eqref", "autoref", "cref", "Cref", "pageref", "nameref", "vref"];
/// Float environments.
pub const FLOAT_ENVS: &[&str] = &[
    "figure",
    "figure*",
    "table",
    "table*",
    "quadro",
    "wrapfigure",
    "wraptable",
    "sidewaysfigure",
    "sidewaystable",
    "algorithm",
];
/// Display-math environments.
pub const MATH_ENVS: &[&str] = &[
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "displaymath",
    "math",
    "flalign",
    "flalign*",
    "alignat",
    "alignat*",
];
/// Environments whose content is not prose.
pub const SKIP_ENVS: &[&str] = &[
    "tikzpicture",
    "tabular",
    "tabular*",
    "tabularx",
    "longtable",
    "thebibliography",
    "filecontents",
    "filecontents*",
    "algorithmic",
    "fichacatalografica",
    "folhadeaprovacao",
    "titlepage",
    "lstlisting",
    "minted",
    "siglas",
    "simbolos",
];
/// List environments.
pub const LIST_ENVS: &[&str] = &["itemize", "enumerate", "description", "alineas", "subalineas", "incisos"];
/// Abstract environments (abnTeX2 and standard classes).
pub const ABSTRACT_ENVS: &[&str] = &["resumo", "abstract", "resumoumacoluna", "resumoduascolunas"];
/// Block-quotation environments.
pub const QUOTE_ENVS: &[&str] = &["citacao", "quote", "quotation"];

/// Kind of a clean-text region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RegionKind {
    /// Section title.
    Heading,
    /// Float caption.
    Caption,
    /// List environment.
    List,
    /// Abstract environment.
    Abstract,
    /// Quotation environment.
    Quote,
    /// Any other environment.
    Env,
}

/// A region of clean text produced by a heading, caption or environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Region {
    /// Kind.
    pub kind: RegionKind,
    /// Command or environment name.
    pub name: String,
    /// Clean-text start.
    pub start: usize,
    /// Clean-text end.
    pub end: usize,
    /// Language code (`pt`, `en`, `es`…).
    pub lang: String,
}

/// A sectioning command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Section {
    /// Depth (`chapter` = 0, `section` = 1…).
    pub level: i32,
    /// Title as clean text.
    pub title: String,
    /// Starred (unnumbered).
    pub starred: bool,
    /// Source offset of the command.
    pub src_offset: usize,
    /// Clean-text offset of the title.
    pub clean_start: usize,
}

/// A citation placeholder in the clean text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CiteSpan {
    /// Clean-text start of the placeholder.
    pub start: usize,
    /// Clean-text end.
    pub end: usize,
    /// Cited keys.
    pub keys: Vec<String>,
    /// Command name (`cite`, `citeonline`…).
    pub command: String,
    /// Source offset of the command.
    pub src_offset: usize,
}

/// Kind of a paragraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ParaKind {
    /// Running text.
    Body,
    /// Section title.
    Heading,
    /// Float caption.
    Caption,
    /// List item.
    Item,
    /// Abstract paragraph.
    Abstract,
    /// Quotation.
    Quote,
}

/// A paragraph of clean text (offsets into [`CleanText::text`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ParaSpan {
    /// Clean-text start.
    pub start: usize,
    /// Clean-text end.
    pub end: usize,
    /// Kind.
    pub kind: ParaKind,
    /// Language code.
    pub lang: String,
    /// Innermost region name (`resumo`, `itemize`, `section`…), empty for plain body text.
    pub region: String,
}

/// Extraction result for one file.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CleanText {
    /// Clean text.
    pub text: String,
    /// Source byte offset of every clean-text byte.
    #[serde(skip)]
    pub offsets: Vec<usize>,
    /// Regions.
    pub regions: Vec<Region>,
    /// Sectioning commands in order.
    pub sections: Vec<Section>,
    /// Citations in order.
    pub cites: Vec<CiteSpan>,
    /// Paragraphs in order.
    pub paragraphs: Vec<ParaSpan>,
}

impl CleanText {
    /// Source offset of clean-text byte `i` (clamped).
    pub fn src(&self, i: usize) -> usize {
        if self.offsets.is_empty() {
            return 0;
        }
        self.offsets[i.min(self.offsets.len() - 1)]
    }

    /// Source location of a clean-text range.
    pub fn location(&self, file: &SourceFile, start: usize, end: Option<usize>) -> Location {
        let s = self.src(start);
        match end {
            None => file.location(s, None),
            Some(e) => file.location(s, Some(self.src(e.max(start + 1) - 1) + 1)),
        }
    }
}

/// Map a babel/polyglossia language name to a short code.
pub fn lang_code(name: &str) -> String {
    let n = name.trim().to_lowercase();
    match n.as_str() {
        "english" | "american" | "british" | "en" | "ukenglish" | "usenglish" | "canadian" | "australian" => {
            "en".into()
        }
        "spanish" | "es" | "spanish-mx" => "es".into(),
        "brazil" | "brazilian" | "portuguese" | "portuges" | "pt" | "pt-br" | "portugues" => "pt".into(),
        "french" | "francais" | "fr" => "fr".into(),
        "german" | "ngerman" | "de" => "de".into(),
        "" => String::new(),
        other => other.to_string(),
    }
}

fn macro_spec(name: &str) -> Option<&'static str> {
    if SECTION_LEVELS.iter().any(|(n, _)| *n == name) {
        return Some("*[{");
    }
    if CITE_PAREN.contains(&name)
        || CITE_TEXT.contains(&name)
        || CITE_AUTHOR.contains(&name)
        || CITE_YEAR.contains(&name)
    {
        return Some("*[[{");
    }
    if CITE_APUD.contains(&name) {
        return Some("[[{{");
    }
    if REF_MACROS.contains(&name) {
        return Some("*{");
    }
    Some(match name {
        "Idem" | "Ibidem" | "opcit" | "passim" | "cfcite" | "nocite" => "{",
        "caption" => "*[{",
        "captionof" => "*{[{",
        "subcaption" => "[{",
        "href" => "{{",
        "foreignlanguage" => "[{{",
        "item" => "[",
        "emph" | "textbf" | "textit" | "textsl" | "textsc" | "underline" | "textup" | "textmd" | "textnormal"
        | "mbox" | "hbox" | "textrm" | "textsf" | "uline" | "textsuperscript" | "textsubscript" | "MakeUppercase"
        | "MakeLowercase" | "fbox" | "sout" | "textmu" | "text" => "{",
        "enquote" => "*{",
        "textcolor" | "colorbox" => "[{{",
        "makebox" => "[[{",
        "parbox" => "[[[{{",
        "raisebox" => "{[[{",
        // consumed and dropped
        "label" | "url" | "nolinkurl" | "input" | "include" | "subfile" | "bibliography" | "bibliographystyle"
        | "graphicspath" | "hypersetup" | "geometry" | "selectlanguage" | "pagestyle" | "thispagestyle" | "titulo"
        | "autor" | "data" | "instituicao" | "local" | "tipotrabalho" | "preambulo" | "texttt" | "linespread"
        | "usetikzlibrary" | "lstset" | "color" | "pagenumbering" | "setmainlanguage" | "setotherlanguage"
        | "newpage" | "includeonly" | "settowidth" | "addto" | "programa" | "area" | "palavraschave" => "{",
        "title" | "author" | "date" | "footnote" | "footnotetext" | "thanks" | "addbibresource" | "documentclass"
        | "usepackage" | "RequirePackage" | "todo" | "orientador" | "coorientador" | "bibitem" | "si" | "num"
        | "printbibliography" | "cleardoublepage" | "chapterprecis" => "[{",
        "SI" | "rule" | "pdfbookmark" => "[{{",
        "includegraphics" => "*[[{",
        "setlength" | "addtolength" | "setcounter" | "addtocounter" | "import" | "subimport" | "definecolor"
        | "newacronym" => match name {
            "definecolor" => "{{{",
            "newacronym" => "[{{{",
            _ => "{{",
        },
        "vspace" | "hspace" => "*{",
        "addcontentsline" => "{{{",
        "resizebox" | "scalebox" => "{{",
        _ => return None,
    })
}

const TEXT_MACROS: &[&str] = &[
    "emph",
    "textbf",
    "textit",
    "textsl",
    "textsc",
    "underline",
    "textup",
    "textmd",
    "textnormal",
    "mbox",
    "hbox",
    "textrm",
    "textsf",
    "uline",
    "enquote",
    "textsuperscript",
    "textsubscript",
    "MakeUppercase",
    "MakeLowercase",
    "fbox",
    "sout",
    "text",
    "makebox",
];
/// Macros whose *last* mandatory argument is the visible text.
const LAST_ARG_MACROS: &[&str] = &["textcolor", "colorbox", "parbox", "raisebox", "href"];

fn symbol(name: &str) -> Option<&'static str> {
    Some(match name {
        "ldots" | "dots" | "textellipsis" => "…",
        "LaTeX" => "LaTeX",
        "TeX" => "TeX",
        "LaTeXe" => "LaTeX2e",
        "BibTeX" => "BibTeX",
        "textendash" => "–",
        "textemdash" => "—",
        "S" | "textsection" => "§",
        "P" => "¶",
        "textdegree" | "degree" => "°",
        "ss" => "ß",
        "ae" => "æ",
        "AE" => "Æ",
        "oe" => "œ",
        "OE" => "Œ",
        "o" => "ø",
        "O" => "Ø",
        "aa" => "å",
        "AA" => "Å",
        "l" => "ł",
        "L" => "Ł",
        "i" => "ı",
        "j" => "ȷ",
        "copyright" | "textcopyright" => "©",
        "textregistered" => "®",
        "texttrademark" => "™",
        "euro" | "texteuro" => "€",
        "pounds" | "textsterling" => "£",
        "textquotedblleft" => "“",
        "textquotedblright" => "”",
        "textquoteleft" => "‘",
        "textquoteright" => "’",
        "guillemotleft" | "guillemetleft" => "«",
        "guillemotright" | "guillemetright" => "»",
        "textbackslash" => "\\",
        "textasciitilde" => "~",
        "textunderscore" => "_",
        "textbar" => "|",
        "textgreater" => ">",
        "textless" => "<",
        "quad" | "qquad" | "enspace" | "thinspace" | "space" | "newline" | "linebreak" | "hfill" | "nobreakspace" => {
            " "
        }
        _ => return None,
    })
}

fn combining(accent: &str) -> Option<char> {
    Some(match accent {
        "'" => '\u{301}',
        "`" => '\u{300}',
        "^" => '\u{302}',
        "~" => '\u{303}',
        "\"" => '\u{308}',
        "=" => '\u{304}',
        "." => '\u{307}',
        "c" => '\u{327}',
        "u" => '\u{306}',
        "v" => '\u{30C}',
        "H" => '\u{30B}',
        "k" => '\u{328}',
        "r" => '\u{30A}',
        "d" => '\u{323}',
        "b" => '\u{331}',
        _ => return None,
    })
}

struct Em<'a> {
    t: &'a str,
    buf: String,
    offs: Vec<usize>,
    regions: Vec<Region>,
    sections: Vec<Section>,
    cites: Vec<CiteSpan>,
    lang: Vec<String>,
    depth: usize,
}

impl<'a> Em<'a> {
    fn pos(&self) -> usize {
        self.buf.len()
    }

    fn put(&mut self, s: &str, offset: usize) {
        self.buf.push_str(s);
        self.offs.extend(std::iter::repeat_n(offset, s.len()));
    }

    fn put_char(&mut self, c: char, offset: usize) {
        self.buf.push(c);
        self.offs.extend(std::iter::repeat_n(offset, c.len_utf8()));
    }

    fn brk(&mut self, offset: usize) {
        self.put("\n\n", offset);
    }

    fn cur_lang(&self) -> String {
        self.lang.last().cloned().unwrap_or_default()
    }

    fn ch(&self, i: usize) -> Option<char> {
        self.t.get(i..).and_then(|s| s.chars().next())
    }

    /// Closing delimiter of a group at `i`, only if it ends before `end`.
    fn group(&self, i: usize, end: usize) -> Option<usize> {
        match_group(self.t, i).filter(|&k| k < end)
    }

    fn walk(&mut self, mut i: usize, end: usize) {
        self.depth += 1;
        if self.depth > 200 {
            self.depth -= 1;
            return; // pathological nesting
        }
        let b = self.t.as_bytes();
        while i < end {
            let Some(c) = self.ch(i) else { break };
            match c {
                '\\' => i = self.control(i, end),
                '{' => match self.group(i, end) {
                    Some(k) => {
                        self.walk(i + 1, k);
                        i = k + 1;
                    }
                    None => i += 1,
                },
                '}' => i += 1,
                '$' => i = self.dollar(i, end),
                '~' | '&' => {
                    self.put(" ", i);
                    i += 1;
                }
                '-' if b.get(i + 1) == Some(&b'-') => {
                    if b.get(i + 2) == Some(&b'-') {
                        self.put("—", i);
                        i += 3;
                    } else {
                        self.put("–", i);
                        i += 2;
                    }
                }
                '`' => {
                    if b.get(i + 1) == Some(&b'`') {
                        self.put("“", i);
                        i += 2;
                    } else {
                        self.put("‘", i);
                        i += 1;
                    }
                }
                '\'' if b.get(i + 1) == Some(&b'\'') => {
                    self.put("”", i);
                    i += 2;
                }
                _ => {
                    self.put_char(c, i);
                    i += c.len_utf8();
                }
            }
        }
        self.depth -= 1;
    }

    fn find_unescaped(&self, pat: &str, from: usize, end: usize) -> Option<usize> {
        let mut p = from;
        while p < end {
            let k = p + self.t[p..end].find(pat)?;
            if crate::source::backslashes_before(self.t.as_bytes(), k).is_multiple_of(2) {
                return Some(k);
            }
            p = k + 1;
        }
        None
    }

    fn dollar(&mut self, i: usize, end: usize) -> usize {
        if self.t[i..].starts_with("$$") {
            match self.find_unescaped("$$", i + 2, end) {
                Some(k) => {
                    self.put(" ", i);
                    k + 2
                }
                None => i + 2,
            }
        } else {
            match self.find_unescaped("$", i + 1, end) {
                Some(k) => {
                    self.put("x", i);
                    k + 1
                }
                None => i + 1,
            }
        }
    }

    fn control(&mut self, i: usize, end: usize) -> usize {
        let Some(n) = self.ch(i + 1) else { return i + 1 };
        if n.is_ascii_alphabetic() || n == '@' {
            let mut j = i + 1;
            let b = self.t.as_bytes();
            while j < end && (b[j].is_ascii_alphabetic() || b[j] == b'@') {
                j += 1;
            }
            let name = &self.t[i + 1..j];
            return self.macro_(name, i, j, end);
        }
        let after = i + 1 + n.len_utf8();
        match n {
            '\\' => {
                self.put(" ", i);
                // optional `\\[2pt]`
                let k = skip_ws(self.t, after);
                if self.t.as_bytes().get(k) == Some(&b'[')
                    && let Some(c) = self.group(k, end)
                {
                    return c + 1;
                }
                after
            }
            '(' => match self.find_unescaped("\\)", after, end) {
                Some(k) => {
                    self.put("x", i);
                    k + 2
                }
                None => after,
            },
            '[' => match self.find_unescaped("\\]", after, end) {
                Some(k) => {
                    self.put(" ", i);
                    k + 2
                }
                None => after,
            },
            '%' | '&' | '$' | '#' | '_' | '{' | '}' => {
                self.put_char(n, i);
                after
            }
            ',' | ';' | ':' | ' ' | '\n' | '\t' => {
                self.put(" ", i);
                after
            }
            '\'' | '`' | '^' | '"' | '~' | '=' | '.' => self.accent(&n.to_string(), i, after, end),
            _ => after,
        }
    }

    /// `\'e`, `\'{e}`, `\c{c}`, `\'{\i}` → composed character.
    fn accent(&mut self, acc: &str, i: usize, mut j: usize, end: usize) -> usize {
        let Some(mark) = combining(acc) else { return j };
        let letter_accent = acc.chars().all(|c| c.is_ascii_alphabetic());
        if letter_accent {
            // `\c c` needs a space; `\c{c}` a group
            while self.t.as_bytes().get(j) == Some(&b' ') {
                j += 1;
            }
        }
        let (base, next) = if self.t.as_bytes().get(j) == Some(&b'{') {
            match self.group(j, end) {
                Some(k) => (self.t[j + 1..k].trim().to_string(), k + 1),
                None => return j,
            }
        } else if self.t[j..].starts_with("\\i") || self.t[j..].starts_with("\\j") {
            (self.t[j..j + 2].to_string(), j + 2)
        } else {
            match self.ch(j) {
                Some(c) if j < end => (c.to_string(), j + c.len_utf8()),
                _ => return j,
            }
        };
        let base = match base.as_str() {
            "\\i" => "i".to_string(),
            "\\j" => "j".to_string(),
            b => b.to_string(),
        };
        let mut chars = base.chars();
        if let Some(c) = chars.next() {
            let composed: String = [c, mark].iter().collect::<String>().nfc().collect();
            self.put(&composed, i);
            let rest: String = chars.collect();
            if !rest.is_empty() {
                self.put(&rest, i);
            }
        }
        next
    }

    /// Consume a definition (`\newcommand{\x}[1]{…}`, `\def\x#1{…}`) without emitting.
    fn definition(&mut self, name: &str, mut j: usize, end: usize) -> usize {
        let b = self.t.as_bytes();
        if b.get(j) == Some(&b'*') {
            j += 1;
        }
        j = skip_ws(self.t, j);
        // the defined name: `{\x}` or `\x`
        if b.get(j) == Some(&b'{') {
            j = self.group(j, end).map_or(end, |k| k + 1);
        } else if b.get(j) == Some(&b'\\') {
            j += 1;
            while j < end && (b[j].is_ascii_alphabetic() || b[j] == b'@') {
                j += 1;
            }
        }
        if name == "def" || name == "gdef" || name == "edef" {
            // parameter text up to the body
            while j < end && b[j] != b'{' {
                j += 1;
            }
            return self.group(j, end).map_or(end, |k| k + 1);
        }
        if name == "let" {
            return j;
        }
        let bodies = if name.ends_with("environment") { 2 } else { 1 };
        let (_, after) = parse_args(self.t, j, bodies, 2);
        after.min(end)
    }

    fn macro_(&mut self, name: &str, i: usize, j: usize, end: usize) -> usize {
        let name = name.to_string();
        let name = name.as_str();
        match name {
            "begin" => return self.env(i, j, end),
            "end" => {
                let (_, after) = parse_args(self.t, j, 1, 0);
                return after.min(end).max(j);
            }
            "par" => {
                self.brk(i);
                return j;
            }
            "newcommand"
            | "renewcommand"
            | "providecommand"
            | "newenvironment"
            | "renewenvironment"
            | "def"
            | "gdef"
            | "edef"
            | "let"
            | "DeclareMathOperator"
            | "newtheorem" => return self.definition(name, j, end),
            _ => {}
        }
        if combining(name).is_some() && name.len() == 1 {
            return self.accent(name, i, j, end);
        }
        let Some(spec) = macro_spec(name) else {
            if let Some(s) = symbol(name) {
                self.put(s, i);
            }
            // starred variants of unknown commands (`\listoffigures*`)
            return if self.t.as_bytes().get(j) == Some(&b'*') { j + 1 } else { j };
        };
        // parse arguments per spec
        let b = self.t.as_bytes();
        let mut k = j;
        let mut star = false;
        let mut args: Vec<(usize, usize, bool)> = Vec::new(); // (content start, close, optional)
        for s in spec.chars() {
            match s {
                '*' => {
                    if b.get(k) == Some(&b'*') {
                        star = true;
                        k += 1;
                    }
                }
                '[' | '{' => {
                    let p = skip_ws(self.t, k);
                    let want = if s == '[' { b'[' } else { b'{' };
                    if p < end && b.get(p) == Some(&want) {
                        match self.group(p, end) {
                            Some(c) => {
                                args.push((p + 1, c, s == '['));
                                k = c + 1;
                            }
                            None => break,
                        }
                    } else if s == '{' {
                        break; // missing mandatory argument: stop (tolerant)
                    }
                }
                _ => {}
            }
        }
        let mandatory: Vec<(usize, usize)> = args.iter().filter(|a| !a.2).map(|a| (a.0, a.1)).collect();
        let t = self.t;
        let raw = |a: &(usize, usize)| t[a.0..a.1].to_string();

        if let Some(&(_, level)) = SECTION_LEVELS.iter().find(|(n, _)| *n == name) {
            self.brk(i);
            let start = self.pos();
            if let Some(last) = mandatory.last() {
                self.walk(last.0, last.1);
            }
            let title = self.buf[start..].trim().to_string();
            let lang = self.cur_lang();
            let end_pos = self.pos();
            self.regions.push(Region { kind: RegionKind::Heading, name: name.into(), start, end: end_pos, lang });
            self.sections.push(Section { level, title, starred: star, src_offset: i, clean_start: start });
            self.brk(i);
            return k;
        }
        if CITE_PAREN.contains(&name)
            || CITE_TEXT.contains(&name)
            || CITE_AUTHOR.contains(&name)
            || CITE_YEAR.contains(&name)
            || CITE_APUD.contains(&name)
            || matches!(name, "Idem" | "Ibidem" | "opcit" | "passim" | "cfcite" | "nocite")
        {
            if name == "nocite" {
                return k;
            }
            let keys: Vec<String> = mandatory
                .iter()
                .flat_map(|a| raw(a).split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>())
                .filter(|s| !s.is_empty())
                .collect();
            let start = self.pos();
            let ph = if CITE_TEXT.contains(&name) {
                "Silva (2020)"
            } else if CITE_AUTHOR.contains(&name) {
                "Silva"
            } else if CITE_YEAR.contains(&name) {
                "2020"
            } else if CITE_APUD.contains(&name) {
                "(SILVA, 2020 apud SOUZA, 2019)"
            } else {
                "(SILVA, 2020)"
            };
            self.put(ph, i);
            let end_pos = self.pos();
            self.cites.push(CiteSpan { start, end: end_pos, keys, command: name.into(), src_offset: i });
            return k;
        }
        if REF_MACROS.contains(&name) {
            self.put("1", i);
            return k;
        }
        match name {
            "foreignlanguage" => {
                let lang = mandatory.first().map(|a| lang_code(&raw(a))).unwrap_or_default();
                let cur = self.cur_lang();
                self.lang.push(if lang.is_empty() { cur } else { lang });
                if let Some(last) = mandatory.last().filter(|_| mandatory.len() >= 2) {
                    self.walk(last.0, last.1);
                }
                self.lang.pop();
                return k;
            }
            "caption" | "captionof" | "subcaption" => {
                self.caption(i, mandatory.last().copied());
                return k;
            }
            "item" => {
                self.brk(i);
                return k;
            }
            _ => {}
        }
        if TEXT_MACROS.contains(&name) {
            for a in &mandatory {
                self.walk(a.0, a.1);
            }
            return k;
        }
        if LAST_ARG_MACROS.contains(&name) {
            if let Some(last) = mandatory.last() {
                self.walk(last.0, last.1);
            }
            return k;
        }
        // everything else with a spec: arguments consumed and dropped
        k
    }

    fn caption(&mut self, i: usize, arg: Option<(usize, usize)>) {
        self.brk(i);
        let start = self.pos();
        if let Some(a) = arg {
            self.walk(a.0, a.1);
        }
        let lang = self.cur_lang();
        let end_pos = self.pos();
        self.regions.push(Region { kind: RegionKind::Caption, name: "caption".into(), start, end: end_pos, lang });
        self.brk(i);
    }

    /// Body end (offset of `\end`) and offset after `\end{name}`.
    fn env_end(&self, name: &str, from: usize, end: usize) -> (usize, usize) {
        let open = format!("\\begin{{{name}}}");
        let close = format!("\\end{{{name}}}");
        let mut depth = 1usize;
        let mut p = from;
        while p < end {
            let nb = self.t[p..end].find(&open).map(|k| p + k);
            let ne = self.t[p..end].find(&close).map(|k| p + k);
            match (nb, ne) {
                (Some(b), Some(e)) if b < e => {
                    depth += 1;
                    p = b + open.len();
                }
                (_, Some(e)) => {
                    depth -= 1;
                    if depth == 0 {
                        return (e, e + close.len());
                    }
                    p = e + close.len();
                }
                _ => break,
            }
        }
        (end, end)
    }

    fn env(&mut self, i: usize, j: usize, end: usize) -> usize {
        let b = self.t.as_bytes();
        let p = skip_ws(self.t, j);
        if b.get(p) != Some(&b'{') {
            return j;
        }
        let Some(close) = self.t[p + 1..end.max(p + 1)].find('}').map(|k| p + 1 + k) else { return j };
        let name = self.t[p + 1..close].trim().to_string();
        let after_name = close + 1;
        let (body_end, after) = self.env_end(&name, after_name, end);
        let n = name.as_str();
        let spec = match n {
            "otherlanguage" | "otherlanguage*" => "{",
            "resumo" | "resumoumacoluna" | "abstract" | "figure" | "table" | "figure*" | "table*" | "quadro"
            | "citacao" | "description" | "itemize" | "enumerate" => "[",
            "minipage" => "[[[{",
            "wrapfigure" => "[{[{",
            _ => "",
        };
        let (mand_n, opt_n) = (spec.matches('{').count(), spec.matches('[').count());
        let (args, mut body_start) =
            parse_args(self.t, after_name, mand_n.max(usize::from(opt_n > 0 && mand_n == 0)), opt_n);
        // `parse_args` needs a mandatory budget to read; with only optional args, drop a
        // spurious brace group it may have read.
        let args: Vec<_> = if mand_n == 0 {
            let opts: Vec<_> = args.iter().filter(|a| a.optional).cloned().collect();
            body_start = opts.last().map_or(after_name, |a| a.end + 1);
            opts
        } else {
            args
        };
        let body_start = body_start.min(body_end);

        if n == "document" {
            self.walk(body_start, body_end);
            return after;
        }
        if SKIP_ENVS.contains(&n) || VERBATIM_ENVS.contains(&n) || MATH_ENVS.contains(&n) {
            self.put(" ", i);
            return after;
        }
        if FLOAT_ENVS.contains(&n) {
            self.brk(i);
            let body = &self.t[body_start..body_end];
            let caps = crate::scan::find_commands(body, &["caption", "captionof", "subcaption"], 2, 1);
            for c in caps {
                let last = c.mandatory().last().map(|a| (body_start + a.start, body_start + a.end));
                self.caption(body_start + c.start, last);
            }
            self.brk(i);
            return after;
        }
        let mut pushed = false;
        if n == "otherlanguage" || n == "otherlanguage*" {
            let l = args.iter().find(|a| !a.optional).map(|a| lang_code(&a.text)).unwrap_or_default();
            let cur = self.cur_lang();
            self.lang.push(if l.is_empty() { cur } else { l });
            pushed = true;
        } else if ABSTRACT_ENVS.contains(&n) {
            let opt = args.iter().find(|a| a.optional).map(|a| a.text.trim().to_lowercase()).unwrap_or_default();
            let l = match opt.as_str() {
                "abstract" => Some("en"),
                "resumen" => Some("es"),
                "résumé" | "resume" => Some("fr"),
                "resumo" => Some("pt"),
                _ if n == "abstract" => None, // the document's own abstract
                _ => Some("pt"),
            };
            if let Some(l) = l {
                self.lang.push(l.to_string());
                pushed = true;
            }
        }
        let kind = if ABSTRACT_ENVS.contains(&n) {
            RegionKind::Abstract
        } else if LIST_ENVS.contains(&n) {
            RegionKind::List
        } else if QUOTE_ENVS.contains(&n) {
            RegionKind::Quote
        } else {
            RegionKind::Env
        };
        self.brk(i);
        let start = self.pos();
        self.walk(body_start, body_end);
        let lang = self.cur_lang();
        let end_pos = self.pos();
        self.regions.push(Region { kind, name, start, end: end_pos, lang });
        self.brk(after.saturating_sub(1).max(i));
        if pushed {
            self.lang.pop();
        }
        after
    }
}

/// View used by the walker: masked text where a comment also swallows its newline (as TeX does).
fn extraction_view(src: &SourceFile) -> String {
    let mut b = src.masked().as_bytes().to_vec();
    for c in src.comments() {
        if let Some(k) = b[c.offset..].iter().position(|&x| x == b'\n') {
            b[c.offset + k] = b' ';
        }
    }
    String::from_utf8(b).unwrap_or_default()
}

/// Extract clean text from one `.tex` file. Only the `document` body is used when the file has
/// one; files with `\documentclass` but no body yield nothing; included files are used whole.
/// `default_lang` is the document's main language code (`pt`, `en`…).
pub fn extract(src: &SourceFile, default_lang: &str) -> CleanText {
    let view = extraction_view(src);
    let mut em = Em {
        t: &view,
        buf: String::new(),
        offs: Vec::new(),
        regions: Vec::new(),
        sections: Vec::new(),
        cites: Vec::new(),
        lang: vec![default_lang.to_string()],
        depth: 0,
    };
    if let Some(b) = view.find("\\begin{document}") {
        let start = b + "\\begin{document}".len();
        let end = view.rfind("\\end{document}").filter(|&e| e >= start).unwrap_or(view.len());
        em.walk(start, end);
    } else if !view.contains("\\documentclass") {
        em.walk(0, view.len());
    }
    let mut ct = CleanText {
        text: em.buf,
        offsets: em.offs,
        regions: em.regions,
        sections: em.sections,
        cites: em.cites,
        paragraphs: Vec::new(),
    };
    ct.paragraphs = paragraphs(&ct, default_lang);
    ct
}

/// Paragraph boundaries: blank lines (`\n[ \t]*\n` plus following whitespace).
fn split_bounds(text: &str) -> Vec<(usize, usize)> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut pos = 0;
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\n' {
            let mut j = i + 1;
            while j < b.len() && matches!(b[j], b' ' | b'\t') {
                j += 1;
            }
            if j < b.len() && b[j] == b'\n' {
                out.push((pos, i));
                let mut k = j + 1;
                while k < b.len() && b[k].is_ascii_whitespace() {
                    k += 1;
                }
                pos = k;
                i = k;
                continue;
            }
        }
        i += 1;
    }
    out.push((pos, b.len()));
    out
}

fn paragraphs(ct: &CleanText, default_lang: &str) -> Vec<ParaSpan> {
    let mut out = Vec::new();
    for (s, e) in split_bounds(&ct.text) {
        let chunk = &ct.text[s..e];
        if chunk.trim().is_empty() {
            continue;
        }
        let s2 = s + (chunk.len() - chunk.trim_start().len());
        let e2 = e - (chunk.len() - chunk.trim_end().len());
        let inside = |r: &Region| r.start <= s2 && e2 <= r.end + 1;
        let best = ct.regions.iter().filter(|r| inside(r)).max_by_key(|r| r.start);
        let (mut kind, mut lang, mut region) = (ParaKind::Body, default_lang.to_string(), String::new());
        if let Some(r) = best {
            lang = r.lang.clone();
            region = r.name.clone();
            kind = match r.kind {
                RegionKind::Heading => ParaKind::Heading,
                RegionKind::Caption => ParaKind::Caption,
                RegionKind::List => ParaKind::Item,
                RegionKind::Abstract => ParaKind::Abstract,
                RegionKind::Quote => ParaKind::Quote,
                RegionKind::Env => ParaKind::Body,
            };
            if r.kind == RegionKind::Env {
                for a in ct.regions.iter().filter(|r| inside(r)) {
                    let k = match a.kind {
                        RegionKind::Abstract => ParaKind::Abstract,
                        RegionKind::Quote => ParaKind::Quote,
                        RegionKind::List => ParaKind::Item,
                        _ => continue,
                    };
                    kind = k;
                    region = a.name.clone();
                }
            }
        }
        out.push(ParaSpan { start: s2, end: e2, kind, lang, region });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Encoding;

    fn ex(t: &str) -> (SourceFile, CleanText) {
        let s = SourceFile::new("a.tex", "/a.tex", t.to_string(), Encoding::Utf8);
        let c = extract(&s, "pt");
        (s, c)
    }

    #[test]
    fn body_headings_cites_and_map() {
        let t = "\\documentclass{article}\n\\begin{document}\n\\section{Introdu\\c{c}\\~ao}\\label{s}\nO texto \\textbf{forte} \\cite[p.~2]{a,b} e $x^2$ -- ok.\n% comentário\nMesmo parágrafo.\n\n\\begin{itemize}\\item um\\end{itemize}\n\\end{document}\n";
        let (s, c) = ex(t);
        assert_eq!(c.sections.len(), 1);
        assert_eq!(c.sections[0].title, "Introdução");
        let kinds: Vec<_> = c.paragraphs.iter().map(|p| p.kind).collect();
        assert_eq!(kinds, [ParaKind::Heading, ParaKind::Body, ParaKind::Item]);
        let body = &c.text[c.paragraphs[1].start..c.paragraphs[1].end];
        assert!(body.starts_with("O texto forte (SILVA, 2020) e x – ok."), "{body}");
        assert!(body.contains("Mesmo parágrafo"), "comment must not split the paragraph: {body}");
        assert_eq!(c.cites[0].keys, ["a", "b"]);
        // map back: "forte" in clean text points at "forte" in the source
        let i = c.text.find("forte").unwrap();
        assert_eq!(&s.text[c.src(i)..c.src(i) + 5], "forte");
        let loc = c.location(&s, i, Some(i + 5));
        assert_eq!(loc.line, 4);
    }

    #[test]
    fn languages_abstracts_floats() {
        let t = "\\begin{resumo}Texto em português.\\end{resumo}\n\\begin{resumo}[Abstract]\\begin{otherlanguage*}{english}English text.\\end{otherlanguage*}\\end{resumo}\n\\begin{figure}\\centering\\includegraphics{x}\\caption{Uma figura}\\label{f}\\end{figure}\n\\begin{tabular}{cc} a & b \\end{tabular}";
        let (_, c) = ex(t);
        let p: Vec<_> = c.paragraphs.iter().map(|p| (&c.text[p.start..p.end], p.kind, p.lang.as_str())).collect();
        assert_eq!(p[0], ("Texto em português.", ParaKind::Abstract, "pt"));
        assert_eq!(p[1], ("English text.", ParaKind::Abstract, "en"));
        assert_eq!(p[2], ("Uma figura", ParaKind::Caption, "pt"));
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn accents_and_symbols() {
        let (_, c) = ex("Jos\\'e e \\\"u \\'{\\i} 50\\% \\LaTeX\\ \\emph{n\\~ao}");
        assert_eq!(c.text.trim(), "José e ü í 50% LaTeX não");
    }

    #[test]
    fn tolerates_garbage() {
        let (_, c) = ex("\\begin{itemize} \\item a { b \\end{enumerate} $ x \\section{");
        assert!(c.text.contains('a'));
    }
}
