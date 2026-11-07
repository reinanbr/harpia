//! Read-only discovery of a LaTeX project: the main file, the include graph, bibliographies,
//! the document language and the clean text of every reachable file.

use crate::bib::{self, BibEntry, BibFile};
use crate::extract::{self, CiteSpan, CleanText, ParaKind, ParaSpan};
use crate::scan::{Command, find_commands, split_keys};
use crate::source::{Encoding, Location, SourceFile};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::path::{Component, Path, PathBuf};

/// Directories never descended into.
pub const SKIP_DIRS: &[&str] = &["node_modules", "__pycache__", "build", "out", "dist", "venv", "target"];
/// Maximum number of files visited.
pub const MAX_FILES: usize = 2000;
/// Preferred names when several files have `\documentclass`.
pub const MAIN_NAMES: &[&str] =
    &["main.tex", "artigo.tex", "article.tex", "paper.tex", "documento.tex", "thesis.tex", "tese.tex"];
/// File-inclusion commands.
pub const INCLUDE_CMDS: &[&str] = &["input", "include", "subfile", "InputIfFileExists"];
/// Every citation command recognized in the source.
pub const CITE_CMDS: &[&str] = &[
    "cite",
    "citep",
    "citet",
    "citeonline",
    "citeauthor",
    "citeyear",
    "citeauthoronline",
    "parencite",
    "Parencite",
    "textcite",
    "Textcite",
    "autocite",
    "Autocite",
    "footcite",
    "citealp",
    "citealt",
    "citeyearpar",
    "Cite",
    "Citet",
    "Citeauthor",
    "nocite",
    "fullcite",
    "smartcite",
    "supercite",
];

/// Why a project could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectError {
    /// The root is not a directory.
    NotADirectory(PathBuf),
    /// No `.tex` file under the root.
    NoTexFiles(PathBuf),
    /// The requested main file does not exist.
    MainNotFound(PathBuf),
    /// No file has both `\documentclass` and `\begin{document}`.
    NoMainFile,
}

impl std::fmt::Display for ProjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotADirectory(p) => write!(f, "not a directory: {}", p.display()),
            Self::NoTexFiles(p) => write!(f, "no .tex file under {}", p.display()),
            Self::MainNotFound(p) => write!(f, "main file not found: {}", p.display()),
            Self::NoMainFile => write!(f, "no main .tex file (with \\documentclass and \\begin{{document}})"),
        }
    }
}

impl std::error::Error for ProjectError {}

/// What is wrong with an inclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncludeProblem {
    /// The target does not exist.
    Missing,
    /// Absolute path or escapes the project root (not read, for safety).
    OutsideRoot,
    /// The file (indirectly) includes itself.
    Cycle,
}

/// An `\input`, `\include`, `\bibliography`… occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Include {
    /// Index (in [`Project::tex`]) of the including file.
    pub file: usize,
    /// Offset of the command.
    pub offset: usize,
    /// Path as written.
    pub raw: String,
    /// Command name.
    pub kind: String,
    /// Resolved absolute path, when found inside the root.
    pub resolved: Option<PathBuf>,
    /// Problem, if any.
    pub problem: Option<IncludeProblem>,
}

impl Include {
    /// `\bibliography` or `\addbibresource`.
    pub fn is_bib(&self) -> bool {
        matches!(self.kind.as_str(), "bibliography" | "addbibresource")
    }
}

/// Observations made while loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectNote {
    /// Several candidate main files; this one was chosen.
    SeveralMains(PathBuf),
    /// A file was decoded as Latin-1.
    NotUtf8(PathBuf),
    /// No `\bibliography`/`\addbibresource`: every `.bib` was used.
    AllBibsUsed,
}

/// How the document language was decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LangSource {
    /// Forced by the caller.
    Forced,
    /// Class or babel options (last language is the main one).
    Babel,
    /// `\setmainlanguage` (polyglossia).
    Polyglossia,
    /// Not declared: decided by the caller's fallback (e.g. language identification).
    Undeclared,
}

/// Options for [`load`].
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Main file relative to the root (auto-detected when `None`).
    pub main: Option<PathBuf>,
    /// Force the document language code (`pt`, `en`…).
    pub lang: Option<String>,
    /// Language used when the source does not declare one.
    pub fallback_lang: Option<String>,
}

/// A `\documentclass`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocClass {
    /// Class name.
    pub name: String,
    /// Options as written.
    pub options: String,
    /// Offset in the main file.
    pub offset: usize,
}

/// A citation key used in the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CiteUse {
    /// File index in [`Project::tex`].
    pub file: usize,
    /// The command.
    pub command: Command,
    /// The key.
    pub key: String,
    /// Offset of the key.
    pub offset: usize,
}

/// A loaded project. `tex[0]` is the main file; `tex` is in document order.
#[derive(Debug, Clone)]
pub struct Project {
    /// Canonical root.
    pub root: PathBuf,
    /// Reachable `.tex` files in document order (DFS from the main file).
    pub tex: Vec<SourceFile>,
    /// Clean text of each file in `tex`.
    pub clean: Vec<CleanText>,
    /// `.tex` files not reachable from the main file.
    pub orphans: Vec<SourceFile>,
    /// Bibliographies.
    pub bibs: Vec<BibFile>,
    /// Inclusions.
    pub includes: Vec<Include>,
    /// Observations.
    pub notes: Vec<ProjectNote>,
    /// Main language code (`pt`, `en`…).
    pub lang: String,
    /// How [`Project::lang`] was decided.
    pub lang_source: LangSource,
}

/// A paragraph view over a file's clean text.
#[derive(Debug, Clone, Copy)]
pub struct Paragraph<'a> {
    /// File index in [`Project::tex`].
    pub index: usize,
    /// The source file.
    pub file: &'a SourceFile,
    /// The file's clean text.
    pub clean: &'a CleanText,
    /// The span.
    pub span: &'a ParaSpan,
}

impl<'a> Paragraph<'a> {
    /// Paragraph text.
    pub fn text(&self) -> &'a str {
        &self.clean.text[self.span.start..self.span.end]
    }

    /// Kind.
    pub fn kind(&self) -> ParaKind {
        self.span.kind
    }

    /// Language code.
    pub fn lang(&self) -> &'a str {
        &self.span.lang
    }

    /// Source offset of byte `rel` of the paragraph.
    pub fn src(&self, rel: usize) -> usize {
        self.clean.src(self.span.start + rel)
    }

    /// Source location of a paragraph-relative range.
    pub fn location(&self, rel_start: usize, rel_end: Option<usize>) -> Location {
        self.clean.location(self.file, self.span.start + rel_start, rel_end.map(|e| self.span.start + e))
    }

    /// First source line.
    pub fn first_line(&self) -> usize {
        self.file.line_col(self.src(0)).0
    }

    /// Number of `\w+` words.
    pub fn word_count(&self) -> usize {
        word_count(self.text())
    }

    /// Citations whose placeholder starts inside the paragraph.
    pub fn cites(&self) -> impl Iterator<Item = &'a CiteSpan> + 'a {
        let (s, e) = (self.span.start, self.span.end);
        self.clean.cites.iter().filter(move |c| s <= c.start && c.start < e)
    }
}

/// Count of alphanumeric runs (`\w+`).
pub fn word_count(text: &str) -> usize {
    let mut n = 0;
    let mut inside = false;
    for c in text.chars() {
        let w = c.is_alphanumeric() || c == '_';
        if w && !inside {
            n += 1;
        }
        inside = w;
    }
    n
}

impl Project {
    /// The main file.
    pub fn main(&self) -> &SourceFile {
        &self.tex[0]
    }

    /// Paragraphs of every reachable file, in document order, optionally filtered by kind.
    pub fn paragraphs(&self, kinds: Option<&[ParaKind]>) -> Vec<Paragraph<'_>> {
        let mut out = Vec::new();
        for (i, (file, clean)) in self.tex.iter().zip(&self.clean).enumerate() {
            for span in &clean.paragraphs {
                if kinds.is_none_or(|k| k.contains(&span.kind)) {
                    out.push(Paragraph { index: i, file, clean, span });
                }
            }
        }
        out
    }

    /// Commands in reachable files (and orphans on request) on the masked text.
    pub fn commands(&self, names: &[&str], mandatory: usize, include_orphans: bool) -> Vec<(&SourceFile, Command)> {
        let files = self.tex.iter().chain(self.orphans.iter().filter(|_| include_orphans));
        files.flat_map(|f| find_commands(f.masked(), names, mandatory, 3).into_iter().map(move |c| (f, c))).collect()
    }

    /// Every cited key, in document order (`\apud` contributes both arguments).
    pub fn cite_uses(&self) -> Vec<CiteUse> {
        let mut out = Vec::new();
        for (i, f) in self.tex.iter().enumerate() {
            let mut cmds = find_commands(f.masked(), CITE_CMDS, 1, 3);
            cmds.extend(find_commands(f.masked(), &["apud", "apudonline"], 2, 3));
            cmds.sort_by_key(|c| c.start);
            for c in cmds {
                for a in c.mandatory() {
                    for (key, off) in split_keys(a) {
                        out.push(CiteUse { file: i, command: c.clone(), key, offset: off });
                    }
                }
            }
        }
        out
    }

    /// Every bibliography entry with its file.
    pub fn bib_entries(&self) -> Vec<(&BibFile, &BibEntry)> {
        self.bibs.iter().flat_map(|b| b.entries.iter().map(move |e| (b, e))).collect()
    }

    /// Masked preamble of the main file (before `\begin{document}`).
    pub fn preamble(&self) -> &str {
        let m = self.main().masked();
        m.find("\\begin{document}").map_or(m, |i| &m[..i])
    }

    /// Loaded packages: name → (options, file, offset of the name).
    pub fn packages(&self) -> BTreeMap<String, (String, &SourceFile, usize)> {
        let mut out = BTreeMap::new();
        for (f, c) in self.commands(&["usepackage", "RequirePackage"], 1, false) {
            let opts = c.first_optional().map(|a| a.text.clone()).unwrap_or_default();
            if let Some(a) = c.first() {
                for (name, off) in split_keys(a) {
                    out.entry(name).or_insert((opts.clone(), f, off));
                }
            }
        }
        out
    }

    /// The main file's `\documentclass`.
    pub fn documentclass(&self) -> Option<DocClass> {
        documentclass(self.main())
    }

    /// Every file (reachable, orphan and `.bib`).
    pub fn all_sources(&self) -> Vec<&SourceFile> {
        self.tex.iter().chain(&self.orphans).chain(self.bibs.iter().map(|b| &b.source)).collect()
    }

    /// SHA-256 over paths and contents (hex).
    pub fn content_hash(&self) -> String {
        let mut files = self.all_sources();
        files.sort_by(|a, b| a.path.cmp(&b.path));
        let mut h = Sha256::new();
        for f in files {
            h.update(crate::source::slash(&f.path).as_bytes());
            h.update([0]);
            h.update(f.text.as_bytes());
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// File by relative path.
    pub fn find_source(&self, path: &Path) -> Option<&SourceFile> {
        self.all_sources().into_iter().find(|s| s.path == path)
    }
}

fn documentclass(f: &SourceFile) -> Option<DocClass> {
    let c = find_commands(f.masked(), &["documentclass"], 1, 1).into_iter().next()?;
    Some(DocClass {
        name: c.first()?.text.trim().to_string(),
        options: c.first_optional().map(|a| a.text.clone()).unwrap_or_default(),
        offset: c.start,
    })
}

/// Main language declared in a preamble: (code, source).
pub fn declared_language(main: &SourceFile) -> Option<(String, LangSource)> {
    let m = main.masked();
    let pre = m.find("\\begin{document}").map_or(m, |i| &m[..i]);
    if let Some(c) = find_commands(pre, &["setmainlanguage", "setdefaultlanguage"], 1, 1).into_iter().next() {
        let code = extract::lang_code(&c.first()?.text);
        if !code.is_empty() {
            return Some((code, LangSource::Polyglossia));
        }
    }
    // babel semantics: the last language option (class options then babel options) is the main one
    let known = |o: &str| {
        let c = extract::lang_code(o);
        matches!(c.as_str(), "en" | "pt" | "es" | "fr" | "de").then_some(c)
    };
    let mut last = None;
    if let Some(dc) = documentclass(main) {
        for o in dc.options.split(',') {
            if let Some(c) = known(o) {
                last = Some(c);
            }
        }
    }
    for c in find_commands(pre, &["usepackage"], 1, 1) {
        if c.first().is_some_and(|a| a.text.trim() == "babel") {
            let opts = c.first_optional().map(|a| a.text.as_str()).unwrap_or("");
            for o in opts.split(',') {
                let o = o.trim().trim_start_matches("main=");
                if let Some(c) = known(o) {
                    last = Some(c);
                }
            }
            // `main=xxx` wins
            if let Some(m) = opts.split(',').find_map(|o| o.trim().strip_prefix("main=")) {
                last = known(m).or(last);
            }
        }
    }
    last.map(|c| (c, LangSource::Babel))
}

fn walk_dir(root: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let (mut tex, mut bib) = (Vec::new(), Vec::new());
    let mut count = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut entries: Vec<_> = rd.filter_map(Result::ok).collect();
        entries.sort_by_key(|e| e.file_name());
        let mut subdirs = Vec::new();
        for e in entries {
            let Ok(meta) = std::fs::symlink_metadata(e.path()) else { continue };
            let name = e.file_name().to_string_lossy().into_owned();
            if meta.file_type().is_symlink() {
                continue;
            }
            if meta.is_dir() {
                if !name.starts_with('.') && !name.starts_with("_minted") && !SKIP_DIRS.contains(&name.as_str()) {
                    subdirs.push(e.path());
                }
                continue;
            }
            if name.ends_with(".tex") {
                tex.push(e.path());
            } else if name.ends_with(".bib") {
                bib.push(e.path());
            }
            count += 1;
            if count > MAX_FILES {
                return (tex, bib);
            }
        }
        stack.extend(subdirs.into_iter().rev());
    }
    tex.sort();
    bib.sort();
    (tex, bib)
}

/// Lexical normalization (`a/./b/../c` → `a/c`).
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

fn within(p: &Path, root: &Path) -> bool {
    let real = std::fs::canonicalize(p).unwrap_or_else(|_| normalize(p));
    real.starts_with(root)
}

fn is_main(f: &SourceFile) -> bool {
    let m = f.masked();
    m.contains("\\documentclass") && m.contains("\\begin{document}")
}

fn rel(root: &Path, p: &Path) -> PathBuf {
    p.strip_prefix(root).map(Path::to_path_buf).unwrap_or_else(|_| p.to_path_buf())
}

/// Load a project rooted at `root`.
pub fn load(root: &Path, opts: &LoadOptions) -> Result<Project, ProjectError> {
    if !root.is_dir() {
        return Err(ProjectError::NotADirectory(root.to_path_buf()));
    }
    let root = std::fs::canonicalize(root).map_err(|_| ProjectError::NotADirectory(root.to_path_buf()))?;
    let (tex_paths, bib_paths) = walk_dir(&root);
    if tex_paths.is_empty() {
        return Err(ProjectError::NoTexFiles(root));
    }
    let mut sources: BTreeMap<PathBuf, SourceFile> = BTreeMap::new();
    let mut notes = Vec::new();
    for p in &tex_paths {
        let real = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        if let Ok(s) = SourceFile::read(&real, &rel(&root, p)) {
            if s.encoding != Encoding::Utf8 {
                notes.push(ProjectNote::NotUtf8(s.path.clone()));
            }
            sources.insert(real, s);
        }
    }
    let main_key = match &opts.main {
        Some(m) => {
            let mp = std::fs::canonicalize(root.join(m)).map_err(|_| ProjectError::MainNotFound(m.clone()))?;
            if !sources.contains_key(&mp) {
                return Err(ProjectError::MainNotFound(m.clone()));
            }
            mp
        }
        None => {
            let cands: Vec<&PathBuf> = sources.iter().filter(|(_, s)| is_main(s)).map(|(k, _)| k).collect();

            match cands.len() {
                0 => return Err(ProjectError::NoMainFile),
                1 => cands[0].clone(),
                _ => {
                    let by_name = MAIN_NAMES
                        .iter()
                        .find_map(|n| cands.iter().find(|c| sources[**c].path.as_os_str() == std::ffi::OsStr::new(n)));
                    let pick = match by_name {
                        Some(c) => (*c).clone(),
                        None => (*cands
                            .iter()
                            .max_by_key(|c| find_commands(sources[**c].masked(), INCLUDE_CMDS, 1, 0).len())
                            .unwrap_or(&cands[0]))
                        .clone(),
                    };
                    notes.push(ProjectNote::SeveralMains(sources[&pick].path.clone()));
                    pick
                }
            }
        }
    };

    struct Walk<'s> {
        root: PathBuf,
        base_dir: PathBuf,
        sources: &'s mut BTreeMap<PathBuf, SourceFile>,
        ordered: Vec<SourceFile>,
        ordered_keys: Vec<PathBuf>,
        visiting: HashSet<PathBuf>,
        includes: Vec<Include>,
        bib_targets: Vec<PathBuf>,
    }

    impl Walk<'_> {
        fn resolve(&self, raw: &str, dir: &Path, exts: &[&str]) -> (Option<PathBuf>, Option<IncludeProblem>) {
            let raw = raw.trim();
            if raw.is_empty() {
                return (None, Some(IncludeProblem::Missing));
            }
            if Path::new(raw).is_absolute() {
                return (None, Some(IncludeProblem::OutsideRoot));
            }
            for ext in exts {
                let name = if !ext.is_empty() && !raw.ends_with(ext) { format!("{raw}{ext}") } else { raw.to_string() };
                let p = dir.join(name);
                if !within(&p, &self.root) {
                    return (None, Some(IncludeProblem::OutsideRoot));
                }
                if p.is_file() {
                    return (Some(std::fs::canonicalize(&p).unwrap_or(p)), None);
                }
            }
            (None, Some(IncludeProblem::Missing))
        }

        fn visit(&mut self, key: PathBuf, src: SourceFile) {
            self.visiting.insert(key.clone());
            let idx = self.ordered.len();
            let masked = src.masked().to_string();
            let src_dir = src.abspath.parent().map(Path::to_path_buf).unwrap_or_default();
            self.ordered.push(src);
            self.ordered_keys.push(key.clone());
            let mut events: Vec<(usize, String, Vec<String>, PathBuf)> = Vec::new();
            for c in find_commands(&masked, INCLUDE_CMDS, 1, 0) {
                if let Some(a) = c.first() {
                    let d = if c.name == "subfile" { src_dir.clone() } else { self.base_dir.clone() };
                    events.push((c.start, c.name.clone(), vec![a.text.clone()], d));
                }
            }
            for c in find_commands(&masked, &["import", "subimport"], 2, 0) {
                let m: Vec<_> = c.mandatory().collect();
                if m.len() == 2 {
                    let d = if c.name == "subimport" { src_dir.clone() } else { self.base_dir.clone() };
                    events.push((c.start, c.name.clone(), vec![m[1].text.clone()], d.join(m[0].text.trim())));
                }
            }
            for c in find_commands(&masked, &["bibliography", "addbibresource"], 1, 1) {
                if let Some(a) = c.first() {
                    let parts = split_keys(a).into_iter().map(|(k, _)| k).collect();
                    events.push((c.start, c.name.clone(), parts, self.base_dir.clone()));
                }
            }
            events.sort_by_key(|e| e.0);
            for (offset, kind, raws, dir) in events {
                for raw in raws {
                    let is_bib = kind == "bibliography" || kind == "addbibresource";
                    if is_bib {
                        let (p, prob) = self.resolve(&raw, &dir, &[".bib", ""]);
                        if let Some(p) = &p {
                            self.bib_targets.push(p.clone());
                        }
                        self.includes.push(Include {
                            file: idx,
                            offset,
                            raw,
                            kind: kind.clone(),
                            resolved: p,
                            problem: prob,
                        });
                        continue;
                    }
                    let (p, prob) = self.resolve(&raw, &dir, &[".tex", ""]);
                    if let Some(pp) = &p
                        && self.visiting.contains(pp)
                    {
                        self.includes.push(Include {
                            file: idx,
                            offset,
                            raw,
                            kind: kind.clone(),
                            resolved: None,
                            problem: Some(IncludeProblem::Cycle),
                        });
                        continue;
                    }
                    self.includes.push(Include {
                        file: idx,
                        offset,
                        raw,
                        kind: kind.clone(),
                        resolved: p.clone(),
                        problem: prob,
                    });
                    if let Some(p) = p {
                        if self.ordered_keys.contains(&p) {
                            continue;
                        }
                        let child = match self.sources.remove(&p) {
                            Some(s) => Some(s),
                            None => SourceFile::read(&p, &rel(&self.root, &p)).ok(),
                        };
                        if let Some(child) = child {
                            self.visit(p, child);
                        }
                    }
                }
            }
            self.visiting.remove(&key);
        }
    }

    let main_src = sources.remove(&main_key).ok_or(ProjectError::NoMainFile)?;
    let base_dir = main_src.abspath.parent().map(Path::to_path_buf).unwrap_or_else(|| root.clone());
    let mut w = Walk {
        root: root.clone(),
        base_dir,
        sources: &mut sources,
        ordered: Vec::new(),
        ordered_keys: Vec::new(),
        visiting: HashSet::new(),
        includes: Vec::new(),
        bib_targets: Vec::new(),
    };
    w.visit(main_key, main_src);
    let Walk { ordered, includes, mut bib_targets, .. } = w;
    let orphans: Vec<SourceFile> = sources.into_values().collect();

    if bib_targets.is_empty() && !bib_paths.is_empty() {
        bib_targets = bib_paths.iter().map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone())).collect();
        notes.push(ProjectNote::AllBibsUsed);
    }
    let mut seen = HashSet::new();
    let mut bibs = Vec::new();
    for p in bib_targets {
        if !seen.insert(p.clone()) {
            continue;
        }
        if let Ok(s) = SourceFile::read(&p, &rel(&root, &p)) {
            bibs.push(bib::parse(s));
        }
    }

    let (lang, lang_source) = match &opts.lang {
        Some(l) => (extract::lang_code(l), LangSource::Forced),
        None => declared_language(&ordered[0])
            .unwrap_or_else(|| (opts.fallback_lang.clone().unwrap_or_else(|| "en".into()), LangSource::Undeclared)),
    };
    let clean = ordered.iter().map(|f| extract::extract(f, &lang)).collect();
    Ok(Project { root, tex: ordered, clean, orphans, bibs, includes, notes, lang, lang_source })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, text: &str) {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, text).unwrap();
    }

    #[test]
    fn loads_include_graph() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "main.tex",
            "\\documentclass[english,brazil]{abntex2}\n\\begin{document}\n\\input{cap/intro}\n\\include{missing}\n\\input{../escape}\n\\cite{a}\\bibliography{refs}\n\\end{document}\n",
        );
        write(d.path(), "cap/intro.tex", "\\section{Introdução}\nTexto \\cite{a,b}.\n\\input{cap/intro}\n");
        write(d.path(), "orphan.tex", "nada");
        write(d.path(), "refs.bib", "@book{a, title={T}}\n");
        let p = load(d.path(), &LoadOptions::default()).unwrap();
        assert_eq!(p.tex.len(), 2);
        assert_eq!(p.tex[1].path, Path::new("cap/intro.tex"));
        assert_eq!(p.orphans.len(), 1);
        assert_eq!(p.bibs.len(), 1);
        assert_eq!(p.lang, "pt");
        let probs: Vec<_> = p.includes.iter().filter_map(|i| i.problem).collect();
        assert!(probs.contains(&IncludeProblem::Missing));
        assert!(probs.contains(&IncludeProblem::OutsideRoot));
        assert!(probs.contains(&IncludeProblem::Cycle));
        let keys: Vec<_> = p.cite_uses().into_iter().map(|c| c.key).collect();
        assert_eq!(keys, ["a", "a", "b"]);
        let heads = p.paragraphs(Some(&[ParaKind::Heading]));
        assert_eq!(heads[0].text(), "Introdução");
        assert_eq!(heads[0].location(0, None).file, Path::new("cap/intro.tex"));
    }

    #[test]
    fn errors() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(
            load(d.path(), &LoadOptions::default()).unwrap_err(),
            ProjectError::NoTexFiles(std::fs::canonicalize(d.path()).unwrap())
        );
        write(d.path(), "a.tex", "x");
        assert_eq!(load(d.path(), &LoadOptions::default()).unwrap_err(), ProjectError::NoMainFile);
    }

    #[test]
    fn language_detection() {
        let s = |t: &str| SourceFile::new("m.tex", "/m.tex", t.into(), Encoding::Utf8);
        assert_eq!(declared_language(&s("\\documentclass[brazil,english]{article}")).unwrap().0, "en");
        assert_eq!(
            declared_language(&s("\\documentclass{article}\\usepackage[english,main=brazilian]{babel}")).unwrap().0,
            "pt"
        );
        assert_eq!(declared_language(&s("\\setmainlanguage{portuguese}")).unwrap().1, LangSource::Polyglossia);
        assert!(declared_language(&s("\\documentclass{article}")).is_none());
    }
}
