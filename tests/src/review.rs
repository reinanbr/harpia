//! A small article reviewer in the spirit of [errei](https://github.com/reinanbr/harpia/blob/main/docs/errei-mapping.md),
//! written only with Harpia's public API.
//!
//! It is deliberately short (a teaching example, not a product): it loads a LaTeX project,
//! then runs a handful of independent rules over the clean text and the bibliography. Every
//! finding points back to `file:line:column` in the original `.tex`/`.bib`, because Harpia
//! keeps a byte map from clean text to source.
//!
//! | rule | what it checks | Harpia API used |
//! |---|---|---|
//! | `project.include` | `\input`/`\include` that are missing, cyclic or outside the root | `publish::load` |
//! | `structure.sections` | the document has sections and an abstract | `CleanText::sections`, `ParaKind` |
//! | `citations.undefined` / `citations.unused` | cite keys vs. bibliography keys | `Project::cite_uses`, `bib_entries` |
//! | `bib.syntax`, `bib.missing-field`, `bib.doi` | BibTeX problems, required fields, DOI shape | `publish::bib` |
//! | `style.long-sentence` | sentences longer than a word limit | `text::segment::sentences` |
//! | `style.repeated-word` | "de de", "the the" | `text::segment::words` |
//! | `style.filler` | filler phrases ("vale ressaltar") | `text::phrases::PhraseMatcher` |
//! | `readability.flesch` | hard-to-read paragraphs | `text::stats::flesch` |
//! | `similarity.overlap` | paragraphs copied from a reference corpus | `text::similarity` |
//!
//! The real errei (a separate project, `errei-rs`) adds pt-BR grammar, ABNT norms and more; it
//! is not needed here.

use harpia::publish::bib::{required_fields, valid_doi};
use harpia::publish::extract::ParaKind;
use harpia::publish::{LoadOptions, Location, Project};
use harpia::text::phrases::PhraseMatcher;
use harpia::text::similarity::{containment, longest_shared_span, shingle_set, shingles, tokens};
use harpia::text::{Lang, LangData, segment, stats};
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

/// How serious a finding is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// A hint; the text may be fine.
    Info,
    /// Probably worth fixing.
    Warning,
    /// Broken: the document will not build or cites something that does not exist.
    Error,
}

/// One problem found in the article.
#[derive(Debug, Clone)]
pub struct Finding {
    /// Rule id, e.g. `citations.undefined`.
    pub rule: &'static str,
    /// Severity.
    pub severity: Severity,
    /// Where, as `file:line:column`.
    pub location: String,
    /// What is wrong, in one sentence.
    pub message: String,
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<8} {:<22} {:<24} {}", format!("{:?}", self.severity), self.rule, self.location, self.message)
    }
}

/// Numbers about the article, useful on their own.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// `.tex` files read.
    pub files: usize,
    /// Body words.
    pub words: usize,
    /// Body sentences.
    pub sentences: usize,
    /// Section headings.
    pub sections: usize,
    /// Citation keys used (with repetitions).
    pub citations: usize,
    /// Bibliography entries.
    pub bib_entries: usize,
    /// Flesch reading ease of the whole body (`None` when too short).
    pub flesch: Option<f64>,
}

/// The review result.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// Findings, most severe first, then in document order.
    pub findings: Vec<Finding>,
    /// Article statistics.
    pub stats: Stats,
}

impl Report {
    /// Findings of one rule.
    pub fn of(&self, rule: &str) -> Vec<&Finding> {
        self.findings.iter().filter(|f| f.rule == rule).collect()
    }

    /// Number of findings at `severity` or above.
    pub fn count_at_least(&self, severity: Severity) -> usize {
        self.findings.iter().filter(|f| f.severity >= severity).count()
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.stats;
        writeln!(
            f,
            "{} files, {} sections, {} words, {} sentences, {} citations, {} references, Flesch {}",
            s.files,
            s.sections,
            s.words,
            s.sentences,
            s.citations,
            s.bib_entries,
            s.flesch.map_or("n/a".into(), |x| format!("{x:.1}"))
        )?;
        for finding in &self.findings {
            writeln!(f, "{finding}")?;
        }
        Ok(())
    }
}

/// Reviewer settings. `Default` is tuned for Brazilian Portuguese academic prose.
#[derive(Debug, Clone)]
pub struct Options {
    /// Language rules (abbreviations, syllables, stopwords).
    pub lang: Lang,
    /// Sentences longer than this many words are flagged.
    pub max_sentence_words: usize,
    /// Paragraphs whose Flesch score is below this are flagged.
    pub min_flesch: f64,
    /// Filler phrases to flag (matched case- and accent-insensitively, whole words).
    pub filler_phrases: Vec<String>,
    /// Reference corpus `(name, text)` for the copy check.
    pub corpus: Vec<(String, String)>,
    /// Word shingle size for the copy check.
    pub shingle_k: usize,
    /// A paragraph is flagged when at least this fraction of its shingles appear in one corpus text.
    pub overlap_threshold: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            lang: Lang::PtBr,
            max_sentence_words: 40,
            min_flesch: 25.0,
            filler_phrases: ["vale ressaltar", "é importante notar", "nos dias de hoje", "cabe destacar"]
                .map(String::from)
                .to_vec(),
            corpus: Vec::new(),
            shingle_k: 5,
            overlap_threshold: 0.5,
        }
    }
}

/// Load the LaTeX project in `root` (main file auto-detected) and review it.
pub fn review_dir(root: &Path, opts: &Options) -> Result<Report, String> {
    let project = harpia::publish::load(root, &LoadOptions::default()).map_err(|e| e.to_string())?;
    Ok(review(&project, opts))
}

/// Review an already loaded project.
pub fn review(project: &Project, opts: &Options) -> Report {
    let lang = opts.lang.data();
    let mut r = Report::default();
    includes(project, &mut r);
    structure(project, &mut r);
    citations(project, &mut r);
    bibliography(project, &mut r);
    prose(project, opts, &lang, &mut r);
    if !opts.corpus.is_empty() {
        overlap(project, opts, &mut r);
    }
    r.stats.files = project.tex.len();
    r.stats.bib_entries = project.bib_entries().len();
    // Stable order: severity (high first), then file and line as written in the location.
    r.findings
        .sort_by(|a, b| b.severity.cmp(&a.severity).then_with(|| natural(&a.location).cmp(&natural(&b.location))));
    r
}

fn at(loc: &Location) -> String {
    format!("{}:{}:{}", harpia::publish::source::slash(&loc.file), loc.line, loc.column.unwrap_or(1))
}

/// Sort key that orders `a.tex:9:1` before `a.tex:10:1`.
fn natural(loc: &str) -> (String, usize, usize) {
    let mut parts = loc.rsplitn(3, ':');
    let col = parts.next().and_then(|c| c.parse().ok()).unwrap_or(0);
    let line = parts.next().and_then(|l| l.parse().ok()).unwrap_or(0);
    (parts.next().unwrap_or("").to_string(), line, col)
}

fn push(r: &mut Report, rule: &'static str, severity: Severity, location: String, message: String) {
    r.findings.push(Finding { rule, severity, location, message });
}

fn includes(project: &Project, r: &mut Report) {
    for inc in &project.includes {
        if let Some(problem) = &inc.problem {
            let file = &project.tex[inc.file];
            let loc = at(&file.location(inc.offset, None));
            push(r, "project.include", Severity::Error, loc, format!("\\{}{{{}}}: {problem:?}", inc.kind, inc.raw));
        }
    }
}

fn structure(project: &Project, r: &mut Report) {
    let sections: usize = project.clean.iter().map(|c| c.sections.len()).sum();
    r.stats.sections = sections;
    let main = at(&project.main().location(0, None));
    if sections == 0 {
        push(
            r,
            "structure.sections",
            Severity::Warning,
            main.clone(),
            "no \\section: the article has no structure".into(),
        );
    }
    if project.paragraphs(Some(&[ParaKind::Abstract])).is_empty() {
        push(r, "structure.abstract", Severity::Warning, main, "no abstract environment".into());
    }
}

fn citations(project: &Project, r: &mut Report) {
    let defined: BTreeSet<&str> = project.bib_entries().iter().map(|(_, e)| e.key.as_str()).collect();
    let uses = project.cite_uses();
    r.stats.citations = uses.len();
    let mut used = BTreeSet::new();
    for u in &uses {
        used.insert(u.key.as_str());
        if !defined.contains(u.key.as_str()) {
            let loc = at(&project.tex[u.file].location(u.offset, Some(u.offset + u.key.len())));
            push(
                r,
                "citations.undefined",
                Severity::Error,
                loc,
                format!("`{}` is cited but not in the bibliography", u.key),
            );
        }
    }
    for (bib, entry) in project.bib_entries() {
        if !used.contains(entry.key.as_str()) {
            let loc = at(&bib.source.location(entry.key_offset, None));
            push(r, "citations.unused", Severity::Info, loc, format!("`{}` is never cited", entry.key));
        }
    }
}

fn bibliography(project: &Project, r: &mut Report) {
    for bib in &project.bibs {
        for p in &bib.problems {
            push(r, "bib.syntax", Severity::Error, at(&bib.source.location(p.offset, None)), format!("{:?}", p.kind));
        }
        for e in &bib.entries {
            let loc = at(&bib.source.location(e.start, None));
            let missing: Vec<&str> = required_fields(&e.kind)
                .unwrap_or(&[])
                .iter()
                // a required field may be satisfied by any of `a|b` alternatives
                .filter(|f| !f.split('|').any(|alt| e.has(alt)))
                .copied()
                .collect();
            if !missing.is_empty() {
                let msg = format!("@{} `{}` lacks {}", e.kind, e.key, missing.join(", "));
                push(r, "bib.missing-field", Severity::Warning, loc.clone(), msg);
            }
            if e.has("doi") && !valid_doi(e.get("doi")) {
                push(
                    r,
                    "bib.doi",
                    Severity::Warning,
                    loc,
                    format!("`{}` has a malformed DOI: {}", e.key, e.get("doi")),
                );
            }
        }
    }
}

fn prose(project: &Project, opts: &Options, lang: &LangData, r: &mut Report) {
    let fillers = PhraseMatcher::new(&opts.filler_phrases).ok();
    let mut body = String::new();
    for p in project.paragraphs(Some(&[ParaKind::Body, ParaKind::Abstract])) {
        let text = p.text();
        body.push_str(text);
        body.push_str("\n\n");

        for s in segment::sentences(text, lang) {
            r.stats.sentences += 1;
            let n = segment::word_count(s.slice(text));
            r.stats.words += n;
            if n > opts.max_sentence_words {
                let loc = at(&p.location(s.start, Some(s.end)));
                let msg = format!("sentence of {n} words (limit {}); consider splitting it", opts.max_sentence_words);
                push(r, "style.long-sentence", Severity::Info, loc, msg);
            }
        }

        // "de de": the same word twice in a row, case-insensitive
        let words: Vec<_> = segment::alpha_words(text).collect();
        for w in words.windows(2) {
            let (a, b) = (w[0].slice(text), w[1].slice(text));
            let only_space = text[w[0].end..w[1].start].chars().all(char::is_whitespace);
            if only_space && a.to_lowercase() == b.to_lowercase() {
                let loc = at(&p.location(w[0].start, Some(w[1].end)));
                push(r, "style.repeated-word", Severity::Warning, loc, format!("repeated word `{a} {b}`"));
            }
        }

        if let Some(m) = &fillers {
            for hit in m.find_iter(text) {
                let loc = at(&p.location(hit.span.start, Some(hit.span.end)));
                let msg = format!("filler phrase `{}`; say the point directly", hit.span.slice(text));
                push(r, "style.filler", Severity::Info, loc, msg);
            }
        }

        if let Some(f) = stats::flesch(text, lang)
            && f.score < opts.min_flesch
        {
            let msg = format!(
                "hard to read (Flesch {:.0} < {:.0}; {:.1} words/sentence, {:.2} syllables/word)",
                f.score, opts.min_flesch, f.asl, f.asw
            );
            push(r, "readability.flesch", Severity::Info, at(&p.location(0, None)), msg);
        }
    }
    r.stats.flesch = stats::flesch(&body, lang).map(|f| f.score);
}

fn overlap(project: &Project, opts: &Options, r: &mut Report) {
    let k = opts.shingle_k;
    let corpus: Vec<_> = opts.corpus.iter().map(|(name, text)| (name, shingle_set(text, k))).collect();
    for p in project.paragraphs(Some(&[ParaKind::Body])) {
        let text = p.text();
        let toks = tokens(text);
        let seq = shingles(&toks, k);
        if seq.is_empty() {
            continue;
        }
        let mine = seq.iter().copied().collect();
        for (name, theirs) in &corpus {
            let c = containment(&mine, theirs);
            if c >= opts.overlap_threshold {
                let span = longest_shared_span(&seq, theirs, &toks, k);
                let (loc, words) = match &span {
                    Some(s) => (at(&p.location(s.span.start, Some(s.span.end))), s.words),
                    None => (at(&p.location(0, None)), 0),
                };
                let msg = format!("{:.0}% of this paragraph matches `{name}` (longest run: {words} words)", c * 100.0);
                push(r, "similarity.overlap", Severity::Warning, loc, msg);
            }
        }
    }
}
