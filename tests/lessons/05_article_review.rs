//! Lesson 5: reviewing a scientific article.
//!
//! `harpia::publish` reads a LaTeX project without running TeX: it follows `\input`, finds the
//! bibliography, extracts clean prose and keeps a byte map from every clean character back to
//! the `.tex` source. On top of that, `harpia_learn::review` is a ~300-line article reviewer in
//! the spirit of errei (citations, bibliography, style, readability, copied passages).
//! Read `tests/src/review.rs` alongside this lesson.
//!
//! The fixture in `tests/fixtures/article/` is a short pt-BR paper with deliberate mistakes.
//!
//! Run: `cargo test -p harpia-learn --test 05_article_review -- --nocapture`
//! Review your own article: `HARPIA_ARTICLE_DIR=/path/to/thesis cargo test -p harpia-learn
//! --test 05_article_review -- --ignored --nocapture`

use harpia::publish::extract::ParaKind;
use harpia::publish::{LoadOptions, load};
use harpia_learn::fixtures;
use harpia_learn::review::{Options, Severity, review, review_dir};

/// Step 1: what the loader sees.
#[test]
fn load_a_latex_project() {
    let project = load(&fixtures().join("article"), &LoadOptions::default()).expect("loads");
    let files: Vec<String> = project.tex.iter().map(|f| f.path.display().to_string()).collect();
    println!("files in document order: {files:?}");
    println!("language: {} ({:?})", project.lang, project.lang_source);
    assert_eq!(files.len(), 2, "main.tex and the \\input-ed secoes/metodo.tex");
    assert_eq!(project.lang, "pt");

    for clean in &project.clean {
        for s in &clean.sections {
            println!("section (level {}): {}", s.level, s.title);
        }
    }
    let abstract_text: Vec<&str> = project.paragraphs(Some(&[ParaKind::Abstract])).iter().map(|p| p.text()).collect();
    println!("abstract: {abstract_text:?}");
    assert!(abstract_text[0].starts_with("Este artigo"));

    // Clean text has no LaTeX commands, but every byte still maps to file:line:column.
    let p = project.paragraphs(Some(&[ParaKind::Body]))[0];
    println!("first paragraph at line {}: {}", p.first_line(), p.text());
    assert!(!p.text().contains("\\textit"));
    assert!(p.cites().any(|c| c.keys == ["silva2020"]));

    let keys: Vec<&str> = project.bib_entries().iter().map(|(_, e)| e.key.as_str()).collect();
    println!("bibliography: {keys:?}");
    assert_eq!(keys.len(), 4);
}

/// Step 2: the full review, with a reference corpus for the copy check.
#[test]
fn review_the_fixture_article() {
    let corpus_text = std::fs::read_to_string(fixtures().join("corpus/enciclopedia.txt")).unwrap();
    let opts = Options { corpus: vec![("enciclopedia.txt".into(), corpus_text)], ..Default::default() };
    let report = review_dir(&fixtures().join("article"), &opts).expect("review runs");
    println!("{report}");

    // Every planted mistake is found, with its location in the source.
    let undefined = report.of("citations.undefined");
    assert_eq!(undefined.len(), 1);
    assert!(undefined[0].message.contains("inexistente2021"));
    assert!(undefined[0].location.starts_with("main.tex:"));

    assert!(report.of("citations.unused").iter().any(|f| f.message.contains("orfao2019")));
    assert!(
        report.of("bib.missing-field").iter().any(|f| f.message.contains("souza2018") && f.message.contains("journal"))
    );
    assert!(report.of("bib.doi").iter().any(|f| f.message.contains("doi-invalido")));
    assert!(report.of("style.repeated-word").iter().any(|f| f.message.contains("de de")));
    assert!(report.of("style.filler").iter().any(|f| f.message.to_lowercase().contains("vale ressaltar")));
    assert_eq!(report.of("style.long-sentence").len(), 1);

    let copied = report.of("similarity.overlap");
    assert_eq!(copied.len(), 1, "the method section copies the encyclopedia");
    assert!(copied[0].location.starts_with("secoes/metodo.tex:"));

    // Nothing is flagged that should not be: the article has sections and an abstract.
    assert!(report.of("structure.sections").is_empty());
    assert!(report.of("structure.abstract").is_empty());
    assert_eq!(report.count_at_least(Severity::Error), 1, "only the undefined citation is an error");

    assert_eq!(report.stats.files, 2);
    assert_eq!(report.stats.bib_entries, 4);
    assert!(report.stats.sections >= 3);
}

/// Step 3: a broken project. Rules are independent, so one problem never hides another.
#[test]
fn review_a_broken_project() -> std::io::Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("main.tex"),
        "\\documentclass{article}\n\\begin{document}\nTexto sem seções que cita \\cite{fantasma}.\n\
         \\input{capitulo_perdido}\n\\end{document}\n",
    )?;
    let report = review_dir(dir.path(), &Options::default()).expect("still reviews");
    println!("{report}");
    assert_eq!(report.of("project.include").len(), 1, "the missing \\input is reported");
    assert_eq!(report.of("structure.sections").len(), 1);
    assert_eq!(report.of("structure.abstract").len(), 1);
    assert_eq!(report.of("citations.undefined").len(), 1);
    Ok(())
}

/// Step 4: your own article. Ignored by default; set `HARPIA_ARTICLE_DIR` to a LaTeX project.
/// For English articles also set `HARPIA_ARTICLE_LANG=en`.
#[test]
#[ignore = "needs HARPIA_ARTICLE_DIR"]
fn review_your_article() {
    let Ok(dir) = std::env::var("HARPIA_ARTICLE_DIR") else {
        eprintln!("set HARPIA_ARTICLE_DIR to a LaTeX project directory");
        return;
    };
    let mut opts = Options::default();
    if std::env::var("HARPIA_ARTICLE_LANG").as_deref() == Ok("en") {
        opts.lang = harpia::text::Lang::En;
        opts.filler_phrases =
            ["it is important to note", "it should be noted", "in today's world"].map(String::from).to_vec();
    }
    let project = load(std::path::Path::new(&dir), &LoadOptions::default()).expect("a LaTeX project");
    println!("{}", review(&project, &opts));
}
