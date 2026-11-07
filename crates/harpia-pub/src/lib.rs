//! # harpia-pub
//!
//! Scientific-publishing primitives for pure-Rust tools that read papers and theses written
//! in LaTeX:
//!
//! - [`project`]: read-only discovery of a project (main file, `\input`/`\include` graph with
//!   missing/out-of-root/cyclic detection, bibliographies, declared document language);
//! - [`source`]: files with a line index and a *masked* view where comments and verbatim are
//!   blanked without shifting byte offsets;
//! - [`scan`]: tolerant command/argument and environment scanning;
//! - [`extract`]: clean prose (headings, captions, lists, abstracts, quotes) with a per-byte
//!   map back to the source, citation placeholders and per-paragraph language;
//! - [`bib`]: tolerant BibTeX parsing with positions, names, required fields and identifier
//!   validators.
//!
//! Nothing here executes TeX or shells out; inputs are treated as untrusted.
//!
//! ```
//! use harpia_pub::extract::{extract, ParaKind};
//! use harpia_pub::source::{Encoding, SourceFile};
//!
//! let src = SourceFile::new("a.tex", "/a.tex", r"\section{Intro} Text \cite{k}.".into(), Encoding::Utf8);
//! let clean = extract(&src, "en");
//! assert_eq!(clean.paragraphs[0].kind, ParaKind::Heading);
//! assert_eq!(clean.cites[0].keys, ["k"]);
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bib;
pub mod extract;
pub mod project;
pub mod scan;
pub mod source;

pub use project::{LoadOptions, Paragraph, Project, ProjectError, load};
pub use source::{Location, SourceFile};
