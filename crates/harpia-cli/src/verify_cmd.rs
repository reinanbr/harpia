//! `harpia verify ...` and `harpia sources ...`: grounding against trusted sources.
//!
//! Nothing leaves the machine without `--online`; every request that does is listed in the
//! disclosure log (`--disclosure`, and the JSON output).

use crate::util::{OutputFormat, emit, read_records, read_text, write_json, write_records};
use clap::{Args, Subcommand, ValueEnum};
use harpia::core::traits::PairClassifier;
use harpia::core::{HarpiaError, NetworkPolicy, Result};
use harpia::eval::calibration::{Calibrated, DecisionRule, Thresholds};
use harpia::local::SequenceClassifier;
use harpia::verify::reference::check_quote;
use harpia::verify::verify::status_name;
use harpia::verify::{
    AnswerReport, ClaimVerifier, Disclosure, FetchConfig, Fetcher, Mode, NliLabels, ReferenceConfig, ReferenceQuery,
    ReferenceResult, ReferenceStatus, ReferenceVerifier, SourceRegistry, Status, Verification, VerifyConfig,
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Parse `--mode`.
pub fn parse_mode(s: &str) -> Result<Mode> {
    match s {
        "live" => Ok(Mode::Live),
        "cached-first" | "cached_first" => Ok(Mode::CachedFirst),
        "frozen" => Ok(Mode::Frozen),
        m => Err(HarpiaError::invalid(format!("unknown --mode `{m}` (live, cached-first, frozen)"))),
    }
}

/// Load a registry (built-in when `path` is `None`).
pub fn load_registry(path: Option<&Path>) -> Result<SourceRegistry> {
    let reg = match path {
        Some(p) => SourceRegistry::load(p)?,
        None => SourceRegistry::builtin(),
    };
    reg.validate()?;
    Ok(reg)
}

/// A fetcher over `registry` with the CLI's network and store settings.
pub fn build_fetcher(registry: SourceRegistry, online: bool, store: Option<PathBuf>, mode: Mode) -> Result<Fetcher> {
    let cfg = FetchConfig {
        policy: if online { NetworkPolicy::Online } else { NetworkPolicy::Offline },
        mode,
        store,
        ..FetchConfig::default()
    };
    Fetcher::new(registry, cfg)
}

/// Options shared by every subcommand that contacts sources.
#[derive(Args, Clone)]
pub struct SourceOpts {
    /// Trusted-source registry (TOML); the built-in registry when omitted.
    #[arg(long)]
    pub sources: Option<String>,
    /// Use only these source ids (repeatable).
    #[arg(long = "source", value_name = "ID")]
    pub only: Vec<String>,
    /// Allow network access (nothing is sent without it).
    #[arg(long)]
    pub online: bool,
    /// Content store directory (record responses; replay them with `--mode frozen`).
    #[arg(long)]
    pub store: Option<String>,
    /// Store mode: live, cached-first or frozen (frozen never uses the network).
    #[arg(long, default_value = "live")]
    pub mode: String,
    /// Write the disclosure log (every request sent or replayed) as JSON here.
    #[arg(long)]
    pub disclosure: Option<String>,
}

impl SourceOpts {
    fn fetcher(&self) -> Result<Fetcher> {
        let mut reg = load_registry(self.sources.as_deref().map(Path::new))?;
        if !self.only.is_empty() {
            let unknown: Vec<&String> = self.only.iter().filter(|id| reg.get(id).is_none()).collect();
            if !unknown.is_empty() {
                return Err(HarpiaError::invalid(format!("unknown source id(s): {unknown:?}")));
            }
            let ids: Vec<&str> = self.only.iter().map(String::as_str).collect();
            reg.restrict(&ids);
        }
        build_fetcher(reg, self.online, self.store.as_ref().map(PathBuf::from), parse_mode(&self.mode)?)
    }

    fn write_disclosure(&self, d: &Disclosure) -> Result<()> {
        if let Some(p) = &self.disclosure {
            write_json(p, &serde_json::to_value(d)?)?;
        }
        Ok(())
    }
}

/// Classifier options for claim verification.
#[derive(Args, Clone)]
pub struct NliOpts {
    /// NLI cross-encoder directory (premise = evidence, hypothesis = claim).
    #[arg(long)]
    pub classifier: String,
    /// Classifier label meaning "supported".
    #[arg(long, default_value = "entailment")]
    pub entailment: String,
    /// Classifier label meaning "contradicted".
    #[arg(long, default_value = "contradiction")]
    pub contradiction: String,
    /// Classifier label meaning "neutral".
    #[arg(long, default_value = "neutral")]
    pub neutral: String,
    /// Calibrated thresholds from `harpia calibrate` (default 0.5 per label).
    #[arg(long)]
    pub thresholds: Option<String>,
    /// Language of claims and evidence (en, pt-br).
    #[arg(long, default_value = "en")]
    pub lang: String,
}

impl NliOpts {
    fn config(&self) -> Result<VerifyConfig> {
        crate::util::lang(&self.lang)?;
        let labels = NliLabels {
            entailment: self.entailment.clone(),
            contradiction: self.contradiction.clone(),
            neutral: self.neutral.clone(),
        };
        let decision = match &self.thresholds {
            Some(p) => Calibrated::load(p)?,
            None => {
                let rule = DecisionRule::nli(&labels.entailment, &labels.contradiction, &labels.neutral);
                Calibrated::fixed(rule.clone(), Thresholds::uniform(&rule, 0.5))
            }
        };
        Ok(VerifyConfig { lang: self.lang.clone(), labels, decision, ..VerifyConfig::default() })
    }

    fn load(&self) -> Result<SequenceClassifier> {
        let clf = SequenceClassifier::load(Path::new(&self.classifier))?;
        let have = clf.labels();
        let missing: Vec<&String> =
            [&self.entailment, &self.contradiction, &self.neutral].into_iter().filter(|l| !have.contains(l)).collect();
        if !missing.is_empty() {
            return Err(HarpiaError::invalid(format!(
                "classifier labels are {have:?}; map them with --entailment/--contradiction/--neutral (missing {missing:?})"
            )));
        }
        Ok(clf)
    }
}

/// When `verify claim|answer` exits with code 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FailOn {
    /// Any claim contradicted by a trusted source.
    Contradicted,
    /// Any claim not supported (contradicted, not enough info, unavailable or skipped).
    Unsupported,
    /// Never (report only).
    Never,
}

fn fails(fail_on: FailOn, statuses: &[&Status]) -> bool {
    match fail_on {
        FailOn::Contradicted => statuses.iter().any(|s| **s == Status::Contradicted),
        FailOn::Unsupported => statuses.iter().any(|s| **s != Status::Supported),
        FailOn::Never => false,
    }
}

fn code(fail: bool) -> ExitCode {
    if fail { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

fn rt() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_current_thread().enable_all().build()?)
}

// ---------------------------------------------------------------- verify

#[derive(Subcommand)]
pub enum VerifyCommand {
    /// Split an answer into claims and check each against trusted sources.
    Answer(AnswerArgs),
    /// Check one claim against trusted sources.
    Claim(ClaimArgs),
    /// Check the claims in a field of every JSONL row; flag or drop contradicted rows.
    Dataset(DatasetArgs),
    /// Check bibliographic references (JSONL or BibTeX) against scholarly sources.
    Reference(ReferenceArgs),
    /// Locate a quote in a local full text (pages separated by form feeds).
    Quote(QuoteArgs),
}

#[derive(Args)]
#[command(after_help = "Example:
  harpia verify answer answer.txt --classifier models/nli --sources trusted.toml --online --disclosure sent.json

Exit codes: 0 ok, 1 a claim failed --fail-on (default: contradicted), 2 usage or environment error.")]
pub struct AnswerArgs {
    /// Answer text file (`-` for stdin).
    pub input: String,
    #[command(flatten)]
    pub nli: NliOpts,
    #[command(flatten)]
    pub src: SourceOpts,
    /// Which outcome exits with code 1.
    #[arg(long, value_enum, default_value_t = FailOn::Contradicted)]
    pub fail_on: FailOn,
}

#[derive(Args)]
#[command(after_help = "Example:
  harpia verify claim 'The harpy eagle is the national bird of Panama.' --classifier models/nli --online")]
pub struct ClaimArgs {
    /// The claim (`-` for stdin).
    pub claim: String,
    /// Extra search queries (repeatable), searched in addition to the claim's keywords.
    #[arg(long = "query")]
    pub queries: Vec<String>,
    #[command(flatten)]
    pub nli: NliOpts,
    #[command(flatten)]
    pub src: SourceOpts,
    /// Which outcome exits with code 1.
    #[arg(long, value_enum, default_value_t = FailOn::Contradicted)]
    pub fail_on: FailOn,
}

#[derive(Args)]
#[command(after_help = "Example:
  harpia verify dataset qa.jsonl --field output --classifier models/nli --online --drop-contradicted -o qa.verified.jsonl")]
pub struct DatasetArgs {
    /// JSONL input (`-` for stdin).
    pub input: String,
    /// Field holding the text to verify.
    #[arg(long, default_value = "output")]
    pub field: String,
    /// Leave out rows with a contradicted claim (otherwise every row is kept and annotated).
    #[arg(long)]
    pub drop_contradicted: bool,
    #[command(flatten)]
    pub nli: NliOpts,
    #[command(flatten)]
    pub src: SourceOpts,
    /// Output JSONL (stdout when omitted).
    #[arg(short, long)]
    pub output: Option<String>,
    /// Write the run summary (JSON) here.
    #[arg(long)]
    pub report: Option<String>,
}

#[derive(Args)]
#[command(after_help = "Examples:
  harpia verify reference --bib refs.bib --online
  harpia verify reference refs.jsonl --online --format json

JSONL rows use the ReferenceQuery fields: key, kind, title, authors (family names), year, venue,
doi, arxiv, isbn. Exit code 1 when a reference is not found or retracted (or divergent, with
--fail-on-divergent). \"Not found\" is not proof that a work does not exist.")]
pub struct ReferenceArgs {
    /// JSONL references (`-` for stdin); omit with --bib.
    #[arg(required_unless_present = "bib", conflicts_with = "bib")]
    pub input: Option<String>,
    /// BibTeX file.
    #[arg(long)]
    pub bib: Option<String>,
    /// Also exit 1 when metadata diverges.
    #[arg(long)]
    pub fail_on_divergent: bool,
    #[command(flatten)]
    pub src: SourceOpts,
    /// Write per-reference results (JSONL) here.
    #[arg(short, long)]
    pub output: Option<String>,
}

#[derive(Args)]
#[command(after_help = "Example:
  pdftotext -layout paper.pdf paper.txt
  harpia verify quote 'readability formulas apply to textbooks' --text paper.txt --pages 12-14

Exit code 1 when the quote is not found (similarity below 0.9).")]
pub struct QuoteArgs {
    /// The quote (`-` for stdin).
    pub quote: String,
    /// Full text; pages separated by form feeds (`\\f`, as `pdftotext` writes them).
    #[arg(long)]
    pub text: String,
    /// Cited page range, e.g. `12-14` or `7`.
    #[arg(long)]
    pub pages: Option<String>,
}

pub fn run(c: VerifyCommand, format: OutputFormat) -> Result<ExitCode> {
    match c {
        VerifyCommand::Answer(a) => answer(a, format),
        VerifyCommand::Claim(a) => claim(a, format),
        VerifyCommand::Dataset(a) => dataset(a, format),
        VerifyCommand::Reference(a) => reference(a, format),
        VerifyCommand::Quote(a) => quote(a, format),
    }
}

fn verification_text(v: &Verification) -> String {
    let mut s = format!("[{} {:.2}] {}", status_name(&v.status), v.confidence, v.claim);
    if let Status::Skipped { reason } = &v.status {
        s += &format!("\n    skipped: {reason}");
    }
    for e in v.evidence.iter().take(2) {
        let passage: String = e.passage.chars().take(200).collect();
        s += &format!("\n    {} [{}] {}: {passage}", e.source, e.label, e.url);
    }
    if v.disagreement {
        s += "\n    note: sources disagree";
    }
    s
}

fn report_text(r: &AnswerReport) -> String {
    let mut lines: Vec<String> = r.claims.iter().map(|c| verification_text(&c.verification)).collect();
    lines.push(format!("{:?}; {} request(s) sent", r.counts, r.disclosure.network_requests()));
    lines.join("\n")
}

fn answer(a: AnswerArgs, format: OutputFormat) -> Result<ExitCode> {
    let text = read_text(&a.input)?;
    let clf = a.nli.load()?;
    let fetcher = a.src.fetcher()?;
    let report = rt()?.block_on(ClaimVerifier::new(&fetcher, &clf, a.nli.config()?).verify_answer(&text));
    a.src.write_disclosure(&report.disclosure)?;
    let statuses: Vec<&Status> = report.claims.iter().map(|c| &c.verification.status).collect();
    let fail = fails(a.fail_on, &statuses);
    emit(format, json!({"report": report, "failed": fail}), || report_text(&report));
    Ok(code(fail))
}

fn claim(a: ClaimArgs, format: OutputFormat) -> Result<ExitCode> {
    let text = if a.claim == "-" { read_text("-")?.trim().to_string() } else { a.claim.clone() };
    let clf = a.nli.load()?;
    let fetcher = a.src.fetcher()?;
    let v = rt()?.block_on(ClaimVerifier::new(&fetcher, &clf, a.nli.config()?).verify_claim(&text, &a.queries));
    let disclosure = fetcher.disclosure();
    a.src.write_disclosure(&disclosure)?;
    let fail = fails(a.fail_on, &[&v.status]);
    emit(format, json!({"verification": v, "disclosure": disclosure, "failed": fail}), || {
        format!("{}\n{} request(s) sent", verification_text(&v), disclosure.network_requests())
    });
    Ok(code(fail))
}

fn dataset(a: DatasetArgs, format: OutputFormat) -> Result<ExitCode> {
    let rows = read_records(&a.input)?;
    let clf = a.nli.load()?;
    let fetcher = a.src.fetcher()?;
    let verifier = ClaimVerifier::new(&fetcher, &clf, a.nli.config()?);
    let (out, report) =
        rt()?.block_on(harpia::verify::dataset::verify_rows(&verifier, rows, &a.field, a.drop_contradicted))?;
    write_records(a.output.as_deref(), &out)?;
    let disclosure = fetcher.disclosure();
    a.src.write_disclosure(&disclosure)?;
    let payload = json!({"report": report, "requests_sent": disclosure.network_requests()});
    if let Some(p) = &a.report {
        write_json(p, &payload)?;
    }
    if a.output.is_some() {
        emit(format, payload, || {
            format!(
                "{} rows: {} kept, {} dropped, {} without `{}`; claims {:?}",
                report.rows, report.kept, report.dropped, report.missing_field, a.field, report.claims
            )
        });
    }
    Ok(ExitCode::SUCCESS)
}

fn reference_status(r: &ReferenceResult) -> &'static str {
    match r.status {
        ReferenceStatus::Confirmed => "confirmed",
        ReferenceStatus::Divergent { .. } => "divergent",
        ReferenceStatus::NotFound => "not_found",
        ReferenceStatus::Unavailable => "unavailable",
        ReferenceStatus::Skipped { .. } => "skipped",
    }
}

fn reference(a: ReferenceArgs, format: OutputFormat) -> Result<ExitCode> {
    let queries: Vec<ReferenceQuery> = match (&a.input, &a.bib) {
        (_, Some(bib)) => harpia::verify::bib::parse(&read_text(bib)?)?.iter().map(|e| e.to_query()).collect(),
        (Some(input), None) => read_records(input)?
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                serde_json::from_value(r).map_err(|e| HarpiaError::invalid(format!("reference row {}: {e}", i + 1)))
            })
            .collect::<Result<_>>()?,
        (None, None) => unreachable!("clap requires input or --bib"),
    };
    let fetcher = a.src.fetcher()?;
    let rv = ReferenceVerifier::new(&fetcher, ReferenceConfig::default());
    let rt = rt()?;
    let mut results = Vec::with_capacity(queries.len());
    for q in &queries {
        let r = rt.block_on(rv.verify(q));
        if format == OutputFormat::Text && a.output.is_none() {
            eprintln!("checked {}", q.key);
        }
        results.push(r);
    }
    let disclosure = fetcher.disclosure();
    a.src.write_disclosure(&disclosure)?;
    if let Some(p) = &a.output {
        let rows: Vec<Value> = results.iter().map(serde_json::to_value).collect::<serde_json::Result<_>>()?;
        write_records(Some(p), &rows)?;
    }
    let fail = results.iter().any(|r| {
        r.retracted
            || matches!(r.status, ReferenceStatus::NotFound)
            || (a.fail_on_divergent && matches!(r.status, ReferenceStatus::Divergent { .. }))
    });
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for r in &results {
        *counts.entry(reference_status(r)).or_default() += 1;
    }
    emit(format, json!({"results": results, "counts": counts, "disclosure": disclosure, "failed": fail}), || {
        let mut lines: Vec<String> = results
            .iter()
            .map(|r| {
                let mut s = format!("[{} {:.2}] {}", reference_status(r), r.confidence, r.key);
                if let ReferenceStatus::Divergent { fields } = &r.status {
                    s += &format!("\n    differs: {}", fields.join("; "));
                }
                if r.retracted {
                    s += "\n    RETRACTED";
                }
                for n in &r.notes {
                    s += &format!("\n    {n}");
                }
                s
            })
            .collect();
        lines.push(format!("{counts:?}; {} request(s) sent", disclosure.network_requests()));
        lines.join("\n")
    });
    Ok(code(fail))
}

fn parse_pages(s: &str) -> Result<(usize, usize)> {
    let bad = || HarpiaError::invalid(format!("--pages `{s}`: use `12` or `12-14`"));
    let (a, b) = s.split_once(['-', '–']).unwrap_or((s, s));
    let a: usize = a.trim().parse().map_err(|_| bad())?;
    let b: usize = b.trim().parse().map_err(|_| bad())?;
    if a == 0 || b < a {
        return Err(bad());
    }
    Ok((a, b))
}

fn quote(a: QuoteArgs, format: OutputFormat) -> Result<ExitCode> {
    let q = if a.quote == "-" { read_text("-")?.trim().to_string() } else { a.quote.clone() };
    let text = read_text(&a.text)?;
    let pages: Vec<String> = text.split('\u{c}').map(String::from).collect();
    let cited = a.pages.as_deref().map(parse_pages).transpose()?;
    let r = check_quote(&q, &pages, cited);
    emit(format, json!({"check": r, "pages": pages.len()}), || match (r.found, r.page, r.page_matches) {
        (true, Some(p), Some(false)) => {
            format!("found on page {p} (similarity {:.2}), outside the cited pages", r.score)
        }
        (true, Some(p), _) => format!("found on page {p} (similarity {:.2})", r.score),
        _ => format!("not found (best similarity {:.2} over {} pages)", r.score, pages.len()),
    });
    Ok(code(!r.found))
}

// ---------------------------------------------------------------- sources

#[derive(Subcommand)]
pub enum SourcesCommand {
    /// List the sources of a registry (built-in when --sources is omitted).
    List(ListArgs),
    /// Validate a registry; with --online, send one small search to each source.
    Check(CheckArgs),
}

#[derive(Args)]
#[command(after_help = "Examples:
  harpia sources list
  harpia sources list --toml > trusted.toml   # start your own registry from the built-in one")]
pub struct ListArgs {
    /// Registry file (TOML).
    #[arg(long)]
    pub sources: Option<String>,
    /// Print the registry as TOML.
    #[arg(long)]
    pub toml: bool,
}

#[derive(Args)]
#[command(after_help = "Example:
  harpia sources check --sources trusted.toml --online

Exit code 1 when a source is misconfigured or (with --online) does not answer.")]
pub struct CheckArgs {
    #[command(flatten)]
    pub src: SourceOpts,
    /// Probe query sent to each source with --online.
    #[arg(long, default_value = "harpy eagle")]
    pub query: String,
}

pub fn sources(c: SourcesCommand, format: OutputFormat) -> Result<ExitCode> {
    match c {
        SourcesCommand::List(a) => list(a, format),
        SourcesCommand::Check(a) => check(a, format),
    }
}

/// Registry entries that only allow hosts (full texts linked from other sources) and have no
/// search adapter by design.
const ALLOWLIST_ONLY: &[&str] = &["open-access-pdf"];

fn adapter_names() -> Vec<String> {
    harpia::verify::adapters::builtin().iter().map(|a| a.name().to_string()).collect()
}

fn list(a: ListArgs, format: OutputFormat) -> Result<ExitCode> {
    let reg = load_registry(a.sources.as_deref().map(Path::new))?;
    if a.toml {
        print!("{}", reg.to_toml()?);
        return Ok(ExitCode::SUCCESS);
    }
    let compiled = adapter_names();
    let rows: Vec<Value> = reg
        .sources
        .iter()
        .map(|s| {
            json!({
                "id": s.id, "name": s.name, "kind": s.kind, "tier": s.tier, "adapter": s.adapter,
                "domains": s.domains, "enabled": s.enabled, "license": s.license,
                "adapter_compiled": compiled.contains(&s.adapter),
                "allowlist_only": ALLOWLIST_ONLY.contains(&s.adapter.as_str()),
                "unavailable": s.unavailable_reason(),
            })
        })
        .collect();
    emit(format, json!({"sources": rows}), || {
        reg.sources
            .iter()
            .map(|s| {
                let state = match (s.unavailable_reason(), compiled.contains(&s.adapter)) {
                    (Some(r), _) => r,
                    (None, false) if ALLOWLIST_ONLY.contains(&s.adapter.as_str()) => "allowlist only".into(),
                    (None, false) => format!("adapter `{}` not compiled in", s.adapter),
                    (None, true) => "ready".into(),
                };
                format!("{:<20} tier {}  {:<16} {:<36} {state}", s.id, s.tier, s.adapter, s.domains.join(","))
            })
            .collect::<Vec<_>>()
            .join("\n")
    });
    Ok(ExitCode::SUCCESS)
}

fn check(a: CheckArgs, format: OutputFormat) -> Result<ExitCode> {
    let fetcher = a.src.fetcher()?;
    let adapters = harpia::verify::adapters::builtin();
    let rt = rt()?;
    let mut rows = Vec::new();
    let mut failed = false;
    for s in &fetcher.registry().sources {
        let mut row = json!({"id": s.id, "adapter": s.adapter});
        let adapter = adapters.iter().find(|ad| ad.name() == s.adapter);
        if s.enabled && s.unavailable_reason().is_none() && ALLOWLIST_ONLY.contains(&s.adapter.as_str()) {
            row["status"] = json!("allowlist-only");
            rows.push(row);
            continue;
        }
        let problem = match (s.unavailable_reason(), adapter) {
            (Some(r), _) => Some(r),
            (None, None) => Some(format!("adapter `{}` is not compiled into this build", s.adapter)),
            (None, Some(_)) if s.domains.is_empty() => Some("no allowed domains".into()),
            _ => None,
        };
        if let Some(p) = problem {
            // a disabled source is a deliberate choice, not a failure
            failed |= s.enabled;
            row["status"] = json!(if s.enabled { "misconfigured" } else { "disabled" });
            row["detail"] = json!(p);
        } else if let (true, Some(ad)) = (a.src.online || a.src.mode != "live", adapter) {
            let t = std::time::Instant::now();
            match rt.block_on(ad.search(&fetcher, s, &a.query, 1)) {
                Ok(works) => {
                    row["status"] = json!("ok");
                    row["results"] = json!(works.len());
                }
                Err(e) => {
                    failed = true;
                    row["status"] = json!("error");
                    row["detail"] = json!(e.to_string());
                }
            }
            row["ms"] = json!(t.elapsed().as_millis() as u64);
        } else {
            row["status"] = json!("configured");
        }
        rows.push(row);
    }
    let disclosure = fetcher.disclosure();
    a.src.write_disclosure(&disclosure)?;
    emit(format, json!({"sources": rows, "failed": failed, "disclosure": disclosure}), || {
        let mut lines: Vec<String> = rows
            .iter()
            .map(|r| {
                let mut s = format!("{:<20} {}", r["id"].as_str().unwrap_or(""), r["status"].as_str().unwrap_or(""));
                if let Some(ms) = r["ms"].as_u64() {
                    s += &format!(" ({ms} ms, {} result(s))", r["results"].as_u64().unwrap_or(0));
                }
                if let Some(d) = r["detail"].as_str() {
                    s += &format!(": {d}");
                }
                s
            })
            .collect();
        if !a.src.online && a.src.mode == "live" {
            lines.push("offline: configuration checked only; pass --online to probe each source".into());
        }
        lines.join("\n")
    });
    Ok(code(failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_and_fail_rules() {
        assert_eq!(parse_pages("12-14").unwrap(), (12, 14));
        assert_eq!(parse_pages("7").unwrap(), (7, 7));
        assert!(parse_pages("14-12").is_err() && parse_pages("x").is_err() && parse_pages("0").is_err());
        let nei = Status::NotEnoughInfo;
        assert!(!fails(FailOn::Contradicted, &[&nei]));
        assert!(fails(FailOn::Unsupported, &[&nei]));
        assert!(fails(FailOn::Contradicted, &[&Status::Supported, &Status::Contradicted]));
        assert!(!fails(FailOn::Never, &[&Status::Contradicted]));
    }
}
