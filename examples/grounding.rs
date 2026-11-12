//! Grounding an LLM answer against a trusted-source registry served by a local mock (a tiny
//! HTTP server in a thread that answers like the Wikipedia search API).
//!
//! The answer is split into claims; each claim becomes a keywords-only query to the allowed
//! sources, the returned passages are scored by an NLI classifier, and every claim gets a
//! status with evidence. Everything sent out is listed in the disclosure log. With the default
//! offline policy nothing is sent and every claim is `Skipped`.
//!
//! The classifier is a word-overlap stand-in; use `harpia::local::SequenceClassifier` with an
//! NLI cross-encoder in practice.
//!
//! `cargo run -p harpia --example grounding --features verify`

use harpia::core::traits::PairClassifier;
use harpia::core::{Label, NetworkPolicy, Scores};
use harpia::verify::verify::status_name;
use harpia::verify::{ClaimVerifier, FetchConfig, Fetcher, Mode, SourceRegistry, VerifyConfig};
use std::io::{Read, Write};
use std::net::TcpListener;

const EXTRACT: &str = "The harpy eagle is a large raptor. It lives in the rainforests of Brazil and Panama. \
                       The harpy eagle mainly eats sloths and monkeys.";

/// Mock Wikipedia: every request gets the same search result.
fn mock_wikipedia() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let body = serde_json::json!({"query": {"pages": [{
                "pageid": 1, "index": 1, "title": "Harpy eagle",
                "fullurl": "https://en.wikipedia.org/wiki/Harpy_eagle", "extract": EXTRACT
            }]}})
            .to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    format!("http://{addr}")
}

/// NLI stand-in: entailment = share of the claim's content words found in the premise;
/// a claim naming a place the premise does not mention is scored as contradicted.
struct Overlap;
impl PairClassifier for Overlap {
    fn labels(&self) -> &[Label] {
        &[]
    }
    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> harpia::Result<Vec<Scores>> {
        Ok(pairs
            .iter()
            .map(|(premise, claim)| {
                let p = premise.to_lowercase();
                let words: Vec<String> = claim
                    .to_lowercase()
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|w| w.len() > 3)
                    .map(String::from)
                    .collect();
                let hit = words.iter().filter(|w| p.contains(w.as_str())).count() as f32 / words.len().max(1) as f32;
                let wrong_place = p.contains("lives in") && claim.contains("Europe");
                let (e, c) = if wrong_place { (0.02, 0.9) } else { (hit * 0.95, 0.02) };
                Scores::new(vec![
                    ("entailment".into(), e),
                    ("neutral".into(), (1.0 - e - c).max(0.0)),
                    ("contradiction".into(), c),
                ])
            })
            .collect())
    }
}

fn main() -> harpia::Result<()> {
    let base = mock_wikipedia();
    let registry = SourceRegistry::from_toml(&format!(
        r#"
[[sources]]
id = "wiki"
name = "Wikipedia (mock)"
kind = "encyclopedia"
domains = ["127.0.0.1"]
tier = 2
adapter = "wikipedia"
base_url = "{base}"
license = "CC-BY-SA-4.0"
"#
    ))?;
    let answer = "The harpy eagle lives in the rainforests of Brazil. \
                  The harpy eagle mainly eats sloths and monkeys. \
                  The harpy eagle can survive for forty years. \
                  The harpy eagle lives in the forests of Europe.";
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;

    // Offline (the default): nothing leaves the machine.
    let offline = Fetcher::new(registry.clone(), FetchConfig::default())?;
    let report = rt.block_on(ClaimVerifier::new(&offline, &Overlap, VerifyConfig::default()).verify_answer(answer));
    println!("offline: {:?}, requests sent: {}\n", report.counts, report.disclosure.network_requests());

    // Online, restricted to the registry's allowlist (the mock is plain http on loopback).
    let cfg = FetchConfig {
        policy: NetworkPolicy::Online,
        mode: Mode::Live,
        allow_loopback_http: true,
        ..Default::default()
    };
    let fetcher = Fetcher::new(registry, cfg)?;
    let report = rt.block_on(ClaimVerifier::new(&fetcher, &Overlap, VerifyConfig::default()).verify_answer(answer));
    for c in &report.claims {
        let v = &c.verification;
        println!("{:<16} {:.2}  {}", status_name(&v.status), v.confidence, v.claim);
        if let Some(ev) = v.evidence.first() {
            println!("                 evidence [{} tier {}] {:?}", ev.source, ev.tier, ev.passage);
        }
        println!("                 queries sent: {:?}", v.queries);
    }
    println!("\ncounts: {:?}", report.counts);
    println!("disclosure: {} requests", report.disclosure.network_requests());
    for e in &report.disclosure.entries {
        println!("  {}", e.url);
    }
    Ok(())
}
