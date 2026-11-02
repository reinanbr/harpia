//! harpia-verify against local mock servers: policy enforcement, claim and reference checks,
//! frozen replay and dataset curation. No real network.

use harpia_core::traits::PairClassifier;
use harpia_core::{Label, NetworkPolicy, Result, Scores};
use harpia_http::testing::ConnectionCounter;
use harpia_verify::adapters::SourceAdapter;
use harpia_verify::adapters::web::fetch_page;
use harpia_verify::{
    ClaimVerifier, FetchConfig, Fetcher, Mode, ReferenceConfig, ReferenceQuery, ReferenceStatus, ReferenceVerifier,
    SourceRegistry, Status, VerifyConfig,
};
use serde_json::json;
use std::time::Duration;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Entailment = share of the claim's longer words found in the passage; contradiction when the
/// passage also contains "not" or "never".
struct Overlap;
impl PairClassifier for Overlap {
    fn labels(&self) -> &[Label] {
        &[]
    }
    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Scores>> {
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
                let negated = p.contains(" not ") || p.contains(" never ");
                let (e, c) = if negated { (0.05, hit * 0.95) } else { (hit * 0.95, 0.02) };
                Scores::new(vec![
                    ("entailment".into(), e),
                    ("neutral".into(), (1.0 - e - c).max(0.0)),
                    ("contradiction".into(), c),
                ])
            })
            .collect())
    }
}

fn registry(server: &MockServer, extra: &str) -> SourceRegistry {
    let base = server.uri();
    SourceRegistry::from_toml(&format!(
        r#"
[[sources]]
id = "wiki"
name = "Mock Wikipedia"
kind = "encyclopedia"
domains = ["127.0.0.1"]
tier = 2
adapter = "wikipedia"
base_url = "{base}"
license = "CC-BY-SA-4.0"

[[sources]]
id = "crossref"
name = "Mock Crossref"
kind = "scholarly-api"
domains = ["127.0.0.1"]
tier = 1
adapter = "crossref"
base_url = "{base}"

[[sources]]
id = "openalex"
name = "Mock OpenAlex"
kind = "scholarly-api"
domains = ["127.0.0.1"]
tier = 1
adapter = "openalex"
base_url = "{base}"
{extra}
"#
    ))
    .unwrap()
}

fn online(store: Option<std::path::PathBuf>, mode: Mode) -> FetchConfig {
    FetchConfig {
        policy: NetworkPolicy::Online,
        mode,
        store,
        allow_loopback_http: true,
        retries: 1,
        backoff: Duration::from_millis(5),
        ..Default::default()
    }
}

fn wiki_body(extract: &str) -> serde_json::Value {
    json!({"query": {"pages": [
        {"pageid": 1, "index": 1, "title": "Harpy eagle", "fullurl": "https://en.wikipedia.org/wiki/Harpy_eagle", "extract": extract}
    ]}})
}

async fn mount_wiki(server: &MockServer, extract: &str) {
    Mock::given(method("GET"))
        .and(path("/w/api.php"))
        .and(query_param("generator", "search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(wiki_body(extract)))
        .mount(server)
        .await;
}

fn wiki_only() -> VerifyConfig {
    VerifyConfig { sources: Some(vec!["wiki".into()]), ..Default::default() }
}

#[tokio::test]
async fn supported_contradicted_and_disclosure() {
    let server = MockServer::start().await;
    mount_wiki(
        &server,
        "Birds are animals. The harpy eagle lives in the rainforests of Brazil and Panama. It hunts sloths.",
    )
    .await;
    let f = Fetcher::new(registry(&server, ""), online(None, Mode::Live)).unwrap();
    let v = ClaimVerifier::new(&f, &Overlap, wiki_only());
    let r = v.verify_claim("The harpy eagle lives in Brazil.", &[]).await;
    assert_eq!(r.status, Status::Supported, "{r:#?}");
    assert!(r.confidence > 0.9);
    let ev = &r.evidence[0];
    assert_eq!((ev.source.as_str(), ev.tier, ev.label.as_str()), ("wiki", 2, "entailment"));
    assert!(ev.passage.contains("rainforests of Brazil"));
    assert_eq!(ev.license.as_deref(), Some("CC-BY-SA-4.0"));
    assert_eq!(ev.content_hash.len(), 64);
    // Keywords only: the full claim never left the machine.
    assert_eq!(r.queries, ["harpy eagle lives brazil"]);
    let d = f.disclosure();
    assert_eq!(d.network_requests(), 1);
    assert!(d.entries[0].url.contains("gsrsearch=harpy eagle lives brazil"));
    assert!(!d.entries[0].url.contains("The harpy"));

    let server2 = MockServer::start().await;
    mount_wiki(&server2, "The harpy eagle does not live in Europe and never lived in Brazil's cities.").await;
    let f2 = Fetcher::new(registry(&server2, ""), online(None, Mode::Live)).unwrap();
    let r = ClaimVerifier::new(&f2, &Overlap, wiki_only()).verify_claim("The harpy eagle lives in Brazil.", &[]).await;
    assert_eq!(r.status, Status::Contradicted, "{r:#?}");
}

#[tokio::test]
async fn offline_skips_without_connections() {
    let counter = ConnectionCounter::start();
    let reg = SourceRegistry::from_toml(&format!(
        "[[sources]]\nid='wiki'\nname='w'\nkind='encyclopedia'\ndomains=['127.0.0.1']\ntier=2\nadapter='wikipedia'\nbase_url='{}'",
        counter.url()
    ))
    .unwrap();
    let f = Fetcher::new(reg, FetchConfig { allow_loopback_http: true, ..Default::default() }).unwrap();
    let r = ClaimVerifier::new(&f, &Overlap, VerifyConfig::default())
        .verify_claim("The harpy eagle lives in Brazil.", &[])
        .await;
    assert!(matches!(r.status, Status::Skipped { .. }), "{r:?}");
    let rr = ReferenceVerifier::new(&f, ReferenceConfig::default())
        .verify(&ReferenceQuery {
            title: "Deep learning".into(),
            doi: "10.1038/nature14539".into(),
            ..Default::default()
        })
        .await;
    assert!(matches!(rr.status, ReferenceStatus::Skipped { .. }));
    let (docs, d) = harpia_verify::rag::trusted_documents(
        &f,
        &harpia_verify::adapters::builtin(),
        "harpy",
        Default::default(),
        &harpia_text::lang::Lang::En.data(),
        3,
    )
    .await
    .unwrap();
    assert!(docs.is_empty() && d.entries.is_empty());
    counter.assert_no_connections();
}

#[tokio::test]
async fn frozen_mode_replays_offline() {
    let server = MockServer::start().await;
    mount_wiki(&server, "The harpy eagle lives in the rainforests of Brazil.").await;
    let dir = tempfile::tempdir().unwrap();
    let store = Some(dir.path().join("store"));
    let live = Fetcher::new(registry(&server, ""), online(store.clone(), Mode::Live)).unwrap();
    let a = ClaimVerifier::new(&live, &Overlap, wiki_only()).verify_claim("Harpy eagles live in Brazil.", &[]).await;
    drop(server); // the mock is gone: only the store can answer now
    // Same registry (store keys are URLs), offline policy, frozen mode.
    let frozen = Fetcher::new(
        live.registry().clone(),
        FetchConfig { policy: NetworkPolicy::Offline, ..online(store, Mode::Frozen) },
    )
    .unwrap();
    let b = ClaimVerifier::new(&frozen, &Overlap, wiki_only()).verify_claim("Harpy eagles live in Brazil.", &[]).await;
    assert_eq!(a.status, Status::Supported);
    assert_eq!(b.status, a.status);
    assert_eq!(b.evidence, a.evidence, "same content, time and hash");
    assert_eq!(frozen.requests_sent(), 0);
    assert!(frozen.disclosure().entries.iter().all(|e| e.from_store));
    // A claim that was never fetched is unavailable in frozen mode, not an error.
    let c = ClaimVerifier::new(&frozen, &Overlap, wiki_only()).verify_claim("Penguins cannot fly at all.", &[]).await;
    assert_eq!(c.status, Status::Unavailable);
}

#[tokio::test]
async fn web_policy_robots_redirects_size_and_type() {
    let server = MockServer::start().await;
    let port = server.address().port();
    Mock::given(path("/robots.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_string("User-agent: *\nDisallow: /private\n"))
        .mount(&server)
        .await;
    let page = "<html><head><title>Harpia</title></head><body><nav>menu</nav><article><p>The harpy eagle lives in Brazil.</p></article></body></html>";
    Mock::given(path("/ok"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(page, "text/html; charset=utf-8"))
        .mount(&server)
        .await;
    Mock::given(path("/private/page"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(page, "text/html"))
        .expect(0)
        .mount(&server)
        .await;
    Mock::given(path("/bounce"))
        .respond_with(ResponseTemplate::new(302).insert_header("location", format!("http://localhost:{port}/ok")))
        .mount(&server)
        .await;
    Mock::given(path("/big"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("x".repeat(5000), "text/html"))
        .mount(&server)
        .await;
    Mock::given(path("/pdf"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("%PDF-1.7", "application/pdf"))
        .mount(&server)
        .await;
    let extra = "[[sources]]\nid='site'\nname='Site'\nkind='docs'\ndomains=['127.0.0.1']\ntier=3\nadapter='web'";
    let f =
        Fetcher::new(registry(&server, extra), FetchConfig { max_bytes: 1000, ..online(None, Mode::Live) }).unwrap();
    let site = f.registry().get("site").unwrap().clone();
    let w = fetch_page(&f, &site, &format!("{}/ok", server.uri())).await.unwrap();
    assert_eq!((w.title.as_str(), w.text.as_str()), ("Harpia", "The harpy eagle lives in Brazil."));
    let e = fetch_page(&f, &site, &format!("{}/private/page", server.uri())).await.unwrap_err();
    assert!(e.to_string().contains("robots.txt"), "{e}");
    let e = fetch_page(&f, &site, &format!("{}/bounce", server.uri())).await.unwrap_err();
    assert!(e.to_string().contains("disallowed"), "redirect off the allowlist: {e}");
    let e = fetch_page(&f, &site, &format!("{}/big", server.uri())).await.unwrap_err();
    assert!(e.to_string().contains("larger than"), "{e}");
    let e = fetch_page(&f, &site, &format!("{}/pdf", server.uri())).await.unwrap_err();
    assert!(e.to_string().contains("content type"), "{e}");
    // Hosts outside the source's domains and plain HTTP to real hosts are refused before any request.
    let e = fetch_page(&f, &site, "https://example.com/").await.unwrap_err();
    assert!(e.to_string().contains("not allowed"), "{e}");
    let mut open = site.clone();
    open.domains = vec!["example.com".into()];
    let e = fetch_page(&f, &open, "http://example.com/").await.unwrap_err();
    assert!(e.to_string().contains("only HTTPS"), "{e}");
    let robots_fetches = f.disclosure().entries.iter().filter(|x| x.purpose == "robots.txt").count();
    assert_eq!(robots_fetches, 1, "robots.txt is cached per host");
}

#[tokio::test]
async fn retry_after_and_circuit_breaker() {
    let server = MockServer::start().await;
    Mock::given(path("/w/api.php"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    mount_wiki(&server, "The harpy eagle lives in Brazil.").await;
    let f = Fetcher::new(registry(&server, ""), online(None, Mode::Live)).unwrap();
    let r = ClaimVerifier::new(&f, &Overlap, wiki_only()).verify_claim("The harpy eagle lives in Brazil.", &[]).await;
    assert_eq!(r.status, Status::Supported, "succeeds after one 429");

    let failing = MockServer::start().await;
    Mock::given(path("/w/api.php")).respond_with(ResponseTemplate::new(503)).mount(&failing).await;
    let cfg = FetchConfig { retries: 0, failures_to_disable: 2, ..online(None, Mode::Live) };
    let f = Fetcher::new(registry(&failing, ""), cfg).unwrap();
    let v = ClaimVerifier::new(&f, &Overlap, wiki_only());
    for _ in 0..3 {
        let r = v.verify_claim("The harpy eagle lives in Brazil.", &[]).await;
        assert_eq!(r.status, Status::Unavailable);
    }
    assert_eq!(f.requests_sent(), 2, "disabled after two failures");
    assert!(f.host_status().iter().any(|h| h.disabled));
}

#[tokio::test]
async fn references_doi_search_and_retraction() {
    let server = MockServer::start().await;
    let item = |title: &str, family: &str, year: i64, doi: &str| {
        json!({"DOI": doi, "title": [title], "author": [{"family": family, "given": "A."}],
               "issued": {"date-parts": [[year]]}, "container-title": ["Nature"], "type": "journal-article"})
    };
    Mock::given(path("/works/10.1038/nature14539"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"message": item("Deep learning", "LeCun", 2015, "10.1038/nature14539")})),
        )
        .mount(&server)
        .await;
    let mut retracted = item("A retracted finding", "Doe", 2019, "10.1000/retracted");
    retracted["updated-by"] = json!([{"type": "retraction", "DOI": "10.1000/notice"}]);
    Mock::given(path("/works/10.1000/retracted"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"message": retracted})))
        .mount(&server)
        .await;
    Mock::given(path("/works/10.9999/missing")).respond_with(ResponseTemplate::new(404)).mount(&server).await;
    Mock::given(path("/works/doi:10.9999/missing")).respond_with(ResponseTemplate::new(404)).mount(&server).await;
    Mock::given(path("/works"))
        .and(query_param("query.bibliographic", "Attention is all you need Vaswani 2017"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({"message": {"items": [item("Attention Is All You Need", "Vaswani", 2018, "10.5555/attn")]}}),
        ))
        .mount(&server)
        .await;
    Mock::given(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"message": {"items": []}, "results": []})))
        .mount(&server)
        .await;
    let f = Fetcher::new(registry(&server, ""), online(None, Mode::Live)).unwrap();
    let rv = ReferenceVerifier::new(&f, ReferenceConfig::default());
    let q = |title: &str, author: &str, year: i32, doi: &str| ReferenceQuery {
        key: title.into(),
        kind: "article".into(),
        title: title.into(),
        authors: vec![author.into()],
        year: Some(year),
        doi: doi.into(),
        ..Default::default()
    };
    let ok = rv.verify(&q("Deep learning", "LeCun", 2015, "10.1038/nature14539")).await;
    assert_eq!(ok.status, ReferenceStatus::Confirmed, "{ok:#?}");
    assert_eq!(ok.consulted, ["crossref"]);
    let r = rv.verify(&q("A retracted finding", "Doe", 2019, "10.1000/retracted")).await;
    assert!(r.retracted && r.status == ReferenceStatus::Confirmed);
    let year_off = rv.verify(&q("Attention is all you need", "Vaswani", 2017, "")).await;
    match &year_off.status {
        ReferenceStatus::Divergent { fields } => assert!(fields.iter().any(|f| f.starts_with("year")), "{fields:?}"),
        s => panic!("expected divergent, got {s:?}"),
    }
    let missing = rv.verify(&q("Quantum plagiarism in hyperbolic nets", "Inventado", 2021, "10.9999/missing")).await;
    assert_eq!(missing.status, ReferenceStatus::NotFound);
    assert!(missing.notes.iter().any(|n| n.contains("10.9999/missing")), "{:?}", missing.notes);
    assert!(missing.confidence >= 0.5);
}

#[tokio::test]
async fn dataset_rows_flagged_and_dropped() {
    let server = MockServer::start().await;
    mount_wiki(&server, "The harpy eagle lives in Brazil. Owls hunt at night. Penguins never fly over Brazil.").await;
    let f = Fetcher::new(registry(&server, ""), online(None, Mode::Live)).unwrap();
    let v = ClaimVerifier::new(&f, &Overlap, wiki_only());
    let rows = vec![
        json!({"id": 1, "output": "The harpy eagle lives in Brazil."}),
        json!({"id": 2, "output": "Penguins fly over Brazil daily."}),
        json!({"id": 3, "other": "no field"}),
    ];
    let (kept, report) = harpia_verify::dataset::verify_rows(&v, rows, "output", true).await.unwrap();
    assert_eq!((report.rows, report.kept, report.dropped, report.missing_field), (3, 2, 1, 1));
    assert_eq!(kept[0]["verification"]["status"], "supported");
    assert_eq!(kept[1]["id"], 3);
}

#[tokio::test]
async fn custom_adapter_without_network() {
    struct Handbook;
    impl SourceAdapter for Handbook {
        fn name(&self) -> &str {
            "handbook"
        }
        fn search<'a>(
            &'a self,
            _f: &'a Fetcher,
            s: &'a harpia_verify::Source,
            _q: &'a str,
            _k: usize,
        ) -> harpia_verify::adapters::BoxFuture<'a, Result<Vec<harpia_verify::Work>>> {
            Box::pin(async move {
                Ok(vec![harpia_verify::Work {
                    source: s.id.clone(),
                    tier: s.tier,
                    title: "Policy".into(),
                    text: "Refunds are processed within 14 days.".into(),
                    ..Default::default()
                }])
            })
        }
    }
    let reg = SourceRegistry::from_toml("[[sources]]\nid='hb'\nname='Handbook'\nkind='custom'\ndomains=['handbook.internal']\ntier=1\nadapter='handbook'").unwrap();
    let f = Fetcher::new(reg, FetchConfig { policy: NetworkPolicy::Online, ..Default::default() }).unwrap();
    let v = ClaimVerifier::new(&f, &Overlap, VerifyConfig::default()).with_adapter(Box::new(Handbook));
    let r = v.verify_claim("Refunds are processed within 14 days.", &[]).await;
    assert_eq!(r.status, Status::Supported);
    assert_eq!(f.requests_sent(), 0);
}

/// Real network, opt-in: `HARPIA_NETWORK_TESTS=1 cargo test -p harpia-verify --features tls -- --ignored`.
#[tokio::test]
#[ignore = "real network; set HARPIA_NETWORK_TESTS=1"]
async fn real_crossref_lookup() {
    if std::env::var("HARPIA_NETWORK_TESTS").as_deref() != Ok("1") {
        return;
    }
    let f =
        Fetcher::new(SourceRegistry::builtin(), FetchConfig { policy: NetworkPolicy::Online, ..Default::default() })
            .unwrap();
    let r = ReferenceVerifier::new(&f, ReferenceConfig::default())
        .verify(&ReferenceQuery {
            key: "lecun2015".into(),
            kind: "article".into(),
            title: "Deep learning".into(),
            authors: vec!["LeCun".into()],
            year: Some(2015),
            doi: "10.1038/nature14539".into(),
            ..Default::default()
        })
        .await;
    assert_eq!(r.status, ReferenceStatus::Confirmed, "{r:#?}");
}
