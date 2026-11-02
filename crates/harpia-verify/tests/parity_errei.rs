//! Opt-in parity with errei's reference classification, replayed from errei's recorded HTTP
//! fixtures (`tests/fixtures/http/refs_artigo_exemplo.json`) against its example bibliography.
//!
//! `HARPIA_ERREI_DIR=~/Desktop/std/plugins/python/errei cargo test -p harpia-verify --test parity_errei -- --ignored`

use harpia_core::NetworkPolicy;
use harpia_verify::{FetchConfig, Fetcher, ReferenceConfig, ReferenceStatus, ReferenceVerifier, SourceRegistry};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn form_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => {
                out.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap_or(b'?'));
                i += 2;
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `(host, path, params without select/mailto)`.
type Key = (String, String, BTreeMap<String, String>);

fn parse_key(url: &str) -> Key {
    let rest = url.split_once("://").map_or(url, |x| x.1);
    let (hostpath, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (host, path) = hostpath.split_once('/').map_or((hostpath, ""), |(h, p)| (h, p));
    let params = query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (form_decode(k), form_decode(v)))
        .filter(|(k, _)| k != "select" && k != "mailto")
        .collect();
    (host.to_string(), format!("/{path}"), params)
}

struct Replay(Vec<(Key, u16, Value)>);

impl Respond for Replay {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        // The mock serves both APIs: /crossref/... and /openalex/... map back to their hosts.
        let path = form_decode(req.url.path());
        let (host, path) = if let Some(p) = path.strip_prefix("/crossref") {
            ("api.crossref.org", p.to_string())
        } else if let Some(p) = path.strip_prefix("/openalex") {
            ("api.openalex.org", p.to_string())
        } else {
            return ResponseTemplate::new(404);
        };
        let params: BTreeMap<String, String> = req
            .url
            .query_pairs()
            .filter(|(k, _)| k != "select" && k != "mailto")
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        let key = (host.to_string(), path, params);
        match self.0.iter().find(|(k, _, _)| *k == key) {
            Some((_, status, Value::String(s))) => ResponseTemplate::new(*status).set_body_string(s.clone()),
            Some((_, status, body)) => ResponseTemplate::new(*status).set_body_json(body.clone()),
            None => ResponseTemplate::new(404)
                .set_body_json(serde_json::json!({"status": "not-found", "key": format!("{key:?}")})),
        }
    }
}

#[tokio::test]
#[ignore = "needs HARPIA_ERREI_DIR"]
async fn errei_reference_classes() {
    let Ok(errei) = std::env::var("HARPIA_ERREI_DIR") else { return };
    let errei = std::path::PathBuf::from(errei);
    let recorded: Vec<Value> = serde_json::from_str(
        &std::fs::read_to_string(errei.join("tests/fixtures/http/refs_artigo_exemplo.json")).unwrap(),
    )
    .unwrap();
    let table = recorded
        .into_iter()
        .map(|r| (parse_key(r["key"].as_str().unwrap()), r["status"].as_u64().unwrap() as u16, r["body"].clone()))
        .collect();
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::any()).respond_with(Replay(table)).mount(&server).await;
    let reg = SourceRegistry::from_toml(&format!(
        r#"
[[sources]]
id = "crossref"
name = "Crossref (replay)"
kind = "scholarly-api"
domains = ["127.0.0.1"]
tier = 1
adapter = "crossref"
base_url = "{0}/crossref"
[[sources]]
id = "openalex"
name = "OpenAlex (replay)"
kind = "scholarly-api"
domains = ["127.0.0.1"]
tier = 1
adapter = "openalex"
base_url = "{0}/openalex"
"#,
        server.uri()
    ))
    .unwrap();
    let cfg = FetchConfig {
        policy: NetworkPolicy::Online,
        allow_loopback_http: true,
        retries: 0,
        backoff: Duration::from_millis(1),
        ..Default::default()
    };
    let f = Fetcher::new(reg, cfg).unwrap();
    let rv = ReferenceVerifier::new(&f, ReferenceConfig::default());
    let bib = harpia_verify::bib::parse(
        &std::fs::read_to_string(errei.join("tests/fixtures/artigo_exemplo/referencias.bib")).unwrap(),
    )
    .unwrap();
    let mut got = BTreeMap::new();
    for e in &bib {
        if got.contains_key(&e.key) || e.key == "nuncacitado2010" {
            continue; // errei checks each key once and only cited entries
        }
        let r = rv.verify(&e.to_query()).await;
        eprintln!("{:<14} {:?} conf={} notes={:?}", e.key, r.status, r.confidence, r.notes);
        got.insert(e.key.clone(), r);
    }
    for key in ["lecun2015", "devlin2019", "flesch1948"] {
        assert_eq!(got[key].status, ReferenceStatus::Confirmed, "{key}: {:#?}", got[key]);
    }
    let inv = &got["inventado2021"];
    assert_eq!(inv.status, ReferenceStatus::NotFound);
    assert!(inv.notes.iter().any(|n| n.contains("10.9999/rbei.2021.0042")), "{:?}", inv.notes);
    assert!(!inv.weakly_indexed);
    let m = &got["martins1996"];
    assert_eq!(m.status, ReferenceStatus::NotFound);
    assert!(m.weakly_indexed, "techreport without DOI is weakly indexed (errei: info severity)");
    assert!(m.confidence < inv.confidence);
}
