//! `harpia rag`, `rerank`, `verify` and `sources` end to end with tiny random models (offline).

use assert_cmd::Command;
use harpia::local::testing;
use predicates::prelude::*;
use serde_json::{Value, json};
use std::fs;
use std::path::Path;

fn harpia() -> Command {
    Command::cargo_bin("harpia").expect("binary built")
}

fn json_out(cmd: &mut Command) -> Value {
    let out = cmd.assert().success().get_output().stdout.clone();
    serde_json::from_slice(&out).unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out)))
}

fn jsonl(path: &Path) -> Vec<Value> {
    fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

#[test]
fn rag_index_append_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let enc = dir.path().join("enc");
    testing::xlm_roberta_classifier(&enc, &["entailment", "neutral", "contradiction"]).unwrap();
    let docs = dir.path().join("docs.jsonl");
    fs::write(
        &docs,
        concat!(
            "{\"id\": \"eagle\", \"text\": \"the harpy eagle eats sloths in the forest\", \"source\": \"a.md\", \"license\": \"CC-BY-4.0\", \"lang\": \"en\"}\n",
            "{\"id\": 7, \"text\": \"hello world\", \"lang\": \"en\"}\n",
        ),
    )
    .unwrap();
    let idx = dir.path().join("idx");
    let v =
        json_out(harpia().args(["rag", "index"]).arg(&docs).arg("--embedder").arg(&enc).arg("-o").arg(&idx).args([
            "--strategy",
            "sentence",
            "--size",
            "200",
            "--index",
            "hnsw",
            "--format",
            "json",
        ]));
    assert_eq!(v["documents"], 2);
    assert_eq!(v["chunks_total"], 2);
    // refuses to overwrite without --append; --append adds and re-indexing replaces
    harpia().args(["rag", "index"]).arg(&docs).arg("--embedder").arg(&enc).arg("-o").arg(&idx).assert().code(2);
    let more = dir.path().join("more.jsonl");
    fs::write(&more, "{\"text\": \"a large bird\", \"lang\": \"pt\"}\n{\"id\": \"7\", \"text\": \"hello again\"}\n")
        .unwrap();
    let v = json_out(
        harpia()
            .args(["rag", "index", "--append", "--format", "json"])
            .arg(&more)
            .arg("--embedder")
            .arg(&enc)
            .arg("-o")
            .arg(&idx),
    );
    assert_eq!(v["chunks_total"], 3);

    let v = json_out(
        harpia()
            .args(["rag", "query"])
            .arg(&idx)
            .args(["harpy eagle sloths", "--k", "10", "--budget", "100", "--format", "json"])
            .arg("--embedder")
            .arg(&enc)
            .arg("--reranker")
            .arg(&enc),
    );
    let hits = v["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 3, "--k above the saved top_k returns more hits");
    assert!(hits.iter().all(|h| h["rerank"].is_number()));
    assert!(v["context"]["text"].as_str().unwrap().contains("[1]"));
    let eagle = hits.iter().find(|h| h["chunk"]["doc_id"] == "eagle").unwrap();
    assert_eq!(eagle["chunk"]["license"], "CC-BY-4.0");
    assert_eq!(eagle["chunk"]["metadata"]["lang"], "en");

    let v = json_out(
        harpia()
            .args(["rag", "query"])
            .arg(&idx)
            .args(["bird", "--filter", r#"{"equals": {"lang": "pt"}}"#, "--format", "json"])
            .arg("--embedder")
            .arg(&enc),
    );
    assert_eq!(v["hits"].as_array().unwrap().len(), 1);
    assert_eq!(v["hits"][0]["chunk"]["doc_id"], "row-1");

    // batch queries produce the input `eval rag` expects
    let qs = dir.path().join("q.jsonl");
    fs::write(&qs, "{\"question\": \"harpy eagle\", \"relevant\": [\"eagle\"]}\n").unwrap();
    let out = dir.path().join("retrieved.jsonl");
    harpia()
        .args(["rag", "query"])
        .arg(&idx)
        .arg("--queries")
        .arg(&qs)
        .arg("--embedder")
        .arg(&enc)
        .arg("-o")
        .arg(&out)
        .assert()
        .success();
    let rows = jsonl(&out);
    assert_eq!(rows[0]["retrieved"].as_array().unwrap().len(), 3);
    assert_eq!(rows[0]["contexts"].as_array().unwrap().len(), 3);
    harpia().args(["eval", "rag"]).arg(&out).args(["--k", "3"]).assert().success();
}

#[test]
fn rerank_query_and_per_row_modes() {
    let dir = tempfile::tempdir().unwrap();
    let enc = dir.path().join("enc");
    testing::xlm_roberta_classifier(&enc, &["irrelevant", "relevant"]).unwrap();
    let docs = dir.path().join("docs.jsonl");
    fs::write(&docs, "{\"text\": \"the harpy eagle\"}\n{\"text\": \"hello world\"}\n{\"text\": \"a bird\"}\n").unwrap();
    let out = harpia()
        .args(["rerank", "--model"])
        .arg(&enc)
        .arg(&docs)
        .args(["--query", "eagle", "--top-n", "2"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let rows: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows.len(), 2);
    assert!(rows[0]["rerank_score"].as_f64() >= rows[1]["rerank_score"].as_f64());

    let cands = dir.path().join("cands.jsonl");
    fs::write(&cands, "{\"query\": \"eagle\", \"documents\": [\"a bird\", {\"text\": \"the harpy eagle\"}]}\n")
        .unwrap();
    let out = dir.path().join("out.jsonl");
    harpia().args(["rerank", "--model"]).arg(&enc).arg(&cands).arg("-o").arg(&out).assert().success();
    let r = &jsonl(&out)[0]["reranked"];
    assert_eq!(r.as_array().unwrap().len(), 2);
    assert!(r[0]["score"].as_f64() >= r[1]["score"].as_f64());
    harpia()
        .args(["rerank", "--model"])
        .arg(&enc)
        .arg(&docs)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("query"));
}

#[test]
fn verify_offline_is_skipped_and_discloses_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let nli = dir.path().join("nli");
    testing::xlm_roberta_classifier(&nli, &["entailment", "neutral", "contradiction"]).unwrap();
    let sent = dir.path().join("sent.json");
    let v = json_out(
        harpia()
            .args(["verify", "claim", "The harpy eagle lives in Brazil.", "--format", "json", "--classifier"])
            .arg(&nli)
            .arg("--disclosure")
            .arg(&sent),
    );
    assert_eq!(v["verification"]["status"], "skipped");
    assert_eq!(v["failed"], false);
    let d: Value = serde_json::from_str(&fs::read_to_string(&sent).unwrap()).unwrap();
    assert_eq!(d["entries"], json!([]));
    // skipped is not "supported"
    harpia()
        .args(["verify", "claim", "The harpy eagle lives in Brazil.", "--fail-on", "unsupported", "--classifier"])
        .arg(&nli)
        .assert()
        .code(1);
    // label names are checked against the classifier
    harpia()
        .args(["verify", "claim", "x", "--entailment", "supports", "--classifier"])
        .arg(&nli)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--entailment"));

    harpia()
        .args(["verify", "answer", "-", "--classifier"])
        .arg(&nli)
        .write_stdin("The harpy eagle is a large bird of prey. It lives in tropical forests of Brazil.")
        .assert()
        .success()
        .stdout(predicate::str::contains("skipped"));

    let rows = dir.path().join("rows.jsonl");
    fs::write(&rows, "{\"output\": \"The harpy eagle eats sloths in the forest.\"}\n{\"other\": 1}\n").unwrap();
    let out = dir.path().join("out.jsonl");
    let report = dir.path().join("report.json");
    harpia()
        .args(["verify", "dataset"])
        .arg(&rows)
        .arg("--classifier")
        .arg(&nli)
        .arg("-o")
        .arg(&out)
        .arg("--report")
        .arg(&report)
        .assert()
        .success();
    let out = jsonl(&out);
    assert_eq!(out[0]["verification"]["status"], "skipped");
    let r: Value = serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(r["report"]["missing_field"], 1);
    assert_eq!(r["requests_sent"], 0);
}

#[test]
fn verify_reference_and_quote() {
    let dir = tempfile::tempdir().unwrap();
    let bib = dir.path().join("refs.bib");
    fs::write(&bib, "@article{lecun2015, title={Deep learning}, author={LeCun, Yann and Bengio, Yoshua}, journal={Nature}, year={2015}, doi={10.1038/nature14539}}\n").unwrap();
    let v = json_out(harpia().args(["verify", "reference", "--format", "json", "--bib"]).arg(&bib));
    assert_eq!(v["results"][0]["key"], "lecun2015");
    assert_eq!(v["disclosure"]["entries"], json!([]));
    let refs = dir.path().join("refs.jsonl");
    fs::write(&refs, "{\"key\": \"k1\", \"title\": \"Deep learning\"}\n{\"key\": \"k2\"}\n").unwrap();
    harpia().args(["verify", "reference"]).arg(&refs).assert().code(2).stderr(predicate::str::contains("row 2"));

    let text = dir.path().join("paper.txt");
    fs::write(&text, "Introduction text.\u{c}As shown, readability formulas apply to textbooks.\u{c}End.").unwrap();
    let v = json_out(
        harpia()
            .args(["verify", "quote", "readability formulas apply to textbooks", "--pages", "2", "--format", "json"])
            .arg("--text")
            .arg(&text),
    );
    assert_eq!(v["check"]["page"], 2);
    assert_eq!(v["check"]["page_matches"], true);
    assert_eq!(v["pages"], 3);
    harpia()
        .args(["verify", "quote", "harpy eagles are reptiles"])
        .arg("--text")
        .arg(&text)
        .assert()
        .code(1)
        .stdout(predicate::str::contains("not found"));
    harpia().args(["verify", "quote", "x", "--pages", "3-1"]).arg("--text").arg(&text).assert().code(2);
}

#[test]
fn sources_list_toml_round_trip_and_check() {
    let dir = tempfile::tempdir().unwrap();
    let out = harpia().args(["sources", "list", "--toml"]).assert().success().get_output().stdout.clone();
    let reg = dir.path().join("trusted.toml");
    fs::write(&reg, &out).unwrap();
    let v = json_out(harpia().args(["sources", "list", "--format", "json", "--sources"]).arg(&reg));
    let ids: Vec<&str> = v["sources"].as_array().unwrap().iter().map(|s| s["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"crossref") && ids.contains(&"wikipedia-pt"));

    let v = json_out(
        harpia().args(["sources", "check", "--format", "json", "--source", "crossref", "--sources"]).arg(&reg),
    );
    assert_eq!(v["sources"].as_array().unwrap().len(), 1);
    assert_eq!(v["sources"][0]["status"], "configured");
    assert_eq!(v["failed"], false);

    // a source with an adapter this build lacks is misconfigured
    let bad = dir.path().join("bad.toml");
    fs::write(&bad, "[[sources]]\nid = \"x\"\nname = \"X\"\nkind = \"custom\"\ndomains = [\"x.org\"]\ntier = 3\nadapter = \"nope\"\n").unwrap();
    harpia()
        .args(["sources", "check", "--sources"])
        .arg(&bad)
        .assert()
        .code(1)
        .stdout(predicate::str::contains("misconfigured"));
    harpia().args(["sources", "check", "--source", "nope"]).assert().code(2);
}
