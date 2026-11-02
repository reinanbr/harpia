//! End-to-end tests of the `harpia` binary.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;

fn harpia() -> Command {
    Command::cargo_bin("harpia").expect("binary built")
}

#[test]
fn segment_sentences_json_from_stdin() {
    let out = harpia()
        .args(["segment", "-", "--lang", "pt-br", "--format", "json"])
        .write_stdin("O Dr. Silva chegou.\nEle saiu.")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["schema"], 1);
    assert_eq!(v["count"], 2);
    assert_eq!(v["spans"][1]["line"], 2);
    assert_eq!(v["spans"][1]["text"], "Ele saiu.");
}

#[test]
fn repair_json_fixes_llm_output() {
    harpia()
        .args(["repair-json", "-"])
        .write_stdin("Sure! {name: 'Ana', tags: ['a',],")
        .assert()
        .success()
        .stdout(predicate::str::contains(r#"{"name":"Ana","tags":["a"]}"#));
    harpia().args(["repair-json", "-"]).write_stdin("no json at all").assert().code(1);
}

#[test]
fn missing_file_is_exit_code_2() {
    harpia().args(["stats", "/definitely/not/here.txt"]).assert().code(2).stderr(predicate::str::contains("error:"));
}

#[test]
fn data_pipeline_filter_dedup_split() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("raw.jsonl");
    let long = "The harpy eagle is one of the largest birds of prey and it lives in the forests of the Americas where it hunts in the canopy.";
    fs::write(
        &input,
        format!("{{\"text\": \"{long}\"}}\n{{\"text\": \"{long}\"}}\n{{\"text\": \"hi\"}}\n{{\"text\": \"mail ana@example.com now please, thanks a lot\"}}\n"),
    )
    .unwrap();
    let filtered = dir.path().join("filtered.jsonl");
    let report = dir.path().join("report.json");
    harpia()
        .args(["data", "filter"])
        .arg(&input)
        .args(["--preset", "minimal", "-o"])
        .arg(&filtered)
        .arg("--report")
        .arg(&report)
        .assert()
        .success()
        .stdout(predicate::str::contains("kept 3 of 4"));
    let rep: serde_json::Value = serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(rep["output"], 3);

    let dedup = dir.path().join("dedup.jsonl");
    harpia()
        .args(["data", "dedup"])
        .arg(&filtered)
        .arg("-o")
        .arg(&dedup)
        .assert()
        .success()
        .stdout(predicate::str::contains("kept 2 of 3"));

    let redacted = dir.path().join("redacted.jsonl");
    harpia().args(["data", "pii"]).arg(&dedup).args(["--redact", "-o"]).arg(&redacted).assert().success();
    assert!(fs::read_to_string(&redacted).unwrap().contains("<EMAIL>"));

    let out_dir = dir.path().join("splits");
    harpia()
        .args(["data", "split"])
        .arg(&redacted)
        .args(["--ratios", "0.5,0.5,0", "--out-dir"])
        .arg(&out_dir)
        .assert()
        .success();
    assert!(out_dir.join("train.jsonl").exists() && out_dir.join("test.jsonl").exists());
}

#[test]
fn data_convert_and_template() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("alpaca.jsonl");
    fs::write(&input, "{\"instruction\": \"Say hi\", \"input\": \"\", \"output\": \"Hi!\"}\n").unwrap();
    harpia()
        .args(["data", "convert"])
        .arg(&input)
        .args(["--to", "messages"])
        .assert()
        .success()
        .stdout(predicate::str::contains(r#""role":"assistant","content":"Hi!""#));
    harpia()
        .args(["data", "template"])
        .arg(&input)
        .args(["--template", "chatml"])
        .assert()
        .success()
        .stdout(predicate::str::contains("<|im_start|>assistant\n⟦Hi!<|im_end|>\n⟧"));
}

#[test]
fn chunk_markdown() {
    harpia()
        .args(["chunk", "-", "--strategy", "markdown", "--size", "100", "--format", "json"])
        .write_stdin("# A\ntext a\n# B\ntext b\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"count\": 2"));
}

#[test]
fn chat_without_online_flag_is_refused() {
    harpia()
        .args(["chat", "--provider", "ollama", "--model", "llama3.2", "hello"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("offline"));
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_and_synth_against_a_mock_openai_compatible_server() {
    use wiremock::matchers::path;
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "model": "local", "choices": [{"finish_reason": "stop", "message": {"content": "olá"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}})))
        .mount(&server)
        .await;
    let base = format!("{}/v1", server.uri());
    let dir = tempfile::tempdir().unwrap();
    let tmpl = dir.path().join("t.toml");
    fs::write(&tmpl, "model = \"x\"\nuser = \"Greet {{ name }}\"\noutput_field = \"greeting\"\n").unwrap();
    let input = dir.path().join("in.jsonl");
    fs::write(&input, "{\"name\": \"Ana\"}\n{\"name\": \"Bia\"}\n").unwrap();
    let out = dir.path().join("out.jsonl");
    let cache = dir.path().join("cache.jsonl");
    let (base2, tmpl2, input2, out2, cache2) = (base.clone(), tmpl.clone(), input.clone(), out.clone(), cache.clone());
    tokio::task::spawn_blocking(move || {
        harpia()
            .args(["chat", "--provider", "openai", "--model", "local", "--online", "--base-url", &base2, "hi"])
            .assert()
            .success()
            .stdout(predicate::str::contains("olá"));
        harpia()
            .args(["data", "synth"])
            .arg(&input2)
            .args(["--provider", "openai", "--model", "local", "--online", "--base-url", &base2, "--template"])
            .arg(&tmpl2)
            .arg("--cache")
            .arg(&cache2)
            .arg("-o")
            .arg(&out2)
            .assert()
            .success()
            .stdout(predicate::str::contains("2 generated"));
    })
    .await
    .unwrap();
    let rows: Vec<serde_json::Value> =
        fs::read_to_string(&out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows[1]["greeting"], "olá");
    assert_eq!(rows[1]["_provenance"]["model"], "local");
}

#[test]
fn local_classify_generate_embed_and_chat() {
    use harpia::local::testing;
    let dir = tempfile::tempdir().unwrap();
    let nli = dir.path().join("nli");
    testing::xlm_roberta_classifier(&nli, &["supports", "neutral", "contradicts"]).unwrap();
    let pairs = dir.path().join("pairs.jsonl");
    fs::write(&pairs, "{\"evidence\": \"the harpy eagle eats sloths\", \"claim\": \"the eagle eats monkeys\"}\n")
        .unwrap();
    let out = harpia()
        .args(["classify", "--model"])
        .arg(&nli)
        .arg(&pairs)
        .args(["--premise", "evidence", "--hypothesis", "claim"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let row: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert!(["supports", "neutral", "contradicts"].contains(&row["label"].as_str().unwrap()));
    assert_eq!(row["scores"].as_object().unwrap().len(), 3);

    harpia()
        .args(["embed", "--model"])
        .arg(&nli)
        .arg(&pairs)
        .args(["--field", "claim"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"embedding\""));

    let lm = dir.path().join("lm");
    testing::qwen2(&lm).unwrap();
    harpia()
        .args(["generate", "--model"])
        .arg(&lm)
        .args(["--choices", "yes,no", "--format", "json", "the bird"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"text\": \"yes\"").or(predicate::str::contains("\"text\": \"no\"")));
    harpia()
        .args(["chat", "--provider", "local", "--model"])
        .arg(&lm)
        .args(["--max-tokens", "4", "hello"])
        .assert()
        .success();
}

#[test]
fn train_classifier_and_lora_sft_end_to_end() {
    use harpia::local::testing;
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    testing::xlm_roberta_classifier(&base, &["entailment", "neutral", "contradiction"]).unwrap();
    let data = dir.path().join("pairs.jsonl");
    let rows: String = (0..12)
        .map(|i| {
            let (h, l) = [("yes", "supports"), ("no", "contradicts"), ("bird", "neutral")][i % 3];
            format!("{{\"premise\": \"the harpy eagle eats sloths\", \"hypothesis\": \"{h}\", \"label\": \"{l}\"}}\n")
        })
        .collect();
    fs::write(&data, rows).unwrap();
    let out = dir.path().join("nli");
    harpia()
        .args(["train", "classifier", "--base"])
        .arg(&base)
        .arg("--train")
        .arg(&data)
        .arg("--val")
        .arg(&data)
        .arg("--out")
        .arg(&out)
        .args(["--hypothesis", "hypothesis", "--labels", "supports,neutral,contradicts"])
        .args([
            "--label-map",
            "supports=entailment,contradicts=contradiction",
            "--epochs",
            "2",
            "--lr",
            "0.001",
            "--batch-size",
            "4",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("best val macro-F1"));
    let cfg: serde_json::Value = serde_json::from_str(&fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["id2label"]["2"], "contradicts");

    let lm = dir.path().join("lm");
    testing::qwen2(&lm).unwrap();
    let chat = dir.path().join("chat.jsonl");
    fs::write(&chat, "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}, {\"role\": \"assistant\", \"content\": \"hello world\"}]}\n").unwrap();
    let adapter = dir.path().join("adapter");
    harpia()
        .args(["train", "sft", "--base"])
        .arg(&lm)
        .arg("--data")
        .arg(&chat)
        .arg("--out")
        .arg(&adapter)
        .args(["--r", "4", "--epochs", "3", "--lr", "0.01"])
        .assert()
        .success()
        .stdout(predicate::str::contains("3 steps"));
    assert!(adapter.join("adapter_config.json").exists());
}
