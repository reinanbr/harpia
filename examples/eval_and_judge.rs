//! Evaluation harness plus LLM-as-judge against mocked providers: a TOML task over a JSONL
//! dataset (few-shot, seeded, cached), two "models" compared on a Markdown leaderboard, then a
//! rubric score and a position-swapped pairwise comparison from a judge.
//!
//! `cargo run -p harpia --example eval_and_judge --features eval`

use harpia::core::traits::JudgeInput;
use harpia::data::JsonlCache;
use harpia::eval::harness::{ProviderBackend, RunOptions, Task, run_task};
use harpia::eval::judge::LlmJudge;
use harpia::eval::report::{RunSummary, leaderboard};
use harpia::eval::testing::{FnProvider, ScriptedProvider};
use std::io::Write;

const TASK: &str = r#"
name = "capitals"
kind = "generation"
dataset = "capitals.jsonl"
prompt = "Capital of {{ country }}?"
target = "{{ capital }}"
metrics = ["exact_match", "token_f1"]
few_shot = 2
seed = 7
"#;

fn capital_of(prompt: &str) -> Option<&'static str> {
    let last = prompt.rsplit("\n\n").next().unwrap_or(prompt);
    [("Brazil", "Brasília"), ("France", "Paris"), ("Japan", "Tokyo"), ("Peru", "Lima"), ("Kenya", "Nairobi")]
        .into_iter()
        .find(|(c, _)| last.contains(c))
        .map(|(_, cap)| cap)
}

fn main() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut f = std::fs::File::create(dir.path().join("capitals.jsonl"))?;
    for (country, capital) in
        [("Brazil", "Brasília"), ("France", "Paris"), ("Japan", "Tokyo"), ("Peru", "Lima"), ("Kenya", "Nairobi")]
    {
        writeln!(f, "{}", serde_json::json!({"country": country, "capital": capital}))?;
    }
    std::fs::write(dir.path().join("capitals.toml"), TASK)?;
    let task = Task::load(dir.path().join("capitals.toml"))?;

    // Two mocked models: one terse and correct, one chatty and sometimes wrong.
    let good = FnProvider::new(|p: &str| capital_of(p).unwrap_or("?").to_string());
    let chatty = FnProvider::new(|p: &str| match capital_of(p) {
        Some("Lima") => "I believe it is Cusco.".to_string(),
        Some(c) => format!("The capital is {c}."),
        None => "?".into(),
    });

    let rt = tokio::runtime::Builder::new_current_thread().build()?;
    let mut opts = RunOptions { cache: Some(JsonlCache::open(dir.path().join("cache.jsonl"))?), concurrency: 4 };
    let a = rt.block_on(run_task(&task, &ProviderBackend { provider: &good, model: "terse".into() }, &mut opts))?;
    let mut opts = RunOptions::default();
    let b = rt.block_on(run_task(&task, &ProviderBackend { provider: &chatty, model: "chatty".into() }, &mut opts))?;
    println!("first prompt (2-shot):\n{}\n", a.samples[0].prompt);
    let runs = [RunSummary::new("terse", &[a]), RunSummary::new("chatty", &[b])];
    println!("{}", leaderboard(&runs, None));

    // LLM-as-judge: rubric score (the first reply is malformed and is retried) ...
    let judge_llm = ScriptedProvider::new([
        "I'd say 4/5",
        "```json\n{\"score\": 4, \"rationale\": \"Correct but adds filler.\"}\n```",
        // ... and a pairwise comparison asked in both orders to cancel position bias.
        "{\"winner\": \"1\", \"rationale\": \"A is concise.\"}",
        "{\"winner\": \"2\", \"rationale\": \"The second answer is concise.\"}",
    ]);
    let judge = LlmJudge::new(&judge_llm, "mock-judge");
    let v = rt.block_on(judge.score(&JudgeInput {
        question: "Capital of Peru?".into(),
        answer: "The capital is Lima.".into(),
        reference: Some("Lima".into()),
        rubric: "1 = wrong, 5 = correct and concise".into(),
    }))?;
    println!("rubric score: {} ({})", v.score, v.rationale);
    let p = rt.block_on(judge.compare(
        "Capital of Peru?",
        "Prefer correct, concise answers.",
        "Lima",
        "I believe it is Cusco.",
    ))?;
    println!("pairwise: winner {:?}, consistent across swaps: {}", p.winner, p.consistent);
    println!("judge calls: {}", judge_llm.calls());
    Ok(())
}
