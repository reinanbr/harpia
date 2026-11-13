//! Lesson 4: grading a model, then correcting it.
//!
//! "Is this model any good?" has three layers, all in `harpia::eval`:
//!
//! 1. **metrics** compare one answer with a reference (exact match, F1, BLEU, chrF, ROUGE);
//! 2. a **classification report** grades labels (accuracy, per-class F1, confusion matrix);
//! 3. the **eval harness** runs a whole task (TOML + JSONL) against any model or API and
//!    produces comparable numbers and a leaderboard.
//!
//! When the grade is bad you **correct** the model: `harpia::train` fine-tunes it with LoRA
//! on the right answers, and you measure again.
//!
//! Run: `cargo test -p harpia-learn --test 04_model_review -- --nocapture`

use harpia::core::traits::{Model, Tokenizer};
use harpia::data::pack::TokenizedSample;
use harpia::eval::classification::ClassificationReport;
use harpia::eval::harness::{ProviderBackend, RunOptions, Task, run_task};
use harpia::eval::metrics;
use harpia::eval::report::{RunSummary, leaderboard};
use harpia::eval::testing::FnProvider;
use harpia::local::{DType, Device, TextGenerator, testing};
use harpia::train::{LoraConfig, LoraTrainConfig, train_lora_sft};

/// Generation metrics: from strict (exact match) to forgiving (chrF, ROUGE-L).
#[test]
fn answer_metrics() {
    let reference = "A harpia come preguiças e macacos.";
    let answers = [
        "a harpia come preguiças e macacos",  // same answer, different case and punctuation
        "A harpia come macacos e preguiças.", // same words, different order
        "O tucano come frutas.",              // wrong
    ];
    println!("{:<40} {:>5} {:>5} {:>5} {:>5} {:>7}", "answer", "EM", "F1", "BLEU", "chrF", "ROUGE-L");
    for a in answers {
        println!(
            "{a:<40} {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>7.2}",
            metrics::exact_match(a, reference),
            metrics::token_f1(a, reference),
            metrics::sentence_bleu(a, &[reference]),
            metrics::chrf(a, reference),
            metrics::rouge_l(a, reference).f1,
        );
    }
    // exact match normalizes case, punctuation and articles before comparing
    assert_eq!(metrics::exact_match(answers[0], reference), 1.0);
    // token F1 ignores order; ROUGE-L (longest common subsequence) does not
    assert_eq!(metrics::token_f1(answers[1], reference), 1.0);
    assert!(metrics::rouge_l(answers[1], reference).f1 < 1.0);
    assert!(metrics::token_f1(answers[2], reference) < 0.4);
}

/// Grading a classifier: never trust accuracy alone on unbalanced labels.
#[test]
fn classification_report() {
    let gold = ["supports", "supports", "supports", "supports", "contradicts", "neutral"];
    let pred = ["supports", "supports", "supports", "supports", "supports", "supports"];
    let r = ClassificationReport::new(&gold, &pred, &["supports", "contradicts", "neutral"]);
    println!("accuracy {:.2}, macro-F1 {:.2}", r.accuracy, r.macro_f1);
    for (label, s) in &r.per_class {
        println!("{label:>12}: P {:.2} R {:.2} F1 {:.2} (n={})", s.precision, s.recall, s.f1, s.support);
    }
    println!("confusion (rows = gold): {:?}", r.confusion);
    // A model that always says "supports" gets 67% accuracy but a poor macro-F1.
    assert!(r.accuracy > 0.6);
    assert!(r.macro_f1 < 0.3);
}

/// The eval harness: one TOML task, two models, one leaderboard. The "models" here are
/// closures behind the `Provider` trait; an API client or a local model plugs in the same way.
#[test]
fn eval_harness_and_leaderboard() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    let rows = [("Brazil", "Brasília"), ("France", "Paris"), ("Japan", "Tokyo"), ("Peru", "Lima")];
    let jsonl: String =
        rows.iter().map(|(c, k)| format!("{}\n", serde_json::json!({"country": c, "capital": k}))).collect();
    std::fs::write(dir.path().join("capitals.jsonl"), jsonl)?;
    std::fs::write(
        dir.path().join("capitals.toml"),
        r#"
name = "capitals"
kind = "generation"
dataset = "capitals.jsonl"
prompt = "Capital of {{ country }}?"
target = "{{ capital }}"
metrics = ["exact_match", "token_f1"]
"#,
    )?;
    let task = Task::load(dir.path().join("capitals.toml"))?;

    let answer = |p: &str| rows.iter().find(|(c, _)| p.contains(c)).map_or("?", |(_, k)| k).to_string();
    let good = FnProvider::new(move |p: &str| answer(p));
    let wrong_on_peru = FnProvider::new(move |p: &str| if p.contains("Peru") { "Cusco".into() } else { answer(p) });

    let rt = tokio::runtime::Builder::new_current_thread().build()?;
    let a = rt.block_on(run_task(
        &task,
        &ProviderBackend { provider: &good, model: "good".into() },
        &mut RunOptions::default(),
    ))?;
    let b = rt.block_on(run_task(
        &task,
        &ProviderBackend { provider: &wrong_on_peru, model: "wrong-on-peru".into() },
        &mut RunOptions::default(),
    ))?;
    assert_eq!(a.metrics["exact_match"], 1.0);
    assert_eq!(b.metrics["exact_match"], 0.75);
    println!("{}", leaderboard(&[RunSummary::new("good", &[a]), RunSummary::new("wrong-on-peru", &[b])], None));
    Ok(())
}

/// Correcting a model: the base model gives the right answer a low probability; a few LoRA
/// steps on that answer raise it. The loss is only on the answer tokens (the prompt is context).
#[test]
fn correct_a_model_with_lora() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    let (base, adapter) = (dir.path().join("base"), dir.path().join("adapter"));
    testing::qwen2(&base)?;
    let lm = TextGenerator::load(&base)?;

    let (prompt, answer) = ("the harpy eagle", " eats sloths in the forest");
    let before = lm.loglikelihood(prompt, answer)?;

    let prompt_len = lm.tokenizer().encode(prompt, false)?.ids.len();
    let ids = lm.tokenizer().encode(&format!("{prompt}{answer}"), false)?.ids;
    let loss_mask = (0..ids.len()).map(|i| i >= prompt_len).collect();
    let cfg = LoraTrainConfig {
        lora: LoraConfig { r: 4, alpha: 8.0, ..Default::default() },
        epochs: 15,
        lr: 2e-2,
        warmup_ratio: 0.0,
        ..Default::default()
    };
    let report = train_lora_sft(&base, &adapter, &cfg, &[TokenizedSample { ids, loss_mask }], |_, _, _| {})?;
    println!(
        "{} steps, {} trainable parameters, loss {:.3} -> {:.3}",
        report.steps,
        report.trainable_params,
        report.losses[0],
        report.losses.last().unwrap()
    );

    // The adapter is a small PEFT-format directory next to the untouched base model.
    let corrected = TextGenerator::load_with_adapter(&base, Some(&adapter), &Device::Cpu, DType::F32)?;
    let after = corrected.loglikelihood(prompt, answer)?;
    println!("log p(answer | prompt): before {before:.3}, after {after:.3}");
    assert!(report.losses.last().unwrap() < &report.losses[0], "training loss went down");
    assert!(after > before, "the corrected model prefers the right answer");
    assert!(adapter.join("adapter_config.json").exists() && adapter.join("adapter_model.safetensors").exists());
    Ok(())
}
