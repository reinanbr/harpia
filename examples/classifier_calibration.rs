//! Cross-encoder (NLI) fine-tuning with label remapping, then per-label threshold calibration:
//! grid search on validation, thresholds saved and applied *frozen* on a separate test set.
//!
//! The base model is a random-weight XLM-RoBERTa classifier in a temporary directory, with
//! `entailment/neutral/contradiction` labels remapped to the task's own labels so the head is
//! reused. For a real run start from an NLI cross-encoder (e.g. an mDeBERTa/XLM-R NLI model).
//!
//! `cargo run -p harpia --release --example classifier_calibration --features train,eval`

use harpia::core::traits::PairClassifier;
use harpia::eval::calibration::{
    Calibrated, DecisionRule, Objective, TestSet, ValidationSet, default_grid, grid_search,
};
use harpia::local::{SequenceClassifier, testing};
use harpia::train::{ClassifierConfig, Example, train_classifier};

fn examples(premise: &str, n: usize) -> Vec<Example> {
    (0..n)
        .map(|i| {
            let (hypothesis, label) = match i % 3 {
                0 => ("yes", "supports"),
                1 => ("no", "contradicts"),
                _ => ("bird", "neutral"),
            };
            Example { premise: premise.into(), hypothesis: Some(hypothesis.into()), label: label.into() }
        })
        .collect()
}

fn scores(clf: &SequenceClassifier, rows: &[Example]) -> harpia::Result<(Vec<harpia::core::Scores>, Vec<String>)> {
    let pairs: Vec<(&str, &str)> =
        rows.iter().map(|e| (e.premise.as_str(), e.hypothesis.as_deref().unwrap_or(""))).collect();
    Ok((clf.classify_pairs(&pairs)?, rows.iter().map(|e| e.label.clone()).collect()))
}

fn main() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    let base = dir.path().join("base");
    let out = dir.path().join("nli");
    testing::xlm_roberta_classifier(&base, &["entailment", "neutral", "contradiction"])?;

    let cfg = ClassifierConfig {
        labels: vec!["supports".into(), "neutral".into(), "contradicts".into()],
        label_map: [("supports", "entailment"), ("neutral", "neutral"), ("contradicts", "contradiction")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect(),
        epochs: 12,
        batch_size: 6,
        lr: 5e-3,
        warmup_ratio: 0.1,
        seed: 42,
        ..Default::default()
    };
    let train = examples("the harpy eagle eats sloths", 30);
    let val = examples("the harpy eagle hunts monkeys", 9);
    let report = train_classifier(&base, &out, &cfg, &train, &val, |_| {})?;
    println!(
        "head reused: {}, {} epochs, best val macro-F1 {:.3}, {:.1}s",
        report.head_reused,
        report.history.len(),
        report.best_val_macro_f1,
        report.seconds
    );

    // Calibrate on validation, freeze, evaluate once on a different test set. A tiny random model
    // trained on 30 toy rows does not generalize to a new premise, so expect a low test score:
    // the point here is the protocol (tune on validation only, then apply frozen).
    let clf = SequenceClassifier::load(&out)?;
    let (vs, vg) = scores(&clf, &val)?;
    let rule = DecisionRule::nli("supports", "contradicts", "neutral");
    let calibrated = grid_search(&rule, &ValidationSet::new(vs, vg)?, &default_grid(), Objective::MacroF1)?;
    println!("thresholds {:?}, validation macro-F1 {:.3}", calibrated.thresholds.0, calibrated.validation_score);
    let path = out.join("thresholds.json");
    calibrated.save(&path)?;

    let frozen = Calibrated::load(&path)?;
    let (ts, tg) = scores(&clf, &examples("a large eagle nests in the forest", 9))?;
    let test = frozen.evaluate(&TestSet::new(ts, tg)?)?;
    println!("test accuracy {:.3}, macro-F1 {:.3} with frozen thresholds", test.accuracy, test.macro_f1);
    println!("model card: {}", out.join("README.md").display());
    Ok(())
}
