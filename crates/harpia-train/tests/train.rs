//! Training on tiny random-weight models (no downloads).

use harpia_core::traits::{Model, PairClassifier, Tokenizer};
use harpia_data::pack::TokenizedSample;
use harpia_local::{DType, Device, SequenceClassifier, TextGenerator, testing};
use harpia_train::{
    ClassifierConfig, Example, LoraConfig, LoraTrainConfig, PreferenceIds, train_classifier, train_lora_dpo,
    train_lora_sft,
};

fn toy_examples(n: usize) -> Vec<Example> {
    (0..n)
        .map(|i| {
            let (h, l) = match i % 3 {
                0 => ("yes", "sustenta"),
                1 => ("no", "contradiz"),
                _ => ("bird", "neutro"),
            };
            Example { premise: "the harpy eagle eats sloths".into(), hypothesis: Some(h.into()), label: l.into() }
        })
        .collect()
}

#[test]
fn classifier_fine_tuning_with_head_remap() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let out = dir.path().join("out");
    testing::xlm_roberta_classifier(&base, &["entailment", "neutral", "contradiction"]).unwrap();
    let cfg = ClassifierConfig {
        labels: vec!["sustenta".into(), "neutro".into(), "contradiz".into()],
        label_map: [("sustenta", "entailment"), ("neutro", "neutral"), ("contradiz", "contradiction")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect(),
        epochs: 12,
        batch_size: 6,
        lr: 5e-3,
        warmup_ratio: 0.1,
        ..Default::default()
    };
    let train = toy_examples(30);
    let val = toy_examples(9);
    let mut steps = 0;
    let report = train_classifier(&base, &out, &cfg, &train, &val, |_| steps += 1).unwrap();
    assert!(report.head_reused);
    assert_eq!(steps, 12 * 5);
    let first = report.history.first().unwrap().train_loss;
    let last = report.history.last().unwrap().train_loss;
    assert!(last < first, "loss should decrease: {first} -> {last}");
    assert!(
        report.best_val_macro_f1 >= 0.99,
        "{:?}",
        report.history.iter().map(|h| h.val_macro_f1).collect::<Vec<_>>()
    );
    assert!(out.join("train.json").exists() && out.join("tokenizer.json").exists());

    // the checkpoint reloads with the new labels and reproduces the best validation score
    let clf = SequenceClassifier::load_with(&out, &Device::Cpu, DType::F32, 192).unwrap();
    assert_eq!(clf.labels(), ["sustenta", "neutro", "contradiz"]);
    let pairs: Vec<(&str, &str)> = val.iter().map(|e| (e.premise.as_str(), e.hypothesis.as_deref().unwrap())).collect();
    let pred: Vec<String> =
        clf.classify_pairs(&pairs).unwrap().iter().map(|s| s.argmax().unwrap().0.to_string()).collect();
    let correct = pred.iter().zip(&val).filter(|(p, e)| **p == e.label).count();
    assert_eq!(correct, val.len());
}

#[test]
fn classifier_with_new_labels_reinitializes_the_head() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    testing::bert_classifier(&base, &["a", "b"]).unwrap();
    let cfg = ClassifierConfig {
        labels: vec!["sustenta".into(), "neutro".into(), "contradiz".into()],
        epochs: 1,
        lr: 1e-3,
        ..Default::default()
    };
    let r = train_classifier(&base, &dir.path().join("out"), &cfg, &toy_examples(6), &toy_examples(3), |_| {}).unwrap();
    assert!(!r.head_reused);
    assert_eq!(SequenceClassifier::load(&dir.path().join("out")).unwrap().labels().len(), 3);
}

fn sft_sample(lm: &TextGenerator) -> TokenizedSample {
    let ids = lm.tokenizer().encode("the harpy eagle eats sloths in the forest", false).unwrap().ids;
    let mask = (0..ids.len()).map(|i| i >= 3).collect();
    TokenizedSample { ids, loss_mask: mask }
}

fn check_lora_sft(base: &std::path::Path, steps: usize) {
    let out = base.parent().unwrap().join("adapter");
    let lm = TextGenerator::load(base).unwrap();
    let sample = sft_sample(&lm);
    let before = lm.loglikelihood("the harpy eagle", "eats sloths in the forest").unwrap();
    let cfg = LoraTrainConfig {
        lora: LoraConfig { r: 4, alpha: 8.0, ..Default::default() },
        epochs: steps,
        lr: 2e-2,
        warmup_ratio: 0.0,
        save_merged: true,
        ..Default::default()
    };
    let report = train_lora_sft(base, &out, &cfg, &[sample], |_, _, _| {}).unwrap();
    assert_eq!(report.steps, steps);
    assert!(report.adapted_weights >= 2 * 7, "every layer's projections are adapted");
    assert!(report.losses.last().unwrap() < report.losses.first().unwrap(), "{:?}", report.losses);
    assert!(out.join("adapter_config.json").exists() && out.join("adapter_model.safetensors").exists());

    let adapted = TextGenerator::load_with_adapter(base, Some(&out), &Device::Cpu, DType::F32).unwrap();
    let merged = TextGenerator::load(&out.join("merged")).unwrap();
    let a = adapted.loglikelihood("the harpy eagle", "eats sloths in the forest").unwrap();
    let m = merged.loglikelihood("the harpy eagle", "eats sloths in the forest").unwrap();
    assert!((a - m).abs() < 1e-3, "adapter vs merged: {a} vs {m}");
    assert!(a > before, "the trained continuation became more likely: {before} -> {a}");
}

#[test]
fn lora_sft_qwen2() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    testing::qwen2(&base).unwrap();
    check_lora_sft(&base, 25);
}

#[test]
fn lora_sft_llama() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    testing::llama(&base).unwrap();
    check_lora_sft(&base, 10);
}

#[test]
fn lora_dpo_prefers_chosen() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    testing::qwen2(&base).unwrap();
    let lm = TextGenerator::load(&base).unwrap();
    let tok = |s: &str| lm.tokenizer().encode(s, false).unwrap().ids;
    let pair = PreferenceIds { prompt: tok("is the bird large"), chosen: tok("yes"), rejected: tok("no") };
    let margin = |m: &TextGenerator| {
        m.loglikelihood("is the bird large", "yes").unwrap() - m.loglikelihood("is the bird large", "no").unwrap()
    };
    let before = margin(&lm);
    let out = dir.path().join("dpo");
    let cfg = LoraTrainConfig {
        lora: LoraConfig { r: 4, alpha: 8.0, ..Default::default() },
        epochs: 15,
        lr: 2e-2,
        warmup_ratio: 0.0,
        beta: 0.5,
        ..Default::default()
    };
    let r = train_lora_dpo(&base, &out, &cfg, &[pair], |_, _, _| {}).unwrap();
    assert_eq!(r.objective, "dpo");
    assert!(r.losses.last().unwrap() < r.losses.first().unwrap());
    let after = margin(&TextGenerator::load_with_adapter(&base, Some(&out), &Device::Cpu, DType::F32).unwrap());
    assert!(after > before, "margin {before} -> {after}");
}

#[test]
fn resuming_reproduces_an_uninterrupted_run() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    testing::xlm_roberta_classifier(&base, &["entailment", "neutral", "contradiction"]).unwrap();
    let cfg = |stop_after: Option<usize>, resume: bool| ClassifierConfig {
        labels: vec!["sustenta".into(), "neutro".into(), "contradiz".into()],
        epochs: 4,
        stop_after,
        batch_size: 4,
        lr: 3e-3,
        dropout: Some(0.2),
        resume,
        ..Default::default()
    };
    let (train, val) = (toy_examples(12), toy_examples(6));
    let a = dir.path().join("straight");
    let ra = train_classifier(&base, &a, &cfg(None, false), &train, &val, |_| {}).unwrap();
    let b = dir.path().join("resumed");
    train_classifier(&base, &b, &cfg(Some(2), false), &train, &val, |_| {}).unwrap();
    let rb = train_classifier(&base, &b, &cfg(None, true), &train, &val, |_| {}).unwrap();
    assert_eq!(ra.history, rb.history);
    let wa = candle_core::safetensors::load(a.join("checkpoint/trainable.safetensors"), &Device::Cpu).unwrap();
    let wb = candle_core::safetensors::load(b.join("checkpoint/trainable.safetensors"), &Device::Cpu).unwrap();
    for (k, t) in &wa {
        let d = (t - &wb[k]).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(d == 0.0, "{k} differs by {d}");
    }
}

#[test]
fn lora_resume_is_exact() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    testing::qwen2(&base).unwrap();
    let lm = TextGenerator::load(&base).unwrap();
    let samples = vec![sft_sample(&lm), sft_sample(&lm)];
    let cfg = |stop_after: Option<usize>, resume: bool| LoraTrainConfig {
        lora: LoraConfig { r: 4, alpha: 8.0, dropout: 0.1, ..Default::default() },
        epochs: 4,
        lr: 1e-2,
        stop_after,
        resume,
        ..Default::default()
    };
    let a = train_lora_sft(&base, &dir.path().join("a"), &cfg(None, false), &samples, |_, _, _| {}).unwrap();
    let out_b = dir.path().join("b");
    train_lora_sft(&base, &out_b, &cfg(Some(2), false), &samples, |_, _, _| {}).unwrap();
    let b = train_lora_sft(&base, &out_b, &cfg(None, true), &samples, |_, _, _| {}).unwrap();
    assert_eq!(a.losses, b.losses);
}
