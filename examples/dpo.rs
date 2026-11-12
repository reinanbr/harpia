//! Direct preference optimization (experimental) with a LoRA adapter on a toy preference set,
//! measuring the chosen-vs-rejected log-likelihood margin before and after.
//!
//! The base model is a random-weight Qwen2 in a temporary directory (no downloads).
//!
//! `cargo run -p harpia --release --example dpo --features train`

use harpia::core::traits::{Model, Tokenizer};
use harpia::local::{DType, Device, TextGenerator, testing};
use harpia::train::{LoraConfig, LoraTrainConfig, PreferenceIds, train_lora_dpo};

const PAIRS: [(&str, &str, &str); 2] =
    [("is the harpy eagle large", " yes", " no"), ("does the harpy eagle eat sloths", " yes", " never")];

fn margin(lm: &TextGenerator) -> harpia::Result<f64> {
    let mut total = 0.0;
    for (prompt, chosen, rejected) in PAIRS {
        total += lm.loglikelihood(prompt, chosen)? - lm.loglikelihood(prompt, rejected)?;
    }
    Ok(total / PAIRS.len() as f64)
}

fn main() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    let base = dir.path().join("base");
    let out = dir.path().join("dpo");
    testing::qwen2(&base)?;
    let lm = TextGenerator::load(&base)?;
    let tok = |s: &str| lm.tokenizer().encode(s, false).map(|e| e.ids);
    let pairs = PAIRS
        .iter()
        .map(|(p, c, r)| Ok(PreferenceIds { prompt: tok(p)?, chosen: tok(c)?, rejected: tok(r)? }))
        .collect::<harpia::Result<Vec<_>>>()?;

    let before = margin(&lm)?;
    let cfg = LoraTrainConfig {
        lora: LoraConfig { r: 4, alpha: 8.0, ..Default::default() },
        epochs: 15,
        lr: 2e-2,
        warmup_ratio: 0.0,
        beta: 0.5,
        ..Default::default()
    };
    let report = train_lora_dpo(&base, &out, &cfg, &pairs, |_, _, _| {})?;
    let after = margin(&TextGenerator::load_with_adapter(&base, Some(&out), &Device::Cpu, DType::F32)?)?;
    println!(
        "{}: {} steps, loss {:.4} -> {:.4}",
        report.objective,
        report.steps,
        report.losses.first().unwrap_or(&f64::NAN),
        report.losses.last().unwrap_or(&f64::NAN)
    );
    println!("mean log p(chosen) - log p(rejected): {before:.3} -> {after:.3}");
    Ok(())
}
