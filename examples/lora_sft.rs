//! LoRA supervised fine-tuning of a tiny decoder, then reloading the adapter (unmerged) and the
//! merged weights and checking they agree.
//!
//! The base model is a random-weight Qwen2 written to a temporary directory, so nothing is
//! downloaded; point `base` at a real Hugging Face directory (e.g. Qwen2.5-0.5B-Instruct) for a
//! real run. The data path mirrors a real one: render with the chat template, tokenize, keep
//! the loss on the assistant tokens only.
//!
//! `cargo run -p harpia --release --example lora_sft --features train`

use harpia::core::traits::{Model, Tokenizer};
use harpia::data::pack::TokenizedSample;
use harpia::local::{DType, Device, TextGenerator, testing};
use harpia::train::{LoraConfig, LoraTrainConfig, train_lora_sft};

fn main() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    let base = dir.path().join("base");
    let out = dir.path().join("adapter");
    testing::qwen2(&base)?;
    let lm = TextGenerator::load(&base)?;

    // prompt + answer, with the loss only on the answer tokens
    let (prompt, answer) = ("the harpy eagle", " eats sloths in the forest");
    let prompt_ids = lm.tokenizer().encode(prompt, false)?.ids;
    let ids = lm.tokenizer().encode(&format!("{prompt}{answer}"), false)?.ids;
    let loss_mask = (0..ids.len()).map(|i| i >= prompt_ids.len()).collect();
    let samples = vec![TokenizedSample { ids, loss_mask }];

    let before = lm.loglikelihood(prompt, answer)?;
    let cfg = LoraTrainConfig {
        lora: LoraConfig { r: 4, alpha: 8.0, ..Default::default() },
        epochs: 20,
        lr: 2e-2,
        warmup_ratio: 0.0,
        save_merged: true,
        ..Default::default()
    };
    let report = train_lora_sft(&base, &out, &cfg, &samples, |step, total, loss| {
        if step % 5 == 0 || step == total {
            eprintln!("step {step:>3}/{total}  loss {loss:.4}");
        }
    })?;
    println!(
        "{} steps, {} adapted weights, {} trainable params, loss {:.4} -> {:.4} in {:.1}s",
        report.steps,
        report.adapted_weights,
        report.trainable_params,
        report.losses.first().unwrap_or(&f64::NAN),
        report.losses.last().unwrap_or(&f64::NAN),
        report.seconds
    );
    println!("wrote {}", out.display());
    for f in ["adapter_config.json", "adapter_model.safetensors", "train.json", "README.md", "merged"] {
        println!("  {f}: {}", out.join(f).exists());
    }

    // reload: base + unmerged adapter, and the merged full weights
    let adapted = TextGenerator::load_with_adapter(&base, Some(&out), &Device::Cpu, DType::F32)?;
    let merged = TextGenerator::load(&out.join("merged"))?;
    let a = adapted.loglikelihood(prompt, answer)?;
    let m = merged.loglikelihood(prompt, answer)?;
    println!("log p(answer | prompt): base {before:.3}, adapter {a:.3}, merged {m:.3}");
    assert!((a - m).abs() < 1e-3, "adapter and merged weights must agree");
    Ok(())
}
