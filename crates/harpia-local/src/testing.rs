//! Tiny random-weight models written to disk in Hugging Face layout, so tests (here and in
//! downstream crates) never download anything. The weights are random: outputs are meaningless,
//! but shapes, plumbing, batching and determinism can be checked.
//!
//! The weights are seeded: every builder writes byte-identical files on every run and platform,
//! so tests and examples built on these models are reproducible.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap, linear};
use candle_transformers::models::{bert, llama, marian, qwen2, xlm_roberta};
use harpia_core::{HarpiaError, Result};
use serde_json::{Value, json};
use std::path::Path;

/// Words of the tiny vocabularies (after the special tokens).
pub const WORDS: &[&str] = &[
    "the",
    "harpy",
    "eagle",
    "eats",
    "sloths",
    "monkeys",
    "in",
    "forest",
    "a",
    "bird",
    "is",
    "large",
    "yes",
    "no",
    "{",
    "}",
    "\"",
    ":",
    "label",
    "supports",
    "contradicts",
    ",",
    "hello",
    "world",
    "o",
    "gavião",
    "come",
    "preguiças",
];

fn write(dir: &Path, name: &str, v: &Value) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| HarpiaError::io(dir, e))?;
    let p = dir.join(name);
    std::fs::write(&p, serde_json::to_string_pretty(v)?).map_err(|e| HarpiaError::io(&p, e))
}

fn ce(e: candle_core::Error) -> HarpiaError {
    HarpiaError::Model { model: "tiny".into(), message: e.to_string() }
}

/// Word-level `tokenizer.json` with `specials` first, then [`WORDS`], and an optional
/// post-processor (`Value::Null` for none).
pub fn tokenizer_json(specials: &[&str], unk: &str, post_processor: Value) -> Value {
    let mut vocab = serde_json::Map::new();
    for (i, t) in specials.iter().chain(WORDS.iter()).enumerate() {
        vocab.insert((*t).to_string(), json!(i));
    }
    let added: Vec<Value> = specials
        .iter()
        .enumerate()
        .map(|(i, t)| json!({"id": i, "content": t, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
        .collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": post_processor,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": unk}
    })
}

/// Vocabulary size of tokenizers built with `specials`.
pub fn vocab_size(specials: &[&str]) -> usize {
    specials.len() + WORDS.len()
}

/// A tiny XLM-RoBERTa sequence classifier (`<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3).
pub fn xlm_roberta_classifier(dir: &Path, labels: &[&str]) -> Result<()> {
    let specials = ["<s>", "<pad>", "</s>", "<unk>"];
    let id2label: serde_json::Map<String, Value> =
        labels.iter().enumerate().map(|(i, l)| (i.to_string(), json!(l))).collect();
    let cfg = json!({
        "model_type": "xlm-roberta", "architectures": ["XLMRobertaForSequenceClassification"],
        "hidden_size": 16, "layer_norm_eps": 1e-5, "attention_probs_dropout_prob": 0.0, "hidden_dropout_prob": 0.0,
        "num_attention_heads": 2, "position_embedding_type": "absolute", "intermediate_size": 32, "hidden_act": "gelu",
        "num_hidden_layers": 2, "vocab_size": vocab_size(&specials), "max_position_embeddings": 64, "type_vocab_size": 1,
        "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2, "id2label": id2label,
    });
    let c: xlm_roberta::Config = serde_json::from_value(cfg.clone())?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    xlm_roberta::XLMRobertaModel::new(&c, vb.pp("roberta")).map_err(ce)?;
    linear(16, 16, vb.pp("classifier.dense")).map_err(ce)?;
    linear(16, labels.len(), vb.pp("classifier.out_proj")).map_err(ce)?;
    write(dir, "config.json", &cfg)?;
    write(
        dir,
        "tokenizer.json",
        &tokenizer_json(
            &specials,
            "<unk>",
            json!({"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0], "trim_offsets": true, "add_prefix_space": false}),
        ),
    )?;
    save_seeded(&vm, &dir.join("model.safetensors"))
}

/// A tiny BERT sequence classifier (`[PAD]`=0, `[UNK]`=1, `[CLS]`=2, `[SEP]`=3).
pub fn bert_classifier(dir: &Path, labels: &[&str]) -> Result<()> {
    let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"];
    let id2label: serde_json::Map<String, Value> =
        labels.iter().enumerate().map(|(i, l)| (i.to_string(), json!(l))).collect();
    let cfg = json!({
        "model_type": "bert", "vocab_size": vocab_size(&specials), "hidden_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "intermediate_size": 32, "hidden_act": "gelu", "hidden_dropout_prob": 0.0,
        "max_position_embeddings": 64, "type_vocab_size": 2, "initializer_range": 0.02, "layer_norm_eps": 1e-12,
        "pad_token_id": 0, "classifier_dropout": null, "id2label": id2label,
    });
    let c: bert::Config = serde_json::from_value(cfg.clone())?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    bert::BertModel::load(vb.pp("bert"), &c).map_err(ce)?;
    linear(16, 16, vb.pp("bert.pooler.dense")).map_err(ce)?;
    linear(16, labels.len(), vb.pp("classifier")).map_err(ce)?;
    write(dir, "config.json", &cfg)?;
    write(
        dir,
        "tokenizer.json",
        &tokenizer_json(
            &specials,
            "[UNK]",
            json!({"type": "BertProcessing", "sep": ["[SEP]", 3], "cls": ["[CLS]", 2]}),
        ),
    )?;
    save_seeded(&vm, &dir.join("model.safetensors"))
}

/// Specials of the tiny decoders: `<|endoftext|>`=0, `<|im_start|>`=1, `<|im_end|>`=2,
/// `<unk>`=3, plus `system`, `user`, `assistant` role words.
pub const DECODER_SPECIALS: &[&str] =
    &["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>", "system", "user", "assistant"];

fn decoder_files(dir: &Path) -> Result<()> {
    // decoding concatenates pieces, like byte-level BPE tokenizers
    let mut tok = tokenizer_json(DECODER_SPECIALS, "<unk>", Value::Null);
    tok["decoder"] = json!({"type": "Fuse"});
    write(dir, "tokenizer.json", &tok)?;
    write(
        dir,
        "tokenizer_config.json",
        &json!({
            "eos_token": "<|im_end|>", "bos_token": null,
            "chat_template": "{% for message in messages %}{{ '<|im_start|> ' + message['role'] + ' ' + message['content'] + ' <|im_end|> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|> assistant ' }}{% endif %}"
        }),
    )
}

/// A tiny Qwen2 causal LM.
pub fn qwen2(dir: &Path) -> Result<()> {
    let cfg = json!({
        "model_type": "qwen2", "vocab_size": vocab_size(DECODER_SPECIALS), "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 128,
        "sliding_window": 128, "max_window_layers": 2, "tie_word_embeddings": false, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6, "use_sliding_window": false, "hidden_act": "silu", "eos_token_id": 2, "bos_token_id": 0,
    });
    let c: qwen2::Config = serde_json::from_value(cfg.clone())?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    qwen2::ModelForCausalLM::new(&c, vb).map_err(ce)?;
    // ModelForCausalLM only creates lm_head when present; create it explicitly
    candle_nn::linear_no_bias(16, c.vocab_size, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu).pp("lm_head"))
        .map_err(ce)?;
    write(dir, "config.json", &cfg)?;
    decoder_files(dir)?;
    save_seeded(&vm, &dir.join("model.safetensors"))
}

/// A tiny Llama causal LM.
pub fn llama(dir: &Path) -> Result<()> {
    let cfg = json!({
        "model_type": "llama", "vocab_size": vocab_size(DECODER_SPECIALS), "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "bos_token_id": 0, "eos_token_id": 2, "tie_word_embeddings": false,
    });
    let c: llama::LlamaConfig = serde_json::from_value(cfg.clone())?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    llama::Llama::load(vb, &c.into_config(false)).map_err(ce)?;
    write(dir, "config.json", &cfg)?;
    decoder_files(dir)?;
    save_seeded(&vm, &dir.join("model.safetensors"))
}

/// A tiny MarianMT model (`</s>`=0, `<unk>`=1, `<pad>`=2; decoder start = pad, as in Marian).
pub fn marian(dir: &Path) -> Result<()> {
    let specials = ["</s>", "<unk>", "<pad>"];
    let v = vocab_size(&specials);
    let cfg = json!({
        "model_type": "marian", "vocab_size": v, "decoder_vocab_size": v, "max_position_embeddings": 64,
        "encoder_layers": 1, "encoder_ffn_dim": 32, "encoder_attention_heads": 2, "decoder_layers": 1,
        "decoder_ffn_dim": 32, "decoder_attention_heads": 2, "use_cache": true, "is_encoder_decoder": true,
        "activation_function": "swish", "d_model": 16, "decoder_start_token_id": 2, "scale_embedding": true,
        "pad_token_id": 2, "eos_token_id": 0, "forced_eos_token_id": 0, "share_encoder_decoder_embeddings": true,
    });
    let c: marian::Config = serde_json::from_value(cfg.clone())?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    marian::MTModel::new(&c, vb).map_err(ce)?;
    write(dir, "config.json", &cfg)?;
    write(
        dir,
        "tokenizer.json",
        &tokenizer_json(
            &specials,
            "<unk>",
            json!({"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}], "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}], "special_tokens": {"</s>": {"id": "</s>", "ids": [0], "tokens": ["</s>"]}}}),
        ),
    )?;
    save_seeded(&vm, &dir.join("model.safetensors"))
}

/// Seed of the tiny models' weights.
const WEIGHT_SEED: u64 = 0x4a52_5059_4541_474c;

/// SplitMix64 with Box-Muller normals: small, portable and stable across versions.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

/// Redraw every non-constant variable from a generator seeded by its name, then save. candle's
/// own initializers are not seedable on CPU. Constant inits (norm weights, zero biases) are kept;
/// the rest are drawn from N(0, std) with `std` fixed by name and shape only: 1 for embedding
/// tables (as `candle_nn::embedding`), `1/sqrt(fan_in)` for weight matrices, and
/// `0.5/sqrt(len)` for vectors (biases).
fn save_seeded(vm: &VarMap, path: &Path) -> Result<()> {
    {
        let data = vm.data().lock().map_err(|_| HarpiaError::invalid("tiny model: poisoned VarMap"))?;
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        for name in names {
            let var = &data[name];
            let values: Vec<f32> = var.flatten_all().and_then(|t| t.to_vec1()).map_err(ce)?;
            if values.windows(2).all(|w| w[0] == w[1]) {
                continue;
            }
            let dims = var.dims();
            let std = if name.contains("embed") || name.ends_with("shared.weight") {
                1.0
            } else if dims.len() >= 2 {
                1.0 / (dims[dims.len() - 1] as f64).sqrt()
            } else {
                0.5 / (values.len() as f64).sqrt()
            };
            let seed = name.bytes().fold(WEIGHT_SEED, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
            let mut rng = SplitMix(seed);
            let fresh: Vec<f32> = (0..values.len()).map(|_| (std * rng.normal()) as f32).collect();
            var.set(&Tensor::from_vec(fresh, var.shape(), &Device::Cpu).map_err(ce)?).map_err(ce)?;
        }
    }
    vm.save(path).map_err(ce)
}
