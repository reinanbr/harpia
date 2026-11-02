//! The training forward passes must match `harpia-local` inference, and gradients must reach
//! every layer (candle-nn's fused kernels have no backward pass, which once made training
//! silently head-only).

use harpia_core::traits::Tokenizer;
use harpia_local::{CausalLm, DType, Device, ModelDir, SequenceClassifier, testing};
use harpia_train::backend::{Mode, TrainStore};
use harpia_train::nn::{DecoderLm, EncoderClassifier};

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn check_encoder(dir: &std::path::Path) {
    let md = ModelDir::open(dir).unwrap();
    let infer = SequenceClassifier::load(dir).unwrap();
    let inputs = vec![
        infer.encode_input("the harpy eagle eats sloths", Some("yes")).unwrap(),
        infer.encode_input("a bird", Some("the forest is large and the eagle is large")).unwrap(),
    ];
    let want = infer.batch_logits(&inputs).unwrap().to_vec2::<f32>().unwrap();
    let store = TrainStore::new(md.tensors(&Device::Cpu).unwrap(), Mode::Full, &Device::Cpu, 0).unwrap();
    let net = EncoderClassifier::new(&md.config, &store, want[0].len()).unwrap();
    let logits = net.logits(&inputs, &Device::Cpu).unwrap();
    let got = logits.to_vec2::<f32>().unwrap();
    assert!(max_diff(&want, &got) < 1e-4, "{want:?} vs {got:?}");
    let grads = logits.sum_all().unwrap().backward().unwrap();
    for (name, v) in store.vars() {
        if name.contains("word_embeddings") || name.contains("layer.0.attention.self.query.weight") {
            assert!(grads.get(v.as_tensor()).is_some(), "no gradient for {name}");
        }
    }
}

#[test]
fn xlm_roberta_training_forward_matches_inference() {
    let dir = tempfile::tempdir().unwrap();
    testing::xlm_roberta_classifier(dir.path(), &["a", "b", "c"]).unwrap();
    check_encoder(dir.path());
}

#[test]
fn bert_training_forward_matches_inference() {
    let dir = tempfile::tempdir().unwrap();
    testing::bert_classifier(dir.path(), &["a", "b"]).unwrap();
    check_encoder(dir.path());
}

fn check_decoder(dir: &std::path::Path) {
    let md = ModelDir::open(dir).unwrap();
    let mut infer =
        CausalLm::from_var_builder(&md, md.var_builder(DType::F32, &Device::Cpu).unwrap(), DType::F32, &Device::Cpu)
            .unwrap();
    let tok = md.tokenizer().unwrap();
    let ids = tok.encode("the harpy eagle eats sloths in the forest", false).unwrap().ids;
    let want = infer.logits_all(&ids).unwrap().to_vec2::<f32>().unwrap();
    let store = TrainStore::new(md.tensors(&Device::Cpu).unwrap(), Mode::Full, &Device::Cpu, 0).unwrap();
    let net = DecoderLm::new(&md.config, &store).unwrap();
    let logits = net.logits(&ids, &Device::Cpu).unwrap();
    assert!(max_diff(&want, &logits.to_vec2::<f32>().unwrap()) < 1e-4);
    let grads = logits.sum_all().unwrap().backward().unwrap();
    for (name, v) in store.vars() {
        if name.contains("embed_tokens")
            || name.contains("layers.0.self_attn.q_proj.weight")
            || name.contains("layers.0.input_layernorm")
        {
            assert!(grads.get(v.as_tensor()).is_some(), "no gradient for {name}");
        }
    }
}

#[test]
fn qwen2_training_forward_matches_inference() {
    let dir = tempfile::tempdir().unwrap();
    testing::qwen2(dir.path()).unwrap();
    check_decoder(dir.path());
}

#[test]
fn llama_training_forward_matches_inference() {
    let dir = tempfile::tempdir().unwrap();
    testing::llama(dir.path()).unwrap();
    check_decoder(dir.path());
}

#[test]
fn dropout_is_seeded_and_off_in_eval() {
    let dir = tempfile::tempdir().unwrap();
    testing::xlm_roberta_classifier(dir.path(), &["a", "b"]).unwrap();
    let md = ModelDir::open(dir.path()).unwrap();
    let store = TrainStore::new(md.tensors(&Device::Cpu).unwrap(), Mode::Frozen, &Device::Cpu, 0).unwrap();
    let ids = vec![vec![0u32, 8, 9, 10, 2]];
    let run = |seed: u64, train: bool| {
        let net = EncoderClassifier::new(&md.config, &store, 2).unwrap().with_dropout(0.5, 0.5, 0.5, seed);
        net.set_training(train);
        net.logits(&ids, &Device::Cpu).unwrap().to_vec2::<f32>().unwrap()
    };
    assert_eq!(run(1, false), run(2, false), "eval ignores dropout");
    assert_eq!(run(7, true), run(7, true), "same seed, same masks");
    assert_ne!(run(7, true), run(8, true));
    assert_ne!(run(7, true), run(7, false));
}
