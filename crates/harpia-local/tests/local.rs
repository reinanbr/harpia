//! Local inference on tiny random-weight models (no downloads).

use harpia_core::traits::{
    ChatRequest, Embedder, GenerationParams, Model, PairClassifier, Provider, Reranker, Translator,
};
use harpia_core::types::Message;
use harpia_local::{EncoderEmbedder, MarianTranslator, ModelCache, Search, SequenceClassifier, TextGenerator, testing};
use harpia_structured::constraint::{Choice, Constraint, JsonSchemaConstraint, PrefixStatus, RegexConstraint};
use std::path::Path;

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

fn check_classifier(dir: &Path) {
    let clf =
        SequenceClassifier::load(dir).unwrap().with_label_aliases([("entailment", "supports")]).with_batch_size(2);
    assert_eq!(clf.labels(), ["supports", "neutral", "contradiction"]);
    let pairs = [
        ("the harpy eagle eats sloths in the forest", "the eagle eats monkeys"),
        ("a bird", "yes"),
        ("the forest is large", "the harpy is a large bird in the forest"),
    ];
    let batched = clf.pair_logits(&pairs).unwrap();
    for (i, p) in pairs.iter().enumerate() {
        let single = clf.pair_logits(&[*p]).unwrap();
        assert!(
            close(&batched[i], &single[0], 1e-4),
            "padding must not change results: {:?} vs {:?}",
            batched[i],
            single[0]
        );
    }
    let scores = clf.classify_pairs(&pairs).unwrap();
    for s in &scores {
        let total: f32 = s.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-5);
    }
    assert_eq!(clf.rerank("bird", &["a bird", "the forest"]).unwrap().len(), 2);
}

#[test]
fn xlm_roberta_pair_classifier() {
    let dir = tempfile::tempdir().unwrap();
    testing::xlm_roberta_classifier(dir.path(), &["entailment", "neutral", "contradiction"]).unwrap();
    check_classifier(dir.path());
    let clf = SequenceClassifier::load(dir.path()).unwrap();
    // RoBERTa pair template: <s> A </s></s> B </s>
    let ids = clf.encode_pair("bird", "yes").unwrap();
    assert_eq!((ids[0], ids[2], ids[3], *ids.last().unwrap()), (0, 2, 2, 2));
}

#[test]
fn bert_pair_classifier() {
    let dir = tempfile::tempdir().unwrap();
    testing::bert_classifier(dir.path(), &["entailment", "neutral", "contradiction"]).unwrap();
    check_classifier(dir.path());
}

#[test]
fn embeddings_are_normalized_and_batch_invariant() {
    let dir = tempfile::tempdir().unwrap();
    testing::xlm_roberta_classifier(dir.path(), &["a", "b"]).unwrap();
    let emb = EncoderEmbedder::load(dir.path()).unwrap();
    let texts = ["the harpy eagle eats sloths", "bird", "the forest is large and the eagle is large"];
    let all = emb.embed(&texts).unwrap();
    assert_eq!(emb.dim(), 16);
    for (i, t) in texts.iter().enumerate() {
        let norm: f32 = all[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        assert!(close(&all[i], &emb.embed(&[t]).unwrap()[0], 1e-4));
    }
}

#[test]
fn model_cache_loads_once() {
    let dir = tempfile::tempdir().unwrap();
    testing::bert_classifier(dir.path(), &["x", "y"]).unwrap();
    let a = ModelCache::get_or_load(dir.path(), SequenceClassifier::load).unwrap();
    let b = ModelCache::get_or_load(dir.path(), SequenceClassifier::load).unwrap();
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    ModelCache::evict(dir.path());
}

fn check_decoder(dir: &Path) {
    let lm = TextGenerator::load(dir).unwrap();
    let greedy = GenerationParams { max_new_tokens: 8, ..Default::default() };
    let a = lm.generate("the harpy eagle", &greedy).unwrap();
    let b = lm.generate("the harpy eagle", &greedy).unwrap();
    assert_eq!(a.token_ids, b.token_ids, "greedy decoding is deterministic");
    assert!(a.token_ids.len() <= 8 && a.logprobs.len() == a.token_ids.len());
    assert!(a.logprobs.iter().all(|l| *l <= 0.0));

    let sampled =
        GenerationParams { max_new_tokens: 8, temperature: 1.0, top_p: 0.9, seed: Some(3), ..Default::default() };
    assert_eq!(
        lm.generate("the bird", &sampled).unwrap().token_ids,
        lm.generate("the bird", &sampled).unwrap().token_ids
    );

    // constrained decoding
    let choice = Choice::new(["supports", "contradicts"]);
    let g = lm
        .generate_constrained("label", &GenerationParams { max_new_tokens: 4, ..Default::default() }, &choice)
        .unwrap();
    assert!(["supports", "contradicts"].contains(&g.text.as_str()), "{:?}", g.text);
    let schema = serde_json::json!({"type": "object", "properties": {"label": {"enum": ["supports", "contradicts"]}}, "required": ["label"], "additionalProperties": false});
    let js = JsonSchemaConstraint::new(&schema).unwrap();
    let g = lm
        .generate_constrained("the eagle", &GenerationParams { max_new_tokens: 16, ..Default::default() }, &js)
        .unwrap();
    assert_eq!(js.check(&g.text), PrefixStatus::Complete, "{:?}", g.text);
    let re = RegexConstraint::new("(yes|no)(,(yes|no))*").unwrap();
    let g = lm.generate_constrained("a", &GenerationParams { max_new_tokens: 5, ..Default::default() }, &re).unwrap();
    assert_ne!(re.check(&g.text), PrefixStatus::Invalid, "{:?}", g.text);

    // full-sequence scoring agrees with the KV-cache path
    let one = lm
        .generate_constrained(
            "the harpy",
            &GenerationParams { max_new_tokens: 1, ..Default::default() },
            &Choice::new(["eagle"]),
        )
        .unwrap();
    let ll = lm.loglikelihood("the harpy", "eagle").unwrap();
    assert!((f64::from(one.logprobs[0]) - ll).abs() < 1e-4, "{} vs {ll}", one.logprobs[0]);

    // chat through the model's template, with JSON output
    let req = ChatRequest {
        model: "tiny".into(),
        messages: vec![Message::user("hello")],
        params: GenerationParams { max_new_tokens: 16, ..Default::default() },
        response_schema: Some(schema),
        tools: vec![],
    };
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let resp = rt.block_on(lm.chat(&req)).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp.message.content).unwrap();
    assert!(v["label"].is_string());
    assert!(lm.render_chat(&req.messages).unwrap().ends_with("<|im_start|> assistant "));
}

#[test]
fn qwen2_generation() {
    let dir = tempfile::tempdir().unwrap();
    testing::qwen2(dir.path()).unwrap();
    check_decoder(dir.path());
}

#[test]
fn llama_generation() {
    let dir = tempfile::tempdir().unwrap();
    testing::llama(dir.path()).unwrap();
    check_decoder(dir.path());
}

#[test]
fn marian_greedy_and_beam() {
    let dir = tempfile::tempdir().unwrap();
    testing::marian(dir.path()).unwrap();
    let mut mt = MarianTranslator::load(dir.path()).unwrap();
    mt.max_new_tokens = 6;
    let a = mt.translate(&["the harpy eagle eats sloths"]).unwrap();
    assert_eq!(a, mt.translate(&["the harpy eagle eats sloths"]).unwrap());
    mt.search = Search::Beam { width: 3 };
    let b = mt.translate(&["the harpy eagle eats sloths", "hello world"]).unwrap();
    assert_eq!(b.len(), 2);
}

/// Real `opus-mt-*` checkpoint straight from the Hugging Face cache (`source.spm` +
/// `vocab.json`, no `tokenizer.json`). Run with
/// `HARPIA_MARIAN_DIR=<snapshot dir> cargo test -p harpia-local -- --ignored marian_spm`.
#[test]
#[ignore = "needs a downloaded opus-mt checkpoint"]
fn marian_spm_real_model() {
    let Ok(dir) = std::env::var("HARPIA_MARIAN_DIR") else { return };
    let mut mt = MarianTranslator::load(std::path::Path::new(&dir)).unwrap();
    mt.prefix = Some(">>por<<".into());
    let out =
        mt.translate(&["The harpy eagle is the largest raptor in the Americas.", "I like to read books."]).unwrap();
    eprintln!("{out:?}");
    assert!(out[0].to_lowercase().contains("harpia"), "{out:?}");
    assert!(out[1].to_lowercase().contains("livros"), "{out:?}");

    // Golden file from transformers (`docs/parity.md`): token ids and greedy translations.
    let Ok(golden) = std::env::var("HARPIA_MARIAN_REF") else { return };
    let rows: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(golden).unwrap()).unwrap();
    for row in &rows {
        let text = row["text"].as_str().unwrap();
        let want: Vec<u32> = serde_json::from_value(row["ids"].clone()).unwrap();
        assert_eq!(mt.encode_source(text).unwrap(), want, "ids for {text:?}");
        assert_eq!(mt.translate_one(text).unwrap(), row["translation"].as_str().unwrap(), "translation of {text:?}");
    }
    eprintln!("{} rows match transformers", rows.len());
}
