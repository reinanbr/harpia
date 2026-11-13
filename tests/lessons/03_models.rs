//! Lesson 3: reading and running models.
//!
//! A Hugging Face model is just a directory: `config.json` (the architecture),
//! `*.safetensors` (the weights), `tokenizer.json` and usually `tokenizer_config.json` (special
//! tokens and the chat template). `harpia::local` reads those directories directly with
//! [candle](https://github.com/huggingface/candle), on the CPU, in pure Rust.
//!
//! The four model families covered here:
//!
//! | family | Harpia type | used for |
//! |---|---|---|
//! | decoder LLM (Qwen2, Llama) | `TextGenerator` | generation, chat, log-probabilities |
//! | encoder classifier (BERT, XLM-R) | `SequenceClassifier` | labels, NLI, quality filters |
//! | encoder embedder | `EncoderEmbedder` | vectors for RAG and search |
//! | seq2seq (MarianMT) | `MarianTranslator` | translation |
//!
//! `harpia::local::testing` writes *tiny random-weight* versions of each into a temp
//! directory, so the outputs below are gibberish, but every call is the real one: point the
//! same code at `Qwen2.5-0.5B-Instruct` or an NLI cross-encoder and it does real work.
//!
//! Run: `cargo test -p harpia-learn --test 03_models -- --nocapture`

use harpia::core::traits::{Embedder, GenerationParams, Model, PairClassifier, Translator};
use harpia::local::{EncoderEmbedder, MarianTranslator, ModelDir, Search, SequenceClassifier, TextGenerator, testing};
use harpia::structured::constraint::Choice;

/// Before loading anything, look inside the directory.
#[test]
fn inspect_a_model_directory() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    testing::xlm_roberta_classifier(dir.path(), &["supports", "neutral", "contradicts"])?;

    let md = ModelDir::open(dir.path())?;
    println!("architecture: {}", md.model_type());
    println!("weights: {:?}", md.weight_files()?.iter().map(|p| p.file_name().unwrap()).collect::<Vec<_>>());
    println!("labels: {:?}", md.labels(3));
    assert_eq!(md.model_type(), "xlm-roberta");
    assert_eq!(md.num_labels(), Some(3));

    // The raw tensors are there too, by name, if you want to look at a layer yourself.
    let tensors = md.tensors(&harpia::local::Device::Cpu)?;
    let emb = &tensors["roberta.embeddings.word_embeddings.weight"];
    println!("word embeddings: {:?}", emb.dims());
    Ok(())
}

/// Decoder LLMs: sample text, read token log-probabilities, score continuations.
#[test]
fn decoder_generation() -> harpia::Result<()> {
    for (name, write) in
        [("qwen2", testing::qwen2 as fn(&std::path::Path) -> harpia::Result<()>), ("llama", testing::llama)]
    {
        let dir = tempfile::tempdir()?;
        write(dir.path())?;
        let lm = TextGenerator::load(dir.path())?;

        // Same seed, same text: sampling is reproducible.
        let params =
            GenerationParams { max_new_tokens: 6, temperature: 0.7, top_p: 0.9, seed: Some(1), ..Default::default() };
        let a = lm.generate("the harpy eagle", &params)?;
        let b = lm.generate("the harpy eagle", &params)?;
        println!("[{name}] {:?}  finish={:?}  logprobs={:?}", a.text, a.finish_reason, a.logprobs);
        assert_eq!(a.text, b.text);
        assert_eq!(a.logprobs.len(), a.token_ids.len(), "one log-probability per generated token");
        assert!(a.logprobs.iter().all(|&l| l <= 0.0));

        // log p(continuation | context): the building block of multiple-choice evals and perplexity.
        let ll = lm.loglikelihood("the harpy eagle", " eats sloths")?;
        println!("[{name}] log p(' eats sloths' | 'the harpy eagle') = {ll:.3}");
        assert!(ll < 0.0);
    }
    Ok(())
}

/// Constrained decoding: the model can only produce one of the allowed answers.
#[test]
fn constrained_generation() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    testing::qwen2(dir.path())?;
    let lm = TextGenerator::load(dir.path())?;
    let yes_no = Choice::new(["yes", "no"]);
    let g = lm.generate_constrained(
        "is a harpy a bird",
        &GenerationParams { max_new_tokens: 4, ..Default::default() },
        &yes_no,
    )?;
    println!("constrained answer: {:?}", g.text);
    assert!(g.text == "yes" || g.text == "no");
    Ok(())
}

/// Encoder classifiers return a probability per label (`Scores`); pairs are (premise, hypothesis).
#[test]
fn classifier_and_cross_encoder() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    testing::bert_classifier(dir.path(), &["entailment", "neutral", "contradiction"])?;
    let clf = SequenceClassifier::load(dir.path())?;
    println!("labels: {:?}", clf.labels());

    let scores = clf.classify_pairs(&[("the harpy eagle eats sloths", "a bird eats sloths")])?;
    let s = &scores[0];
    for (label, p) in s.iter() {
        println!("{label:>14}: {p:.3}");
    }
    let total: f32 = s.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-4, "softmax probabilities sum to 1");
    let (best, _) = s.argmax().unwrap();
    assert!(["entailment", "neutral", "contradiction"].contains(&best));
    Ok(())
}

/// Embeddings: one normalized vector per text; cosine similarity is then a dot product.
#[test]
fn embeddings() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    testing::xlm_roberta_classifier(dir.path(), &["a", "b"])?;
    let emb = EncoderEmbedder::load(dir.path())?;
    let v = emb.embed(&["the harpy eagle", "the harpy eagle", "hello world"])?;
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    println!("dim {}, cos(same) {:.3}, cos(different) {:.3}", emb.dim(), dot(&v[0], &v[1]), dot(&v[0], &v[2]));
    assert!((dot(&v[0], &v[1]) - 1.0).abs() < 1e-4);
    Ok(())
}

/// MarianMT translation, greedy or beam search.
#[test]
fn translation() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    testing::marian(dir.path())?;
    let mut mt = MarianTranslator::load(dir.path())?;
    mt.max_new_tokens = 5;
    let greedy = mt.translate(&["the harpy eagle eats sloths"])?;
    mt.search = Search::Beam { width: 3 };
    let beam = mt.translate(&["the harpy eagle eats sloths"])?;
    println!("greedy {greedy:?}\nbeam   {beam:?}");
    assert_eq!(greedy.len(), 1);
    Ok(())
}
