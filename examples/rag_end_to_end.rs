//! RAG end to end: chunk documents → embed → hybrid retrieval (dense + BM25, reciprocal rank
//! fusion) → rerank → assemble a cited context within a token budget → answer → score the
//! retrieval and the answer's faithfulness.
//!
//! To stay self-contained the embedder, reranker, NLI classifier and LLM are small stand-ins
//! implementing the same traits as the real ones. In practice use
//! `harpia::local::EncoderEmbedder` (e.g. e5 / bge), `harpia::local::SequenceClassifier` as a
//! cross-encoder reranker and NLI model, and any `Provider` for the answer.
//!
//! `cargo run -p harpia --example rag_end_to_end --features data,eval`

use harpia::core::traits::{Embedder, PairClassifier, Provider, Reranker};
use harpia::core::{Document, Label, Message, Scores};
use harpia::data::chunk::SentenceChunker;
use harpia::data::rag::{Fusion, MetadataFilter, RagConfig, RagPipeline};
use harpia::eval::calibration::{Calibrated, DecisionRule, Thresholds};
use harpia::eval::rag::{RagExample, faithfulness, retrieval_report};
use harpia::eval::testing::FnProvider;
use harpia::text::Lang;
use serde_json::json;

/// Bag of hashed words (FNV-1a): texts sharing words get similar vectors.
struct HashedWords;
impl Embedder for HashedWords {
    fn dim(&self) -> usize {
        128
    }
    fn embed(&self, texts: &[&str]) -> harpia::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0; 128];
                for w in t.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
                    let h = w
                        .bytes()
                        .fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
                    v[(h % 128) as usize] += 1.0;
                }
                v
            })
            .collect())
    }
}

/// Reranker: fraction of the query's words present in the passage.
struct WordOverlap;
impl Reranker for WordOverlap {
    fn rerank(&self, query: &str, docs: &[&str]) -> harpia::Result<Vec<f32>> {
        let q: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 3)
            .map(String::from)
            .collect();
        Ok(docs
            .iter()
            .map(|d| {
                let d = d.to_lowercase();
                q.iter().filter(|w| d.contains(w.as_str())).count() as f32 / q.len().max(1) as f32
            })
            .collect())
    }
}

/// NLI stand-in: entailment when the hypothesis' content words all appear in the premise.
struct Entails;
impl PairClassifier for Entails {
    fn labels(&self) -> &[Label] {
        &[]
    }
    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> harpia::Result<Vec<Scores>> {
        let rerank = WordOverlap;
        pairs
            .iter()
            .map(|(premise, hyp)| {
                let e = rerank.rerank(hyp, &[premise])?[0];
                Ok(Scores::new(vec![("entailment".into(), e), ("neutral".into(), 1.0 - e)]))
            })
            .collect()
    }
}

fn docs() -> Vec<Document> {
    let mut eagle = Document::new(
        "harpy",
        "The harpy eagle lives in tropical lowland rainforests from Mexico to northern Argentina. \
         It hunts sloths and monkeys in the canopy. Its talons are as large as a grizzly bear's claws. \
         Females weigh up to nine kilograms.",
    );
    eagle.source = Some("field-guide.md".into());
    eagle.license = Some("CC-BY-4.0".into());
    eagle.metadata.insert("topic".into(), "birds".into());
    let mut sloth = Document::new(
        "sloth",
        "Sloths move slowly through the canopy and sleep most of the day. Algae grow in their fur.",
    );
    sloth.metadata.insert("topic".into(), "mammals".into());
    let mut piranha = Document::new("piranha", "Piranhas live in South American rivers and hunt in groups.");
    piranha.metadata.insert("topic".into(), "fish".into());
    vec![eagle, sloth, piranha]
}

fn main() -> harpia::Result<()> {
    let embedder = HashedWords;
    let reranker = WordOverlap;
    let chunker = SentenceChunker { max_chars: 120, overlap: 0, lang: Lang::En.data() };
    let config = RagConfig { fusion: Fusion::Rrf { k: 60.0 }, top_k: 3, ..Default::default() };
    let mut rag = RagPipeline::new(Box::new(chunker), &embedder, config).with_reranker(&reranker);
    let n = rag.add_documents(&docs())?;
    println!("indexed {n} chunks");

    let question = "Where does the harpy eagle live and what does it hunt?";
    for hit in rag.retrieve(question, None)? {
        println!(
            "  {:.3} (dense {:?}, bm25 {:?}, rerank {:?}) {}",
            hit.score, hit.dense, hit.lexical, hit.rerank, hit.chunk.text
        );
    }

    // metadata filters restrict the search space
    let only_birds = MetadataFilter { equals: [("topic".to_string(), json!("birds"))].into(), ..Default::default() };
    let words = |s: &str| s.split_whitespace().count();
    let ctx = rag.context(question, 60, &words, Some(&only_birds))?;
    println!("\ncontext ({} tokens, {} dropped):\n{}", ctx.tokens, ctx.dropped, ctx.text);
    for c in &ctx.citations {
        println!("  [{}] {} {}..{} {:?} {:?}", c.n, c.doc_id, c.span.start, c.span.end, c.source, c.license);
    }

    // answer with any Provider
    let llm = FnProvider::new(|_: &str| {
        "The harpy eagle lives in tropical lowland rainforests [1]. It hunts sloths and monkeys [2].".to_string()
    });
    let prompt = format!("Answer with citations using only this context.\n\n{}\n\nQuestion: {question}", ctx.text);
    let rt = tokio::runtime::Builder::new_current_thread().build()?;
    let req = harpia::core::traits::ChatRequest {
        model: "mock".into(),
        messages: vec![Message::user(prompt)],
        ..Default::default()
    };
    let answer = rt.block_on(llm.chat(&req))?.message.content;
    println!("\nanswer: {answer}");

    // evaluate: retrieval against gold chunk ids, then faithfulness of the answer to the context
    let retrieved: Vec<String> = rag.retrieve(question, None)?.into_iter().map(|h| h.chunk.id).collect();
    let relevant: Vec<String> = retrieved.iter().filter(|id| id.starts_with("harpy")).cloned().collect();
    let report = retrieval_report(&[RagExample { retrieved, relevant, ..Default::default() }], 3);
    println!("\nretrieval: {}", serde_json::to_string(&report)?);
    let rule = Calibrated::fixed(
        DecisionRule::argmax(&["entailment", "neutral"], "neutral"),
        Thresholds(std::collections::BTreeMap::new()),
    );
    let contexts: Vec<String> =
        ctx.citations.iter().filter_map(|c| rag.chunk(&c.chunk_id)).map(|c| c.text.clone()).collect();
    let f = faithfulness(&Entails, &rule, "entailment", &answer, &contexts, &Lang::En.data())?;
    println!("faithfulness: {:.2} ({} sentences)", f.score, f.sentences.len());
    Ok(())
}
