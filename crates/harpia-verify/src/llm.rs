//! Optional LLM assistance (feature `llm`, off by default): claim decomposition, query
//! rewriting and evidence summaries through any `Provider`. The LLM never sets a verification
//! status; callers record its use in `Verification::llm_assistance`.

use crate::claims::Claim;
use crate::verify::Evidence;
use harpia_core::traits::{ChatRequest, GenerationParams, Provider};
use harpia_core::{HarpiaError, Message, Result};
use serde::Deserialize;

async fn ask_json<P: Provider, T: serde::de::DeserializeOwned>(p: &P, model: &str, prompt: String) -> Result<T> {
    let req = ChatRequest {
        model: model.into(),
        messages: vec![Message::user(prompt)],
        params: GenerationParams { max_new_tokens: 512, ..Default::default() },
        ..Default::default()
    };
    let reply = p.chat(&req).await?.message.content;
    harpia_structured::extract::parse_llm_json(&reply)
        .map_err(|e| HarpiaError::Validation(format!("LLM reply is not the requested JSON: {e}")))
}

#[derive(Deserialize)]
struct Claims {
    claims: Vec<String>,
}

#[derive(Deserialize)]
struct Queries {
    queries: Vec<String>,
}

/// Decompose `answer` into atomic, self-contained factual claims.
pub async fn decompose_claims<P: Provider>(p: &P, model: &str, answer: &str) -> Result<Vec<Claim>> {
    let prompt = format!(
        "Split the text into atomic factual claims. Each claim must be one self-contained sentence \
         (resolve pronouns), checkable against a reference source. Skip opinions and questions.\n\n\
         Text:\n{answer}\n\nReply with only JSON: {{\"claims\": [\"...\"]}}"
    );
    let c: Claims = ask_json(p, model, prompt).await?;
    Ok(c.claims
        .into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .map(|text| Claim { text, span: None })
        .collect())
}

/// Search queries (keywords, not full sentences) that would find evidence for `claim`.
pub async fn rewrite_queries<P: Provider>(p: &P, model: &str, claim: &str, n: usize) -> Result<Vec<String>> {
    let prompt = format!(
        "Write up to {n} short keyword search queries (3-8 words each, no full sentences) to find \
         reference material that supports or refutes this claim:\n{claim}\n\nReply with only JSON: {{\"queries\": [\"...\"]}}"
    );
    let q: Queries = ask_json(p, model, prompt).await?;
    Ok(q.queries
        .into_iter()
        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|s| !s.is_empty())
        .take(n)
        .collect())
}

/// A short plain-language summary of the evidence (for reports; never used for the status).
pub async fn summarize_evidence<P: Provider>(p: &P, model: &str, claim: &str, evidence: &[Evidence]) -> Result<String> {
    let mut prompt = format!("Claim: {claim}\n\nEvidence passages:\n");
    for (i, e) in evidence.iter().enumerate() {
        prompt += &format!("[{}] ({}, tier {}) {}\n", i + 1, e.source, e.tier, e.passage);
    }
    prompt += "\nIn two sentences, say what the passages state about the claim, citing [n]. Do not add facts.";
    let req = ChatRequest { model: model.into(), messages: vec![Message::user(prompt)], ..Default::default() };
    Ok(p.chat(&req).await?.message.content.trim().to_string())
}
