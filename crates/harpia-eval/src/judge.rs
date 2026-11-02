//! LLM-as-judge: rubric scoring, pairwise comparison with position swap, and agreement
//! statistics across judges.
//!
//! Verdicts are requested as JSON and parsed with `harpia_structured` (fence stripping and
//! repair), with corrective retries when the reply does not parse or the score is off-scale.

use crate::cost::CostTracker;
use harpia_core::traits::{ChatRequest, GenerationParams, Judge, JudgeInput, Provider, Verdict};
use harpia_core::{HarpiaError, Message, Result};
use serde::{Deserialize, Serialize};

/// An LLM judge over any [`Provider`].
pub struct LlmJudge<'a, P> {
    provider: &'a P,
    /// Model name sent to the provider.
    pub model: String,
    /// Inclusive score range of the rubric.
    pub scale: (f64, f64),
    /// Extra attempts after an unparseable or off-scale reply.
    pub retries: usize,
    /// Decoding parameters (greedy by default, for repeatability).
    pub params: GenerationParams,
    /// Optional cost/latency tracking.
    pub tracker: Option<&'a CostTracker>,
}

impl<P> std::fmt::Debug for LlmJudge<'_, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmJudge").field("model", &self.model).field("scale", &self.scale).finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
struct RawVerdict {
    score: f64,
    #[serde(default)]
    rationale: String,
}

#[derive(Deserialize)]
struct RawPreference {
    winner: String,
    #[serde(default)]
    rationale: String,
}

/// Which answer a pairwise judge preferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preference {
    /// The first answer.
    A,
    /// The second answer.
    B,
    /// Neither.
    Tie,
}

/// Result of a position-swapped pairwise comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseVerdict {
    /// Final preference: the agreed winner, or `Tie` when the two orders disagree.
    pub winner: Preference,
    /// Whether both presentation orders agreed (a position-bias signal when `false`).
    pub consistent: bool,
    /// Preference with A shown first.
    pub forward: Preference,
    /// Preference with B shown first, mapped back to the original A/B.
    pub swapped: Preference,
    /// Rationales of both calls.
    pub rationales: [String; 2],
}

impl<'a, P: Provider> LlmJudge<'a, P> {
    /// New judge with a `1..=10` scale and two retries.
    pub fn new(provider: &'a P, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            scale: (1.0, 10.0),
            retries: 2,
            params: GenerationParams { max_new_tokens: 512, ..GenerationParams::default() },
            tracker: None,
        }
    }

    async fn ask(&self, messages: Vec<Message>) -> Result<String> {
        let req =
            ChatRequest { model: self.model.clone(), messages, params: self.params.clone(), ..Default::default() };
        let resp = match self.tracker {
            Some(t) => t.chat(self.provider, &req).await?.0,
            None => self.provider.chat(&req).await?,
        };
        Ok(resp.message.content)
    }

    async fn ask_json<T: serde::de::DeserializeOwned>(
        &self,
        mut messages: Vec<Message>,
        check: impl Fn(&T) -> std::result::Result<(), String>,
    ) -> Result<T> {
        let mut last_err = String::new();
        for _ in 0..=self.retries {
            let reply = self.ask(messages.clone()).await?;
            let problem = match harpia_structured::extract::parse_llm_json::<T>(&reply) {
                Ok(v) => match check(&v) {
                    Ok(()) => return Ok(v),
                    Err(e) => e,
                },
                Err(e) => e.to_string(),
            };
            last_err = problem.clone();
            messages.push(Message::assistant(reply));
            messages.push(Message::user(format!(
                "Your reply was not usable ({problem}). Answer again with only the JSON object."
            )));
        }
        Err(HarpiaError::Validation(format!("judge reply unusable after {} attempts: {last_err}", self.retries + 1)))
    }

    /// Score one answer on the rubric.
    pub async fn score(&self, input: &JudgeInput) -> Result<Verdict> {
        let (lo, hi) = self.scale;
        let mut prompt = format!(
            "You are a strict grader. Grade the answer using the rubric.\n\nRubric:\n{}\n\nQuestion:\n{}\n\nAnswer:\n{}\n",
            input.rubric, input.question, input.answer
        );
        if let Some(r) = &input.reference {
            prompt += &format!("\nReference answer:\n{r}\n");
        }
        prompt += &format!(
            "\nReply with only a JSON object: {{\"score\": <number from {lo} to {hi}>, \"rationale\": \"<one or two sentences>\"}}"
        );
        let raw: RawVerdict = self
            .ask_json(vec![Message::user(prompt)], |v: &RawVerdict| {
                if v.score.is_finite() && v.score >= lo && v.score <= hi {
                    Ok(())
                } else {
                    Err(format!("score {} outside {lo}..={hi}", v.score))
                }
            })
            .await?;
        Ok(Verdict { score: raw.score, rationale: raw.rationale })
    }

    async fn prefer(&self, question: &str, rubric: &str, first: &str, second: &str) -> Result<(Preference, String)> {
        let prompt = format!(
            "Compare two answers to the question using the rubric. Ignore answer order and length.\n\nRubric:\n{rubric}\n\n\
             Question:\n{question}\n\n[Answer 1]\n{first}\n\n[Answer 2]\n{second}\n\n\
             Reply with only a JSON object: {{\"winner\": \"1\" | \"2\" | \"tie\", \"rationale\": \"<one sentence>\"}}"
        );
        let raw: RawPreference = self
            .ask_json(vec![Message::user(prompt)], |v: &RawPreference| match v.winner.trim() {
                "1" | "2" | "tie" => Ok(()),
                w => Err(format!("winner {w:?} is not \"1\", \"2\" or \"tie\"")),
            })
            .await?;
        let p = match raw.winner.trim() {
            "1" => Preference::A,
            "2" => Preference::B,
            _ => Preference::Tie,
        };
        Ok((p, raw.rationale))
    }

    /// Compare `a` and `b` twice (both orders); disagreement between orders yields a tie.
    pub async fn compare(&self, question: &str, rubric: &str, a: &str, b: &str) -> Result<PairwiseVerdict> {
        let (forward, r1) = self.prefer(question, rubric, a, b).await?;
        let (back, r2) = self.prefer(question, rubric, b, a).await?;
        let swapped = match back {
            Preference::A => Preference::B,
            Preference::B => Preference::A,
            Preference::Tie => Preference::Tie,
        };
        let consistent = forward == swapped;
        Ok(PairwiseVerdict {
            winner: if consistent { forward } else { Preference::Tie },
            consistent,
            forward,
            swapped,
            rationales: [r1, r2],
        })
    }
}

impl<P: Provider> Judge for LlmJudge<'_, P> {
    async fn judge(&self, input: &JudgeInput) -> Result<Verdict> {
        self.score(input).await
    }
}

/// Agreement between two raters over the same items.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Agreement {
    /// Items compared.
    pub n: usize,
    /// Fraction of identical labels.
    pub percent: f64,
    /// Cohen's kappa (chance-corrected; 1 = perfect, 0 = chance level).
    pub kappa: f64,
}

/// Percent agreement and Cohen's kappa for categorical labels.
///
/// ```
/// let a = harpia_eval::judge::agreement(&["x", "y", "x", "y"], &["x", "y", "y", "y"]);
/// assert_eq!(a.percent, 0.75);
/// assert!((a.kappa - 0.5).abs() < 1e-12);
/// ```
pub fn agreement<S: AsRef<str> + PartialEq>(a: &[S], b: &[S]) -> Agreement {
    let n = a.len().min(b.len());
    if n == 0 {
        return Agreement::default();
    }
    let po = a.iter().zip(b).filter(|(x, y)| x.as_ref() == y.as_ref()).count() as f64 / n as f64;
    let mut labels: Vec<&str> = a[..n].iter().chain(&b[..n]).map(AsRef::as_ref).collect();
    labels.sort_unstable();
    labels.dedup();
    let pe: f64 = labels
        .iter()
        .map(|l| {
            let pa = a[..n].iter().filter(|x| x.as_ref() == *l).count() as f64 / n as f64;
            let pb = b[..n].iter().filter(|x| x.as_ref() == *l).count() as f64 / n as f64;
            pa * pb
        })
        .sum();
    let kappa = if (1.0 - pe).abs() < f64::EPSILON { 1.0 } else { (po - pe) / (1.0 - pe) };
    Agreement { n, percent: po, kappa }
}

/// Pearson correlation of two score lists (`NaN` when either is constant).
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return f64::NAN;
    }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        cov += (a[i] - ma) * (b[i] - mb);
        va += (a[i] - ma).powi(2);
        vb += (b[i] - mb).powi(2);
    }
    cov / (va.sqrt() * vb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::ScriptedProvider;

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
    }

    #[test]
    fn score_with_retry_on_bad_reply() {
        let p = ScriptedProvider::new([
            "not json at all",
            "```json\n{\"score\": 42}\n```",
            "{\"score\": 7, \"rationale\": \"ok\"}",
        ]);
        let judge = LlmJudge::new(&p, "m");
        let v = block_on(judge.score(&JudgeInput {
            question: "q".into(),
            answer: "a".into(),
            reference: None,
            rubric: "r".into(),
        }))
        .unwrap();
        assert_eq!(v.score, 7.0);
        assert_eq!(p.calls(), 3);
        let p = ScriptedProvider::new(["nope"; 3]);
        let mut judge = LlmJudge::new(&p, "m");
        judge.retries = 2;
        assert!(block_on(judge.score(&JudgeInput::default())).is_err());
    }

    #[test]
    fn pairwise_swaps_positions() {
        // A judge with position bias always says "1": inconsistent → tie.
        let biased = ScriptedProvider::new(["{\"winner\": \"1\"}", "{\"winner\": \"1\"}"]);
        let v = block_on(LlmJudge::new(&biased, "m").compare("q", "r", "a", "b")).unwrap();
        assert_eq!((v.winner, v.consistent), (Preference::Tie, false));
        // A consistent judge prefers A in both orders.
        let fair = ScriptedProvider::new(["{\"winner\": \"1\"}", "{\"winner\": \"2\"}"]);
        let v = block_on(LlmJudge::new(&fair, "m").compare("q", "r", "a", "b")).unwrap();
        assert_eq!((v.winner, v.consistent), (Preference::A, true));
    }

    #[test]
    fn agreement_stats() {
        let a = agreement(&["x", "x"], &["x", "x"]);
        assert_eq!((a.percent, a.kappa), (1.0, 1.0));
        assert!((pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-12);
        assert!(pearson(&[1.0, 1.0], &[1.0, 2.0]).is_nan());
    }
}
