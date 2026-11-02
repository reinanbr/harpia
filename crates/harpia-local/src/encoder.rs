//! Encoder models: sequence/pair classification (incl. NLI cross-encoders and rerankers) and
//! sentence embeddings, for BERT and XLM-RoBERTa (also RoBERTa) checkpoints.

use crate::loader::{ModelDir, model_err};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear};
use candle_transformers::models::{bert, xlm_roberta};
use harpia_core::traits::{Embedder, PairClassifier, Reranker};
use harpia_core::{HarpiaError, Label, Result, Scores};
use harpia_data::tokenize::HfTokenizer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::{TruncationDirection, TruncationParams, TruncationStrategy};

enum Arch {
    XlmRoberta(xlm_roberta::XLMRobertaModel),
    Bert(bert::BertModel),
}

enum Head {
    XlmRoberta { dense: Linear, out_proj: Linear },
    Bert { pooler: Linear, classifier: Linear },
}

fn is_roberta(model_type: &str) -> bool {
    matches!(model_type, "xlm-roberta" | "roberta" | "camembert" | "xlm_roberta")
}

fn ce(dir: &Path) -> impl Fn(candle_core::Error) -> HarpiaError + '_ {
    move |e| model_err(dir, e)
}

/// Shared encoder plumbing: tokenizer with truncation, batching, forward pass.
struct EncoderCore {
    arch: Arch,
    tokenizer: tokenizers::Tokenizer,
    pad_id: u32,
    device: Device,
    dir: PathBuf,
    batch_size: usize,
}

impl EncoderCore {
    fn tokenizer_with_truncation(tok: &HfTokenizer, max_len: usize) -> Result<tokenizers::Tokenizer> {
        let mut t = tok.inner().clone();
        t.with_padding(None);
        t.with_truncation(Some(TruncationParams {
            max_length: max_len,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        }))
        .map_err(|e| HarpiaError::Tokenizer(e.to_string()))?;
        Ok(t)
    }

    fn encode(&self, first: &str, second: Option<&str>) -> Result<Vec<u32>> {
        let enc = match second {
            Some(s) => self.tokenizer.encode((first, s), true),
            None => self.tokenizer.encode(first, true),
        }
        .map_err(|e| HarpiaError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Hidden states `(batch, seq, hidden)` and the attention mask `(batch, seq)` for a padded batch.
    fn forward(&self, batch: &[Vec<u32>]) -> Result<(Tensor, Tensor)> {
        let max = batch.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut ids = Vec::with_capacity(batch.len() * max);
        let mut mask = Vec::with_capacity(batch.len() * max);
        for seq in batch {
            ids.extend(seq.iter().copied());
            ids.extend(std::iter::repeat_n(self.pad_id, max - seq.len()));
            mask.extend(std::iter::repeat_n(1u32, seq.len()));
            mask.extend(std::iter::repeat_n(0u32, max - seq.len()));
        }
        let d = &self.dir;
        let ids = Tensor::from_vec(ids, (batch.len(), max), &self.device).map_err(ce(d))?;
        let mask = Tensor::from_vec(mask, (batch.len(), max), &self.device).map_err(ce(d))?;
        let types = ids.zeros_like().map_err(ce(d))?;
        let hidden = match &self.arch {
            Arch::XlmRoberta(m) => m.forward(&ids, &mask, &types, None, None, None),
            Arch::Bert(m) => m.forward(&ids, &types, Some(&mask)),
        }
        .map_err(ce(d))?;
        Ok((hidden, mask))
    }

    /// Run `f` over length-sorted batches and return results in input order.
    fn batched<T, F>(&self, inputs: Vec<Vec<u32>>, mut f: F) -> Result<Vec<T>>
    where
        F: FnMut(&[Vec<u32>]) -> Result<Vec<T>>,
    {
        let mut order: Vec<usize> = (0..inputs.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(inputs[i].len()));
        let mut out: Vec<Option<T>> = (0..inputs.len()).map(|_| None).collect();
        for chunk in order.chunks(self.batch_size.max(1)) {
            let batch: Vec<Vec<u32>> = chunk.iter().map(|&i| inputs[i].clone()).collect();
            for (&i, r) in chunk.iter().zip(f(&batch)?) {
                out[i] = Some(r);
            }
        }
        Ok(out.into_iter().map(|o| o.expect("every input processed")).collect())
    }
}

/// A sequence or pair classifier (`*ForSequenceClassification` checkpoints).
///
/// Pair inputs follow the NLI convention: `(premise, hypothesis)` — the evidence first, the
/// claim second — encoded with the model's own pair template.
///
/// ```no_run
/// use harpia_local::SequenceClassifier;
/// use harpia_core::traits::PairClassifier;
/// let nli = SequenceClassifier::load("models/errei-cita-v1".as_ref())?
///     .with_label_aliases([("sustenta", "supports"), ("contradiz", "contradicts"), ("neutro", "neutral")]);
/// let scores = nli.classify_pairs(&[("The bird eats sloths.", "Harpy eagles hunt sloths.")])?;
/// println!("{:?}", scores[0].argmax());
/// # Ok::<(), harpia_core::HarpiaError>(())
/// ```
pub struct SequenceClassifier {
    core: EncoderCore,
    head: Head,
    labels: Vec<Label>,
}

impl std::fmt::Debug for SequenceClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceClassifier")
            .field("model", &self.core.dir)
            .field("labels", &self.labels)
            .finish_non_exhaustive()
    }
}

impl SequenceClassifier {
    /// Load from a model directory on the CPU in f32 with `max_len` 512.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with(dir, &Device::Cpu, DType::F32, 512)
    }

    /// Load with an explicit device, dtype and maximum sequence length.
    pub fn load_with(dir: &Path, device: &Device, dtype: DType, max_len: usize) -> Result<Self> {
        let md = ModelDir::open(dir)?;
        let num_labels = md.num_labels().ok_or_else(|| model_err(dir, "config.json has no id2label/num_labels"))?;
        let labels = md.labels(num_labels);
        let vb = md.var_builder(dtype, device)?;
        let tok = md.tokenizer()?;
        let (arch, head, pad_id) = if is_roberta(md.model_type()) {
            let cfg: xlm_roberta::Config = md.config_as()?;
            let pad = cfg.pad_token_id;
            let model = xlm_roberta::XLMRobertaModel::new(&cfg, vb.pp("roberta")).map_err(ce(dir))?;
            let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("classifier.dense")).map_err(ce(dir))?;
            let out_proj = linear(cfg.hidden_size, num_labels, vb.pp("classifier.out_proj")).map_err(ce(dir))?;
            (Arch::XlmRoberta(model), Head::XlmRoberta { dense, out_proj }, pad)
        } else if md.model_type() == "bert" {
            let cfg: bert::Config = md.config_as()?;
            let pad = cfg.pad_token_id as u32;
            let model = bert::BertModel::load(vb.clone(), &cfg).map_err(ce(dir))?;
            let pooler = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("bert.pooler.dense")).map_err(ce(dir))?;
            let classifier = linear(cfg.hidden_size, num_labels, vb.pp("classifier")).map_err(ce(dir))?;
            (Arch::Bert(model), Head::Bert { pooler, classifier }, pad)
        } else {
            return Err(model_err(
                dir,
                format!("unsupported encoder model_type `{}` (bert, roberta, xlm-roberta)", md.model_type()),
            ));
        };
        Ok(Self {
            core: EncoderCore {
                arch,
                tokenizer: EncoderCore::tokenizer_with_truncation(&tok, max_len)?,
                pad_id,
                device: device.clone(),
                dir: dir.to_path_buf(),
                batch_size: 32,
            },
            head,
            labels,
        })
    }

    /// Rename labels (e.g. `entailment → supports`). Unknown names are ignored.
    pub fn with_label_aliases<I, A, B>(mut self, aliases: I) -> Self
    where
        I: IntoIterator<Item = (A, B)>,
        A: AsRef<str>,
        B: Into<String>,
    {
        let map: HashMap<String, String> =
            aliases.into_iter().map(|(a, b)| (a.as_ref().to_lowercase(), b.into())).collect();
        for l in &mut self.labels {
            if let Some(new) = map.get(&l.to_lowercase()) {
                *l = new.clone();
            }
        }
        self
    }

    /// Batch size used internally (default 32).
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.core.batch_size = n.max(1);
        self
    }

    fn head_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let d = &self.core.dir;
        let cls = hidden.get_on_dim(1, 0).map_err(ce(d))?.contiguous().map_err(ce(d))?;
        match &self.head {
            Head::XlmRoberta { dense, out_proj } => {
                let x = dense.forward(&cls).map_err(ce(d))?.tanh().map_err(ce(d))?;
                out_proj.forward(&x).map_err(ce(d))
            }
            Head::Bert { pooler, classifier } => {
                let x = pooler.forward(&cls).map_err(ce(d))?.tanh().map_err(ce(d))?;
                classifier.forward(&x).map_err(ce(d))
            }
        }
    }

    fn logits_for(&self, inputs: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>> {
        self.core.batched(inputs, |batch| {
            let (hidden, _) = self.core.forward(batch)?;
            let logits = self.head_logits(&hidden)?;
            logits.to_dtype(DType::F32).and_then(|t| t.to_vec2::<f32>()).map_err(ce(&self.core.dir))
        })
    }

    /// Raw logits for `(premise, hypothesis)` pairs, in label order.
    pub fn pair_logits(&self, pairs: &[(&str, &str)]) -> Result<Vec<Vec<f32>>> {
        let inputs = pairs.iter().map(|(a, b)| self.core.encode(a, Some(b))).collect::<Result<Vec<_>>>()?;
        self.logits_for(inputs)
    }

    /// Raw logits for single texts.
    pub fn logits(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let inputs = texts.iter().map(|t| self.core.encode(t, None)).collect::<Result<Vec<_>>>()?;
        self.logits_for(inputs)
    }

    fn to_scores(&self, logits: Vec<Vec<f32>>) -> Vec<Scores> {
        logits.into_iter().map(|row| Scores::new(self.labels.iter().cloned().zip(softmax(&row)).collect())).collect()
    }

    /// Softmax scores for single texts.
    pub fn classify(&self, texts: &[&str]) -> Result<Vec<Scores>> {
        Ok(self.to_scores(self.logits(texts)?))
    }

    /// Token ids the model would see for a pair (for inspection and tests).
    pub fn encode_pair(&self, premise: &str, hypothesis: &str) -> Result<Vec<u32>> {
        self.core.encode(premise, Some(hypothesis))
    }
}

/// Numerically stable softmax.
pub fn softmax(xs: &[f32]) -> Vec<f32> {
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = xs.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

impl PairClassifier for SequenceClassifier {
    fn labels(&self) -> &[Label] {
        &self.labels
    }

    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Scores>> {
        Ok(self.to_scores(self.pair_logits(pairs)?))
    }
}

/// Cross-encoder reranking: one-logit models return the logit; multi-label models return the
/// probability of the last label (the "relevant"/"entailment"-style class by convention).
impl Reranker for SequenceClassifier {
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        let pairs: Vec<(&str, &str)> = documents.iter().map(|d| (query, *d)).collect();
        let logits = self.pair_logits(&pairs)?;
        Ok(logits
            .into_iter()
            .map(|row| if row.len() == 1 { row[0] } else { *softmax(&row).last().expect("labels") })
            .collect())
    }
}

/// How token states become one sentence vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    /// Mean over non-padding tokens (sentence-transformers default).
    #[default]
    Mean,
    /// The first (`[CLS]`/`<s>`) token.
    Cls,
}

/// Sentence embeddings from a BERT/XLM-R encoder (e.g. sentence-transformers checkpoints).
pub struct EncoderEmbedder {
    core: EncoderCore,
    dim: usize,
    pooling: Pooling,
    normalize: bool,
}

impl std::fmt::Debug for EncoderEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderEmbedder").field("model", &self.core.dir).field("dim", &self.dim).finish_non_exhaustive()
    }
}

impl EncoderEmbedder {
    /// Load with mean pooling and L2 normalization, CPU, f32, `max_len` 512.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with(dir, &Device::Cpu, DType::F32, 512, Pooling::Mean, true)
    }

    /// Load with explicit options.
    pub fn load_with(
        dir: &Path,
        device: &Device,
        dtype: DType,
        max_len: usize,
        pooling: Pooling,
        normalize: bool,
    ) -> Result<Self> {
        let md = ModelDir::open(dir)?;
        let vb: VarBuilder = md.var_builder(dtype, device)?;
        let tok = md.tokenizer()?;
        let (arch, dim, pad_id) = if is_roberta(md.model_type()) {
            let cfg: xlm_roberta::Config = md.config_as()?;
            let vb =
                if vb.contains_tensor("roberta.embeddings.word_embeddings.weight") { vb.pp("roberta") } else { vb };
            (
                Arch::XlmRoberta(xlm_roberta::XLMRobertaModel::new(&cfg, vb).map_err(ce(dir))?),
                cfg.hidden_size,
                cfg.pad_token_id,
            )
        } else if md.model_type() == "bert" {
            let cfg: bert::Config = md.config_as()?;
            (Arch::Bert(bert::BertModel::load(vb, &cfg).map_err(ce(dir))?), cfg.hidden_size, cfg.pad_token_id as u32)
        } else {
            return Err(model_err(dir, format!("unsupported encoder model_type `{}`", md.model_type())));
        };
        Ok(Self {
            core: EncoderCore {
                arch,
                tokenizer: EncoderCore::tokenizer_with_truncation(&tok, max_len)?,
                pad_id,
                device: device.clone(),
                dir: dir.to_path_buf(),
                batch_size: 32,
            },
            dim,
            pooling,
            normalize,
        })
    }
}

impl Embedder for EncoderEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let inputs = texts.iter().map(|t| self.core.encode(t, None)).collect::<Result<Vec<_>>>()?;
        let d = self.core.dir.clone();
        self.core.batched(inputs, |batch| {
            let (hidden, mask) = self.core.forward(batch)?;
            let hidden = hidden.to_dtype(DType::F32).map_err(ce(&d))?;
            let pooled = match self.pooling {
                Pooling::Cls => hidden.get_on_dim(1, 0).map_err(ce(&d))?,
                Pooling::Mean => {
                    let m = mask.to_dtype(DType::F32).and_then(|m| m.unsqueeze(2)).map_err(ce(&d))?;
                    let sum = hidden.broadcast_mul(&m).and_then(|x| x.sum(1)).map_err(ce(&d))?;
                    let count = m.sum(1).map_err(ce(&d))?;
                    sum.broadcast_div(&count).map_err(ce(&d))?
                }
            };
            let pooled = if self.normalize {
                let norm =
                    pooled.sqr().and_then(|x| x.sum_keepdim(D::Minus1)).and_then(|x| x.sqrt()).map_err(ce(&d))?;
                pooled.broadcast_div(&norm).map_err(ce(&d))?
            } else {
                pooled
            };
            pooled.to_vec2::<f32>().map_err(ce(&d))
        })
    }
}
