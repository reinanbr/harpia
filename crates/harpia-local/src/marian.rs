//! MarianMT translation (`opus-mt-*`), greedy (with KV cache) or beam search.
//!
//! Tokenization reads the SentencePiece files Hugging Face ships (`source.spm`, `target.spm`,
//! `vocab.json`) the way transformers' `MarianTokenizer` does, with no Python. A `tokenizer.json`
//! (and optional `tokenizer-target.json`) takes precedence when present. Sentences are translated
//! one at a time: candle's Marian decoder has no encoder padding mask, so batching would change
//! results.

use crate::loader::{ModelDir, model_err};
use crate::sampling::log_softmax;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::marian;
use harpia_core::traits::Translator;
use harpia_core::{HarpiaError, Result};
use harpia_data::spm::SpmModel;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Decoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Search {
    /// Pick the most likely token at every step.
    Greedy,
    /// Keep the `width` best hypotheses (length-normalized log-probability).
    Beam {
        /// Beam width.
        width: usize,
    },
}

/// A MarianMT translator.
pub struct MarianTranslator {
    model: Mutex<marian::MTModel>,
    tok: MarianTok,
    start: u32,
    eos: u32,
    pad: u32,
    device: Device,
    dir: PathBuf,
    /// Prepended to every source text (multilingual models), e.g. `">>por<<"`.
    pub prefix: Option<String>,
    /// Decoding strategy (default greedy).
    pub search: Search,
    /// Maximum generated tokens per sentence.
    pub max_new_tokens: usize,
}

impl std::fmt::Debug for MarianTranslator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MarianTranslator")
            .field("model", &self.dir)
            .field("search", &self.search)
            .finish_non_exhaustive()
    }
}

impl MarianTranslator {
    /// Load on the CPU in f32.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with(dir, &Device::Cpu, DType::F32)
    }

    /// Load with an explicit device and dtype.
    pub fn load_with(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let md = ModelDir::open(dir)?;
        let cfg: marian::Config = md.config_as()?;
        let vb = md.var_builder(dtype, device)?;
        let model = marian::MTModel::new(&cfg, vb).map_err(|e| model_err(dir, e))?;
        let tok = if !dir.join("tokenizer.json").is_file() && dir.join("source.spm").is_file() {
            MarianTok::spm(dir, cfg.eos_token_id)?
        } else {
            let source = md.tokenizer()?.inner().clone();
            let target_path = dir.join("tokenizer-target.json");
            let target = if target_path.is_file() {
                tokenizers::Tokenizer::from_file(&target_path).map_err(tok_err)?
            } else {
                source.clone()
            };
            MarianTok::Json { source, target: Box::new(target) }
        };
        Ok(Self {
            model: Mutex::new(model),
            tok,
            start: cfg.decoder_start_token_id,
            eos: cfg.eos_token_id,
            pad: cfg.pad_token_id,
            device: device.clone(),
            dir: dir.to_path_buf(),
            prefix: None,
            search: Search::Greedy,
            max_new_tokens: 256,
        })
    }

    fn err(&self, e: impl std::fmt::Display) -> HarpiaError {
        model_err(&self.dir, e)
    }

    fn last_logits(
        &self,
        model: &mut marian::MTModel,
        ids: &[u32],
        enc: &Tensor,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let input = Tensor::new(ids, &self.device).and_then(|t| t.unsqueeze(0)).map_err(|e| self.err(e))?;
        let logits = model.decode(&input, enc, start_pos).map_err(|e| self.err(e))?;
        let logits = logits.squeeze(0).map_err(|e| self.err(e))?;
        let n = logits.dim(0).map_err(|e| self.err(e))?;
        let mut v = logits
            .get(n - 1)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| self.err(e))?;
        if let Some(p) = v.get_mut(self.pad as usize) {
            *p = f32::NEG_INFINITY; // never generate padding
        }
        Ok(v)
    }

    /// Source token ids for `text` (with [`Self::prefix`] and the trailing `</s>`), as fed to the
    /// encoder.
    pub fn encode_source(&self, text: &str) -> Result<Vec<u32>> {
        match &self.prefix {
            Some(p) => self.tok.encode(&format!("{p} {text}")),
            None => self.tok.encode(text),
        }
    }

    /// Translate one text.
    pub fn translate_one(&self, text: &str) -> Result<String> {
        let ids = self.encode_source(text)?;
        let mut model = self.model.lock().expect("marian lock");
        model.reset_kv_cache();
        let input = Tensor::new(ids.as_slice(), &self.device).and_then(|t| t.unsqueeze(0)).map_err(|e| self.err(e))?;
        let enc = model.encoder().forward(&input, 0).map_err(|e| self.err(e))?;
        let out = match self.search {
            Search::Greedy => {
                let mut tokens = vec![self.start];
                for i in 0..self.max_new_tokens {
                    let ctx = if i == 0 { tokens.len() } else { 1 };
                    let start_pos = tokens.len() - ctx;
                    let logits = self.last_logits(&mut model, &tokens[start_pos..], &enc, start_pos)?;
                    let next = argmax(&logits);
                    if next == self.eos {
                        break;
                    }
                    tokens.push(next);
                }
                tokens
            }
            Search::Beam { width } => self.beam(&mut model, &enc, width.max(1))?,
        };
        let body: Vec<u32> = out.into_iter().skip(1).filter(|&t| t != self.eos && t != self.pad).collect();
        self.tok.decode(&body)
    }

    fn beam(&self, model: &mut marian::MTModel, enc: &Tensor, width: usize) -> Result<Vec<u32>> {
        // (tokens, sum log-prob, finished)
        let mut beams: Vec<(Vec<u32>, f32, bool)> = vec![(vec![self.start], 0.0, false)];
        for _ in 0..self.max_new_tokens {
            if beams.iter().all(|b| b.2) {
                break;
            }
            let mut next: Vec<(Vec<u32>, f32, bool)> = Vec::new();
            for (toks, score, done) in &beams {
                if *done {
                    next.push((toks.clone(), *score, true));
                    continue;
                }
                model.reset_kv_cache();
                let lp = log_softmax(&self.last_logits(model, toks, enc, 0)?);
                let mut idx: Vec<usize> = (0..lp.len()).collect();
                idx.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
                for &t in idx.iter().take(width) {
                    let mut nt = toks.clone();
                    nt.push(t as u32);
                    next.push((nt, score + lp[t], t as u32 == self.eos));
                }
            }
            let norm = |b: &(Vec<u32>, f32, bool)| b.1 / (b.0.len() as f32 - 1.0).max(1.0);
            next.sort_by(|a, b| norm(b).total_cmp(&norm(a)));
            next.truncate(width);
            beams = next;
        }
        Ok(beams.into_iter().next().map(|b| b.0).unwrap_or_default())
    }
}

fn tok_err(e: impl std::fmt::Display) -> HarpiaError {
    HarpiaError::Tokenizer(e.to_string())
}

/// Source/target tokenization.
enum MarianTok {
    /// `tokenizer.json` (+ optional `tokenizer-target.json`).
    Json { source: tokenizers::Tokenizer, target: Box<tokenizers::Tokenizer> },
    /// `source.spm` pieces mapped through `vocab.json`, as `MarianTokenizer` does.
    Spm { source: tokenizers::Tokenizer, vocab: HashMap<String, u32>, pieces: HashMap<u32, String>, unk: u32, eos: u32 },
}

impl MarianTok {
    fn spm(dir: &Path, eos: u32) -> Result<Self> {
        let source = SpmModel::load(dir.join("source.spm"))?.to_tokenizer()?.inner().clone();
        let path = dir.join("vocab.json");
        let raw = std::fs::read(&path).map_err(|e| HarpiaError::io(&path, e))?;
        let vocab: HashMap<String, u32> = serde_json::from_slice(&raw)?;
        let unk = vocab.get("<unk>").copied().unwrap_or(1);
        let pieces = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        Ok(Self::Spm { source, vocab, pieces, unk, eos })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Json { source, .. } => Ok(source.encode(text, true).map_err(tok_err)?.get_ids().to_vec()),
            Self::Spm { source, vocab, unk, eos, .. } => {
                // A leading `>>lang<<` code is looked up directly, the rest goes through SentencePiece.
                let (code, rest) = match text.strip_prefix(">>").and_then(|t| t.find("<<").map(|i| i + 4)) {
                    Some(end) => (Some(&text[..end]), &text[end..]),
                    None => (None, text),
                };
                let enc = source.encode(rest, false).map_err(tok_err)?;
                let mut ids: Vec<u32> = code
                    .into_iter()
                    .chain(enc.get_tokens().iter().map(String::as_str))
                    .map(|t| vocab.get(t).copied().unwrap_or(*unk))
                    .collect();
                ids.push(*eos);
                Ok(ids)
            }
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::Json { target, .. } => target.decode(ids, true).map_err(tok_err),
            Self::Spm { pieces, unk, eos, .. } => {
                let text: String = ids
                    .iter()
                    .filter(|&&i| i != *unk && i != *eos)
                    .filter_map(|i| pieces.get(i))
                    .map(String::as_str)
                    .collect();
                Ok(text.replace('\u{2581}', " ").trim().to_string())
            }
        }
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0)
}

impl Translator for MarianTranslator {
    fn translate(&self, texts: &[&str]) -> Result<Vec<String>> {
        texts.iter().map(|t| self.translate_one(t)).collect()
    }
}
