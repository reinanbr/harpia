//! Differentiable forward passes for training.
//!
//! candle-nn's fused kernels (softmax, layer/RMS norm, rotary embeddings) are applied without a
//! backward pass, so gradients stop silently inside candle-transformers' models. These modules
//! re-implement the supported architectures with composed tensor ops (all differentiable), using
//! Hugging Face weight names. Tests check that their outputs match `harpia-local` inference.

use crate::backend::TrainStore;
use candle_core::{D, DType, Device, Result, Tensor};
use serde_json::Value;

/// Linear layer with an optional LoRA update: `y = x·Wᵀ + b + s·(x·Aᵀ)·Bᵀ`.
pub struct Lin {
    w: Tensor,
    b: Option<Tensor>,
    lora: Option<(Tensor, Tensor, f64)>,
}

impl Lin {
    fn load(st: &TrainStore, prefix: &str, out: usize, inp: usize, bias: bool) -> Result<Self> {
        let wname = format!("{prefix}.weight");
        let w = st.param(&wname, &[out, inp])?;
        let b = if bias { Some(st.param(&format!("{prefix}.bias"), &[out])?) } else { None };
        let lora = st.lora_of(&wname).map(|(a, b)| (a, b, st.scale()));
        Ok(Self { w, b, lora })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_d(x, None)
    }

    /// Forward with optional LoRA dropout on the adapter input (PEFT: `B(A(dropout(x)))`).
    fn forward_d(&self, x: &Tensor, lora_drop: Option<&Dropout>) -> Result<Tensor> {
        // flatten leading dims: one 2-D GEMM is much faster than a batched broadcast matmul
        let dims = x.dims().to_vec();
        let inp = *dims.last().expect("rank >= 1");
        let x2 = x.reshape(((), inp))?;
        let mut y = x2.matmul(&self.w.t()?)?;
        if let Some(b) = &self.b {
            y = y.broadcast_add(b)?;
        }
        if let Some((a, b, s)) = &self.lora {
            let xin = match lora_drop {
                Some(d) => d.apply(&x2)?,
                None => x2.clone(),
            };
            y = (y + (xin.matmul(&a.t()?)?.matmul(&b.t()?)? * *s)?)?;
        }
        let mut out = dims;
        *out.last_mut().expect("rank >= 1") = y.dim(1)?;
        y.reshape(out)
    }
}

/// Inverted dropout with a seeded generator; a no-op when `p == 0` or training is off.
pub struct Dropout {
    p: f64,
    on: std::sync::atomic::AtomicBool,
    rng: std::sync::Mutex<harpia_data::rng::SplitMix64>,
}

impl Dropout {
    /// Dropout with probability `p`, off until [`Dropout::set_training`] turns it on.
    pub fn new(p: f64, seed: u64) -> Self {
        Self {
            p,
            on: std::sync::atomic::AtomicBool::new(false),
            rng: std::sync::Mutex::new(harpia_data::rng::SplitMix64::new(seed)),
        }
    }

    /// Restart the generator from `seed`.
    pub fn reseed(&self, seed: u64) {
        *self.rng.lock().expect("dropout rng") = harpia_data::rng::SplitMix64::new(seed);
    }

    /// Enable (training) or disable (evaluation).
    pub fn set_training(&self, on: bool) {
        self.on.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Apply to `x`.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        if self.p <= 0.0 || !self.on.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(x.clone());
        }
        let keep = 1.0 - self.p;
        let n = x.elem_count();
        let mask: Vec<f32> = {
            let mut r = self.rng.lock().expect("dropout rng");
            (0..n).map(|_| if r.next_f64() < keep { (1.0 / keep) as f32 } else { 0.0 }).collect()
        };
        x.mul(&Tensor::from_vec(mask, x.shape(), x.device())?.to_dtype(x.dtype())?)
    }
}

fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)?.broadcast_mul(w)?.broadcast_add(b)
}

fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(ms + eps)?.sqrt()?)?.broadcast_mul(w)
}

/// Softmax over the last dimension (the max is a constant shift, so it is detached).
fn softmax(x: &Tensor) -> Result<Tensor> {
    let m = x.max_keepdim(D::Minus1)?.detach();
    let e = x.broadcast_sub(&m)?.exp()?;
    e.broadcast_div(&e.sum_keepdim(D::Minus1)?)
}

fn act(name: &str, x: &Tensor) -> Result<Tensor> {
    match name {
        "gelu_new" | "gelu_pytorch_tanh" => x.gelu(),
        "relu" => x.relu(),
        "silu" | "swish" => x.silu(),
        _ => x.gelu_erf(),
    }
}

fn usize_of(cfg: &Value, k: &str) -> Result<usize> {
    cfg[k].as_u64().map(|x| x as usize).ok_or_else(|| candle_core::Error::Msg(format!("config.json: missing `{k}`")))
}

// ------------------------------------------------------------------ encoders

struct EncLayer {
    q: Lin,
    k: Lin,
    v: Lin,
    attn_out: Lin,
    attn_ln: (Tensor, Tensor),
    inter: Lin,
    out: Lin,
    out_ln: (Tensor, Tensor),
}

enum Head {
    Roberta { dense: Lin, out_proj: Lin },
    Bert { pooler: Lin, classifier: Lin },
}

/// A BERT / RoBERTa / XLM-R sequence classifier with differentiable forward.
pub struct EncoderClassifier {
    roberta: bool,
    word: Tensor,
    pos: Tensor,
    types: Tensor,
    emb_ln: (Tensor, Tensor),
    layers: Vec<EncLayer>,
    head: Head,
    heads: usize,
    eps: f64,
    act: String,
    pad: u32,
    hidden_drop: Dropout,
    attn_drop: Dropout,
    cls_drop: Dropout,
}

impl EncoderClassifier {
    /// Build from `config.json` values and the store; `num_labels` outputs.
    pub fn new(cfg: &Value, st: &TrainStore, num_labels: usize) -> Result<Self> {
        let mt = cfg["model_type"].as_str().unwrap_or("");
        let roberta = mt != "bert";
        let p = if roberta { "roberta" } else { "bert" };
        let h = usize_of(cfg, "hidden_size")?;
        let inter = usize_of(cfg, "intermediate_size")?;
        let n_layers = usize_of(cfg, "num_hidden_layers")?;
        let ln = |name: &str| -> Result<(Tensor, Tensor)> {
            Ok((st.param(&format!("{name}.weight"), &[h])?, st.param(&format!("{name}.bias"), &[h])?))
        };
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let l = format!("{p}.encoder.layer.{i}");
            layers.push(EncLayer {
                q: Lin::load(st, &format!("{l}.attention.self.query"), h, h, true)?,
                k: Lin::load(st, &format!("{l}.attention.self.key"), h, h, true)?,
                v: Lin::load(st, &format!("{l}.attention.self.value"), h, h, true)?,
                attn_out: Lin::load(st, &format!("{l}.attention.output.dense"), h, h, true)?,
                attn_ln: ln(&format!("{l}.attention.output.LayerNorm"))?,
                inter: Lin::load(st, &format!("{l}.intermediate.dense"), inter, h, true)?,
                out: Lin::load(st, &format!("{l}.output.dense"), h, inter, true)?,
                out_ln: ln(&format!("{l}.output.LayerNorm"))?,
            });
        }
        let head = if roberta {
            Head::Roberta {
                dense: Lin::load(st, "classifier.dense", h, h, true)?,
                out_proj: Lin::load(st, "classifier.out_proj", num_labels, h, true)?,
            }
        } else {
            Head::Bert {
                pooler: Lin::load(st, "bert.pooler.dense", h, h, true)?,
                classifier: Lin::load(st, "classifier", num_labels, h, true)?,
            }
        };
        Ok(Self {
            roberta,
            word: st.param(&format!("{p}.embeddings.word_embeddings.weight"), &[usize_of(cfg, "vocab_size")?, h])?,
            pos: st.param(
                &format!("{p}.embeddings.position_embeddings.weight"),
                &[usize_of(cfg, "max_position_embeddings")?, h],
            )?,
            types: st.param(
                &format!("{p}.embeddings.token_type_embeddings.weight"),
                &[usize_of(cfg, "type_vocab_size")?, h],
            )?,
            emb_ln: ln(&format!("{p}.embeddings.LayerNorm"))?,
            layers,
            head,
            heads: usize_of(cfg, "num_attention_heads")?,
            eps: cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12),
            act: cfg["hidden_act"].as_str().unwrap_or("gelu").to_string(),
            pad: cfg["pad_token_id"].as_u64().unwrap_or(if roberta { 1 } else { 0 }) as u32,
            hidden_drop: Dropout::new(cfg["hidden_dropout_prob"].as_f64().unwrap_or(0.1), 1),
            attn_drop: Dropout::new(cfg["attention_probs_dropout_prob"].as_f64().unwrap_or(0.1), 2),
            cls_drop: Dropout::new(
                cfg["classifier_dropout"].as_f64().or(cfg["hidden_dropout_prob"].as_f64()).unwrap_or(0.1),
                3,
            ),
        })
    }

    /// Override the dropout probabilities (`hidden`, `attention`, `classifier`) and seed.
    pub fn with_dropout(mut self, hidden: f64, attention: f64, classifier: f64, seed: u64) -> Self {
        self.hidden_drop = Dropout::new(hidden, seed);
        self.attn_drop = Dropout::new(attention, seed.wrapping_add(1));
        self.cls_drop = Dropout::new(classifier, seed.wrapping_add(2));
        self
    }

    /// Re-seed every dropout generator (called per epoch so a resumed run draws the same masks).
    pub fn reseed_dropout(&self, seed: u64) {
        for (i, d) in [&self.hidden_drop, &self.attn_drop, &self.cls_drop].into_iter().enumerate() {
            d.reseed(seed.wrapping_mul(31).wrapping_add(i as u64));
        }
    }

    /// Turn dropout on (training) or off (evaluation, the default).
    pub fn set_training(&self, on: bool) {
        for d in [&self.hidden_drop, &self.attn_drop, &self.cls_drop] {
            d.set_training(on);
        }
    }

    /// Logits `(batch, labels)` for encoded inputs (padded internally).
    pub fn logits(&self, batch: &[Vec<u32>], device: &Device) -> Result<Tensor> {
        let b = batch.len();
        let s = batch.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let mut ids = Vec::with_capacity(b * s);
        let mut pos = Vec::with_capacity(b * s);
        let mut mask = Vec::with_capacity(b * s);
        for seq in batch {
            for j in 0..s {
                let real = j < seq.len();
                ids.push(if real { seq[j] } else { self.pad });
                mask.push(if real { 0f32 } else { -1e9 });
                // RoBERTa positions start after the padding index; BERT counts from zero
                pos.push(if self.roberta { if real { self.pad + 1 + j as u32 } else { self.pad } } else { j as u32 });
            }
        }
        let h = self.word.dim(1)?;
        let idx = Tensor::from_vec(ids, b * s, device)?;
        let pidx = Tensor::from_vec(pos, b * s, device)?;
        let tidx = Tensor::zeros(b * s, DType::U32, device)?;
        let x = (self.word.index_select(&idx, 0)? + self.pos.index_select(&pidx, 0)?)?;
        let x = (x + self.types.index_select(&tidx, 0)?)?.reshape((b, s, h))?;
        let mut x = self.hidden_drop.apply(&layer_norm(&x, &self.emb_ln.0, &self.emb_ln.1, self.eps)?)?;
        let mask = Tensor::from_vec(mask, (b, 1, 1, s), device)?;
        let hd = h / self.heads;
        let split = |t: Tensor| -> Result<Tensor> { t.reshape((b, s, self.heads, hd))?.transpose(1, 2)?.contiguous() };
        for l in &self.layers {
            let q = split(l.q.forward(&x)?)?;
            let k = split(l.k.forward(&x)?)?;
            let v = split(l.v.forward(&x)?)?;
            let scores = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?.broadcast_add(&mask)?;
            let probs = self.attn_drop.apply(&softmax(&scores)?)?;
            let ctx = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b, s, h))?;
            let attn = self.hidden_drop.apply(&l.attn_out.forward(&ctx)?)?;
            x = layer_norm(&(attn + &x)?, &l.attn_ln.0, &l.attn_ln.1, self.eps)?;
            let ff = self.hidden_drop.apply(&l.out.forward(&act(&self.act, &l.inter.forward(&x)?)?)?)?;
            x = layer_norm(&(ff + &x)?, &l.out_ln.0, &l.out_ln.1, self.eps)?;
        }
        let cls = x.narrow(1, 0, 1)?.squeeze(1)?;
        match &self.head {
            Head::Roberta { dense, out_proj } => {
                let h = dense.forward(&self.cls_drop.apply(&cls)?)?.tanh()?;
                out_proj.forward(&self.cls_drop.apply(&h)?)
            }
            Head::Bert { pooler, classifier } => {
                classifier.forward(&self.cls_drop.apply(&pooler.forward(&cls)?.tanh()?)?)
            }
        }
    }
}

// ------------------------------------------------------------------ decoders

struct DecLayer {
    in_ln: Tensor,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    post_ln: Tensor,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// A Qwen2 / Llama causal LM with differentiable forward.
pub struct DecoderLm {
    embed: Tensor,
    layers: Vec<DecLayer>,
    norm: Tensor,
    lm_head: Lin,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
    act: String,
    inv_freq: Vec<f32>,
    lora_drop: Dropout,
}

fn inv_freq(cfg: &Value, head_dim: usize) -> Vec<f32> {
    let theta = cfg["rope_theta"].as_f64().unwrap_or(10_000.0);
    let base: Vec<f64> = (0..head_dim).step_by(2).map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64)).collect();
    let rs = &cfg["rope_scaling"];
    if rs["rope_type"].as_str().or(rs["type"].as_str()) != Some("llama3") {
        return base.into_iter().map(|f| f as f32).collect();
    }
    // Llama 3 frequency scaling
    let factor = rs["factor"].as_f64().unwrap_or(8.0);
    let low = rs["low_freq_factor"].as_f64().unwrap_or(1.0);
    let high = rs["high_freq_factor"].as_f64().unwrap_or(4.0);
    let orig = rs["original_max_position_embeddings"].as_f64().unwrap_or(8192.0);
    let (low_wl, high_wl) = (orig / low, orig / high);
    base.into_iter()
        .map(|f| {
            let wl = 2.0 * std::f64::consts::PI / f;
            let v = if wl < high_wl {
                f
            } else if wl > low_wl {
                f / factor
            } else {
                let smooth = (orig / wl - low) / (high - low);
                (1.0 - smooth) * f / factor + smooth * f
            };
            v as f32
        })
        .collect()
}

impl DecoderLm {
    /// Build from `config.json` values and the store (`qwen2`, `llama`, `mistral`).
    pub fn new(cfg: &Value, st: &TrainStore) -> Result<Self> {
        let qwen = cfg["model_type"].as_str() == Some("qwen2");
        let h = usize_of(cfg, "hidden_size")?;
        let heads = usize_of(cfg, "num_attention_heads")?;
        let kv_heads = cfg["num_key_value_heads"].as_u64().map(|x| x as usize).unwrap_or(heads);
        let head_dim = h / heads;
        let inter = usize_of(cfg, "intermediate_size")?;
        let vocab = usize_of(cfg, "vocab_size")?;
        let mut layers = Vec::new();
        for i in 0..usize_of(cfg, "num_hidden_layers")? {
            let l = format!("model.layers.{i}");
            layers.push(DecLayer {
                in_ln: st.param(&format!("{l}.input_layernorm.weight"), &[h])?,
                q: Lin::load(st, &format!("{l}.self_attn.q_proj"), heads * head_dim, h, qwen)?,
                k: Lin::load(st, &format!("{l}.self_attn.k_proj"), kv_heads * head_dim, h, qwen)?,
                v: Lin::load(st, &format!("{l}.self_attn.v_proj"), kv_heads * head_dim, h, qwen)?,
                o: Lin::load(st, &format!("{l}.self_attn.o_proj"), h, heads * head_dim, false)?,
                post_ln: st.param(&format!("{l}.post_attention_layernorm.weight"), &[h])?,
                gate: Lin::load(st, &format!("{l}.mlp.gate_proj"), inter, h, false)?,
                up: Lin::load(st, &format!("{l}.mlp.up_proj"), inter, h, false)?,
                down: Lin::load(st, &format!("{l}.mlp.down_proj"), h, inter, false)?,
            });
        }
        let embed = st.param("model.embed_tokens.weight", &[vocab, h])?;
        let lm_head = if st.has("lm_head.weight") {
            Lin::load(st, "lm_head", vocab, h, false)?
        } else {
            Lin { w: embed.clone(), b: None, lora: None }
        };
        Ok(Self {
            embed,
            layers,
            norm: st.param("model.norm.weight", &[h])?,
            lm_head,
            heads,
            kv_heads,
            head_dim,
            eps: cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            act: cfg["hidden_act"].as_str().unwrap_or("silu").to_string(),
            inv_freq: inv_freq(cfg, head_dim),
            lora_drop: Dropout::new(0.0, 0),
        })
    }

    /// LoRA dropout probability and seed (PEFT `lora_dropout`).
    pub fn with_lora_dropout(mut self, p: f64, seed: u64) -> Self {
        self.lora_drop = Dropout::new(p, seed);
        self
    }

    /// Re-seed the LoRA dropout generator.
    pub fn reseed_dropout(&self, seed: u64) {
        self.lora_drop.reseed(seed);
    }

    /// Turn LoRA dropout on (training) or off (evaluation, the default).
    pub fn set_training(&self, on: bool) {
        self.lora_drop.set_training(on);
    }

    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half = self.head_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        x.broadcast_mul(cos)? + rot.broadcast_mul(sin)?
    }

    /// Logits `(seq, vocab)` for every position of `ids`.
    pub fn logits(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        let s = ids.len();
        let h = self.embed.dim(1)?;
        let mut x = self.embed.index_select(&Tensor::new(ids, device)?, 0)?.reshape((1, s, h))?;
        let freqs: Vec<f32> = (0..s).flat_map(|p| self.inv_freq.iter().map(move |f| p as f32 * f)).collect();
        let freqs = Tensor::from_vec(freqs, (s, self.head_dim / 2), device)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
        let (cos, sin) =
            (freqs.cos()?.reshape((1, 1, s, self.head_dim))?, freqs.sin()?.reshape((1, 1, s, self.head_dim))?);
        let mask: Vec<f32> = (0..s).flat_map(|i| (0..s).map(move |j| if j > i { -1e9 } else { 0.0 })).collect();
        let mask = Tensor::from_vec(mask, (1, 1, s, s), device)?;
        let rep = self.heads / self.kv_heads;
        for l in &self.layers {
            let hn = rms_norm(&x, &l.in_ln, self.eps)?;
            let q =
                l.q.forward_d(&hn, Some(&self.lora_drop))?
                    .reshape((1, s, self.heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
            let k =
                l.k.forward_d(&hn, Some(&self.lora_drop))?
                    .reshape((1, s, self.kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
            let v =
                l.v.forward_d(&hn, Some(&self.lora_drop))?
                    .reshape((1, s, self.kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
            let q = self.rope(&q, &cos, &sin)?;
            let k = self.rope(&k, &cos, &sin)?;
            let expand = |t: Tensor| -> Result<Tensor> {
                if rep == 1 {
                    return Ok(t);
                }
                t.unsqueeze(2)?.broadcast_as((1, self.kv_heads, rep, s, self.head_dim))?.reshape((
                    1,
                    self.heads,
                    s,
                    self.head_dim,
                ))
            };
            let (k, v) = (expand(k)?, expand(v)?);
            let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?.broadcast_add(&mask)?;
            let ctx = softmax(&scores)?.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
                1,
                s,
                self.heads * self.head_dim,
            ))?;
            x = (x + l.o.forward_d(&ctx, Some(&self.lora_drop))?)?;
            let hn = rms_norm(&x, &l.post_ln, self.eps)?;
            let d = Some(&self.lora_drop);
            let ff = l.down.forward_d(&(act(&self.act, &l.gate.forward_d(&hn, d)?)? * l.up.forward_d(&hn, d)?)?, d)?;
            x = (x + ff)?;
        }
        self.lm_head.forward(&rms_norm(&x, &self.norm, self.eps)?)?.squeeze(0)
    }
}
