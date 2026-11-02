//! Decoder LLMs (Qwen2 and Llama families): generation with KV cache, sampling, stop sequences,
//! constrained decoding, per-token log-probabilities, chat templates and log-likelihood scoring.
//!
//! Generation is synchronous and CPU-bound. The [`Provider`] implementation runs it inside the
//! calling task; from a multi-threaded async runtime, prefer `tokio::task::spawn_blocking` with
//! an `Arc<TextGenerator>` for long generations.

use crate::loader::{ModelDir, model_err};
use crate::sampling::{Sampler, apply_penalties, log_softmax};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};
use candle_transformers::models::{llama, qwen2};
use harpia_core::traits::{
    ChatChunk, ChatRequest, ChatResponse, ChatStream, ChatTemplate, FinishReason, Generation, GenerationParams, Model,
    Provider, Tokenizer,
};
use harpia_core::types::{Message, TokenUsage};
use harpia_core::{HarpiaError, Result};
use harpia_data::template::JinjaChatTemplate;
use harpia_data::tokenize::HfTokenizer;
use harpia_structured::constraint::{Constraint, JsonSchemaConstraint, PrefixStatus};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

struct LlamaLm {
    model: llama::Llama,
    cfg: llama::Config,
    cache: llama::Cache,
    dtype: DType,
    device: Device,
}

enum Lm {
    Qwen2 { base: qwen2::Model, lm_head: Linear },
    Llama(Box<LlamaLm>),
}

impl Lm {
    fn reset(&mut self) -> candle_core::Result<()> {
        match self {
            Lm::Qwen2 { base, .. } => base.clear_kv_cache(),
            Lm::Llama(l) => l.cache = llama::Cache::new(true, l.dtype, &l.cfg, &l.device)?,
        }
        Ok(())
    }

    fn device(&self) -> Device {
        match self {
            Lm::Qwen2 { lm_head, .. } => lm_head.weight().device().clone(),
            Lm::Llama(l) => l.device.clone(),
        }
    }

    /// Logits of the last position after feeding `ids` starting at `offset`.
    fn forward_last(&mut self, ids: &[u32], offset: usize) -> candle_core::Result<Vec<f32>> {
        let x = Tensor::new(ids, &self.device())?.unsqueeze(0)?;
        match self {
            Lm::Qwen2 { base, lm_head } => {
                let h = base.forward(&x, offset, None)?;
                let last = h.narrow(1, ids.len() - 1, 1)?;
                lm_head.forward(&last)?.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()
            }
            Lm::Llama(l) => l.model.forward(&x, offset, &mut l.cache)?.squeeze(0)?.to_dtype(DType::F32)?.to_vec1(),
        }
    }

    /// Logits for every position of `ids` (fresh cache).
    fn forward_all(&mut self, ids: &[u32]) -> candle_core::Result<Vec<Vec<f32>>> {
        self.reset()?;
        match self {
            Lm::Qwen2 { base, lm_head } => {
                let x = Tensor::new(ids, &lm_head.weight().device().clone())?.unsqueeze(0)?;
                let h = base.forward(&x, 0, None)?;
                lm_head.forward(&h)?.squeeze(0)?.to_dtype(DType::F32)?.to_vec2()
            }
            Lm::Llama(_) => {
                // candle's Llama returns only the last position: feed tokens one by one
                let mut out = Vec::with_capacity(ids.len());
                for (i, &t) in ids.iter().enumerate() {
                    out.push(self.forward_last(&[t], i)?);
                }
                Ok(out)
            }
        }
    }
}

/// The bare causal LM (no tokenizer), for training: build it from any `VarBuilder` and get
/// differentiable logits for every position.
pub struct CausalLm(Lm);

impl CausalLm {
    /// Build the architecture named in `md` (`qwen2`, `llama`, `mistral`) from `vb`.
    pub fn from_var_builder(md: &ModelDir, vb: VarBuilder, dtype: DType, device: &Device) -> Result<Self> {
        let dir = md.root.as_path();
        let ce = |e: candle_core::Error| model_err(dir, e);
        let lm = match md.model_type() {
            "qwen2" => {
                let cfg: qwen2::Config = md.config_as()?;
                let base = qwen2::Model::new(&cfg, vb.clone()).map_err(ce)?;
                let lm_head = if vb.contains_tensor("lm_head.weight") {
                    linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head")).map_err(ce)?
                } else {
                    Linear::new(
                        vb.get((cfg.vocab_size, cfg.hidden_size), "model.embed_tokens.weight").map_err(ce)?,
                        None,
                    )
                };
                Lm::Qwen2 { base, lm_head }
            }
            "llama" | "mistral" => {
                let lc: llama::LlamaConfig = md.config_as()?;
                let cfg = lc.into_config(false);
                let model = llama::Llama::load(vb, &cfg).map_err(ce)?;
                let cache = llama::Cache::new(true, dtype, &cfg, device).map_err(ce)?;
                Lm::Llama(Box::new(LlamaLm { model, cfg, cache, dtype, device: device.clone() }))
            }
            other => return Err(model_err(dir, format!("unsupported decoder model_type `{other}` (qwen2, llama)"))),
        };
        Ok(Self(lm))
    }

    /// Logits `(seq, vocab)` for every position of `ids`, keeping the autograd graph. Qwen2 runs
    /// one pass; Llama feeds tokens one at a time (candle's Llama returns only the last position).
    pub fn logits_all(&mut self, ids: &[u32]) -> candle_core::Result<Tensor> {
        self.0.reset()?;
        match &mut self.0 {
            Lm::Qwen2 { base, lm_head } => {
                let x = Tensor::new(ids, &lm_head.weight().device().clone())?.unsqueeze(0)?;
                lm_head.forward(&base.forward(&x, 0, None)?)?.squeeze(0)
            }
            Lm::Llama(l) => {
                let mut rows = Vec::with_capacity(ids.len());
                for (i, &t) in ids.iter().enumerate() {
                    let x = Tensor::new(&[t], &l.device)?.unsqueeze(0)?;
                    rows.push(l.model.forward(&x, i, &mut l.cache)?.squeeze(0)?);
                }
                Tensor::stack(&rows, 0)
            }
        }
    }
}

/// A local causal language model.
///
/// ```no_run
/// use harpia_local::TextGenerator;
/// use harpia_core::traits::{GenerationParams, Model};
/// let lm = TextGenerator::load("models/Qwen2.5-0.5B-Instruct".as_ref())?;
/// let out = lm.generate("The harpy eagle", &GenerationParams { max_new_tokens: 20, ..Default::default() })?;
/// println!("{}", out.text);
/// # Ok::<(), harpia_core::HarpiaError>(())
/// ```
pub struct TextGenerator {
    lm: Mutex<Lm>,
    tokenizer: HfTokenizer,
    template: Option<JinjaChatTemplate>,
    eos: Vec<u32>,
    id: String,
    dir: PathBuf,
    vocab: OnceLock<Vec<String>>,
}

impl std::fmt::Debug for TextGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextGenerator").field("id", &self.id).field("eos", &self.eos).finish_non_exhaustive()
    }
}

impl TextGenerator {
    /// Load on the CPU in f32.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with(dir, &Device::Cpu, DType::F32)
    }

    /// Load with an explicit device and dtype (e.g. `DType::BF16` on GPU).
    pub fn load_with(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        Self::load_with_adapter(dir, None, device, dtype)
    }

    /// Load with a PEFT-format LoRA adapter (`adapter_config.json` +
    /// `adapter_model.safetensors`, as written by `harpia-train`) merged into the weights.
    pub fn load_with_adapter(dir: &Path, adapter: Option<&Path>, device: &Device, dtype: DType) -> Result<Self> {
        let md = ModelDir::open(dir)?;
        let mut tensors = md.tensors(device)?;
        if let Some(a) = adapter {
            crate::adapter::merge_lora(&mut tensors, a, device)?;
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let lm = CausalLm::from_var_builder(&md, vb, dtype, device)?.0;
        let tokenizer = md.tokenizer()?;
        let mut eos = md.token_ids("eos_token_id");
        if let Ok(gc) = std::fs::read_to_string(dir.join("generation_config.json"))
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&gc)
        {
            let gmd = ModelDir { root: dir.to_path_buf(), config: v };
            eos.extend(gmd.token_ids("eos_token_id"));
        }
        let tc = md.tokenizer_config();
        if let Some(tok) = tc.as_ref().and_then(|c| {
            c["eos_token"].as_str().map(String::from).or_else(|| c["eos_token"]["content"].as_str().map(String::from))
        }) && let Some(id) = tokenizer.token_to_id(&tok)
        {
            eos.push(id);
        }
        eos.sort_unstable();
        eos.dedup();
        let template = if tc.as_ref().is_some_and(|c| !c["chat_template"].is_null()) {
            Some(JinjaChatTemplate::from_tokenizer_config(dir.join("tokenizer_config.json"))?)
        } else {
            None
        };
        Ok(Self {
            lm: Mutex::new(lm),
            tokenizer,
            template,
            eos,
            id: dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "local".into()),
            dir: dir.to_path_buf(),
            vocab: OnceLock::new(),
        })
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &HfTokenizer {
        &self.tokenizer
    }

    /// End-of-sequence token ids.
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos
    }

    /// Render `messages` with the model's chat template (generation prompt appended).
    pub fn render_chat(&self, messages: &[Message]) -> Result<String> {
        let t =
            self.template.as_ref().ok_or_else(|| model_err(&self.dir, "tokenizer_config.json has no chat_template"))?;
        t.render(messages, true)
    }

    fn vocab(&self) -> &[String] {
        self.vocab.get_or_init(|| self.tokenizer.vocab_strings())
    }

    /// Generate from token ids, calling `on_token(text_delta)` as tokens arrive. With
    /// `constraint`, only continuations that keep the output a valid prefix are allowed, and
    /// end-of-sequence is allowed only once the output is complete.
    pub fn generate_with(
        &self,
        prompt_ids: &[u32],
        params: &GenerationParams,
        constraint: Option<&dyn Constraint>,
        mut on_token: impl FnMut(&str),
    ) -> Result<Generation> {
        if prompt_ids.is_empty() {
            return Err(HarpiaError::invalid("empty prompt"));
        }
        let ce = |e: candle_core::Error| model_err(&self.dir, e);
        let mut lm = self.lm.lock().expect("model lock");
        lm.reset().map_err(ce)?;
        let mut logits = lm.forward_last(prompt_ids, 0).map_err(ce)?;
        let prompt_len = prompt_ids.len();
        let mut sampler = Sampler::new(params);
        let mut generated: Vec<u32> = Vec::new();
        let mut logprobs = Vec::new();
        let mut text = String::new();
        let mut finish = FinishReason::Length;
        let vocab = if constraint.is_some() { Some(self.vocab()) } else { None };
        for step in 0..params.max_new_tokens as usize {
            let raw = log_softmax(&logits);
            apply_penalties(&mut logits, &generated, params);
            let choice = match (constraint, vocab) {
                (Some(c), Some(v)) => {
                    let complete = c.check(&text) == PrefixStatus::Complete;
                    let eos = &self.eos;
                    let allowed = |t: u32| {
                        if eos.contains(&t) {
                            return complete;
                        }
                        let piece = v.get(t as usize).map(String::as_str).unwrap_or("");
                        !piece.is_empty() && c.check(&format!("{text}{piece}")) != PrefixStatus::Invalid
                    };
                    sampler.choose(&logits, Some(&allowed))
                }
                _ => sampler.choose(&logits, None),
            };
            let Some(tok) = choice else {
                finish = FinishReason::Stop;
                break;
            };
            if self.eos.contains(&tok) {
                finish = FinishReason::Stop;
                break;
            }
            generated.push(tok);
            logprobs.push(raw[tok as usize]);
            let new_text = self.tokenizer.decode(&generated, true)?;
            if let Some(delta) = new_text.strip_prefix(text.as_str()) {
                on_token(delta);
            }
            text = new_text;
            if let Some(stop) = params.stop.iter().find(|s| !s.is_empty() && text.ends_with(s.as_str())) {
                text.truncate(text.len() - stop.len());
                finish = FinishReason::Stop;
                break;
            }
            logits = lm.forward_last(&[tok], prompt_len + step).map_err(ce)?;
        }
        Ok(Generation {
            text,
            usage: TokenUsage {
                input_tokens: prompt_ids.len() as u64,
                output_tokens: generated.len() as u64,
                cached_input_tokens: 0,
            },
            token_ids: generated,
            logprobs,
            finish_reason: finish,
        })
    }

    /// Generate text constrained by `constraint` (JSON Schema, regex, choices).
    pub fn generate_constrained(
        &self,
        prompt: &str,
        params: &GenerationParams,
        constraint: &dyn Constraint,
    ) -> Result<Generation> {
        let ids = self.tokenizer.encode(prompt, true)?.ids;
        self.generate_with(&ids, params, Some(constraint), |_| {})
    }

    fn chat_generation(&self, request: &ChatRequest, on_token: impl FnMut(&str)) -> Result<Generation> {
        let prompt = self.render_chat(&request.messages)?;
        let ids = self.tokenizer.encode(&prompt, false)?.ids;
        let constraint = request.response_schema.as_ref().map(JsonSchemaConstraint::new).transpose()?;
        self.generate_with(&ids, &request.params, constraint.as_ref().map(|c| c as &dyn Constraint), on_token)
    }

    fn to_response(&self, request: &ChatRequest, g: Generation) -> ChatResponse {
        let mut message = Message::assistant(g.text);
        if !request.tools.is_empty() {
            message.tool_calls = harpia_structured::toolcall::parse_tool_calls(&message.content);
        }
        let finish = if message.tool_calls.is_empty() { g.finish_reason } else { FinishReason::ToolCalls };
        ChatResponse { model: self.id.clone(), message, finish_reason: finish, usage: g.usage }
    }
}

impl Model for TextGenerator {
    fn id(&self) -> &str {
        &self.id
    }

    fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<Generation> {
        let ids = self.tokenizer.encode(prompt, true)?.ids;
        self.generate_with(&ids, params, None, |_| {})
    }

    /// Sum of log-probabilities of `continuation`'s tokens given `context` (tokenized
    /// separately and concatenated, as in lm-eval-harness).
    fn loglikelihood(&self, context: &str, continuation: &str) -> Result<f64> {
        let ctx = self.tokenizer.encode(context, true)?.ids;
        let cont = self.tokenizer.encode(continuation, false)?.ids;
        if ctx.is_empty() {
            return Err(HarpiaError::invalid("loglikelihood needs a non-empty context"));
        }
        let full: Vec<u32> = ctx.iter().chain(&cont).copied().collect();
        let mut lm = self.lm.lock().expect("model lock");
        let logits = lm.forward_all(&full).map_err(|e| model_err(&self.dir, e))?;
        let mut total = 0.0f64;
        for i in ctx.len()..full.len() {
            total += f64::from(log_softmax(&logits[i - 1])[full[i] as usize]);
        }
        Ok(total)
    }
}

impl Provider for TextGenerator {
    fn name(&self) -> &str {
        "local"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let g = self.chat_generation(request, |_| {})?;
        Ok(self.to_response(request, g))
    }

    /// Generates the whole reply, then yields one chunk per decoded piece followed by a final
    /// chunk with usage. Use [`TextGenerator::generate_with`] for live token callbacks.
    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let mut pieces = Vec::new();
        let g = self.chat_generation(request, |d| pieces.push(d.to_string()))?;
        let resp = self.to_response(request, g);
        let mut chunks: Vec<Result<ChatChunk>> =
            pieces.into_iter().map(|d| Ok(ChatChunk { delta: d, ..Default::default() })).collect();
        chunks.push(Ok(ChatChunk {
            delta: String::new(),
            tool_calls: resp.message.tool_calls,
            usage: Some(resp.usage),
            finish_reason: Some(resp.finish_reason),
        }));
        Ok(Box::pin(futures_util_stream(chunks)))
    }
}

fn futures_util_stream(items: Vec<Result<ChatChunk>>) -> impl futures_core::Stream<Item = Result<ChatChunk>> + Send {
    struct Iter(std::vec::IntoIter<Result<ChatChunk>>);
    impl futures_core::Stream for Iter {
        type Item = Result<ChatChunk>;
        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Ready(self.0.next())
        }
    }
    Iter(items.into_iter())
}
