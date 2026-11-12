//! The model registry and method dispatch shared by `serve --stdio` and `serve --http`.
//!
//! Models named on the command line are loaded once at startup and stay loaded. Over stdio the
//! client (the parent process) may also name a model directory by path; it is loaded on first
//! use and cached. Over HTTP only the names registered on the command line are accepted.

use harpia::core::traits::{
    Embedder, FinishReason, Generation, GenerationParams, PairClassifier, Reranker, Tokenizer, Translator,
};
use harpia::core::types::ToolCall;
use harpia::core::{HarpiaError, Message, NetworkPolicy, Role, TokenUsage};
use harpia::data::chunk::RecursiveChunker;
use harpia::data::filter::{QualityFilter, QualityThresholds};
use harpia::data::rag::RagPipeline;
use harpia::local::{EncoderEmbedder, MarianTranslator, Search, SequenceClassifier, TextGenerator};
use harpia::structured::constraint::{Choice, Constraint, JsonSchemaConstraint, RegexConstraint};
use harpia::text::Lang;
use harpia::verify::{
    ClaimVerifier, FetchConfig, Fetcher, Mode, ReferenceConfig, ReferenceQuery, ReferenceVerifier, SourceRegistry,
    VerifyConfig,
};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// Every method served over JSON-RPC, in documentation order.
pub const METHODS: &[&str] = &[
    "info",
    "generate",
    "chat",
    "embed",
    "rerank",
    "classify",
    "translate",
    "retrieve",
    "verify.claim",
    "verify.answer",
    "verify.reference",
    "data.filter",
    "structured.repair",
    "shutdown",
];

// ---------------------------------------------------------------- errors

/// A JSON-RPC error object.
#[derive(Debug, Clone)]
pub struct RpcError {
    /// JSON-RPC error code.
    pub code: i64,
    /// Message.
    pub message: String,
    /// Extra data (`{"kind": ...}` for Harpia errors).
    pub data: Option<Value>,
}

/// Standard and Harpia-specific error codes (see `docs/serve-protocol.md`).
pub mod code {
    /// Invalid JSON.
    pub const PARSE: i64 = -32700;
    /// Not a valid request object.
    pub const INVALID_REQUEST: i64 = -32600;
    /// Unknown method.
    pub const METHOD_NOT_FOUND: i64 = -32601;
    /// Bad or missing parameters.
    pub const INVALID_PARAMS: i64 = -32602;
    /// Any other Harpia error.
    pub const HARPIA: i64 = -32000;
    /// Network access refused by the offline policy.
    pub const OFFLINE: i64 = -32001;
    /// A model failed to load or run.
    pub const MODEL: i64 = -32002;
    /// The requested model, index or feature is not available on this server.
    pub const NOT_AVAILABLE: i64 = -32003;
    /// No JSON could be extracted (structured.repair).
    pub const NO_JSON: i64 = -32004;
}

impl RpcError {
    /// New error.
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), data: None }
    }

    /// Invalid parameters.
    pub fn params(message: impl Into<String>) -> Self {
        Self::new(code::INVALID_PARAMS, message)
    }

    /// Not available on this server.
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(code::NOT_AVAILABLE, message)
    }

    /// The JSON-RPC `error` member.
    pub fn to_json(&self) -> Value {
        let mut v = json!({"code": self.code, "message": self.message});
        if let Some(d) = &self.data {
            v["data"] = d.clone();
        }
        v
    }
}

impl From<HarpiaError> for RpcError {
    fn from(e: HarpiaError) -> Self {
        let (code, kind) = match &e {
            HarpiaError::InvalidArgument(_) => (code::INVALID_PARAMS, "invalid_argument"),
            HarpiaError::Validation(_) => (code::INVALID_PARAMS, "validation"),
            HarpiaError::Json(_) => (code::INVALID_PARAMS, "json"),
            HarpiaError::NetworkDisabled { .. } => (code::OFFLINE, "network_disabled"),
            HarpiaError::Model { .. } => (code::MODEL, "model"),
            HarpiaError::Io { .. } => (code::HARPIA, "io"),
            HarpiaError::Tokenizer(_) => (code::HARPIA, "tokenizer"),
            HarpiaError::Template(_) => (code::HARPIA, "template"),
            _ => (code::HARPIA, "other"),
        };
        Self { code, message: e.to_string(), data: Some(json!({"kind": kind})) }
    }
}

impl From<serde_json::Error> for RpcError {
    fn from(e: serde_json::Error) -> Self {
        Self::params(e.to_string())
    }
}

type RpcResult<T> = std::result::Result<T, RpcError>;

// ---------------------------------------------------------------- model slots

type Loader<T> = Box<dyn Fn(&Path) -> harpia::core::Result<T> + Send + Sync>;

/// Models of one kind: registered names plus a cache of everything loaded so far.
pub struct Slot<T> {
    kind: &'static str,
    named: Vec<(String, PathBuf)>,
    cache: Mutex<HashMap<PathBuf, Arc<T>>>,
    load: Loader<T>,
}

impl<T> Slot<T> {
    fn new(kind: &'static str, named: Vec<(String, PathBuf)>, load: Loader<T>) -> Self {
        Self { kind, named, cache: Mutex::new(HashMap::new()), load }
    }

    /// Registered names, default first.
    pub fn names(&self) -> Vec<String> {
        self.named.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Load every registered model now (startup), so configuration errors surface early.
    fn preload(&self) -> harpia::core::Result<()> {
        for (_, p) in &self.named {
            self.get_path(p)?;
        }
        Ok(())
    }

    fn get_path(&self, path: &Path) -> harpia::core::Result<Arc<T>> {
        let mut cache = self.cache.lock().expect("model cache lock");
        if let Some(m) = cache.get(path) {
            return Ok(m.clone());
        }
        let m = Arc::new((self.load)(path)?);
        cache.insert(path.to_path_buf(), m.clone());
        Ok(m)
    }

    /// Resolve `name` (`None`, `""` or `"default"` → the first registered model). Unknown names
    /// are treated as directories when `allow_paths`.
    pub fn resolve(&self, name: Option<&str>, allow_paths: bool) -> RpcResult<(String, Arc<T>)> {
        let (label, path) = match name {
            None | Some("" | "default") => {
                let (n, p) = self.named.first().ok_or_else(|| {
                    RpcError::unavailable(format!(
                        "no {} configured (start the server with --{})",
                        self.kind, self.kind
                    ))
                })?;
                (n.clone(), p.clone())
            }
            Some(n) => match self.named.iter().find(|(k, _)| k == n) {
                Some((k, p)) => (k.clone(), p.clone()),
                None if allow_paths && Path::new(n).is_dir() => {
                    let p = PathBuf::from(n);
                    let label = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| n.into());
                    (label, p)
                }
                None => {
                    let hint = if allow_paths { "; not a directory either" } else { "" };
                    return Err(RpcError::unavailable(format!(
                        "unknown {} `{n}` (available: {}{hint})",
                        self.kind,
                        self.names().join(", ")
                    )));
                }
            },
        };
        Ok((label, self.get_path(&path)?))
    }
}

/// `[name=]dir` → (name, dir); the name defaults to the directory's last component.
pub fn parse_spec(spec: &str) -> (String, PathBuf) {
    match spec.split_once('=') {
        Some((n, p)) if !n.is_empty() && !n.contains(['/', '\\']) => (n.to_string(), PathBuf::from(p)),
        _ => {
            let p = PathBuf::from(spec);
            let n = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| spec.to_string());
            (n, p)
        }
    }
}

// ---------------------------------------------------------------- engine

/// Server configuration (from the command line).
#[derive(Debug, Clone, Default)]
pub struct EngineConfig {
    /// Decoder models.
    pub decoders: Vec<String>,
    /// Embedding models.
    pub embedders: Vec<String>,
    /// Sequence classifiers (also used for NLI in verification).
    pub classifiers: Vec<String>,
    /// Cross-encoder rerankers.
    pub rerankers: Vec<String>,
    /// MarianMT translators.
    pub translators: Vec<String>,
    /// RAG indexes saved with `RagPipeline::save`.
    pub rag: Vec<String>,
    /// Rerank RAG results with the default reranker.
    pub rag_rerank: bool,
    /// Trusted-source registry (TOML); the built-in registry when absent.
    pub sources: Option<PathBuf>,
    /// Allow outbound network for verification.
    pub online: bool,
    /// Verification content store.
    pub store: Option<PathBuf>,
    /// Verification store mode.
    pub mode: Mode,
    /// Default language (`en`, `pt-br`).
    pub lang: String,
}

/// Loaded models and verification state, shared by every request.
pub struct Engine {
    /// Decoders.
    pub decoders: Slot<TextGenerator>,
    /// Embedders.
    pub embedders: Slot<EncoderEmbedder>,
    /// Classifiers.
    pub classifiers: Slot<SequenceClassifier>,
    /// Rerankers.
    pub rerankers: Slot<SequenceClassifier>,
    /// Translators (locked: prefix and beam width are set per request).
    pub translators: Slot<Mutex<MarianTranslator>>,
    /// RAG indexes.
    pub rag: Slot<RagPipeline<'static>>,
    cfg: EngineConfig,
    fetcher: OnceLock<std::result::Result<Arc<Fetcher>, String>>,
}

/// Owns an `Arc` so a long-lived `RagPipeline<'static>` can borrow the embedder.
struct SharedEmbedder(Arc<EncoderEmbedder>);

impl Embedder for SharedEmbedder {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn embed(&self, texts: &[&str]) -> harpia::core::Result<Vec<Vec<f32>>> {
        self.0.embed(texts)
    }
}

struct SharedReranker(Arc<SequenceClassifier>);

impl Reranker for SharedReranker {
    fn rerank(&self, query: &str, documents: &[&str]) -> harpia::core::Result<Vec<f32>> {
        self.0.rerank(query, documents)
    }
}

impl Engine {
    /// Build the engine and load every model and index named in `cfg`.
    pub fn new(cfg: EngineConfig) -> harpia::core::Result<Self> {
        if Lang::parse(&cfg.lang).is_none() {
            return Err(HarpiaError::invalid(format!("unknown language `{}` (use en or pt-br)", cfg.lang)));
        }
        let specs = |v: &[String]| v.iter().map(|s| parse_spec(s)).collect::<Vec<_>>();
        let embedders = Slot::new("embedder", specs(&cfg.embedders), Box::new(EncoderEmbedder::load));
        embedders.preload()?;
        let rerankers = Slot::new("reranker", specs(&cfg.rerankers), Box::new(SequenceClassifier::load));
        rerankers.preload()?;
        let rag_loader: Loader<RagPipeline<'static>> = if cfg.rag.is_empty() {
            Box::new(|_| Err(HarpiaError::invalid("no RAG index configured")))
        } else {
            let (_, emb) = embedders
                .resolve(None, false)
                .map_err(|_| HarpiaError::invalid("--rag needs an --embedder (the one the index was built with)"))?;
            // Leaked once per server: the pipelines live as long as the process.
            let emb: &'static dyn Embedder = Box::leak(Box::new(SharedEmbedder(emb)));
            let rr: Option<&'static dyn Reranker> = if cfg.rag_rerank {
                let (_, r) = rerankers
                    .resolve(None, false)
                    .map_err(|_| HarpiaError::invalid("--rag-rerank needs a --reranker"))?;
                Some(Box::leak(Box::new(SharedReranker(r))))
            } else {
                None
            };
            Box::new(move |p: &Path| {
                let pipe = RagPipeline::load(p, Box::new(RecursiveChunker::new(1200)), emb)?;
                Ok(match rr {
                    Some(r) => pipe.with_reranker(r),
                    None => pipe,
                })
            })
        };
        let engine = Self {
            decoders: Slot::new("decoder", specs(&cfg.decoders), Box::new(TextGenerator::load)),
            embedders,
            classifiers: Slot::new("classifier", specs(&cfg.classifiers), Box::new(SequenceClassifier::load)),
            rerankers,
            translators: Slot::new(
                "translator",
                specs(&cfg.translators),
                Box::new(|p| MarianTranslator::load(p).map(Mutex::new)),
            ),
            rag: Slot::new("rag", specs(&cfg.rag), rag_loader),
            cfg,
            fetcher: OnceLock::new(),
        };
        engine.decoders.preload()?;
        engine.classifiers.preload()?;
        engine.translators.preload()?;
        engine.rag.preload()?;
        Ok(engine)
    }

    fn fetcher(&self) -> RpcResult<Arc<Fetcher>> {
        self.fetcher
            .get_or_init(|| {
                let registry = match &self.cfg.sources {
                    Some(p) => SourceRegistry::load(p).map_err(|e| e.to_string())?,
                    None => SourceRegistry::builtin(),
                };
                let fc = FetchConfig {
                    policy: if self.cfg.online { NetworkPolicy::Online } else { NetworkPolicy::Offline },
                    mode: self.cfg.mode,
                    store: self.cfg.store.clone(),
                    ..FetchConfig::default()
                };
                Fetcher::new(registry, fc).map(Arc::new).map_err(|e| e.to_string())
            })
            .clone()
            .map_err(|e| RpcError::new(code::HARPIA, e))
    }

    /// `info`: version, methods, models and network policy.
    pub fn info(&self) -> Value {
        json!({
            "name": "harpia",
            "version": env!("CARGO_PKG_VERSION"),
            "protocol": "jsonrpc-2.0",
            "methods": METHODS,
            "models": {
                "decoders": self.decoders.names(),
                "embedders": self.embedders.names(),
                "classifiers": self.classifiers.names(),
                "rerankers": self.rerankers.names(),
                "translators": self.translators.names(),
                "rag": self.rag.names(),
            },
            "network": if self.cfg.online { "online" } else { "offline" },
        })
    }

    /// Dispatch one JSON-RPC call. `on_delta` receives generated text as it arrives when the
    /// call asks for `"stream": true`.
    pub async fn call(
        &self,
        method: &str,
        params: &Value,
        allow_paths: bool,
        on_delta: &mut dyn FnMut(&str),
    ) -> RpcResult<Value> {
        let empty = json!({});
        let p = if params.is_null() { &empty } else { params };
        if !p.is_object() {
            return Err(RpcError::params("params must be an object"));
        }
        match method {
            "info" => Ok(self.info()),
            "generate" => self.generate(p, allow_paths, on_delta),
            "chat" => self.chat(p, allow_paths, on_delta),
            "embed" => self.embed(p, allow_paths),
            "rerank" => self.rerank(p, allow_paths),
            "classify" => self.classify(p, allow_paths),
            "translate" => self.translate(p, allow_paths),
            "retrieve" => self.retrieve(p, allow_paths),
            "verify.claim" => self.verify_claim(p, allow_paths).await,
            "verify.answer" => self.verify_answer(p, allow_paths).await,
            "verify.reference" => self.verify_reference(p).await,
            "data.filter" => self.data_filter(p),
            "structured.repair" => structured_repair(p),
            "shutdown" => Ok(Value::Null),
            other => Err(RpcError::new(code::METHOD_NOT_FOUND, format!("unknown method `{other}`"))),
        }
    }

    // ------------------------------------------------------------ use

    fn generate(&self, p: &Value, allow_paths: bool, on_delta: &mut dyn FnMut(&str)) -> RpcResult<Value> {
        let (name, lm) = self.decoders.resolve(opt_str(p, "model")?, allow_paths)?;
        let prompt = req_str(p, "prompt")?;
        let ids = if p.get("chat").and_then(Value::as_bool).unwrap_or(false) {
            chat_ids(&lm, &[Message::user(prompt)])?
        } else {
            lm.tokenizer().encode(prompt, true).map_err(RpcError::from)?.ids
        };
        let params = gen_params(p, lm.default_params())?;
        let constraint = constraint(p)?;
        let g = run(&lm, &ids, &params, constraint.as_deref(), stream_flag(p), on_delta)?;
        let mut out = json!({
            "model": name,
            "text": g.text,
            "finish_reason": g.finish_reason,
            "usage": g.usage,
        });
        if p.get("logprobs").and_then(Value::as_bool).unwrap_or(false) {
            out["token_ids"] = json!(g.token_ids);
            out["logprobs"] = json!(g.logprobs);
        }
        Ok(out)
    }

    fn chat(&self, p: &Value, allow_paths: bool, on_delta: &mut dyn FnMut(&str)) -> RpcResult<Value> {
        let (name, lm) = self.decoders.resolve(opt_str(p, "model")?, allow_paths)?;
        let messages = parse_messages(p.get("messages").ok_or_else(|| RpcError::params("missing `messages`"))?)?;
        let ids = chat_ids(&lm, &messages)?;
        let params = gen_params(p, lm.default_params())?;
        let constraint = constraint(p)?;
        let g = run(&lm, &ids, &params, constraint.as_deref(), stream_flag(p), on_delta)?;
        let wants_tools = p.get("tools").and_then(Value::as_array).is_some_and(|t| !t.is_empty());
        let reply = to_reply(g, wants_tools);
        Ok(json!({
            "model": name,
            "message": reply.message,
            "finish_reason": reply.finish_reason,
            "usage": reply.usage,
        }))
    }

    fn embed(&self, p: &Value, allow_paths: bool) -> RpcResult<Value> {
        let (name, emb) = self.embedders.resolve(opt_str(p, "model")?, allow_paths)?;
        let texts = str_list(p, &["texts", "input"])?;
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vecs = emb.embed(&refs)?;
        Ok(json!({"model": name, "dim": emb.dim(), "embeddings": vecs}))
    }

    fn rerank(&self, p: &Value, allow_paths: bool) -> RpcResult<Value> {
        let (name, rr) = self.rerankers.resolve(opt_str(p, "model")?, allow_paths)?;
        let query = req_str(p, "query")?;
        let docs = document_texts(p.get("documents").ok_or_else(|| RpcError::params("missing `documents`"))?)?;
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let scores = rr.rerank(query, &refs)?;
        let top_n = p.get("top_n").and_then(Value::as_u64).map(|n| n as usize);
        Ok(json!({"model": name, "results": rank(&scores, top_n)}))
    }

    fn classify(&self, p: &Value, allow_paths: bool) -> RpcResult<Value> {
        let (name, clf) = self.classifiers.resolve(opt_str(p, "model")?, allow_paths)?;
        let scores = if let Some(pairs) = p.get("pairs") {
            let pairs: Vec<(String, String)> = serde_json::from_value(pairs.clone())
                .map_err(|e| RpcError::params(format!("`pairs` must be [[premise, hypothesis], ...]: {e}")))?;
            let refs: Vec<(&str, &str)> = pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            clf.classify_pairs(&refs)?
        } else {
            let texts = str_list(p, &["texts", "input"])?;
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            clf.classify(&refs)?
        };
        let thresholds = p.get("thresholds").and_then(Value::as_object);
        let fallback = opt_str(p, "fallback")?.unwrap_or("neutral");
        let results: Vec<Value> = scores
            .iter()
            .map(|s| {
                let mut ranked: Vec<(&str, f32)> = s.iter().collect();
                ranked.sort_by(|x, y| y.1.total_cmp(&x.1));
                let label = match thresholds {
                    None => ranked.first().map(|r| r.0).unwrap_or(fallback),
                    Some(t) => ranked
                        .iter()
                        .find(|(l, pr)| t.get(*l).and_then(Value::as_f64).is_some_and(|th| f64::from(*pr) >= th))
                        .map(|r| r.0)
                        .unwrap_or(fallback),
                };
                let map: serde_json::Map<String, Value> = s.iter().map(|(l, pr)| (l.to_string(), json!(pr))).collect();
                json!({"label": label, "scores": map})
            })
            .collect();
        Ok(json!({"model": name, "labels": clf.labels(), "results": results}))
    }

    fn translate(&self, p: &Value, allow_paths: bool) -> RpcResult<Value> {
        let (name, mt) = self.translators.resolve(opt_str(p, "model")?, allow_paths)?;
        let texts = str_list(p, &["texts", "input"])?;
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let beams = p.get("beams").and_then(Value::as_u64).unwrap_or(1) as usize;
        let mut mt = mt.lock().expect("translator lock");
        mt.prefix = opt_str(p, "prefix")?.map(String::from);
        mt.search = if beams > 1 { Search::Beam { width: beams } } else { Search::Greedy };
        let out = mt.translate(&refs)?;
        Ok(json!({"model": name, "translations": out}))
    }

    fn retrieve(&self, p: &Value, allow_paths: bool) -> RpcResult<Value> {
        let (name, rag) = self.rag.resolve(opt_str(p, "index")?, allow_paths)?;
        let query = req_str(p, "query")?;
        let filter = match p.get("filter") {
            Some(f) => Some(serde_json::from_value(f.clone())?),
            None => None,
        };
        let mut hits = rag.retrieve(query, filter.as_ref())?;
        if let Some(k) = p.get("k").and_then(Value::as_u64) {
            hits.truncate(k as usize);
        }
        let mut out = json!({"index": name, "hits": hits});
        if let Some(budget) = p.get("budget").and_then(Value::as_u64) {
            let counter: Box<dyn Fn(&str) -> usize> = match opt_str(p, "tokenizer")? {
                Some(dec) => {
                    let (_, lm) = self.decoders.resolve(Some(dec), allow_paths)?;
                    Box::new(move |t: &str| lm.tokenizer().count(t).unwrap_or_else(|_| t.split_whitespace().count()))
                }
                None => Box::new(|t: &str| t.split_whitespace().count()),
            };
            out["context"] = json!(harpia::data::rag::assemble(&hits, budget as usize, &*counter));
        }
        Ok(out)
    }

    // ------------------------------------------------------------ verify

    fn verify_config(&self, p: &Value) -> RpcResult<VerifyConfig> {
        let lang = opt_str(p, "lang")?.unwrap_or(&self.cfg.lang);
        if Lang::parse(lang).is_none() {
            return Err(RpcError::params(format!("unknown language `{lang}` (use en or pt-br)")));
        }
        let mut vc = VerifyConfig { lang: lang.to_string(), ..VerifyConfig::default() };
        if let Some(s) = p.get("sources") {
            vc.sources = Some(serde_json::from_value(s.clone())?);
        }
        Ok(vc)
    }

    async fn verify_claim(&self, p: &Value, allow_paths: bool) -> RpcResult<Value> {
        let claim = req_str(p, "claim")?;
        let queries: Vec<String> = match p.get("queries") {
            Some(q) => serde_json::from_value(q.clone())?,
            None => Vec::new(),
        };
        let (_, clf) = self.classifiers.resolve(opt_str(p, "classifier")?, allow_paths)?;
        let fetcher = self.fetcher()?;
        let before = fetcher.disclosure().entries.len();
        let v = ClaimVerifier::new(&fetcher, &*clf, self.verify_config(p)?).verify_claim(claim, &queries).await;
        let disclosure = &fetcher.disclosure().entries[before..];
        Ok(json!({"verification": v, "disclosure": disclosure}))
    }

    async fn verify_answer(&self, p: &Value, allow_paths: bool) -> RpcResult<Value> {
        let answer = req_str(p, "answer")?;
        let (_, clf) = self.classifiers.resolve(opt_str(p, "classifier")?, allow_paths)?;
        let fetcher = self.fetcher()?;
        let report = ClaimVerifier::new(&fetcher, &*clf, self.verify_config(p)?).verify_answer(answer).await;
        Ok(serde_json::to_value(report)?)
    }

    async fn verify_reference(&self, p: &Value) -> RpcResult<Value> {
        let refs: Vec<ReferenceQuery> = match (p.get("reference"), p.get("references")) {
            (Some(r), None) => vec![serde_json::from_value(r.clone())?],
            (None, Some(rs)) => serde_json::from_value(rs.clone())?,
            _ => return Err(RpcError::params("pass exactly one of `reference` or `references`")),
        };
        let fetcher = self.fetcher()?;
        let before = fetcher.disclosure().entries.len();
        let rv = ReferenceVerifier::new(&fetcher, ReferenceConfig::default());
        let mut results = Vec::with_capacity(refs.len());
        for r in &refs {
            results.push(rv.verify(r).await);
        }
        let disclosure = &fetcher.disclosure().entries[before..];
        Ok(json!({"results": results, "disclosure": disclosure}))
    }

    // ------------------------------------------------------------ data

    fn data_filter(&self, p: &Value) -> RpcResult<Value> {
        let texts = str_list(p, &["texts", "input"])?;
        let thresholds = match (p.get("thresholds"), opt_str(p, "preset")?) {
            (Some(t), _) => serde_json::from_value::<QualityThresholds>(t.clone())?,
            (None, Some(name)) => QualityThresholds::preset(name)?,
            (None, None) => QualityThresholds::preset("minimal")?,
        };
        let lang = opt_str(p, "lang")?.unwrap_or(&self.cfg.lang);
        let lang = Lang::parse(lang).ok_or_else(|| RpcError::params(format!("unknown language `{lang}`")))?;
        let filter = QualityFilter::new(thresholds, lang.data());
        use harpia::core::traits::{DataFilter, FilterDecision};
        let decisions: Vec<Value> = texts
            .iter()
            .map(|t| match filter.decide(t) {
                FilterDecision::Keep => json!({"keep": true}),
                FilterDecision::Drop(reason) => json!({"keep": false, "reason": reason}),
            })
            .collect();
        let kept = decisions.iter().filter(|d| d["keep"] == true).count();
        Ok(json!({"decisions": decisions, "kept": kept, "dropped": texts.len() - kept}))
    }
}

fn structured_repair(p: &Value) -> RpcResult<Value> {
    let text = req_str(p, "text")?;
    match harpia::structured::repair::repair(text) {
        Ok(v) => Ok(json!({"value": v})),
        Err(e) => Err(RpcError::new(code::NO_JSON, e.to_string())),
    }
}

// ---------------------------------------------------------------- generation helpers

/// Token ids of `messages` rendered with the model's chat template.
pub fn chat_ids(lm: &TextGenerator, messages: &[Message]) -> RpcResult<Vec<u32>> {
    let prompt = lm.render_chat(messages)?;
    Ok(lm.tokenizer().encode(&prompt, false)?.ids)
}

/// Generate, forwarding deltas to `on_delta` when `stream`.
pub fn run(
    lm: &TextGenerator,
    ids: &[u32],
    params: &GenerationParams,
    constraint: Option<&dyn Constraint>,
    stream: bool,
    on_delta: &mut dyn FnMut(&str),
) -> RpcResult<Generation> {
    Ok(lm.generate_with(ids, params, constraint, |d| {
        if stream {
            on_delta(d);
        }
    })?)
}

/// An assistant reply built from a generation.
pub struct Reply {
    /// The assistant message (with parsed tool calls when tools were offered).
    pub message: Message,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Token accounting.
    pub usage: TokenUsage,
}

/// Turn a generation into an assistant message, parsing tool calls when tools were offered.
pub fn to_reply(g: Generation, wants_tools: bool) -> Reply {
    let mut message = Message::assistant(g.text);
    if wants_tools {
        message.tool_calls = harpia::structured::toolcall::parse_tool_calls(&message.content);
    }
    let finish_reason = if message.tool_calls.is_empty() { g.finish_reason } else { FinishReason::ToolCalls };
    Reply { message, finish_reason, usage: g.usage }
}

fn stream_flag(p: &Value) -> bool {
    p.get("stream").and_then(Value::as_bool).unwrap_or(false)
}

/// Decoding parameters from a request; unset fields keep `defaults` (the model's
/// `generation_config.json`). Accepts OpenAI names (`max_tokens`, `max_completion_tokens`).
pub fn gen_params(p: &Value, defaults: GenerationParams) -> RpcResult<GenerationParams> {
    let mut g = defaults;
    let num = |k: &str| -> RpcResult<Option<f64>> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v.as_f64().map(Some).ok_or_else(|| RpcError::params(format!("`{k}` must be a number"))),
        }
    };
    let uint = |k: &str| -> RpcResult<Option<u64>> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => {
                v.as_u64().map(Some).ok_or_else(|| RpcError::params(format!("`{k}` must be a non-negative integer")))
            }
        }
    };
    for k in ["max_tokens", "max_new_tokens", "max_completion_tokens"] {
        if let Some(v) = uint(k)? {
            g.max_new_tokens = u32::try_from(v).map_err(|_| RpcError::params(format!("`{k}` is too large")))?;
        }
    }
    if let Some(v) = num("temperature")? {
        g.temperature = v as f32;
    }
    if let Some(v) = num("top_p")? {
        g.top_p = v as f32;
    }
    if let Some(v) = uint("top_k")? {
        g.top_k = v as u32;
    }
    if let Some(v) = num("min_p")? {
        g.min_p = v as f32;
    }
    if let Some(v) = num("repetition_penalty")? {
        g.repetition_penalty = v as f32;
    }
    if let Some(v) = num("frequency_penalty")? {
        g.frequency_penalty = v as f32;
    }
    if let Some(v) = num("presence_penalty")? {
        g.presence_penalty = v as f32;
    }
    if let Some(v) = uint("seed")? {
        g.seed = Some(v);
    }
    match p.get("stop") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => g.stop = vec![s.clone()],
        Some(v) => {
            g.stop = serde_json::from_value(v.clone())
                .map_err(|_| RpcError::params("`stop` must be a string or an array of strings"))?
        }
    }
    Ok(g)
}

/// Output constraint from `schema`, `regex`, `choices`, or OpenAI's `response_format`.
pub fn constraint(p: &Value) -> RpcResult<Option<Box<dyn Constraint>>> {
    let schema = match (p.get("schema"), p.get("response_format")) {
        (Some(s), _) if !s.is_null() => Some(s.clone()),
        (_, Some(rf)) => match rf.get("type").and_then(Value::as_str) {
            Some("json_schema") => Some(
                rf.pointer("/json_schema/schema")
                    .cloned()
                    .ok_or_else(|| RpcError::params("response_format.json_schema.schema is required"))?,
            ),
            Some("json_object") => Some(json!({"type": "object"})),
            Some("text") | None => None,
            Some(other) => return Err(RpcError::params(format!("unsupported response_format type `{other}`"))),
        },
        _ => None,
    };
    if let Some(s) = schema {
        return Ok(Some(Box::new(JsonSchemaConstraint::new(&s)?)));
    }
    if let Some(r) = opt_str(p, "regex")? {
        return Ok(Some(Box::new(RegexConstraint::new(r)?)));
    }
    if let Some(c) = p.get("choices").filter(|c| !c.is_null()) {
        let opts: Vec<String> =
            serde_json::from_value(c.clone()).map_err(|_| RpcError::params("`choices` must be an array of strings"))?;
        if opts.is_empty() {
            return Err(RpcError::params("`choices` is empty"));
        }
        return Ok(Some(Box::new(Choice::new(opts))));
    }
    Ok(None)
}

/// Parse chat messages: `content` may be a string or OpenAI content parts (text parts are
/// concatenated); assistant `tool_calls` may use OpenAI's `function` shape.
pub fn parse_messages(v: &Value) -> RpcResult<Vec<Message>> {
    let arr = v.as_array().ok_or_else(|| RpcError::params("`messages` must be an array"))?;
    if arr.is_empty() {
        return Err(RpcError::params("`messages` is empty"));
    }
    arr.iter()
        .enumerate()
        .map(|(i, m)| {
            let role = m.get("role").and_then(Value::as_str).unwrap_or_default();
            let role = match role {
                "developer" => Role::System,
                r => Role::parse(r).ok_or_else(|| RpcError::params(format!("messages[{i}]: unknown role `{r}`")))?,
            };
            let content = match m.get("content") {
                None | Some(Value::Null) => String::new(),
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| match p {
                        Value::String(s) => Some(s.as_str()),
                        _ if p.get("type").and_then(Value::as_str) == Some("text") => {
                            p.get("text").and_then(Value::as_str)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
                Some(_) => return Err(RpcError::params(format!("messages[{i}]: `content` must be a string or parts"))),
            };
            let mut msg = Message::new(role, content);
            msg.name = m.get("name").and_then(Value::as_str).map(String::from);
            msg.tool_call_id = m.get("tool_call_id").and_then(Value::as_str).map(String::from);
            if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                msg.tool_calls = calls.iter().map(parse_tool_call).collect();
            }
            Ok(msg)
        })
        .collect()
}

fn parse_tool_call(c: &Value) -> ToolCall {
    let f = c.get("function").unwrap_or(c);
    let arguments = match f.get("arguments") {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
        Some(v) => v.clone(),
        None => Value::Null,
    };
    ToolCall {
        id: c.get("id").and_then(Value::as_str).map(String::from),
        name: f.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
        arguments,
    }
}

/// Indices sorted by descending score, optionally cut to `top_n`.
pub fn rank(scores: &[f32], top_n: Option<usize>) -> Vec<Value> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    idx.truncate(top_n.unwrap_or(scores.len()));
    idx.into_iter().map(|i| json!({"index": i, "score": scores[i]})).collect()
}

/// Documents as strings or `{"text": ...}` objects.
pub fn document_texts(v: &Value) -> RpcResult<Vec<String>> {
    let arr = v.as_array().ok_or_else(|| RpcError::params("`documents` must be an array"))?;
    arr.iter()
        .map(|d| match d {
            Value::String(s) => Ok(s.clone()),
            _ => d
                .get("text")
                .and_then(Value::as_str)
                .map(String::from)
                .ok_or_else(|| RpcError::params("each document must be a string or {\"text\": ...}")),
        })
        .collect()
}

// ---------------------------------------------------------------- param helpers

fn req_str<'a>(p: &'a Value, k: &str) -> RpcResult<&'a str> {
    p.get(k).and_then(Value::as_str).ok_or_else(|| RpcError::params(format!("missing string `{k}`")))
}

fn opt_str<'a>(p: &'a Value, k: &str) -> RpcResult<Option<&'a str>> {
    match p.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(RpcError::params(format!("`{k}` must be a string"))),
    }
}

/// A string or array of strings under the first present key of `keys`.
pub fn str_list(p: &Value, keys: &[&str]) -> RpcResult<Vec<String>> {
    let Some(v) = keys.iter().find_map(|k| p.get(*k)) else {
        return Err(RpcError::params(format!("missing `{}`", keys[0])));
    };
    match v {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(_) => serde_json::from_value(v.clone())
            .map_err(|_| RpcError::params(format!("`{}` must be a string or an array of strings", keys[0]))),
        _ => Err(RpcError::params(format!("`{}` must be a string or an array of strings", keys[0]))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_and_params() {
        assert_eq!(parse_spec("qwen=models/q").0, "qwen");
        assert_eq!(parse_spec("models/qwen2.5").0, "qwen2.5");
        assert_eq!(parse_spec("/a/b=c/d").1, PathBuf::from("/a/b=c/d"));
        let g = gen_params(&json!({"max_tokens": 7, "temperature": 0.5, "stop": "\n"}), GenerationParams::default())
            .unwrap();
        assert_eq!((g.max_new_tokens, g.temperature, g.stop.clone()), (7, 0.5, vec!["\n".to_string()]));
        assert!(gen_params(&json!({"temperature": "hot"}), GenerationParams::default()).is_err());
    }

    #[test]
    fn messages_accept_openai_parts() {
        let m = parse_messages(&json!([
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\":1}"}}]}
        ]))
        .unwrap();
        assert_eq!(m[0].role, Role::System);
        assert_eq!(m[1].content, "hi there");
        assert_eq!(m[2].tool_calls[0].arguments, json!({"x": 1}));
        assert!(parse_messages(&json!([{"role": "robot", "content": "x"}])).is_err());
    }

    #[test]
    fn ranking_and_constraints() {
        let r = rank(&[0.1, 0.9, 0.5], Some(2));
        assert_eq!(r[0]["index"], 1);
        assert_eq!(r.len(), 2);
        assert!(constraint(&json!({"response_format": {"type": "json_object"}})).unwrap().is_some());
        assert!(constraint(&json!({"response_format": {"type": "text"}})).unwrap().is_none());
        assert!(constraint(&json!({"choices": []})).is_err());
    }
}
