//! Weight store for training: decides, per weight, whether it is frozen, fully trainable, or
//! frozen plus a LoRA update, and tracks every trainable variable.

use candle_core::{DType, Device, Tensor, Var};
use harpia_data::rng::SplitMix64;
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

/// Which weights are trained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Every weight becomes a trainable variable (full fine-tuning).
    Full,
    /// Base weights are frozen; only LoRA matrices (and new weights) are trained.
    Frozen,
}

/// The `A` (`r × in`) and `B` (`out × r`) matrices of one LoRA-adapted weight.
#[derive(Debug, Clone)]
pub struct LoraPair {
    /// Down projection, `r × in`.
    pub a: Var,
    /// Up projection, `out × r`, initialized to zero.
    pub b: Var,
}

/// Weights of a model being trained.
#[derive(Debug)]
pub struct TrainStore {
    base: HashMap<String, Tensor>,
    mode: Mode,
    device: Device,
    seed: u64,
    vars: Mutex<BTreeMap<String, Var>>,
    lora: BTreeMap<String, LoraPair>,
    scale: f64,
}

fn name_seed(name: &str, seed: u64) -> u64 {
    name.bytes().fold(seed ^ 0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

impl TrainStore {
    /// Store over `base` weights (cast to f32 on `device`).
    pub fn new(base: HashMap<String, Tensor>, mode: Mode, device: &Device, seed: u64) -> candle_core::Result<Self> {
        let base = base
            .into_iter()
            .map(|(k, t)| Ok((k, t.to_dtype(DType::F32)?.to_device(device)?)))
            .collect::<candle_core::Result<_>>()?;
        Ok(Self {
            base,
            mode,
            device: device.clone(),
            seed,
            vars: Mutex::new(BTreeMap::new()),
            lora: BTreeMap::new(),
            scale: 1.0,
        })
    }

    /// Add LoRA adapters (keyed by the full weight name, e.g. `model.layers.0.self_attn.q_proj.weight`).
    pub fn with_lora(mut self, lora: BTreeMap<String, LoraPair>, scale: f64) -> Self {
        self.lora = lora;
        self.scale = scale;
        self
    }

    /// Base weights.
    pub fn base(&self) -> &HashMap<String, Tensor> {
        &self.base
    }

    /// LoRA pairs.
    pub fn lora(&self) -> &BTreeMap<String, LoraPair> {
        &self.lora
    }

    /// LoRA scale (`alpha / r`).
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Device of the weights.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Full or new trainable variables created so far, by name.
    pub fn vars(&self) -> BTreeMap<String, Var> {
        self.vars.lock().expect("vars lock").clone()
    }

    /// Every trainable variable: full/new weights plus LoRA matrices.
    pub fn trainable(&self) -> Vec<(String, Var)> {
        let mut out: Vec<(String, Var)> = self.vars().into_iter().collect();
        for (k, p) in &self.lora {
            out.push((format!("{k}#lora_A"), p.a.clone()));
            out.push((format!("{k}#lora_B"), p.b.clone()));
        }
        out
    }

    /// `true` if the base checkpoint has `name`.
    pub fn has(&self, name: &str) -> bool {
        self.base.contains_key(name)
    }

    /// The weight `name` with `shape`: a trainable variable in [`Mode::Full`], a frozen tensor in
    /// [`Mode::Frozen`]. A weight that is missing or has another shape (e.g. a classification head
    /// with a new number of labels) becomes a new trainable variable, initialized
    /// deterministically: biases to zero, matrices uniform in `±1/sqrt(fan_in)`.
    pub fn param(&self, name: &str, shape: &[usize]) -> candle_core::Result<Tensor> {
        let mut vars = self.vars.lock().expect("vars lock");
        if let Some(v) = vars.get(name) {
            return Ok(v.as_tensor().clone());
        }
        match self.base.get(name) {
            Some(t) if t.dims() == shape => match self.mode {
                Mode::Full => {
                    let v = Var::from_tensor(t)?;
                    vars.insert(name.to_string(), v.clone());
                    Ok(v.as_tensor().clone())
                }
                Mode::Frozen => Ok(t.detach()),
            },
            _ => {
                let n: usize = shape.iter().product();
                let data: Vec<f32> = if name.ends_with("bias") {
                    vec![0.0; n]
                } else {
                    let bound = 1.0 / (*shape.last().unwrap_or(&1) as f64).sqrt();
                    let mut rng = SplitMix64::new(name_seed(name, self.seed));
                    (0..n).map(|_| ((rng.next_f64() * 2.0 - 1.0) * bound) as f32).collect()
                };
                let v = Var::from_tensor(&Tensor::from_vec(data, shape, &self.device)?)?;
                vars.insert(name.to_string(), v.clone());
                Ok(v.as_tensor().clone())
            }
        }
    }

    /// LoRA matrices `(A, B)` of weight `name`, if adapted.
    pub fn lora_of(&self, name: &str) -> Option<(Tensor, Tensor)> {
        self.lora.get(name).map(|p| (p.a.as_tensor().clone(), p.b.as_tensor().clone()))
    }

    /// Current value of every weight (LoRA merged), for export.
    pub fn merged(&self) -> candle_core::Result<HashMap<String, Tensor>> {
        let mut out = self.base.clone();
        for (k, v) in self.vars() {
            out.insert(k, v.as_detached_tensor());
        }
        for (k, p) in &self.lora {
            let delta = (p.b.as_tensor().matmul(p.a.as_tensor())? * self.scale)?;
            out.insert(k.clone(), (&self.base[k] + delta)?.detach());
        }
        Ok(out)
    }
}
