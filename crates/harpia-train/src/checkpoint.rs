//! Resumable training state: trainable weights, optimizer moments and loop state, written to
//! `<out>/checkpoint/` at the end of every epoch.

use crate::optim::{GroupedAdamW, err};
use candle_core::{Tensor, Var};
use harpia_core::{HarpiaError, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Directory of the resumable state inside an output directory.
pub fn checkpoint_dir(out: &Path) -> PathBuf {
    out.join("checkpoint")
}

/// Save `vars`, the optimizer and `state` (loop counters, RNG, history) under `dir`.
pub fn save(dir: &Path, vars: &[(String, Var)], opt: &GroupedAdamW, state: &Value) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| HarpiaError::io(dir, e))?;
    let weights: HashMap<String, Tensor> = vars.iter().map(|(n, v)| (n.clone(), v.as_detached_tensor())).collect();
    candle_core::safetensors::save(&weights, dir.join("trainable.safetensors.tmp")).map_err(err)?;
    let (moments, t) = opt.state()?;
    candle_core::safetensors::save(&moments, dir.join("optimizer.safetensors.tmp")).map_err(err)?;
    let mut state = state.clone();
    state["optimizer_step"] = Value::from(t);
    std::fs::write(dir.join("trainer_state.json.tmp"), serde_json::to_string_pretty(&state)?)
        .map_err(|e| HarpiaError::io(dir, e))?;
    // rename last, so a crash never leaves a mix of epochs
    for f in ["trainable.safetensors", "optimizer.safetensors", "trainer_state.json"] {
        std::fs::rename(dir.join(format!("{f}.tmp")), dir.join(f)).map_err(|e| HarpiaError::io(dir, e))?;
    }
    Ok(())
}

/// Restore a checkpoint into `vars` and `opt`; returns the saved loop state, or `None` when
/// there is no checkpoint.
pub fn load(dir: &Path, vars: &[(String, Var)], opt: &mut GroupedAdamW) -> Result<Option<Value>> {
    let state_path = dir.join("trainer_state.json");
    if !state_path.is_file() {
        return Ok(None);
    }
    let state: Value =
        serde_json::from_str(&std::fs::read_to_string(&state_path).map_err(|e| HarpiaError::io(&state_path, e))?)?;
    let device = vars.first().map(|(_, v)| v.as_tensor().device().clone()).unwrap_or(candle_core::Device::Cpu);
    let weights = candle_core::safetensors::load(dir.join("trainable.safetensors"), &device).map_err(err)?;
    for (name, var) in vars {
        let t = weights.get(name).ok_or_else(|| HarpiaError::invalid(format!("checkpoint lacks `{name}`")))?;
        var.set(t).map_err(err)?;
    }
    let moments = candle_core::safetensors::load(dir.join("optimizer.safetensors"), &device).map_err(err)?;
    opt.load_state(&moments, state["optimizer_step"].as_i64().unwrap_or(0) as i32)?;
    Ok(Some(state))
}
