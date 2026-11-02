//! Shared CLI helpers: input reading, output formatting.

use harpia::core::{HarpiaError, Result};
use harpia::text::Lang;
use serde_json::Value;
use std::io::{BufRead, Read, Write};
use std::path::Path;

/// Output format of every subcommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text.
    Text,
    /// JSON with `"schema": 1`.
    Json,
}

/// Read a whole file, or stdin for `-`.
pub fn read_text(path: &str) -> Result<String> {
    if path == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        Ok(s)
    } else {
        std::fs::read_to_string(path).map_err(|e| HarpiaError::io(path, e))
    }
}

/// JSONL records from a file (gzip/zstd by extension) or stdin for `-`.
pub fn read_records(path: &str) -> Result<Vec<Value>> {
    if path == "-" {
        let stdin = std::io::stdin();
        let mut out = Vec::new();
        for (i, line) in stdin.lock().lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(&line).map_err(|e| HarpiaError::Record {
                path: None,
                line: i as u64 + 1,
                message: e.to_string(),
            })?);
        }
        Ok(out)
    } else {
        harpia::data::read_jsonl(Path::new(path))
    }
}

/// Write JSONL to a file, or stdout when `out` is `None` or `-`.
pub fn write_records(out: Option<&str>, records: &[Value]) -> Result<()> {
    match out {
        None | Some("-") => {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            for r in records {
                serde_json::to_writer(&mut lock, r)?;
                lock.write_all(b"\n")?;
            }
            Ok(())
        }
        Some(p) => {
            let mut w = harpia::data::JsonlWriter::create(p)?;
            for r in records {
                w.write(r)?;
            }
            w.finish()
        }
    }
}

/// Print a JSON payload with the schema version, or a text rendering.
pub fn emit(format: OutputFormat, mut payload: Value, text: impl FnOnce() -> String) {
    match format {
        OutputFormat::Json => {
            if let Value::Object(m) = &mut payload {
                m.insert("schema".into(), Value::from(1));
            }
            println!("{}", serde_json::to_string_pretty(&payload).expect("serializable"));
        }
        OutputFormat::Text => {
            let t = text();
            if !t.is_empty() {
                println!("{t}");
            }
        }
    }
}

/// Parse `--lang`.
pub fn lang(s: &str) -> Result<Lang> {
    Lang::parse(s).ok_or_else(|| HarpiaError::invalid(format!("unknown language `{s}` (use en or pt-br)")))
}

/// Write JSON to a path.
pub fn write_json(path: &str, v: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(v)?).map_err(|e| HarpiaError::io(path, e))
}
