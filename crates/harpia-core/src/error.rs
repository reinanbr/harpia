//! The unified error type.

use std::path::PathBuf;

/// Result alias used across Harpia.
pub type Result<T, E = HarpiaError> = std::result::Result<T, E>;

/// Every error produced by Harpia crates.
///
/// Variants carry enough context (path, line, host, model id) to act on the error, and never
/// contain secrets such as API keys.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarpiaError {
    /// I/O failure, with the path involved when known.
    #[error("I/O error{}: {source}", path_suffix(.path))]
    Io {
        /// File involved, if any.
        path: Option<PathBuf>,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A record in a dataset could not be parsed.
    #[error("invalid record at {}line {line}: {message}", path_prefix(.path))]
    Record {
        /// Dataset file, if known.
        path: Option<PathBuf>,
        /// 1-based line (or row) number.
        line: u64,
        /// What went wrong.
        message: String,
    },

    /// JSON (de)serialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid argument or configuration supplied by the caller.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A schema or constraint was violated.
    #[error("validation failed: {0}")]
    Validation(String),

    /// A template could not be rendered.
    #[error("template error: {0}")]
    Template(String),

    /// Tokenizer failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// A model failed to load or run.
    #[error("model error ({model}): {message}")]
    Model {
        /// Model identifier or path.
        model: String,
        /// What went wrong.
        message: String,
    },

    /// A network request was attempted under [`NetworkPolicy::Offline`](crate::NetworkPolicy).
    #[error("network access is disabled (offline policy); refused request to host {host}")]
    NetworkDisabled {
        /// Host that would have been contacted.
        host: String,
    },

    /// A remote service failed.
    #[error("remote error from {host}: {message}")]
    Remote {
        /// Host that failed.
        host: String,
        /// Status or error description (never includes credentials).
        message: String,
    },

    /// The functionality exists behind a Cargo feature that is not enabled.
    #[error("{what} requires the `{feature}` feature of `{krate}`")]
    FeatureDisabled {
        /// What was requested.
        what: &'static str,
        /// Feature to enable.
        feature: &'static str,
        /// Crate that owns the feature.
        krate: &'static str,
    },

    /// Anything else.
    #[error("{0}")]
    Other(String),
}

impl HarpiaError {
    /// I/O error with a path.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io { path: Some(path.into()), source }
    }

    /// Invalid argument helper.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }
}

impl From<std::io::Error> for HarpiaError {
    fn from(source: std::io::Error) -> Self {
        Self::Io { path: None, source }
    }
}

fn path_suffix(path: &Option<PathBuf>) -> String {
    path.as_ref().map(|p| format!(" ({})", p.display())).unwrap_or_default()
}

fn path_prefix(path: &Option<PathBuf>) -> String {
    path.as_ref().map(|p| format!("{}: ", p.display())).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_readable() {
        let e = HarpiaError::Record { path: Some("a.jsonl".into()), line: 3, message: "bad".into() };
        assert_eq!(e.to_string(), "invalid record at a.jsonl: line 3: bad");
        let e = HarpiaError::NetworkDisabled { host: "api.example.org".into() };
        assert!(e.to_string().contains("offline"));
        let e = HarpiaError::FeatureDisabled { what: "PDF reading", feature: "pdf", krate: "harpia-data" };
        assert_eq!(e.to_string(), "PDF reading requires the `pdf` feature of `harpia-data`");
    }
}
