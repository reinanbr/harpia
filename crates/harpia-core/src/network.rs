//! Explicit network policy. Harpia never opens a socket unless the caller says so.

use serde::{Deserialize, Serialize};

/// Whether a component may use the network.
///
/// The default is [`NetworkPolicy::Offline`]: components that would need the network fail fast
/// with [`HarpiaError::NetworkDisabled`](crate::HarpiaError::NetworkDisabled) and open no socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetworkPolicy {
    /// No outbound connections (default).
    #[default]
    Offline,
    /// Outbound connections allowed.
    Online,
}

impl NetworkPolicy {
    /// `true` when outbound connections are allowed.
    pub fn is_online(self) -> bool {
        matches!(self, Self::Online)
    }

    /// Returns an error for `host` when offline.
    pub fn check(self, host: &str) -> crate::Result<()> {
        if self.is_online() { Ok(()) } else { Err(crate::HarpiaError::NetworkDisabled { host: host.to_owned() }) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_by_default() {
        assert_eq!(NetworkPolicy::default(), NetworkPolicy::Offline);
        assert!(NetworkPolicy::default().check("x.org").is_err());
        assert!(NetworkPolicy::Online.check("x.org").is_ok());
    }
}
