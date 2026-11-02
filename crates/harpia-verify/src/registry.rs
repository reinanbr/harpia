//! The trusted-source registry: which sources may be contacted, on which hosts, with which
//! trust tier and politeness settings.
//!
//! The registry is configuration, not Harpia's judgment: the built-in list ([`SourceRegistry::builtin`])
//! is a starting point that users extend, restrict or replace with TOML:
//!
//! ```toml
//! [[sources]]
//! id = "my-docs"
//! name = "Internal handbook"
//! kind = "docs"
//! domains = ["docs.example.org", "*.example.org"]
//! tier = 1
//! adapter = "web"
//! search_url = "https://docs.example.org/search?q={query}"
//! ```

use harpia_core::{HarpiaError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Broad kind of a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    /// Scholarly metadata APIs (Crossref, OpenAlex...).
    ScholarlyApi,
    /// Encyclopedias (Wikipedia, Wikidata).
    Encyclopedia,
    /// Government sites.
    Government,
    /// Standards bodies.
    StandardsBody,
    /// Technical documentation.
    Docs,
    /// News wires.
    NewsWire,
    /// Anything else.
    Custom,
}

/// One trusted source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// Stable id.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Kind.
    pub kind: SourceKind,
    /// Allowed hosts: exact (`api.crossref.org`) or suffix wildcards (`*.example.org`, which
    /// also matches `example.org`).
    pub domains: Vec<String>,
    /// Trust tier: 1 = primary / peer-reviewed, 2 = curated reference, 3 = general.
    pub tier: u8,
    /// Minimum milliseconds between requests to each of this source's hosts.
    #[serde(default)]
    pub min_interval_ms: u64,
    /// Environment variable with a contact e-mail (sent as `mailto` where the API asks for it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact_env: Option<String>,
    /// Environment variable with an optional API key (sent as a header when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Environment variable that must be set for the source to be used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_env: Option<String>,
    /// Adapter name (`crossref`, `openalex`, `semantic-scholar`, `arxiv`, `europepmc`, `doaj`,
    /// `wikipedia`, `wikidata`, `web`, `open-access-pdf`, or a custom adapter's id).
    pub adapter: String,
    /// Language for encyclopedia adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    /// Content license of the source, recorded with evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Search URL template for `web` sources (`{query}` is replaced, URL-encoded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_url: Option<String>,
    /// Override of the adapter's API base URL (mirrors, tests).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Disabled sources stay in the file but are never contacted.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub enabled: bool,
}

fn yes() -> bool {
    true
}
fn is_true(b: &bool) -> bool {
    *b
}

impl Source {
    /// Whether `host` is one of this source's domains.
    pub fn allows_host(&self, host: &str) -> bool {
        let host = host.trim_end_matches('.').to_ascii_lowercase();
        self.domains.iter().any(|d| {
            let d = d.to_ascii_lowercase();
            match d.strip_prefix("*.") {
                Some(suffix) => host == suffix || host.ends_with(&format!(".{suffix}")),
                None => host == d,
            }
        })
    }

    /// Why the source cannot be used right now (missing required environment variable), if so.
    pub fn unavailable_reason(&self) -> Option<String> {
        if !self.enabled {
            return Some(format!("source `{}` is disabled", self.id));
        }
        match &self.required_env {
            Some(v) if std::env::var(v).map_or(true, |s| s.trim().is_empty()) => {
                Some(format!("source `{}` needs the environment variable {v}", self.id))
            }
            _ => None,
        }
    }

    /// Contact e-mail from `contact_env`, if set.
    pub fn contact(&self) -> Option<String> {
        self.contact_env.as_ref().and_then(|v| std::env::var(v).ok()).filter(|s| !s.trim().is_empty())
    }

    /// API key from `api_key_env`, if set.
    pub fn api_key(&self) -> Option<String> {
        self.api_key_env.as_ref().and_then(|v| std::env::var(v).ok()).filter(|s| !s.trim().is_empty())
    }
}

/// The set of sources a pipeline may use.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRegistry {
    /// Sources, in preference order.
    #[serde(default)]
    pub sources: Vec<Source>,
}

impl SourceRegistry {
    /// The built-in default list (Crossref, OpenAlex, Semantic Scholar, arXiv, Europe PMC, DOAJ,
    /// Wikipedia en/pt, Wikidata, open-access full texts).
    pub fn builtin() -> Self {
        Self::from_toml(include_str!("default_sources.toml")).expect("built-in registry parses")
    }

    /// Parse TOML (validated).
    pub fn from_toml(text: &str) -> Result<Self> {
        let r: Self = toml::from_str(text).map_err(|e| HarpiaError::invalid(format!("source registry: {e}")))?;
        r.validate()?;
        Ok(r)
    }

    /// Load a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Self::from_toml(&std::fs::read_to_string(path).map_err(|e| HarpiaError::io(path, e))?)
    }

    /// Serialize to TOML.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| HarpiaError::invalid(e.to_string()))
    }

    /// Check ids are unique, tiers are 1..=3 and domains are plain host names.
    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for s in &self.sources {
            if !seen.insert(&s.id) {
                return Err(HarpiaError::invalid(format!("duplicate source id `{}`", s.id)));
            }
            if !(1..=3).contains(&s.tier) {
                return Err(HarpiaError::invalid(format!("source `{}`: tier must be 1, 2 or 3", s.id)));
            }
            if s.domains.is_empty() {
                return Err(HarpiaError::invalid(format!("source `{}` has no domains", s.id)));
            }
            for d in &s.domains {
                let bare = d.strip_prefix("*.").unwrap_or(d);
                if bare.is_empty() || bare.contains(['/', ':', '*', ' ']) {
                    return Err(HarpiaError::invalid(format!("source `{}`: `{d}` is not a host name", s.id)));
                }
            }
        }
        Ok(())
    }

    /// Add sources from `other`, replacing ones with the same id.
    pub fn extend(&mut self, other: SourceRegistry) {
        for s in other.sources {
            match self.sources.iter_mut().find(|x| x.id == s.id) {
                Some(x) => *x = s,
                None => self.sources.push(s),
            }
        }
    }

    /// Keep only the listed ids (in the given order).
    pub fn restrict(&mut self, ids: &[&str]) {
        let mut kept = Vec::new();
        for id in ids {
            if let Some(s) = self.sources.iter().find(|s| s.id == *id) {
                kept.push(s.clone());
            }
        }
        self.sources = kept;
    }

    /// Source by id.
    pub fn get(&self, id: &str) -> Option<&Source> {
        self.sources.iter().find(|s| s.id == id)
    }

    /// Whether any enabled source allows `host`.
    pub fn allows_host(&self, host: &str) -> bool {
        self.sources.iter().any(|s| s.enabled && s.allows_host(host))
    }

    /// Sources using `adapter`, enabled only.
    pub fn by_adapter<'a>(&'a self, adapter: &'a str) -> impl Iterator<Item = &'a Source> + 'a {
        self.sources.iter().filter(move |s| s.enabled && s.adapter == adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_and_host_rules() {
        let r = SourceRegistry::builtin();
        assert!(r.get("crossref").unwrap().allows_host("api.crossref.org"));
        assert!(!r.get("crossref").unwrap().allows_host("evil-crossref.org"));
        let oa = r.get("open-access-pdf").unwrap();
        assert!(
            oa.allows_host("www.scielo.br") && oa.allows_host("scielo.br") && !oa.allows_host("scielo.br.evil.com")
        );
        assert!(r.allows_host("EN.wikipedia.org."));
        assert!(!r.allows_host("example.com"));
    }

    #[test]
    fn extend_restrict_validate() {
        let mut r = SourceRegistry::builtin();
        r.extend(
            SourceRegistry::from_toml(
                r#"
[[sources]]
id = "crossref"
name = "Crossref mirror"
kind = "scholarly-api"
domains = ["crossref.example.org"]
tier = 1
adapter = "crossref"
[[sources]]
id = "handbook"
name = "Handbook"
kind = "docs"
domains = ["*.example.org"]
tier = 2
adapter = "web"
"#,
            )
            .unwrap(),
        );
        assert_eq!(r.get("crossref").unwrap().name, "Crossref mirror");
        r.restrict(&["handbook", "crossref", "missing"]);
        assert_eq!(r.sources.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), ["handbook", "crossref"]);
        let back = SourceRegistry::from_toml(&r.to_toml().unwrap()).unwrap();
        assert_eq!(back, r);
        for bad in [
            "[[sources]]\nid='a'\nname='a'\nkind='docs'\ndomains=['https://x.org']\ntier=1\nadapter='web'",
            "[[sources]]\nid='a'\nname='a'\nkind='docs'\ndomains=['x.org']\ntier=4\nadapter='web'",
            "[[sources]]\nid='a'\nname='a'\nkind='docs'\ndomains=['x.org']\ntier=1\nadapter='web'\nsurprise=1",
        ] {
            assert!(SourceRegistry::from_toml(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn required_env_gates_sources() {
        let s = Source {
            required_env: Some("HARPIA_TEST_SURELY_UNSET_VAR".into()),
            ..SourceRegistry::builtin().sources[0].clone()
        };
        assert!(s.unavailable_reason().unwrap().contains("HARPIA_TEST_SURELY_UNSET_VAR"));
        assert!(SourceRegistry::builtin().sources[0].unavailable_reason().is_none());
    }
}
