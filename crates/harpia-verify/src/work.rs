//! Records returned by source adapters.

use serde::{Deserialize, Serialize};

/// An author.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Author {
    /// Family name (surname).
    pub family: String,
    /// Full display name, when known.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub full: String,
}

impl Author {
    /// From a display name: the last word is taken as the family name.
    pub fn from_display(name: &str) -> Self {
        let name = name.trim();
        Self { family: name.split_whitespace().last().unwrap_or("").to_string(), full: name.to_string() }
    }
}

/// A document from a trusted source: a scholarly work, an encyclopedia article or a web page.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Work {
    /// Source id from the registry.
    pub source: String,
    /// Trust tier of that source.
    pub tier: u8,
    /// Title.
    pub title: String,
    /// Authors in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authors: Vec<Author>,
    /// Publication year.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub year: Option<i32>,
    /// Journal / venue.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub venue: String,
    /// DOI (lowercase, no `https://doi.org/` prefix).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub doi: String,
    /// Canonical URL.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub url: String,
    /// Abstract, extract or main text used as evidence.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    /// Retraction / withdrawal flag, when the source reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retracted: Option<bool>,
    /// Correction / erratum notices reported by the source.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corrections: Vec<String>,
    /// Open-access full-text (PDF) URL reported by the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf_url: Option<String>,
    /// Content license, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// When the content was retrieved (RFC 3339).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub retrieved_at: String,
    /// SHA-256 of the response the record came from.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content_hash: String,
}

/// Identifiers used for direct lookups.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum WorkId {
    /// A DOI.
    Doi(String),
    /// An arXiv identifier.
    Arxiv(String),
    /// An ISBN.
    Isbn(String),
    /// A PubMed id.
    Pmid(String),
}

/// Outcome of a direct lookup.
#[derive(Debug, Clone, PartialEq)]
pub enum Lookup {
    /// Found.
    Found(Box<Work>),
    /// The source answered that it does not know the id.
    NotFound,
    /// The source could not be asked (network error, frozen-store miss, not supported...).
    Unavailable(String),
}

/// Strip `doi:` / resolver prefixes and surrounding punctuation, keeping the original case (DOIs
/// are case-insensitive, but they are sent to APIs as written).
///
/// ```
/// assert_eq!(harpia_verify::work::clean_doi(" https://doi.org/10.18653/v1/N19-1423. "), "10.18653/v1/N19-1423");
/// ```
pub fn clean_doi(doi: &str) -> String {
    let d = doi.trim().trim_end_matches(['.', ',', ';']);
    let lower = d.to_ascii_lowercase();
    let prefix = ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"]
        .iter()
        .find(|p| lower.starts_with(*p))
        .map_or(0, |p| p.len());
    d[prefix..].trim().to_string()
}

/// [`clean_doi`] in lowercase, for comparisons.
///
/// ```
/// assert_eq!(harpia_verify::work::normalize_doi(" https://doi.org/10.1038/NATURE14539. "), "10.1038/nature14539");
/// ```
pub fn normalize_doi(doi: &str) -> String {
    clean_doi(doi).to_ascii_lowercase()
}

/// Whether `doi` looks like a DOI (`10.<registrant>/<suffix>`).
pub fn valid_doi(doi: &str) -> bool {
    let d = normalize_doi(doi);
    let Some(rest) = d.strip_prefix("10.") else { return false };
    let Some((reg, suffix)) = rest.split_once('/') else { return false };
    reg.len() >= 4
        && reg.chars().all(|c| c.is_ascii_digit() || c == '.')
        && !suffix.trim().is_empty()
        && !d.contains(char::is_whitespace)
}

/// Whether `id` looks like an arXiv identifier (new `2101.00001v2` or old `hep-th/9901001` style).
pub fn valid_arxiv(id: &str) -> bool {
    let id = id.trim().trim_start_matches("arXiv:").trim_start_matches("arxiv:");
    let new_style = {
        let (main, ver) = id.split_once('v').map_or((id, ""), |(m, v)| (m, v));
        let ok_ver = ver.is_empty() || ver.chars().all(|c| c.is_ascii_digit());
        main.len() >= 9
            && main.split_once('.').is_some_and(|(a, b)| {
                a.len() == 4
                    && a.chars().all(|c| c.is_ascii_digit())
                    && (4..=5).contains(&b.len())
                    && b.chars().all(|c| c.is_ascii_digit())
            })
            && ok_ver
    };
    let old_style = id.split_once('/').is_some_and(|(arch, num)| {
        !arch.is_empty()
            && arch.chars().all(|c| c.is_ascii_lowercase() || c == '-' || c == '.')
            && num.len() == 7
            && num.chars().all(|c| c.is_ascii_digit())
    });
    new_style || old_style
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers() {
        assert!(valid_doi("10.1038/nature14539") && valid_doi("doi:10.18653/v1/N19-1423"));
        assert!(!valid_doi("10.12/x") && !valid_doi("nature14539") && !valid_doi("10.1038/"));
        assert!(valid_arxiv("arXiv:1706.03762v7") && valid_arxiv("2101.00001") && valid_arxiv("hep-th/9901001"));
        assert!(valid_arxiv("0704.0001"), "4-digit numbers are the pre-2015 format");
        assert!(!valid_arxiv("1706.03") && !valid_arxiv("foo") && !valid_arxiv("17060.03762"));
        assert_eq!(Author::from_display("Yann LeCun").family, "LeCun");
    }
}
