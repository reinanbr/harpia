//! Per-language data: abbreviations and stopwords, bundled for English and Brazilian
//! Portuguese and extensible by the user.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

/// A bundled language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Lang {
    /// English.
    En,
    /// Brazilian Portuguese.
    PtBr,
}

impl Lang {
    /// Parse `"en"`, `"pt-br"`, `"pt_BR"`, `"pt"`.
    pub fn parse(s: &str) -> Option<Lang> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "en" | "en-us" | "en-gb" | "english" => Some(Lang::En),
            "pt" | "pt-br" | "portuguese" | "português" => Some(Lang::PtBr),
            _ => None,
        }
    }

    /// Shared bundled data for this language (cheap to clone).
    pub fn data(self) -> Arc<LangData> {
        match self {
            Lang::En => EN.clone(),
            Lang::PtBr => PT_BR.clone(),
        }
    }
}

/// Word lists that drive segmentation and statistics.
#[derive(Debug, Clone, Default)]
pub struct LangData {
    /// Lowercase abbreviations without the final period (`"dr"`, `"e.g"`).
    pub abbreviations: HashSet<String>,
    /// Lowercase function words.
    pub stopwords: HashSet<String>,
    /// Syllable heuristic used by readability formulas.
    pub syllables: SyllableRule,
}

/// Which syllable-counting heuristic to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SyllableRule {
    /// English vowel groups with silent-e handling.
    #[default]
    English,
    /// Portuguese vowel nuclei with simple hiatus rules.
    Portuguese,
}

impl LangData {
    /// Build from line-oriented lists (`#` comments and blank lines ignored).
    pub fn from_lists(abbreviations: &str, stopwords: &str, syllables: SyllableRule) -> Self {
        Self { abbreviations: parse_list(abbreviations), stopwords: parse_list(stopwords), syllables }
    }

    /// Add user abbreviations (lowercase, without final period).
    pub fn with_abbreviations<I, S>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.abbreviations.extend(items.into_iter().map(|s| s.as_ref().to_lowercase()));
        self
    }

    /// Add user stopwords.
    pub fn with_stopwords<I, S>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.stopwords.extend(items.into_iter().map(|s| s.as_ref().to_lowercase()));
        self
    }

    /// `true` if `word` (any case) is a stopword.
    pub fn is_stopword(&self, word: &str) -> bool {
        self.stopwords.contains(&word.to_lowercase())
    }
}

/// Parse a line-oriented word list.
pub fn parse_list(text: &str) -> HashSet<String> {
    text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_lowercase).collect()
}

static EN: LazyLock<Arc<LangData>> = LazyLock::new(|| {
    Arc::new(LangData::from_lists(
        include_str!("../data/en/abbreviations.txt"),
        include_str!("../data/en/stopwords.txt"),
        SyllableRule::English,
    ))
});

static PT_BR: LazyLock<Arc<LangData>> = LazyLock::new(|| {
    Arc::new(LangData::from_lists(
        include_str!("../data/pt-br/abbreviations.txt"),
        include_str!("../data/pt-br/stopwords.txt"),
        SyllableRule::Portuguese,
    ))
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_lists_load() {
        let pt = Lang::PtBr.data();
        assert!(pt.abbreviations.contains("dr"));
        assert!(pt.is_stopword("Não"));
        assert!(Lang::En.data().is_stopword("the"));
        assert_eq!(Lang::parse("pt_BR"), Some(Lang::PtBr));
    }

    #[test]
    fn user_extension() {
        let d = (*Lang::En.data()).clone().with_abbreviations(["Approx"]);
        assert!(d.abbreviations.contains("approx"));
    }
}
