//! [`Tokenizer`] backed by Hugging Face `tokenizers`
//! (feature `tokenizers`, pure-Rust build without `onig`/`esaxx` C++).

use harpia_core::traits::{Encoding, Tokenizer};
use harpia_core::{HarpiaError, Result, Span};
use std::path::Path;

/// A `tokenizer.json` tokenizer.
#[derive(Clone)]
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer").field("vocab_size", &self.inner.get_vocab_size(true)).finish()
    }
}

fn err(e: impl std::fmt::Display) -> HarpiaError {
    HarpiaError::Tokenizer(e.to_string())
}

impl HfTokenizer {
    /// Load `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| HarpiaError::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    /// Parse a `tokenizer.json` document from memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self { inner: tokenizers::Tokenizer::from_bytes(bytes).map_err(err)? })
    }

    /// Access the underlying tokenizer.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    /// The vocabulary as strings indexed by id (decoded per token), for constrained decoding.
    pub fn vocab_strings(&self) -> Vec<String> {
        let n = self.inner.get_vocab_size(true) as u32;
        (0..n).map(|i| self.inner.decode(&[i], false).unwrap_or_default()).collect()
    }

    fn convert(enc: &tokenizers::Encoding, pair_offset: Option<usize>) -> Encoding {
        let type_ids = enc.get_type_ids().to_vec();
        let offsets = enc
            .get_offsets()
            .iter()
            .zip(enc.get_type_ids())
            .zip(enc.get_special_tokens_mask())
            .map(|((&(s, e), &t), &sp)| {
                if sp == 1 {
                    Span::default()
                } else {
                    let shift = if t == 1 { pair_offset.unwrap_or(0) } else { 0 };
                    Span::new(s + shift, e + shift)
                }
            })
            .collect();
        Encoding {
            ids: enc.get_ids().to_vec(),
            offsets,
            type_ids,
            special: enc.get_special_tokens_mask().iter().map(|&m| m == 1).collect(),
        }
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        let enc = self.inner.encode(text, add_special_tokens).map_err(err)?;
        Ok(Self::convert(&enc, None))
    }

    /// Offsets of the second sequence are relative to `second`.
    fn encode_pair(&self, first: &str, second: &str, add_special_tokens: bool) -> Result<Encoding> {
        let enc = self.inner.encode((first, second), add_special_tokens).map_err(err)?;
        Ok(Self::convert(&enc, None))
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(ids, skip_special_tokens).map_err(err)
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny word-level tokenizer built in memory (no downloads).
    pub(crate) fn tiny() -> HfTokenizer {
        let json = r#"{
          "version": "1.0", "truncation": null, "padding": null,
          "added_tokens": [{"id": 0, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
          "normalizer": {"type": "Lowercase"},
          "pre_tokenizer": {"type": "Whitespace"},
          "post_processor": null, "decoder": null,
          "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "harpia": 3, ".": 4}, "unk_token": "[UNK]"}
        }"#;
        HfTokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    #[test]
    fn offsets_are_byte_spans() {
        let t = tiny();
        let text = "Hello harpia world.";
        let e = t.encode(text, false).unwrap();
        assert_eq!(e.ids, [1, 3, 2, 4]);
        assert_eq!(e.offsets[1].slice(text), "harpia");
        assert_eq!(t.count("hello unknownword").unwrap(), 2);
        assert_eq!(t.token_to_id("world"), Some(2));
    }
}
