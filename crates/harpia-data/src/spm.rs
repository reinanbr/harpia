//! SentencePiece (`.spm` / `.model`) loading without Python: a minimal protobuf reader for
//! `ModelProto`, turned into a Hugging Face `tokenizers` Unigram tokenizer the same way
//! transformers' `SpmConverter` does (precompiled normalization charsmap + Metaspace).

use crate::tokenize::HfTokenizer;
use harpia_core::{HarpiaError, Result};
use serde_json::{Value, json};
use std::path::Path;

/// One vocabulary entry of a SentencePiece model.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// Piece text (`▁` marks a word start).
    pub piece: String,
    /// Log-probability score (Unigram).
    pub score: f32,
    /// 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte.
    pub kind: u32,
}

/// The parts of a SentencePiece `ModelProto` needed for tokenization.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpmModel {
    /// Vocabulary in id order.
    pub pieces: Vec<Piece>,
    /// `trainer_spec.model_type`: 1 unigram, 2 BPE, 3 word, 4 char.
    pub model_type: u32,
    /// `normalizer_spec.precompiled_charsmap`.
    pub charsmap: Vec<u8>,
    /// `normalizer_spec.add_dummy_prefix`.
    pub add_dummy_prefix: bool,
    /// `normalizer_spec.remove_extra_whitespaces`: trim and collapse runs of spaces.
    pub remove_extra_whitespaces: bool,
    /// `trainer_spec.byte_fallback`.
    pub byte_fallback: bool,
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self.b.get(self.i).ok_or_else(|| HarpiaError::Tokenizer("truncated protobuf".into()))?;
            self.i += 1;
            v |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(HarpiaError::Tokenizer("bad varint".into()))
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        let s = self.b.get(self.i..self.i + n).ok_or_else(|| HarpiaError::Tokenizer("truncated protobuf".into()))?;
        self.i += n;
        Ok(s)
    }

    /// Next `(field, wire_type)`, or `None` at the end.
    fn key(&mut self) -> Result<Option<(u64, u64)>> {
        if self.i >= self.b.len() {
            return Ok(None);
        }
        let k = self.varint()?;
        Ok(Some((k >> 3, k & 7)))
    }

    fn skip(&mut self, wire: u64) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.i += 8,
            2 => {
                self.bytes()?;
            }
            5 => self.i += 4,
            w => return Err(HarpiaError::Tokenizer(format!("unsupported protobuf wire type {w}"))),
        }
        Ok(())
    }
}

fn parse_piece(b: &[u8]) -> Result<Piece> {
    let mut r = Reader { b, i: 0 };
    let mut p = Piece { piece: String::new(), score: 0.0, kind: 1 };
    while let Some((f, w)) = r.key()? {
        match (f, w) {
            (1, 2) => p.piece = String::from_utf8_lossy(r.bytes()?).into_owned(),
            (2, 5) => {
                let s = r.b.get(r.i..r.i + 4).ok_or_else(|| HarpiaError::Tokenizer("truncated score".into()))?;
                p.score = f32::from_le_bytes(s.try_into().expect("4 bytes"));
                r.i += 4;
            }
            (3, 0) => p.kind = r.varint()? as u32,
            (_, w) => r.skip(w)?,
        }
    }
    Ok(p)
}

impl SpmModel {
    /// Parse a serialized `ModelProto`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader { b: bytes, i: 0 };
        let mut m =
            SpmModel { model_type: 1, add_dummy_prefix: true, remove_extra_whitespaces: true, ..Default::default() };
        while let Some((f, w)) = r.key()? {
            match (f, w) {
                (1, 2) => m.pieces.push(parse_piece(r.bytes()?)?),
                (2, 2) => {
                    // trainer_spec: model_type = 3, byte_fallback = 35
                    let mut t = Reader { b: r.bytes()?, i: 0 };
                    while let Some((tf, tw)) = t.key()? {
                        match (tf, tw) {
                            (3, 0) => m.model_type = t.varint()? as u32,
                            (35, 0) => m.byte_fallback = t.varint()? != 0,
                            (_, w) => t.skip(w)?,
                        }
                    }
                }
                (3, 2) => {
                    // normalizer_spec: precompiled_charsmap = 2, add_dummy_prefix = 3,
                    // remove_extra_whitespaces = 4
                    let mut n = Reader { b: r.bytes()?, i: 0 };
                    while let Some((nf, nw)) = n.key()? {
                        match (nf, nw) {
                            (2, 2) => m.charsmap = n.bytes()?.to_vec(),
                            (3, 0) => m.add_dummy_prefix = n.varint()? != 0,
                            (4, 0) => m.remove_extra_whitespaces = n.varint()? != 0,
                            (_, w) => n.skip(w)?,
                        }
                    }
                }
                (_, w) => r.skip(w)?,
            }
        }
        if m.pieces.is_empty() {
            return Err(HarpiaError::Tokenizer("SentencePiece model has no pieces".into()));
        }
        Ok(m)
    }

    /// Load a `.spm` / `.model` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Self::parse(&std::fs::read(path).map_err(|e| HarpiaError::io(path, e))?)
    }

    /// `tokenizer.json` equivalent (Unigram models only), as built by transformers'
    /// `SpmConverter`: precompiled charsmap normalization, Metaspace pre-tokenization and
    /// decoding. With `remove_extra_whitespaces` the ends are trimmed and inner runs of spaces
    /// collapsed, as SentencePiece itself does.
    pub fn to_tokenizer_json(&self) -> Result<Value> {
        if self.model_type != 1 {
            return Err(HarpiaError::Tokenizer(format!(
                "only Unigram SentencePiece models are supported (model_type {})",
                self.model_type
            )));
        }
        let unk = self.pieces.iter().position(|p| p.kind == 2).unwrap_or(0);
        let vocab: Vec<Value> = self.pieces.iter().map(|p| json!([p.piece, p.score])).collect();
        let mut normalizers = Vec::new();
        if !self.charsmap.is_empty() {
            normalizers.push(json!({"type": "Precompiled", "precompiled_charsmap": base64(&self.charsmap)}));
        }
        if self.remove_extra_whitespaces {
            normalizers.push(json!({"type": "Strip", "strip_left": true, "strip_right": true}));
            normalizers.push(json!({"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}));
        }
        let scheme = if self.add_dummy_prefix { "always" } else { "never" };
        let added: Vec<Value> = self
            .pieces
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == 3 || p.kind == 2)
            .map(|(i, p)| json!({"id": i, "content": p.piece, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
            .collect();
        Ok(json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
            "normalizer": {"type": "Sequence", "normalizers": normalizers},
            "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": scheme, "split": true},
            "post_processor": null,
            "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": scheme, "split": true},
            "model": {"type": "Unigram", "unk_id": unk, "vocab": vocab, "byte_fallback": self.byte_fallback}
        }))
    }

    /// A ready tokenizer.
    pub fn to_tokenizer(&self) -> Result<HfTokenizer> {
        HfTokenizer::from_bytes(serde_json::to_vec(&self.to_tokenizer_json()?)?.as_slice())
    }
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n =
            (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use harpia_core::traits::Tokenizer;

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn field_bytes(field: u64, data: &[u8], out: &mut Vec<u8>) {
        varint((field << 3) | 2, out);
        varint(data.len() as u64, out);
        out.extend_from_slice(data);
    }

    /// Build a tiny ModelProto by hand.
    fn tiny_model() -> Vec<u8> {
        let mut model = Vec::new();
        for (piece, score, kind) in [
            ("<unk>", 0.0f32, 2u64),
            ("</s>", 0.0, 3),
            ("▁the", -1.0, 1),
            ("▁harp", -2.0, 1),
            ("y", -2.5, 1),
            ("▁", -3.0, 1),
            ("h", -4.0, 1),
            ("a", -4.0, 1),
            ("r", -4.0, 1),
            ("p", -4.0, 1),
            ("t", -4.0, 1),
            ("e", -4.0, 1),
        ] {
            let mut p = Vec::new();
            field_bytes(1, piece.as_bytes(), &mut p);
            varint((2 << 3) | 5, &mut p);
            p.extend_from_slice(&score.to_le_bytes());
            varint(3 << 3, &mut p);
            varint(kind, &mut p);
            field_bytes(1, &p, &mut model);
        }
        let mut trainer = Vec::new();
        varint(3 << 3, &mut trainer);
        varint(1, &mut trainer);
        field_bytes(2, &trainer, &mut model);
        model
    }

    #[test]
    fn parses_and_tokenizes() {
        let m = SpmModel::parse(&tiny_model()).unwrap();
        assert_eq!(m.pieces.len(), 12);
        assert_eq!((m.pieces[3].piece.as_str(), m.pieces[3].score), ("▁harp", -2.0));
        let tok = m.to_tokenizer().unwrap();
        let enc = tok.encode("the harpy", false).unwrap();
        assert_eq!(enc.ids, [2, 3, 4]);
        assert_eq!(tok.decode(&enc.ids, true).unwrap(), "the harpy");
        assert_eq!(tok.encode("  the   harpy  ", false).unwrap().ids, [2, 3, 4]);
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
    }
}
