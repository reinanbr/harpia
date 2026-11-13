//! Lesson 2: tokenization.
//!
//! Models do not read text, they read token ids. This lesson builds a small WordPiece
//! tokenizer by hand (the same `tokenizer.json` format every Hugging Face model ships), then
//! shows the four things training and inference need from it:
//!
//! 1. ids and **byte offsets** back into the text,
//! 2. **pair encoding** (premise, hypothesis) for cross-encoders,
//! 3. **chat templates** that turn a conversation into one string, with the spans the
//!    assistant wrote,
//! 4. an **assistant-only loss mask** and **packing** several samples into fixed-length rows.
//!
//! With a real model you skip step 0 and call `HfTokenizer::from_file("model/tokenizer.json")`.
//!
//! Run: `cargo test -p harpia-learn --test 02_tokenization -- --nocapture`

use harpia::core::traits::{ChatTemplate, Tokenizer};
use harpia::core::types::Message;
use harpia::data::pack::{PackOptions, TokenizedSample, pack, packing_efficiency};
use harpia::data::template::{JinjaChatTemplate, loss_mask};
use harpia::data::tokenize::HfTokenizer;
use serde_json::json;

/// A WordPiece tokenizer (BERT style): unknown words are split into known sub-words marked `##`.
fn wordpiece() -> HfTokenizer {
    let vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]", // special tokens first
        "o",
        "gavi",
        "##ão",
        "real",
        "harp",
        "##ia",
        "##s",
        "come",
        "preguiça",
        "voa",
        "alto",
        "a",
        "ave",
        "é",
        "grande",
        ".",
    ];
    let vocab: serde_json::Map<_, _> = vocab.iter().enumerate().map(|(i, t)| (t.to_string(), json!(i))).collect();
    let specials: Vec<_> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        .iter()
        .enumerate()
        .map(|(i, t)| json!({"id": i, "content": t, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
        .collect();
    let tokenizer_json = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": specials,
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        // adds [CLS] ... [SEP] (and [SEP] between the two texts of a pair)
        "post_processor": {"type": "BertProcessing", "sep": ["[SEP]", 3], "cls": ["[CLS]", 2]},
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
        "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 100, "vocab": vocab}
    });
    HfTokenizer::from_bytes(tokenizer_json.to_string().as_bytes()).expect("valid tokenizer.json")
}

#[test]
fn encode_decode_and_offsets() -> harpia::Result<()> {
    let tok = wordpiece();
    let text = "O gavião come preguiças.";
    let enc = tok.encode(text, true)?;

    // Each id has a byte span into `text` (empty for the special tokens the post-processor added).
    for (id, span) in enc.ids.iter().zip(&enc.offsets) {
        println!("{id:>3}  {:?}", span.slice(text));
    }
    let pieces: Vec<&str> = enc.offsets.iter().map(|s| s.slice(text)).collect();
    // "gavião" is not in the vocabulary, but "gavi" + "##ão" are; "preguiças" = "preguiça" + "##s".
    assert_eq!(pieces, ["", "O", "gavi", "ão", "come", "preguiça", "s", ".", ""]);
    assert_eq!(enc.special, [true, false, false, false, false, false, false, false, true]);

    // Decoding joins the sub-words again (the normalizer lowercased the text).
    assert_eq!(tok.decode(&enc.ids, true)?, "o gavião come preguiças.");

    // Counting tokens is what context-window budgets and costs are made of.
    assert_eq!(tok.count("a ave voa alto.")?, 5);
    // Unknown words become [UNK] rather than failing.
    assert_eq!(tok.encode("tucano", false)?.ids, [tok.token_to_id("[UNK]").unwrap()]);
    Ok(())
}

/// Cross-encoders (NLI, rerankers) read two texts at once; `type_ids` tell them apart.
#[test]
fn pair_encoding() -> harpia::Result<()> {
    let tok = wordpiece();
    let enc = tok.encode_pair("a harpia é grande.", "a ave voa alto.", true)?;
    let tokens: Vec<String> = enc.ids.iter().map(|&id| tok.decode(&[id], false).unwrap()).collect();
    println!("{tokens:?}\n{:?}", enc.type_ids);
    assert_eq!(enc.ids.first(), tok.token_to_id("[CLS]").as_ref());
    assert_eq!(enc.type_ids.iter().filter(|&&t| t == 1).count(), 6, "5 tokens of the second text + its [SEP]");
    Ok(())
}

/// A chat template renders a conversation into the exact string the model was trained on, and
/// tells you which characters the assistant produced (the only ones to train on).
#[test]
fn chat_template_and_loss_mask() -> harpia::Result<()> {
    let dir = tempfile::tempdir()?;
    // A tiny decoder model directory: we only need its tokenizer.json and tokenizer_config.json.
    harpia::local::testing::qwen2(dir.path())?;
    let tok = HfTokenizer::from_file(dir.path().join("tokenizer.json"))?;
    // The model's own Jinja template, read from tokenizer_config.json (as transformers does).
    let template = JinjaChatTemplate::from_tokenizer_config(dir.path().join("tokenizer_config.json"))?;

    let chat =
        [Message::system("a bird"), Message::user("the harpy eagle eats"), Message::assistant("sloths in the forest")];
    let rendered = template.render_with_spans(&chat)?;
    println!("rendered: {}", rendered.text);
    let trained: Vec<&str> = rendered.assistant_spans.iter().map(|s| s.slice(&rendered.text)).collect();
    println!("assistant spans: {trained:?}");
    assert!(trained[0].contains("sloths in the forest"));

    // Token-level mask: true only for tokens inside an assistant span.
    let enc = tok.encode(&rendered.text, false)?;
    let mask = loss_mask(&enc.offsets, &rendered.assistant_spans);
    let learned: Vec<String> =
        enc.ids.iter().zip(&mask).filter(|(_, m)| **m).map(|(&id, _)| tok.decode(&[id], false).unwrap()).collect();
    println!("tokens with loss: {learned:?}");
    assert!(learned.iter().any(|t| t == "sloths"));
    assert!(!learned.iter().any(|t| t == "harpy"), "the user's words are context, not targets");
    Ok(())
}

/// Packing puts several short samples into one fixed-length row so no compute is wasted on
/// padding. `cu_seqlens` and `position_ids` keep the samples from attending to each other.
#[test]
fn packing() -> harpia::Result<()> {
    let samples: Vec<TokenizedSample> = [5usize, 3, 7, 2, 6]
        .iter()
        .enumerate()
        .map(|(i, &n)| TokenizedSample { ids: vec![10 + i as u32; n], loss_mask: vec![true; n] })
        .collect();
    let rows = pack(samples, PackOptions::new(8, 0))?;
    for r in &rows {
        println!("{:?}  boundaries {:?}", r.ids, r.cu_seqlens);
    }
    println!("efficiency {:.0}%", packing_efficiency(&rows) * 100.0);
    assert!(rows.iter().all(|r| r.ids.len() == 8));
    // 23 real tokens in rows of 8: best-fit packing needs only 3 rows (5 unpacked).
    assert_eq!(rows.len(), 3);
    Ok(())
}
