//! Lesson 1: text.
//!
//! Almost every LLM pipeline starts with plain text: cleaning it, cutting it into words and
//! sentences, measuring it, and comparing it with other text. `harpia::text` does all of that
//! in pure Rust and always reports **byte spans** into the original string, so you can point
//! back at the exact place in the source.
//!
//! Run: `cargo test -p harpia-learn --test 01_text -- --nocapture`

use harpia::text::{Lang, fuzzy, normalize, segment, similarity, stats};

/// Normalization returns `Cow<str>`: when nothing changes, nothing is copied.
#[test]
fn normalize_text() {
    // NFKC folds compatibility forms: the full-width "ＡＢＣ" becomes "ABC".
    assert_eq!(normalize::nfkc("ＡＢＣ"), "ABC");
    // Accent stripping and case folding are what you want for matching keys, not for display.
    assert_eq!(normalize::strip_accents("gavião-real"), "gaviao-real");
    assert_eq!(normalize::fold_case("Harpia HARPYJA"), "harpia harpyja");
    assert_eq!(normalize::collapse_whitespace("a  \t b\n\nc"), "a b c");

    // `match_key` lowercases and strips accents: a stable key to compare "the same" strings.
    assert_eq!(normalize::match_key("Gavião REAL"), normalize::match_key("gaviao real"));
}

/// Words and sentences come out as `Span`s (byte ranges), not copies.
#[test]
fn words_and_sentences() {
    let text = "O Dr. Silva viu 2,5 mil aves. Ele contou: \"Que harpia enorme!\" Depois foi embora.";
    let pt = Lang::PtBr.data();

    let words: Vec<&str> = segment::words(text).map(|s| s.slice(text)).take(5).collect();
    println!("first words: {words:?}");
    assert_eq!(words, ["O", "Dr", "Silva", "viu", "2,5"]);

    // "Dr." is a known pt-BR abbreviation and "2,5" a decimal, so neither ends a sentence.
    let sentences: Vec<&str> = segment::sentences(text, &pt).map(|s| s.slice(text)).collect();
    for s in &sentences {
        println!("sentence: {s}");
    }
    assert_eq!(sentences.len(), 3);
    assert_eq!(sentences[0], "O Dr. Silva viu 2,5 mil aves.");

    // A span is just (start, end) in bytes: you can map it to a line and column yourself.
    let span = segment::sentences(text, &pt).nth(1).unwrap();
    assert_eq!(&text[span.start..span.end], sentences[1]);
}

/// Readability and quality signals are what dataset filters (and article reviewers) use.
#[test]
fn readability_and_signals() {
    let easy = "O gavião come. A ave voa alto. O ninho fica na árvore. A mata é grande. \
                O filhote cresce. O pai caça. A mãe cuida. A floresta vive. O sol nasce. \
                A chuva cai. O rio corre. A vida segue.";
    let hard = "A conservação da biodiversidade neotropical pressupõe a manutenção da \
                conectividade estrutural e funcional entre remanescentes florestais \
                heterogêneos, considerando simultaneamente as interações tróficas \
                multiescalares. Consequentemente, intervenções administrativas \
                descoordenadas comprometem irreversivelmente a resiliência ecossistêmica.";
    let pt = Lang::PtBr.data();
    let fe = stats::flesch(easy, &pt).expect("long enough");
    let fh = stats::flesch(hard, &pt).expect("long enough");
    println!("Flesch (pt-BR): easy {:.1}, hard {:.1}", fe.score, fh.score);
    assert!(fe.score > fh.score, "short words and sentences read more easily");

    // `signals` bundles the character, line and stopword ratios used by quality filters.
    let s = stats::signals(easy, &pt);
    println!("words {}, mean word length {:.2}, stopword ratio {:.2}", s.words, s.mean_word_len, s.stopword_ratio);
    assert!(s.words > 20);

    // Lexical diversity: MATTR over a sliding window (1.0 = no word repeats in a window).
    // It needs at least 20 words.
    let repetitive = "a ave voa a ave voa a ave voa a ave voa a ave voa a ave voa a ave voa";
    assert!(stats::mattr(easy, 10).unwrap() > stats::mattr(repetitive, 10).unwrap());
}

/// Fuzzy matching (like Python's rapidfuzz) is handy for titles, names and quotes.
#[test]
fn fuzzy_matching() {
    let a = "Aves da Amazônia";
    let b = "Aves da Amazonia (2a edição)";
    println!("ratio {:.2}, partial {:.2}", fuzzy::ratio(a, b), fuzzy::partial_ratio(a, b));
    assert!(fuzzy::partial_ratio("Aves da Amazonia", b) > 0.95, "the short string is inside the long one");
    // Word order does not matter for token_sort_ratio.
    assert!((fuzzy::token_sort_ratio("Silva Ana", "Ana Silva") - 1.0).abs() < 1e-9);
}

/// Shingles + Jaccard/containment find copied passages; MinHash makes it scale.
#[test]
fn similarity_and_copy_detection() {
    let source = "a harpia vive nas florestas tropicais e caça preguiças e macacos no dossel";
    let copied = "segundo o texto, a harpia vive nas florestas tropicais e caça preguiças";
    let other = "o tucano come frutas e sementes nas bordas da mata";

    let k = 4; // 4-word shingles
    let (s, c, o) =
        (similarity::shingle_set(source, k), similarity::shingle_set(copied, k), similarity::shingle_set(other, k));
    let copied_in_source = similarity::containment(&c, &s);
    println!("containment copied->source {copied_in_source:.2}, other->source {:.2}", similarity::containment(&o, &s));
    assert!(copied_in_source > 0.5);
    assert_eq!(similarity::containment(&o, &s), 0.0);

    // Where exactly is the copied run? `longest_shared_span` answers with a byte span.
    let toks = similarity::tokens(copied);
    let seq = similarity::shingles(&toks, k);
    let run = similarity::longest_shared_span(&seq, &s, &toks, k).unwrap();
    println!("longest shared run ({} words): {:?}", run.words, run.span.slice(copied));
    assert!(run.span.slice(copied).starts_with("a harpia vive"));

    // MinHash estimates Jaccard from fixed-size signatures (what near-dedup uses at scale).
    let mh = similarity::MinHasher::new(128, 42);
    let est = similarity::minhash_similarity(&mh.signature_of(source, k), &mh.signature_of(copied, k));
    println!("MinHash estimate {est:.2} vs exact Jaccard {:.2}", similarity::jaccard(&s, &c));
}
