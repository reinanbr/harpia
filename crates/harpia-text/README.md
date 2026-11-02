# harpia-text

Text primitives used by Harpia's data curation, RAG and evaluation:

- Unicode normalization returning `Cow<str>` (no allocation when nothing changes);
- zero-copy word and sentence segmentation with byte offsets and per-language abbreviation
  lists (English and Brazilian Portuguese bundled);
- quality signals for data filtering (Flesch, MATTR, repetition, symbol ratios, ...);
- shingles, Jaccard/containment, longest shared span, fingerprint index, MinHash;
- rapidfuzz-style fuzzy matching and BM25.
