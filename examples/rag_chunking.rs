//! Chunk a Markdown document for RAG and rank chunks with BM25, keeping byte spans for citations.
//!
//! `cargo run -p harpia --example rag_chunking`

use harpia::core::traits::Chunker;
use harpia::data::chunk::MarkdownChunker;
use harpia::text::Lang;
use harpia::text::bm25::Bm25;

const DOC: &str = "# Harpy eagle\nThe harpy eagle is a large raptor of Neotropical forests.\n\n## Diet\nIt hunts sloths and monkeys in the canopy, and sometimes birds.\n\n## Conservation\nDeforestation is the main threat; nests need tall emergent trees.\n";

fn main() -> harpia::Result<()> {
    let chunks = MarkdownChunker { max_chars: 200 }.chunk(DOC)?;
    let texts: Vec<&str> = chunks.iter().map(|c| c.span.slice(DOC)).collect();
    let index = Bm25::new(&texts, &Lang::En.data(), Default::default());
    for (i, score) in index.search("which animals does it hunt in the canopy", 2) {
        let c = &chunks[i];
        println!(
            "{score:.3} [{}..{}] {} :: {}",
            c.span.start,
            c.span.end,
            c.headings.join(" > "),
            texts[i].replace('\n', " ")
        );
    }
    Ok(())
}
