//! Print the paragraphs extracted from a LaTeX project: `cargo run -p harpia-pub --example dump -- <dir>`.
fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let p = harpia_pub::load(std::path::Path::new(&dir), &Default::default()).expect("load");
    println!(
        "lang={} ({:?}) files={} bibs={} entries={}",
        p.lang,
        p.lang_source,
        p.tex.len(),
        p.bibs.len(),
        p.bib_entries().len()
    );
    for para in p.paragraphs(None) {
        let loc = para.location(0, None);
        let t: String = para.text().chars().take(110).collect();
        println!("{:>4} {:<8} {:<3} {}", loc.line, format!("{:?}", para.kind()), para.lang(), t.replace('\n', " "));
    }
}
