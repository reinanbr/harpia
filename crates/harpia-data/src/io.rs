//! Streaming dataset IO: JSONL and CSV, with transparent gzip (`.gz`) and zstd (`.zst`)
//! decompression. Memory use is constant in the file size.

use harpia_core::{HarpiaError, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

/// Open `path` for reading, decompressing by extension (`.gz`, `.zst`/`.zstd`).
pub fn open_reader(path: &Path) -> Result<Box<dyn BufRead + Send>> {
    let file = File::open(path).map_err(|e| HarpiaError::io(path, e))?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let inner: Box<dyn Read + Send> = match ext.as_str() {
        "gz" | "gzip" => Box::new(flate2::read::MultiGzDecoder::new(BufReader::new(file))),
        "zst" | "zstd" => Box::new(
            ruzstd::decoding::StreamingDecoder::new(BufReader::new(file))
                .map_err(|e| HarpiaError::Other(format!("{}: zstd: {e}", path.display())))?,
        ),
        _ => Box::new(file),
    };
    Ok(Box::new(BufReader::with_capacity(1 << 16, inner)))
}

/// Iterator over typed JSONL records. Blank lines are skipped; errors carry the line number.
pub struct JsonlReader<T> {
    path: Option<PathBuf>,
    reader: Box<dyn BufRead + Send>,
    line: u64,
    buf: String,
    _t: PhantomData<fn() -> T>,
}

impl<T: DeserializeOwned> JsonlReader<T> {
    /// Open a JSONL file (optionally compressed).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Ok(Self {
            path: Some(path.to_path_buf()),
            reader: open_reader(path)?,
            line: 0,
            buf: String::new(),
            _t: PhantomData,
        })
    }

    /// Read from any buffered reader (e.g. stdin).
    pub fn from_reader(reader: impl BufRead + Send + 'static) -> Self {
        Self { path: None, reader: Box::new(reader), line: 0, buf: String::new(), _t: PhantomData }
    }
}

impl<T: DeserializeOwned> Iterator for JsonlReader<T> {
    /// `(1-based line number, record)`.
    type Item = Result<(u64, T)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();
            match self.reader.read_line(&mut self.buf) {
                Ok(0) => return None,
                Ok(_) => {
                    self.line += 1;
                    let line = self.buf.trim();
                    if line.is_empty() {
                        continue;
                    }
                    return Some(serde_json::from_str(line).map(|v| (self.line, v)).map_err(|e| HarpiaError::Record {
                        path: self.path.clone(),
                        line: self.line,
                        message: e.to_string(),
                    }));
                }
                Err(e) => {
                    return Some(Err(HarpiaError::Io { path: self.path.clone(), source: e }));
                }
            }
        }
    }
}

/// Read a whole JSONL file into memory.
pub fn read_jsonl<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    JsonlReader::open(path)?.map(|r| r.map(|(_, v)| v)).collect()
}

/// Buffered JSONL writer that flushes every `flush_every` records.
pub struct JsonlWriter {
    out: BufWriter<Box<dyn Write + Send>>,
    path: Option<PathBuf>,
    written: u64,
    flush_every: u64,
}

impl JsonlWriter {
    /// Create (truncate) `path`. A `.gz` extension writes gzip.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Self::open(path.as_ref(), false)
    }

    /// Append to `path` (created if missing). Not supported for `.gz`.
    pub fn append(path: impl AsRef<Path>) -> Result<Self> {
        Self::open(path.as_ref(), true)
    }

    fn open(path: &Path, append: bool) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| HarpiaError::io(parent, e))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(path)
            .map_err(|e| HarpiaError::io(path, e))?;
        let gz = path.extension().is_some_and(|e| e == "gz");
        if gz && append {
            return Err(HarpiaError::invalid("appending to a .gz file is not supported"));
        }
        let w: Box<dyn Write + Send> = if gz {
            Box::new(flate2::write::GzEncoder::new(file, flate2::Compression::default()))
        } else {
            Box::new(file)
        };
        Ok(Self {
            out: BufWriter::with_capacity(1 << 16, w),
            path: Some(path.to_path_buf()),
            written: 0,
            flush_every: 1000,
        })
    }

    /// Write to any writer (e.g. stdout).
    pub fn from_writer(w: impl Write + Send + 'static) -> Self {
        Self { out: BufWriter::new(Box::new(w)), path: None, written: 0, flush_every: 1000 }
    }

    /// Flush after this many records (default 1000; 1 = after every record).
    pub fn flush_every(mut self, n: u64) -> Self {
        self.flush_every = n.max(1);
        self
    }

    /// Write one record.
    pub fn write<T: Serialize + ?Sized>(&mut self, record: &T) -> Result<()> {
        serde_json::to_writer(&mut self.out, record)?;
        self.out.write_all(b"\n").map_err(|e| self.io_err(e))?;
        self.written += 1;
        if self.written.is_multiple_of(self.flush_every) {
            self.flush()?;
        }
        Ok(())
    }

    /// Records written so far.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Flush buffers to the OS.
    pub fn flush(&mut self) -> Result<()> {
        self.out.flush().map_err(|e| self.io_err(e))
    }

    /// Flush and close (dropping the writer finishes gzip streams).
    pub fn finish(mut self) -> Result<()> {
        self.flush()
    }

    fn io_err(&self, e: std::io::Error) -> HarpiaError {
        HarpiaError::Io { path: self.path.clone(), source: e }
    }
}

impl Drop for JsonlWriter {
    fn drop(&mut self) {
        let _ = self.out.flush();
    }
}

/// Read a CSV/TSV file (delimiter inferred from extension: `.tsv` → tab) as JSON objects with
/// string values, keyed by the header row.
pub fn read_csv(path: impl AsRef<Path>) -> Result<Vec<serde_json::Map<String, serde_json::Value>>> {
    let path = path.as_ref();
    let name = path.to_string_lossy().to_ascii_lowercase();
    let delim = if name.ends_with(".tsv") || name.ends_with(".tsv.gz") { b'\t' } else { b',' };
    let mut rdr = csv::ReaderBuilder::new().delimiter(delim).flexible(true).from_reader(open_reader(path)?);
    let headers = rdr.headers().map_err(|e| HarpiaError::Other(format!("{}: {e}", path.display())))?.clone();
    let mut out = Vec::new();
    for (i, rec) in rdr.records().enumerate() {
        let rec = rec.map_err(|e| HarpiaError::Record {
            path: Some(path.into()),
            line: i as u64 + 2,
            message: e.to_string(),
        })?;
        out.push(
            headers
                .iter()
                .zip(rec.iter())
                .map(|(h, v)| (h.to_string(), serde_json::Value::String(v.to_string())))
                .collect(),
        );
    }
    Ok(out)
}

/// Get a string field by dotted path (`"a.b"`, array indexes allowed: `"messages.0.content"`).
pub fn get_path<'a>(v: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    path.split('.').try_fold(v, |cur, key| match cur {
        serde_json::Value::Object(m) => m.get(key),
        serde_json::Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
        _ => None,
    })
}

/// Set a field by dotted path on objects (creates intermediate objects).
pub fn set_path(v: &mut serde_json::Value, path: &str, value: serde_json::Value) -> Result<()> {
    let mut cur = v;
    let parts: Vec<&str> = path.split('.').collect();
    for (i, key) in parts.iter().enumerate() {
        let obj =
            cur.as_object_mut().ok_or_else(|| HarpiaError::invalid(format!("`{path}`: not an object at `{key}`")))?;
        if i == parts.len() - 1 {
            obj.insert((*key).to_string(), value);
            return Ok(());
        }
        cur = obj.entry(key.to_string()).or_insert_with(|| serde_json::Value::Object(Default::default()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn jsonl_roundtrip_plain_and_gzip() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a.jsonl", "a.jsonl.gz"] {
            let p = dir.path().join(name);
            let mut w = JsonlWriter::create(&p).unwrap();
            w.write(&json!({"text": "olá"})).unwrap();
            w.write(&json!({"text": "mundo"})).unwrap();
            w.finish().unwrap();
            let rows: Vec<Value> = read_jsonl(&p).unwrap();
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0]["text"], "olá");
        }
    }

    #[test]
    fn errors_have_line_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.jsonl");
        std::fs::write(&p, "{\"a\":1}\n\n{oops}\n").unwrap();
        let err = read_jsonl::<Value>(&p).unwrap_err().to_string();
        assert!(err.contains("line 3"), "{err}");
    }

    #[test]
    fn csv_and_paths() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.tsv");
        std::fs::write(&p, "a\tb\n1\tx\n").unwrap();
        let rows = read_csv(&p).unwrap();
        assert_eq!(rows[0]["b"], "x");
        let mut v = json!({"m": [{"c": "hi"}]});
        assert_eq!(get_path(&v, "m.0.c").unwrap(), "hi");
        set_path(&mut v, "x.y", json!(1)).unwrap();
        assert_eq!(v["x"]["y"], 1);
    }
}
