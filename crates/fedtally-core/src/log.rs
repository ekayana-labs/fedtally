//! The log file: one signed record per line, chained by hash.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::vector::Digest;

pub const GENESIS: Digest = Digest([0u8; 32]);

/// The head after appending `line` to `head`.
pub fn chain(head: &Digest, line: &[u8]) -> Digest {
    let mut hasher = blake3::Hasher::new_derive_key("fedtally v1 log");
    hasher.update(&head.0);
    hasher.update(line);
    Digest(*hasher.finalize().as_bytes())
}

/// An append-only local replica. Each append reaches the disk before it returns.
pub struct LogFile {
    file: File,
    path: PathBuf,
}

impl LogFile {
    pub fn open(path: &Path) -> io::Result<LogFile> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        // The file itself has to survive a power cut, not only what's in it.
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            File::open(dir)?.sync_all()?;
        }
        Ok(LogFile {
            file,
            path: path.to_owned(),
        })
    }

    pub fn append(&mut self, line: &[u8]) -> io::Result<()> {
        let mut framed = Vec::with_capacity(line.len() + 1);
        framed.extend_from_slice(line);
        framed.push(b'\n');
        self.file.write_all(&framed)?;
        self.file.sync_data()
    }

    /// Reads every complete line. A torn final line from a crash is cut off,
    /// so the next append starts clean.
    pub fn read_all(&mut self) -> io::Result<Vec<Vec<u8>>> {
        let mut lines = Vec::new();
        let mut reader = BufReader::new(File::open(&self.path)?);
        let mut good = 0u64;
        loop {
            let mut line = Vec::new();
            let n = reader.read_until(b'\n', &mut line)?;
            if n == 0 || line.last() != Some(&b'\n') {
                break;
            }
            good += n as u64;
            line.pop();
            lines.push(line);
        }
        if good != self.file.metadata()?.len() {
            self.file.set_len(good)?;
            self.file.sync_data()?;
        }
        Ok(lines)
    }
}

pub fn read_lines(path: &Path) -> io::Result<Vec<Vec<u8>>> {
    let reader = BufReader::new(File::open(path)?);
    let mut lines = Vec::new();
    for line in reader.split(b'\n') {
        lines.push(line?);
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_depends_on_order() {
        let a = chain(&chain(&GENESIS, b"x"), b"y");
        let b = chain(&chain(&GENESIS, b"y"), b"x");
        assert_ne!(a, b);
    }

    #[test]
    fn torn_tail_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");
        {
            let mut log = LogFile::open(&path).unwrap();
            log.append(b"one").unwrap();
            log.append(b"two").unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(b"thr");
        std::fs::write(&path, bytes).unwrap();

        let mut log = LogFile::open(&path).unwrap();
        assert_eq!(
            log.read_all().unwrap(),
            vec![b"one".to_vec(), b"two".to_vec()]
        );
        log.append(b"three").unwrap();
        assert_eq!(
            read_lines(&path).unwrap(),
            vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()]
        );
    }
}
