//! Update vectors on disk: chunked files of little-endian field elements.
//!
//! Everything here works one chunk at a time, so memory use depends on the
//! chunk size and the thread count, not on the model size or the site count.

use std::cell::RefCell;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

use rand_chacha::ChaCha12Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::field::{Fp, Fp2, P};

#[cfg(target_endian = "big")]
compile_error!("the vector format is little-endian and is read without conversion");

const WORD: usize = size_of::<u64>();

/// A 32 byte BLAKE3 digest, shown as hex.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    pub fn of(data: &[u8]) -> Digest {
        Digest(*blake3::hash(data).as_bytes())
    }

    pub fn file(path: &Path) -> io::Result<Digest> {
        let mut hasher = blake3::Hasher::new();
        hasher.update_reader(File::open(path)?)?;
        Ok(Digest(*hasher.finalize().as_bytes()))
    }

    pub fn hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.hex()[..12])
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.hex())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Digest, D::Error> {
        use serde::de::Error;
        let text = String::deserialize(d)?;
        let mut out = [0u8; 32];
        hex::decode_to_slice(&text, &mut out).map_err(D::Error::custom)?;
        Ok(Digest(out))
    }
}

/// How a vector divides into chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub coords: u64,
    pub chunk_coords: u64,
}

impl Shape {
    pub fn chunks(&self) -> u64 {
        self.coords.div_ceil(self.chunk_coords)
    }

    pub fn range(&self, chunk: u64) -> (u64, u64) {
        let start = chunk * self.chunk_coords;
        (start, (start + self.chunk_coords).min(self.coords))
    }

    pub fn chunk_bytes(&self, chunk: u64) -> usize {
        let (start, end) = self.range(chunk);
        (end - start) as usize * WORD
    }

    fn offset(&self, chunk: u64) -> u64 {
        chunk * self.chunk_coords * WORD as u64
    }
}

/// Chunk digests and the root that commits to them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub root: Digest,
    pub chunks: Vec<Digest>,
}

impl Manifest {
    pub fn new(chunks: Vec<Digest>) -> Manifest {
        Manifest {
            root: Self::root_of(&chunks),
            chunks,
        }
    }

    pub fn root_of(chunks: &[Digest]) -> Digest {
        let mut hasher = blake3::Hasher::new_derive_key("fedtally v1 manifest");
        hasher.update(&(chunks.len() as u64).to_le_bytes());
        for chunk in chunks {
            hasher.update(&chunk.0);
        }
        Digest(*hasher.finalize().as_bytes())
    }
}

pub fn digest_chunk(data: &[u64]) -> Digest {
    Digest::of(bytemuck::cast_slice(data))
}

pub fn read_chunk(file: &File, shape: Shape, chunk: u64, buf: &mut Vec<u64>) -> io::Result<()> {
    buf.resize(shape.chunk_bytes(chunk) / WORD, 0);
    file.read_exact_at(bytemuck::cast_slice_mut(buf), shape.offset(chunk))
}

pub fn chunk_bytes(data: &[u64]) -> &[u8] {
    bytemuck::cast_slice(data)
}

pub fn write_chunk_bytes(file: &File, shape: Shape, chunk: u64, data: &[u8]) -> io::Result<()> {
    file.write_all_at(data, shape.offset(chunk))
}

thread_local! {
    static SCRATCH: RefCell<(Vec<u64>, Vec<u64>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// Fills `path` chunk by chunk, in parallel.
pub fn build<F>(path: &Path, shape: Shape, fill: F) -> io::Result<Manifest>
where
    F: Fn(u64, &mut [u64], &mut Vec<u64>) -> io::Result<()> + Sync,
{
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    let chunks: io::Result<Vec<Digest>> = (0..shape.chunks())
        .into_par_iter()
        .map(|chunk| {
            SCRATCH.with_borrow_mut(|(buf, scratch)| {
                buf.clear();
                buf.resize(shape.chunk_bytes(chunk) / WORD, 0);
                fill(chunk, buf, scratch)?;
                file.write_all_at(bytemuck::cast_slice(buf), shape.offset(chunk))?;
                Ok(digest_chunk(buf))
            })
        })
        .collect();
    file.sync_data()?;
    Ok(Manifest::new(chunks?))
}

/// Sums vectors: `add` are added, `sub` subtracted.
pub fn combine(out: &Path, shape: Shape, add: &[&Path], sub: &[&Path]) -> io::Result<Manifest> {
    let open =
        |paths: &[&Path]| -> io::Result<Vec<File>> { paths.iter().map(File::open).collect() };
    let (add, sub) = (open(add)?, open(sub)?);
    build(out, shape, |chunk, buf, scratch| {
        for file in &add {
            read_chunk(file, shape, chunk, scratch)?;
            for (acc, v) in buf.iter_mut().zip(scratch.iter()) {
                *acc = (Fp::new(*acc) + Fp::new(*v)).value();
            }
        }
        for file in &sub {
            read_chunk(file, shape, chunk, scratch)?;
            for (acc, v) in buf.iter_mut().zip(scratch.iter()) {
                *acc = (Fp::new(*acc) - Fp::new(*v)).value();
            }
        }
        Ok(())
    })
}

pub fn manifest_of(path: &Path, shape: Shape) -> io::Result<Manifest> {
    let file = File::open(path)?;
    let chunks: io::Result<Vec<Digest>> = (0..shape.chunks())
        .into_par_iter()
        .map(|chunk| {
            SCRATCH.with_borrow_mut(|(buf, _)| {
                read_chunk(&file, shape, chunk, buf)?;
                Ok(digest_chunk(buf))
            })
        })
        .collect();
    Ok(Manifest::new(chunks?))
}

/// Adds the pseudorandom mask stream for `seed` to one chunk.
/// Each chunk has its own stream, so chunks can be masked in any order.
pub fn apply_mask(seed: &[u8; 32], chunk: u64, values: &mut [u64]) {
    let mut rng = ChaCha12Rng::from_seed(*seed);
    rng.set_stream(chunk);
    for v in values {
        *v = (Fp::new(*v) + uniform(&mut rng)).value();
    }
}

fn uniform(rng: &mut ChaCha12Rng) -> Fp {
    loop {
        let v = rng.next_u64();
        if v < P {
            return Fp::new(v);
        }
    }
}

/// The fingerprint of `values` at `rho`, as a polynomial in `rho`.
pub fn fingerprint_chunk(values: &[u64], rho: Fp2) -> Fp2 {
    values
        .iter()
        .rev()
        .fold(Fp2::ZERO, |acc, v| (acc * rho).add_base(Fp::new(*v)))
}

fn combine_fingerprints(
    shape: Shape,
    rho: Fp2,
    chunk_value: impl Fn(u64) -> io::Result<Fp2> + Sync,
) -> io::Result<Fp2> {
    let step = rho.pow(shape.chunk_coords);
    (0..shape.chunks())
        .into_par_iter()
        .map(|chunk| Ok(step.pow(chunk) * chunk_value(chunk)?))
        .try_reduce(|| Fp2::ZERO, |a, b| Ok(a + b))
}

pub fn fingerprint_file(path: &Path, shape: Shape, rho: Fp2) -> io::Result<Fp2> {
    let file = File::open(path)?;
    combine_fingerprints(shape, rho, |chunk| {
        SCRATCH.with_borrow_mut(|(buf, _)| {
            read_chunk(&file, shape, chunk, buf)?;
            Ok(fingerprint_chunk(buf, rho))
        })
    })
}

/// The fingerprint of the mask stream for `seed`, with no file behind it.
pub fn fingerprint_mask(seed: &[u8; 32], shape: Shape, rho: Fp2) -> io::Result<Fp2> {
    combine_fingerprints(shape, rho, |chunk| {
        SCRATCH.with_borrow_mut(|(buf, _)| {
            buf.clear();
            buf.resize(shape.chunk_bytes(chunk) / WORD, 0);
            apply_mask(seed, chunk, buf);
            Ok(fingerprint_chunk(buf, rho))
        })
    })
}

/// The challenge point derived from the log head. Its `b` limb is never zero,
/// which is what makes the pad coordinates hide a fingerprint.
pub fn challenge(head: &Digest) -> Fp2 {
    let mut reader = blake3::Hasher::new_derive_key("fedtally v1 challenge")
        .update(&head.0)
        .finalize_xof();
    let mut limb = || loop {
        let mut bytes = [0u8; 8];
        reader.fill(&mut bytes);
        let v = u64::from_le_bytes(bytes);
        if v < P {
            return Fp::new(v);
        }
    };
    let a = limb();
    loop {
        let b = limb();
        if b != Fp::ZERO {
            return Fp2 { a, b };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, shape: Shape, values: &[u64]) -> Manifest {
        build(path, shape, |chunk, buf, _| {
            let (start, end) = shape.range(chunk);
            buf.copy_from_slice(&values[start as usize..end as usize]);
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn parallel_fingerprint_matches_the_polynomial() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v");
        let values: Vec<u64> = (0..1000u64).map(|i| i * i + 7).collect();
        let shape = Shape {
            coords: 1000,
            chunk_coords: 64,
        };
        write(&path, shape, &values);
        let rho = challenge(&Digest::of(b"x"));
        assert_eq!(
            fingerprint_file(&path, shape, rho).unwrap(),
            fingerprint_chunk(&values, rho)
        );
    }

    #[test]
    fn fingerprint_is_linear_over_combine() {
        let dir = tempfile::tempdir().unwrap();
        let shape = Shape {
            coords: 500,
            chunk_coords: 100,
        };
        let (a, b, c) = (
            dir.path().join("a"),
            dir.path().join("b"),
            dir.path().join("c"),
        );
        write(&a, shape, &(0..500).map(|i| P - 1 - i).collect::<Vec<_>>());
        write(&b, shape, &(0..500).map(|i| i * 3).collect::<Vec<_>>());
        combine(&c, shape, &[&a, &b], &[]).unwrap();
        let rho = challenge(&Digest::of(b"y"));
        let fp = |p: &Path| fingerprint_file(p, shape, rho).unwrap();
        assert_eq!(fp(&c), fp(&a) + fp(&b));
        let d = dir.path().join("d");
        combine(&d, shape, &[&c], &[&b]).unwrap();
        assert_eq!(fp(&d), fp(&a));
    }

    #[test]
    fn one_changed_coordinate_changes_the_fingerprint() {
        let rho = challenge(&Digest::of(b"z"));
        let mut values = vec![5u64; 200];
        let before = fingerprint_chunk(&values, rho);
        values[133] += 1;
        assert_ne!(before, fingerprint_chunk(&values, rho));
    }

    #[test]
    fn masks_are_per_chunk_and_deterministic() {
        let seed = [9u8; 32];
        let (mut a, mut b, mut c) = (vec![0u64; 16], vec![0u64; 16], vec![0u64; 16]);
        apply_mask(&seed, 3, &mut a);
        apply_mask(&seed, 3, &mut b);
        apply_mask(&seed, 4, &mut c);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|v| *v < P));
    }

    #[test]
    fn mask_fingerprint_matches_a_materialised_mask() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m");
        let shape = Shape {
            coords: 300,
            chunk_coords: 64,
        };
        let seed = [1u8; 32];
        build(&path, shape, |chunk, buf, _| {
            apply_mask(&seed, chunk, buf);
            Ok(())
        })
        .unwrap();
        let rho = challenge(&Digest::of(b"m"));
        assert_eq!(
            fingerprint_mask(&seed, shape, rho).unwrap(),
            fingerprint_file(&path, shape, rho).unwrap()
        );
    }

    #[test]
    fn digests_of_files_and_bytes_agree_and_print_as_hex() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        std::fs::write(&path, b"contents").unwrap();
        let digest = Digest::of(b"contents");
        assert_eq!(Digest::file(&path).unwrap(), digest);
        assert_eq!(digest.to_string().len(), 64);
        assert_eq!(digest.to_string(), digest.hex());
        assert_eq!(format!("{digest:?}"), digest.hex()[..12]);
    }

    #[test]
    fn built_files_have_the_full_length_and_chunks_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v");
        let shape = Shape {
            coords: 100,
            chunk_coords: 64,
        };
        let manifest = build(&path, shape, |chunk, buf, _| {
            buf.fill(chunk + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 800);
        assert_ne!(manifest.chunks[0], manifest.chunks[1]);
        assert_eq!(manifest.chunks[0], digest_chunk(&[1u64; 64]));

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut buf = Vec::new();
        read_chunk(&file, shape, 1, &mut buf).unwrap();
        assert_eq!(buf, vec![2u64; 36]);
        assert_eq!(chunk_bytes(&buf).len(), 288);
        assert_eq!(chunk_bytes(&buf)[0], 2);

        let replacement = vec![9u64; 36];
        write_chunk_bytes(&file, shape, 1, chunk_bytes(&replacement)).unwrap();
        read_chunk(&file, shape, 1, &mut buf).unwrap();
        assert_eq!(buf, replacement);
        read_chunk(&file, shape, 0, &mut buf).unwrap();
        assert_eq!(buf, vec![1u64; 64]);
    }

    #[test]
    fn manifest_commits_to_order() {
        let (a, b) = (Digest::of(b"a"), Digest::of(b"b"));
        assert_ne!(Manifest::root_of(&[a, b]), Manifest::root_of(&[b, a]));
        assert_ne!(Manifest::root_of(&[a]), Manifest::root_of(&[a, a]));
    }

    #[test]
    fn manifest_of_matches_build() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v");
        let shape = Shape {
            coords: 130,
            chunk_coords: 32,
        };
        let built = write(&path, shape, &(0..130).collect::<Vec<_>>());
        assert_eq!(manifest_of(&path, shape).unwrap(), built);
    }

    #[test]
    fn challenge_never_lands_in_the_base_field() {
        for i in 0..1000u32 {
            assert_ne!(challenge(&Digest::of(&i.to_le_bytes())).b, Fp::ZERO);
        }
    }
}
