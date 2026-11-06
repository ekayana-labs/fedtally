//! Update vectors on disk: chunked files of little-endian field elements.
//!
//! Everything here works one chunk at a time, so memory use depends on the
//! chunk size and the thread count, not on the model size or the site count.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use chacha20::ChaCha12Rng;
use chacha20::rand_core::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::field::{Fp, Fp2, P};
use crate::merkle;

#[cfg(target_endian = "big")]
compile_error!("the vector format is little-endian and is read without conversion");

const WORD: usize = size_of::<u64>();
/// How much of a file is read at once to hash it on every core.
const FILE_BLOCK: usize = 1 << 23;
/// Below this a chunk is hashed where it is, since handing leaves to other
/// threads would cost more than hashing them.
const PARALLEL_HASH_BYTES: usize = 1 << 16;
/// Coordinates in one Merkle leaf.
pub const LEAF_COORDS: u64 = (merkle::LEAF_BYTES / WORD) as u64;

/// A 32 byte digest, shown as hex.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    pub fn of(data: &[u8]) -> Digest {
        Digest(*blake3::hash(data).as_bytes())
    }

    pub fn file(path: &Path) -> io::Result<Digest> {
        let mut hasher = blake3::Hasher::new();
        let mut file = File::open(path)?;
        let mut block = Vec::with_capacity(FILE_BLOCK);
        loop {
            block.clear();
            let mut part = file.by_ref().take(FILE_BLOCK as u64);
            if part.read_to_end(&mut block)? == 0 {
                break;
            }
            hasher.update_rayon(&block);
        }
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

    pub fn leaves(&self) -> u64 {
        self.coords.div_ceil(LEAF_COORDS)
    }

    /// True when every chunk is a whole power-of-two number of leaves. The
    /// root is then the plain RFC 6962 tree over the vector's leaves, and a
    /// single leaf can be proven against it.
    pub fn aligned(&self) -> bool {
        self.chunk_coords.is_multiple_of(LEAF_COORDS)
            && (self.chunk_coords / LEAF_COORDS).is_power_of_two()
    }
}

/// One leaf of a vector, with the hashes that tie it to the vector's root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafProof {
    pub index: u64,
    pub values: Vec<u64>,
    pub path: Vec<merkle::Hash>,
}

impl LeafProof {
    /// True when this is leaf `index` of the vector with that root and shape.
    pub fn fits(&self, root: &Digest, shape: Shape) -> bool {
        let (start, end) = leaf_range(shape, self.index);
        self.index < shape.leaves()
            && self.values.len() as u64 == end - start
            && merkle::climb(
                merkle::leaf(bytemuck::cast_slice(&self.values)),
                self.index,
                shape.leaves(),
                &self.path,
            ) == Some(root.0)
    }
}

fn leaf_range(shape: Shape, leaf: u64) -> (u64, u64) {
    let start = leaf * LEAF_COORDS;
    (start, (start + LEAF_COORDS).min(shape.coords))
}

fn unaligned() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "single leaves can only be proven when chunks are 128 * 2^k coordinates",
    )
}

/// Proves leaf `leaf` of the vector at `path`, whose chunk digests are `manifest`.
pub fn prove_leaf(
    path: &Path,
    shape: Shape,
    manifest: &Manifest,
    leaf: u64,
) -> io::Result<LeafProof> {
    if !shape.aligned() || leaf >= shape.leaves() {
        return Err(unaligned());
    }
    let per_chunk = shape.chunk_coords / LEAF_COORDS;
    let (chunk, within) = (leaf / per_chunk, (leaf % per_chunk) as usize);
    let mut data = Vec::new();
    read_chunk(&File::open(path)?, shape, chunk, &mut data)?;
    let leaves: Vec<merkle::Hash> = bytemuck::cast_slice::<u64, u8>(&data)
        .chunks(merkle::LEAF_BYTES)
        .map(merkle::leaf)
        .collect();
    let mut hashes = merkle::path(leaves, within);
    let chunk_roots = manifest.chunks.iter().map(|c| c.0).collect();
    hashes.extend(merkle::path(chunk_roots, chunk as usize));
    let from = within * LEAF_COORDS as usize;
    let to = (from + LEAF_COORDS as usize).min(data.len());
    Ok(LeafProof {
        index: leaf,
        values: data[from..to].to_vec(),
        path: hashes,
    })
}

/// The first leaf where `aggregate` is not the sum of `add` minus the sum of
/// `sub`, if there is one.
pub fn first_wrong_leaf(
    shape: Shape,
    add: &[&Path],
    sub: &[&Path],
    aggregate: &Path,
) -> io::Result<Option<u64>> {
    let every: Vec<u64> = (0..shape.chunks()).collect();
    first_wrong_leaf_in(shape, add, sub, aggregate, &every)
}

/// `first_wrong_leaf` over some chunks only, so the files need hold only those.
pub fn first_wrong_leaf_in(
    shape: Shape,
    add: &[&Path],
    sub: &[&Path],
    aggregate: &Path,
    chunks: &[u64],
) -> io::Result<Option<u64>> {
    if !shape.aligned() {
        return Err(unaligned());
    }
    let open =
        |paths: &[&Path]| -> io::Result<Vec<File>> { paths.iter().map(File::open).collect() };
    let (add, sub, aggregate) = (open(add)?, open(sub)?, File::open(aggregate)?);
    let per_chunk = shape.chunk_coords / LEAF_COORDS;
    let wrong: io::Result<Vec<Option<u64>>> = chunks
        .par_iter()
        .map(|&chunk| {
            SCRATCH.with_borrow_mut(|(buf, scratch)| {
                buf.clear();
                buf.resize(shape.chunk_bytes(chunk) / WORD, 0);
                sum_chunk(shape, chunk, &add, &sub, buf, scratch)?;
                read_chunk(&aggregate, shape, chunk, scratch)?;
                let differs = buf
                    .iter()
                    .zip(scratch.iter())
                    .position(|(sum, stated)| Fp::new(*sum) != Fp::new(*stated));
                Ok(differs.map(|at| chunk * per_chunk + at as u64 / LEAF_COORDS))
            })
        })
        .collect();
    Ok(wrong?.into_iter().flatten().min())
}

/// Whether the aggregate is not the sum at one leaf. The files need hold only
/// that leaf.
pub fn wrong_at_leaf(
    shape: Shape,
    add: &[&Path],
    sub: &[&Path],
    aggregate: &Path,
    leaf: u64,
) -> io::Result<bool> {
    if !shape.aligned() || leaf >= shape.leaves() {
        return Err(unaligned());
    }
    let start = leaf * LEAF_COORDS;
    let length = (shape.coords - start).min(LEAF_COORDS) as usize * WORD;
    let read = |path: &Path| -> io::Result<Vec<Fp>> {
        let mut bytes = [0u8; merkle::LEAF_BYTES];
        File::open(path)?.read_exact_at(&mut bytes[..length], start * WORD as u64)?;
        let (words, _) = bytes[..length].as_chunks::<WORD>();
        Ok(words
            .iter()
            .map(|word| Fp::new(u64::from_le_bytes(*word)))
            .collect())
    };
    let mut sum = vec![Fp::new(0); length / WORD];
    for path in add {
        for (sum, value) in sum.iter_mut().zip(read(path)?) {
            *sum = *sum + value;
        }
    }
    for path in sub {
        for (sum, value) in sum.iter_mut().zip(read(path)?) {
            *sum = *sum - value;
        }
    }
    Ok(sum != read(aggregate)?)
}

/// Chunk digests and the root that commits to them. Each chunk digest is the
/// Merkle root of that chunk, and the root is the tree over all of them.
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
        Digest(merkle::root(chunks.iter().map(|c| c.0).collect()))
    }
}

pub fn digest_chunk(data: &[u64]) -> Digest {
    digest_chunk_bytes(bytemuck::cast_slice(data))
}

/// Transfers hashing a chunk across the pool at this moment.
static FANNED_OUT: AtomicUsize = AtomicUsize::new(0);

/// With more transfers than this at once, each hashes on its own thread.
/// Sharing the pool between many of them is slower than not sharing it.
const FAN_OUT_LIMIT: usize = 4;

/// The Merkle root of one chunk. A caller on the thread pool is already one of
/// many hashing chunks side by side. Any other caller, such as a transfer
/// checking one chunk at a time, has its leaves hashed across the pool while
/// few transfers are running.
pub fn digest_chunk_bytes(bytes: &[u8]) -> Digest {
    let alone = rayon::current_thread_index().is_none() && bytes.len() >= PARALLEL_HASH_BYTES;
    let leaves = if alone && FANNED_OUT.fetch_add(1, Ordering::Relaxed) < FAN_OUT_LIMIT {
        let leaves = bytes
            .par_chunks(merkle::LEAF_BYTES)
            .with_min_len(16)
            .map(merkle::leaf)
            .collect();
        FANNED_OUT.fetch_sub(1, Ordering::Relaxed);
        leaves
    } else {
        if alone {
            FANNED_OUT.fetch_sub(1, Ordering::Relaxed);
        }
        bytes.chunks(merkle::LEAF_BYTES).map(merkle::leaf).collect()
    };
    Digest(merkle::root(leaves))
}

pub fn read_chunk(file: &File, shape: Shape, chunk: u64, buf: &mut Vec<u64>) -> io::Result<()> {
    buf.resize(shape.chunk_bytes(chunk) / WORD, 0);
    file.read_exact_at(bytemuck::cast_slice_mut(buf), shape.offset(chunk))
}

/// Reads a chunk as the bytes that go on the wire.
pub fn read_chunk_bytes(file: &File, shape: Shape, chunk: u64) -> io::Result<Vec<u8>> {
    let mut bytes = vec![0u8; shape.chunk_bytes(chunk)];
    file.read_exact_at(&mut bytes, shape.offset(chunk))?;
    Ok(bytes)
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
        sum_chunk(shape, chunk, &add, &sub, buf, scratch)
    })
}

/// Adds one chunk of every file in `add` to `acc` and takes off every file in `sub`.
fn sum_chunk(
    shape: Shape,
    chunk: u64,
    add: &[File],
    sub: &[File],
    acc: &mut [u64],
    scratch: &mut Vec<u64>,
) -> io::Result<()> {
    for file in add {
        read_chunk(file, shape, chunk, scratch)?;
        for (acc, v) in acc.iter_mut().zip(scratch.iter()) {
            *acc = (Fp::new(*acc) + Fp::new(*v)).value();
        }
    }
    for file in sub {
        read_chunk(file, shape, chunk, scratch)?;
        for (acc, v) in acc.iter_mut().zip(scratch.iter()) {
            *acc = (Fp::new(*acc) - Fp::new(*v)).value();
        }
    }
    Ok(())
}

/// The chunks of `file` that are not the ones `manifest` names.
pub fn missing_chunks(file: &File, shape: Shape, manifest: &Manifest) -> io::Result<BTreeSet<u64>> {
    (0..shape.chunks())
        .into_par_iter()
        .filter_map(|chunk| {
            SCRATCH.with_borrow_mut(|(buf, _)| {
                if let Err(e) = read_chunk(file, shape, chunk, buf) {
                    return Some(Err(e));
                }
                (manifest.chunks.get(chunk as usize) != Some(&digest_chunk(buf)))
                    .then_some(Ok(chunk))
            })
        })
        .collect()
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
    let mut rng = mask_stream(seed, chunk);
    let mut mask = [0u64; MASK_BLOCK];
    for part in values.chunks_mut(MASK_BLOCK) {
        let mask = &mut mask[..part.len()];
        fill_uniform(&mut rng, mask);
        for (v, m) in part.iter_mut().zip(mask.iter()) {
            *v = (Fp::new(*v) + Fp::new(*m)).value();
        }
    }
}

/// Takes the mask stream for `seed` back off one chunk.
pub fn remove_mask(seed: &[u8; 32], chunk: u64, values: &mut [u64]) {
    let mut rng = mask_stream(seed, chunk);
    let mut mask = [0u64; MASK_BLOCK];
    for part in values.chunks_mut(MASK_BLOCK) {
        let mask = &mut mask[..part.len()];
        fill_uniform(&mut rng, mask);
        for (v, m) in part.iter_mut().zip(mask.iter()) {
            *v = (Fp::new(*v) - Fp::new(*m)).value();
        }
    }
}

/// Writes the mask stream itself, as `apply_mask` would add it to zeros.
fn fill_mask(seed: &[u8; 32], chunk: u64, values: &mut [u64]) {
    let mut rng = mask_stream(seed, chunk);
    for part in values.chunks_mut(MASK_BLOCK) {
        fill_uniform(&mut rng, part);
    }
}

/// Mask coordinates drawn at a time. They stay in the first-level cache
/// between being drawn and being added.
const MASK_BLOCK: usize = 512;

fn mask_stream(seed: &[u8; 32], chunk: u64) -> ChaCha12Rng {
    let mut rng = ChaCha12Rng::from_seed(*seed);
    rng.set_stream(chunk);
    rng
}

/// Fills `out` with field elements: the generator's 64-bit words in order,
/// leaving out any that are not below p. Nearly always the keystream is the
/// answer as it stands, since a word is left out once in 2^32.
fn fill_uniform(rng: &mut impl Rng, out: &mut [u64]) {
    rng.fill_bytes(bytemuck::cast_slice_mut(out));
    if out.iter().all(|v| *v < P) {
        return;
    }
    let mut kept = 0;
    for i in 0..out.len() {
        if out[i] < P {
            out[kept] = out[i];
            kept += 1;
        }
    }
    while kept < out.len() {
        let v = rng.next_u64();
        if v < P {
            out[kept] = v;
            kept += 1;
        }
    }
}

/// Coordinates fingerprinted between reductions.
const BLOCK: usize = 256;

/// The first powers of `rho`, limb by limb. With them the fingerprint of a
/// block is two dot products over F_p, and blocks are joined by Horner's rule.
struct Powers {
    a: Vec<Fp>,
    b: Vec<Fp>,
    step: Fp2,
}

impl Powers {
    fn new(rho: Fp2, longest: usize) -> Powers {
        let count = longest.min(BLOCK);
        let (mut a, mut b) = (Vec::with_capacity(count), Vec::with_capacity(count));
        let mut power = Fp2::ONE;
        for _ in 0..count {
            a.push(power.a);
            b.push(power.b);
            power = power * rho;
        }
        Powers { a, b, step: power }
    }

    fn fingerprint(&self, values: &[u64]) -> Fp2 {
        values.chunks(BLOCK).rev().fold(Fp2::ZERO, |acc, block| {
            let [a, b] = Fp::dot(block, [&self.a, &self.b]);
            acc * self.step + Fp2 { a, b }
        })
    }
}

/// The fingerprint of `values` at `rho`, as a polynomial in `rho`.
pub fn fingerprint_chunk(values: &[u64], rho: Fp2) -> Fp2 {
    Powers::new(rho, values.len()).fingerprint(values)
}

/// Fingerprints a vector a chunk at a time, in any order. The vector's
/// fingerprint is the sum over its chunks, so it can be worked out as the
/// chunks arrive, with no file behind them.
pub struct Fingerprinter {
    step: Fp2,
    powers: Powers,
}

impl Fingerprinter {
    pub fn new(shape: Shape, rho: Fp2) -> Fingerprinter {
        Fingerprinter {
            step: rho.pow(shape.chunk_coords),
            powers: Powers::new(rho, shape.chunk_coords.min(shape.coords) as usize),
        }
    }

    /// What chunk `chunk` adds to the fingerprint.
    pub fn chunk(&self, chunk: u64, values: &[u64]) -> Fp2 {
        self.step.pow(chunk) * self.powers.fingerprint(values)
    }

    /// The same, for a chunk as it came off the wire.
    pub fn chunk_bytes(&self, chunk: u64, bytes: &[u8]) -> Fp2 {
        match bytemuck::try_cast_slice(bytes) {
            Ok(values) => self.chunk(chunk, values),
            Err(_) => {
                let words = bytes.as_chunks::<WORD>().0;
                let values: Vec<u64> = words.iter().map(|w| u64::from_le_bytes(*w)).collect();
                self.chunk(chunk, &values)
            }
        }
    }
}

fn combine_fingerprints(
    shape: Shape,
    rho: Fp2,
    chunk_values: impl Fn(u64, &mut Vec<u64>) -> io::Result<()> + Sync,
) -> io::Result<Fp2> {
    let prints = Fingerprinter::new(shape, rho);
    (0..shape.chunks())
        .into_par_iter()
        .map(|chunk| {
            SCRATCH.with_borrow_mut(|(buf, _)| {
                chunk_values(chunk, buf)?;
                Ok(prints.chunk(chunk, buf))
            })
        })
        .try_reduce(|| Fp2::ZERO, |a, b| Ok(a + b))
}

pub fn fingerprint_file(path: &Path, shape: Shape, rho: Fp2) -> io::Result<Fp2> {
    let file = File::open(path)?;
    combine_fingerprints(shape, rho, |chunk, buf| {
        read_chunk(&file, shape, chunk, buf)
    })
}

/// The fingerprint of the mask stream for `seed`, with no file behind it.
pub fn fingerprint_mask(seed: &[u8; 32], shape: Shape, rho: Fp2) -> io::Result<Fp2> {
    combine_fingerprints(shape, rho, |chunk, buf| {
        buf.resize(shape.chunk_bytes(chunk) / WORD, 0);
        fill_mask(seed, chunk, buf);
        Ok(())
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

    /// The mask as it was first defined: one word at a time.
    fn mask_word_by_word(seed: &[u8; 32], chunk: u64, values: &mut [u64]) {
        use rand_core::{RngCore, SeedableRng};
        let mut rng = rand_chacha::ChaCha12Rng::from_seed(*seed);
        rng.set_stream(chunk);
        for v in values {
            let mask = loop {
                let word = rng.next_u64();
                if word < P {
                    break Fp::new(word);
                }
            };
            *v = (Fp::new(*v) + mask).value();
        }
    }

    #[test]
    fn masks_are_the_chacha12_stream_taken_a_word_at_a_time() {
        let seed = [0x5a; 32];
        for (chunk, len) in [
            (0u64, 0usize),
            (1, 1),
            (2, 511),
            (3, 512),
            (4, 513),
            (5, 5000),
        ] {
            let start: Vec<u64> = (0..len as u64).map(|i| P - 1 - i * 3).collect();
            let (mut fast, mut slow) = (start.clone(), start);
            apply_mask(&seed, chunk, &mut fast);
            mask_word_by_word(&seed, chunk, &mut slow);
            assert_eq!(fast, slow, "chunk {chunk}");

            let mut stream = vec![9u64; len];
            fill_mask(&seed, chunk, &mut stream);
            let mut zeros = vec![0u64; len];
            mask_word_by_word(&seed, chunk, &mut zeros);
            assert_eq!(stream, zeros);
        }
    }

    /// Hands out fixed words, so words outside the field can be put anywhere.
    struct Words(std::vec::IntoIter<u64>);

    impl chacha20::rand_core::TryRng for Words {
        type Error = std::convert::Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            unreachable!("masks are drawn as whole words")
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            Ok(self.0.next().unwrap())
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
            for word in dst.as_chunks_mut::<8>().0 {
                *word = self.0.next().unwrap().to_le_bytes();
            }
            Ok(())
        }
    }

    #[test]
    fn words_outside_the_field_are_left_out_wherever_they_fall() {
        let high = [P, P + 1, u64::MAX];
        for misses in [
            vec![0],
            vec![7],
            vec![0, 1, 2],
            vec![3, 7, 8, 9],
            vec![7, 8],
        ] {
            let words: Vec<u64> = (0..20u64)
                .map(|i| match misses.iter().position(|m| *m == i) {
                    Some(n) => high[n % high.len()],
                    None => i * 1000 + 1,
                })
                .collect();
            let expected: Vec<u64> = words.iter().copied().filter(|w| *w < P).take(8).collect();
            let mut out = [0u64; 8];
            let mut rng = Words(words.into_iter());
            fill_uniform(&mut rng, &mut out);
            assert_eq!(out.as_slice(), expected, "misses at {misses:?}");
        }
    }

    #[test]
    fn block_fingerprints_match_horners_rule() {
        let rho = challenge(&Digest::of(b"blocks"));
        let horner = |values: &[u64]| {
            values
                .iter()
                .rev()
                .fold(Fp2::ZERO, |acc, v| (acc * rho).add_base(Fp::new(*v)))
        };
        for len in [0, 1, 3, 4, 5, BLOCK - 1, BLOCK, BLOCK + 1, 3 * BLOCK + 77] {
            let values: Vec<u64> = (0..len as u64)
                .map(|i| u64::MAX - i * 0x1234_5678_9abc)
                .collect();
            assert_eq!(fingerprint_chunk(&values, rho), horner(&values), "{len}");
        }
    }

    #[test]
    fn chunks_fingerprinted_one_by_one_add_up_to_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v");
        let shape = Shape {
            coords: 1000,
            chunk_coords: 300,
        };
        let values: Vec<u64> = (0..1000u64).map(|i| P - 1 - i * i).collect();
        write(&path, shape, &values);
        let rho = challenge(&Digest::of(b"stream"));
        let prints = Fingerprinter::new(shape, rho);
        let file = File::open(&path).unwrap();
        let mut total = Fp2::ZERO;
        for chunk in (0..shape.chunks()).rev() {
            let bytes = read_chunk_bytes(&file, shape, chunk).unwrap();
            assert_eq!(bytes.len(), shape.chunk_bytes(chunk));
            // One byte in, the bytes are no longer aligned as coordinates.
            let mut shifted = vec![0u8];
            shifted.extend_from_slice(&bytes);
            assert_eq!(
                prints.chunk_bytes(chunk, &shifted[1..]),
                prints.chunk_bytes(chunk, &bytes)
            );
            total = total + prints.chunk_bytes(chunk, &bytes);
        }
        assert_eq!(total, fingerprint_file(&path, shape, rho).unwrap());
        assert_eq!(total, fingerprint_chunk(&values, rho));
    }

    #[test]
    fn a_chunk_hashes_the_same_on_and_off_the_thread_pool() {
        let bytes: Vec<u8> = (0..PARALLEL_HASH_BYTES * 3 + 100)
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        let leaves = bytes.chunks(merkle::LEAF_BYTES).map(merkle::leaf).collect();
        let expected = Digest(merkle::root(leaves));
        assert_eq!(digest_chunk_bytes(&bytes), expected);
        let pooled = rayon::join(|| digest_chunk_bytes(&bytes), || ()).0;
        assert_eq!(pooled, expected);
    }

    #[test]
    fn a_file_longer_than_one_read_has_the_digest_of_its_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        let bytes: Vec<u8> = (0..FILE_BLOCK * 2 + 17).map(|i| (i % 253) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(Digest::file(&path).unwrap(), Digest::of(&bytes));
        std::fs::write(&path, b"").unwrap();
        assert_eq!(Digest::file(&path).unwrap(), Digest::of(b""));
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
    fn the_root_is_the_rfc_6962_tree_over_one_kib_leaves() {
        let dir = tempfile::tempdir().unwrap();
        let values: Vec<u64> = (0..1500u64).map(|i| i * 7 + 1).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&values);
        let whole = merkle::root(bytes.chunks(merkle::LEAF_BYTES).map(merkle::leaf).collect());
        for chunk_coords in [128, 256, 512, 2048] {
            let shape = Shape {
                coords: 1500,
                chunk_coords,
            };
            let manifest = write(&dir.path().join("v"), shape, &values);
            assert_eq!(manifest.root.0, whole, "chunks of {chunk_coords}");
        }
    }

    #[test]
    fn every_leaf_can_be_proven_against_the_root_and_only_in_its_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v");
        for (coords, chunk_coords) in [
            (1, 128),
            (128, 128),
            (129, 128),
            (1500, 256),
            (5000, 1024),
            (4096, 512),
        ] {
            let shape = Shape {
                coords,
                chunk_coords,
            };
            assert!(shape.aligned());
            let values: Vec<u64> = (0..coords).map(|i| i * 31 + 5).collect();
            let manifest = write(&path, shape, &values);
            let whole: Vec<merkle::Hash> = bytemuck::cast_slice::<u64, u8>(&values)
                .chunks(merkle::LEAF_BYTES)
                .map(merkle::leaf)
                .collect();
            assert_eq!(shape.leaves() as usize, whole.len());
            for leaf in 0..shape.leaves() {
                let proof = prove_leaf(&path, shape, &manifest, leaf).unwrap();
                assert!(
                    proof.fits(&manifest.root, shape),
                    "{coords}/{chunk_coords} leaf {leaf}"
                );
                assert_eq!(proof.path, merkle::path(whole.clone(), leaf as usize));
                let (start, end) = leaf_range(shape, leaf);
                assert_eq!(proof.values, values[start as usize..end as usize]);

                let mut moved = proof.clone();
                moved.index = (leaf + 1) % shape.leaves();
                assert!(shape.leaves() == 1 || !moved.fits(&manifest.root, shape));
                let mut edited = proof.clone();
                edited.values[0] ^= 1;
                assert!(!edited.fits(&manifest.root, shape));
                let mut short = proof.clone();
                short.values.pop();
                assert!(!short.fits(&manifest.root, shape));
            }
            assert!(prove_leaf(&path, shape, &manifest, shape.leaves()).is_err());
        }
    }

    #[test]
    fn leaves_are_only_proven_for_aligned_shapes() {
        for (chunk_coords, aligned) in [
            (128, true),
            (256, true),
            (1 << 19, true),
            (64, false),
            (384, false),
            (130, false),
        ] {
            let shape = Shape {
                coords: 10_000,
                chunk_coords,
            };
            assert_eq!(shape.aligned(), aligned, "{chunk_coords}");
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v");
        let shape = Shape {
            coords: 500,
            chunk_coords: 100,
        };
        let manifest = write(&path, shape, &vec![1; 500]);
        assert!(prove_leaf(&path, shape, &manifest, 0).is_err());
        assert!(first_wrong_leaf(shape, &[&path], &[], &path).is_err());
    }

    #[test]
    fn a_wrong_sum_is_found_from_the_parts_that_are_held() {
        let dir = tempfile::tempdir().unwrap();
        let shape = Shape {
            coords: 300,
            chunk_coords: 128,
        };
        let at = |name: &str| dir.path().join(name);
        let (a, m, sum) = (at("a"), at("m"), at("sum"));
        write(&a, shape, &(0..300).map(|i| i * 7).collect::<Vec<_>>());
        write(&m, shape, &(0..300).map(|i| i + 9).collect::<Vec<_>>());
        combine(&sum, shape, &[&a], &[&m]).unwrap();
        let file = OpenOptions::new().write(true).open(&sum).unwrap();
        file.write_all_at(&5u64.to_le_bytes(), 8 * 130).unwrap();

        let check = |chunks: &[u64]| first_wrong_leaf_in(shape, &[&a], &[&m], &sum, chunks);
        assert_eq!(check(&[0, 2]).unwrap(), None);
        assert_eq!(check(&[2, 1]).unwrap(), Some(1));
        for (leaf, wrong) in [(0, false), (1, true), (2, false)] {
            assert_eq!(
                wrong_at_leaf(shape, &[&a], &[&m], &sum, leaf).unwrap(),
                wrong
            );
        }
        assert!(wrong_at_leaf(shape, &[&a], &[&m], &sum, 3).is_err());
    }

    #[test]
    fn a_wrong_sum_is_traced_to_its_first_leaf() {
        let dir = tempfile::tempdir().unwrap();
        let shape = Shape {
            coords: 3000,
            chunk_coords: 512,
        };
        let at = |name: &str| dir.path().join(name);
        let (a, b, m, sum) = (at("a"), at("b"), at("m"), at("sum"));
        write(&a, shape, &(0..3000).map(|i| P - 1 - i).collect::<Vec<_>>());
        write(&b, shape, &(0..3000).map(|i| i * 7).collect::<Vec<_>>());
        write(&m, shape, &(0..3000).map(|i| i + 9).collect::<Vec<_>>());
        combine(&sum, shape, &[&a, &b], &[&m]).unwrap();
        assert_eq!(
            first_wrong_leaf(shape, &[&a, &b], &[&m], &sum).unwrap(),
            None
        );

        for (coordinate, leaf) in [(0u64, 0u64), (127, 0), (128, 1), (1700, 13), (2999, 23)] {
            let wrong = at("wrong");
            std::fs::copy(&sum, &wrong).unwrap();
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&wrong)
                .unwrap();
            let chunk = coordinate / 512;
            let mut buf = Vec::new();
            read_chunk(&file, shape, chunk, &mut buf).unwrap();
            let slot = (coordinate % 512) as usize;
            buf[slot] = (Fp::new(buf[slot]) + Fp::ONE).value();
            write_chunk_bytes(&file, shape, chunk, chunk_bytes(&buf)).unwrap();
            assert_eq!(
                first_wrong_leaf(shape, &[&a, &b], &[&m], &wrong).unwrap(),
                Some(leaf),
                "coordinate {coordinate}"
            );
        }
        // Leaving a vector out is caught at the first leaf it would have changed.
        assert_eq!(
            first_wrong_leaf(shape, &[&a], &[&m], &sum).unwrap(),
            Some(0)
        );
    }

    #[test]
    fn the_chunks_a_file_lacks_are_the_ones_that_differ_from_the_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v");
        let shape = Shape {
            coords: 1000,
            chunk_coords: 64,
        };
        let manifest = write(&path, shape, &(0..1000).collect::<Vec<_>>());
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert!(missing_chunks(&file, shape, &manifest).unwrap().is_empty());
        for chunk in [3, 15] {
            let zeros = vec![0u8; shape.chunk_bytes(chunk)];
            write_chunk_bytes(&file, shape, chunk, &zeros).unwrap();
        }
        assert_eq!(
            missing_chunks(&file, shape, &manifest).unwrap(),
            BTreeSet::from([3, 15])
        );
        file.set_len(0).unwrap();
        assert!(missing_chunks(&file, shape, &manifest).is_err());
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
