//! Tensor layouts, fixed-point encoding and the model file.
//!
//! A model file is a plain safetensors file with tensors in layout order, so
//! the same model always has the same bytes and the same digest.

use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::field::Fp;
use crate::keys::random_bytes;
use crate::privacy::{Privacy, Sampler};
use crate::vector::{self, Manifest, Shape};

/// Random coordinates appended to each update. They hide its fingerprint.
pub const PAD: u64 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dtype {
    F32,
    F64,
}

impl Dtype {
    pub const fn width(self) -> u64 {
        match self {
            Dtype::F32 => 4,
            Dtype::F64 => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
}

impl TensorSpec {
    pub fn coords(&self) -> u64 {
        self.shape.iter().product()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    pub tensors: Vec<TensorSpec>,
}

/// A contiguous piece of one tensor.
struct Run {
    tensor: usize,
    tensor_offset: u64,
    len: u64,
}

impl Layout {
    pub fn coords(&self) -> u64 {
        self.tensors.iter().map(TensorSpec::coords).sum()
    }

    pub fn shape(&self, chunk_coords: u64) -> Shape {
        Shape {
            coords: self.coords() + PAD,
            chunk_coords,
        }
    }

    fn runs(&self, start: u64, end: u64) -> Vec<Run> {
        let mut runs = Vec::new();
        let mut at = 0;
        for (tensor, spec) in self.tensors.iter().enumerate() {
            let next = at + spec.coords();
            let (lo, hi) = (start.max(at), end.min(next));
            if lo < hi {
                runs.push(Run {
                    tensor,
                    tensor_offset: lo - at,
                    len: hi - lo,
                });
            }
            at = next;
        }
        runs
    }

    fn header(&self) -> Vec<u8> {
        let mut map = serde_json::Map::new();
        let mut at = 0;
        for spec in &self.tensors {
            let next = at + spec.coords() * spec.dtype.width();
            map.insert(
                spec.name.clone(),
                serde_json::json!({"dtype": spec.dtype, "shape": spec.shape, "data_offsets": [at, next]}),
            );
            at = next;
        }
        let mut header = serde_json::to_vec(&map).unwrap_or_default();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.append(&mut header);
        out
    }
}

/// Fixed-point parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Encoding {
    pub frac_bits: u32,
    pub clip: f64,
}

impl Encoding {
    pub fn scale(&self) -> f64 {
        (1u64 << self.frac_bits) as f64
    }

    /// Bits needed for the largest weighted sum over `total_weight`.
    pub fn sum_bits(&self, total_weight: u64) -> u32 {
        let clip = (self.clip.max(1.0) * self.scale()).log2().ceil() as u32;
        clip + (u64::BITS - total_weight.leading_zeros())
    }

    #[inline]
    fn encode(&self, delta: f64, weight: u64, clipped: &mut u64) -> Fp {
        let value = if delta.is_nan() {
            *clipped += 1;
            0.0
        } else if delta.abs() > self.clip {
            *clipped += 1;
            delta.signum() * self.clip
        } else {
            delta
        };
        Fp::from_i64((value * self.scale()).round() as i64 * weight as i64)
    }

    #[inline]
    pub fn decode(&self, v: Fp, total_weight: u64) -> f64 {
        v.centered() as f64 / (self.scale() * total_weight as f64)
    }

    /// The largest magnitude an honest coordinate can have at `weight`.
    pub fn bound(&self, weight: u64) -> u64 {
        (self.clip * self.scale()).round() as u64 * weight
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("model file does not match the layout: {0}")]
    Mismatch(String),
}

/// A read-only model file, read a piece at a time.
pub struct ModelFile {
    file: File,
    data_start: u64,
    layout: Layout,
}

impl ModelFile {
    pub fn open(path: &Path, layout: &Layout) -> Result<ModelFile, ModelError> {
        let mut file = File::open(path)?;
        let expected = layout.header();
        let mut header = vec![0u8; expected.len()];
        file.read_exact(&mut header)?;
        if header != expected {
            return Err(ModelError::Mismatch("header differs".into()));
        }
        let bytes: u64 = layout
            .tensors
            .iter()
            .map(|t| t.coords() * t.dtype.width())
            .sum();
        if file.metadata()?.len() != expected.len() as u64 + bytes {
            return Err(ModelError::Mismatch("wrong length".into()));
        }
        Ok(ModelFile {
            file,
            data_start: expected.len() as u64,
            layout: layout.clone(),
        })
    }

    fn tensor_start(&self, tensor: usize) -> u64 {
        self.data_start
            + self.layout.tensors[..tensor]
                .iter()
                .map(|t| t.coords() * t.dtype.width())
                .sum::<u64>()
    }

    fn read_run(&self, run: &Run, out: &mut [f64]) -> io::Result<()> {
        let spec = &self.layout.tensors[run.tensor];
        let at = self.tensor_start(run.tensor) + run.tensor_offset * spec.dtype.width();
        let mut bytes = vec![0u8; run.len as usize * spec.dtype.width() as usize];
        self.file.read_exact_at(&mut bytes, at)?;
        match spec.dtype {
            Dtype::F32 => {
                for (o, b) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *o = f32::from_le_bytes(*b) as f64;
                }
            }
            Dtype::F64 => {
                for (o, b) in out.iter_mut().zip(bytes.as_chunks::<8>().0) {
                    *o = f64::from_le_bytes(*b);
                }
            }
        }
        Ok(())
    }
}

/// Borrowed tensor data from whoever trained.
#[derive(Clone, Copy)]
pub enum TensorView<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl TensorView<'_> {
    fn len(&self) -> usize {
        match self {
            TensorView::F32(v) => v.len(),
            TensorView::F64(v) => v.len(),
        }
    }

    #[inline]
    fn get(&self, i: usize) -> f64 {
        match self {
            TensorView::F32(v) => v[i] as f64,
            TensorView::F64(v) => v[i],
        }
    }
}

pub struct EncodeRequest<'a> {
    pub layout: &'a Layout,
    pub encoding: Encoding,
    pub chunk_coords: u64,
    pub base: &'a Path,
    pub local: &'a [TensorView<'a>],
    pub weight: u64,
    pub pad: [Fp; PAD as usize],
    pub seeds: &'a [[u8; 32]],
    pub out: &'a Path,
}

pub struct Encoded {
    pub manifest: Manifest,
    pub clipped: u64,
}

fn check_views(layout: &Layout, local: &[TensorView<'_>]) -> Result<(), ModelError> {
    if local.len() != layout.tensors.len() {
        return Err(ModelError::Mismatch("wrong tensor count".into()));
    }
    for (view, spec) in local.iter().zip(&layout.tensors) {
        if view.len() as u64 != spec.coords() {
            return Err(ModelError::Mismatch(format!(
                "wrong size for {}",
                spec.name
            )));
        }
    }
    Ok(())
}

/// Calls `each` with local - base for the model coordinates in `start..end`.
fn for_deltas(
    base: &ModelFile,
    local: &[TensorView<'_>],
    (start, end): (u64, u64),
    mut each: impl FnMut(f64),
) -> io::Result<()> {
    let mut base_values = Vec::new();
    for run in base.layout.runs(start, end) {
        base_values.resize(run.len as usize, 0.0);
        base.read_run(&run, &mut base_values)?;
        let view = local[run.tensor];
        for (k, b) in base_values.iter().enumerate() {
            each(view.get(run.tensor_offset as usize + k) - b);
        }
    }
    Ok(())
}

/// The pad coordinates that fall in a chunk, then the masks over all of it.
fn finish_chunk(
    shape: Shape,
    plain: u64,
    chunk: u64,
    filled: usize,
    buf: &mut [u64],
    pad: &[Fp; PAD as usize],
    seeds: &[[u8; 32]],
) {
    let (start, end) = shape.range(chunk);
    for (n, coord) in (start.max(plain)..end).enumerate() {
        buf[filled + n] = pad[(coord - plain) as usize].value();
    }
    for seed in seeds {
        vector::apply_mask(seed, chunk, buf);
    }
}

/// Writes weight * (local - base) plus masks, one chunk at a time.
pub fn encode_update(req: &EncodeRequest<'_>) -> Result<Encoded, ModelError> {
    check_views(req.layout, req.local)?;
    let base = ModelFile::open(req.base, req.layout)?;
    let shape = req.layout.shape(req.chunk_coords);
    let plain = req.layout.coords();
    let clipped = std::sync::atomic::AtomicU64::new(0);
    let manifest = vector::build(req.out, shape, |chunk, buf, _| {
        let (mut n, mut count) = (0usize, 0);
        for_deltas(&base, req.local, shape.range(chunk), |delta| {
            buf[n] = req.encoding.encode(delta, req.weight, &mut count).value();
            n += 1;
        })?;
        finish_chunk(shape, plain, chunk, n, buf, &req.pad, req.seeds);
        clipped.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    })?;
    Ok(Encoded {
        manifest,
        clipped: clipped.into_inner(),
    })
}

pub struct PrivateRequest<'a> {
    pub layout: &'a Layout,
    pub frac_bits: u32,
    pub chunk_coords: u64,
    pub base: &'a Path,
    pub local: &'a [TensorView<'a>],
    pub privacy: &'a Privacy,
    pub pad: [Fp; PAD as usize],
    pub seeds: &'a [[u8; 32]],
    pub out: &'a Path,
}

const TWO_53: f64 = (1u64 << 53) as f64;

/// Writes the update a private run takes from a site: local - base, clipped to
/// the plan's L2 bound, rounded onto the grid at random so the rounding has no
/// bias, plus discrete Gaussian noise, plus masks.
///
/// The noise and the rounding use fresh randomness that is never stored. Before
/// returning, the norm of the rounded vector is checked in integers, so the
/// bound the privacy accounting relies on holds whatever floating point did.
pub fn encode_private(req: &PrivateRequest<'_>) -> Result<Encoded, ModelError> {
    check_views(req.layout, req.local)?;
    let base = ModelFile::open(req.base, req.layout)?;
    let shape = req.layout.shape(req.chunk_coords);
    let plain = req.layout.coords();
    let scale = (1u64 << req.frac_bits) as f64;

    let square_sum: io::Result<f64> = (0..shape.chunks())
        .into_par_iter()
        .map(|chunk| {
            let mut sum = 0.0;
            for_deltas(&base, req.local, shape.range(chunk), |delta| {
                if delta.is_finite() {
                    sum += delta * delta;
                }
            })?;
            Ok(sum)
        })
        .try_reduce(|| 0.0, |a, b| Ok(a + b));
    let norm = square_sum?.sqrt() * scale;
    let clip = req.privacy.clip as f64;
    let clipped = norm > clip;
    let mut factor = if clipped { clip / norm } else { 1.0 } * scale;

    let limit = u128::from(req.privacy.sensitivity(plain)).pow(2);
    let sampler = Sampler::new(req.privacy.noise);
    loop {
        let (rounding_seed, noise_seed) = (random_bytes::<32>(), random_bytes::<32>());
        let squares = Mutex::new(0u128);
        let manifest = vector::build(req.out, shape, |chunk, buf, _| {
            let mut rounding = ChaCha20Rng::from_seed(rounding_seed);
            rounding.set_stream(chunk);
            let mut noise = ChaCha20Rng::from_seed(noise_seed);
            noise.set_stream(chunk);
            let (mut n, mut sum) = (0usize, 0u128);
            for_deltas(&base, req.local, shape.range(chunk), |delta| {
                let exact = if delta.is_finite() {
                    delta * factor
                } else {
                    0.0
                };
                let floor = exact.floor();
                let up = ((rounding.next_u64() >> 11) as f64) < (exact - floor) * TWO_53;
                let rounded = floor as i64 + i64::from(up);
                sum += u128::from(rounded.unsigned_abs()).pow(2);
                buf[n] = Fp::from_i64(rounded + sampler.sample(&mut noise)).value();
                n += 1;
            })?;
            finish_chunk(shape, plain, chunk, n, buf, &req.pad, req.seeds);
            *squares.lock().unwrap_or_else(PoisonError::into_inner) += sum;
            Ok(())
        })?;
        if squares.into_inner().unwrap_or_else(PoisonError::into_inner) <= limit {
            return Ok(Encoded {
                manifest,
                clipped: u64::from(clipped),
            });
        }
        factor *= 1.0 - 1.0 / 1024.0;
    }
}

/// Writes the next model: base + aggregate / (scale * total_weight).
pub fn apply_aggregate(
    layout: &Layout,
    encoding: Encoding,
    chunk_coords: u64,
    base: &Path,
    aggregate: &Path,
    total_weight: u64,
    out: &Path,
) -> Result<(), ModelError> {
    let base = ModelFile::open(base, layout)?;
    let aggregate = File::open(aggregate)?;
    let shape = layout.shape(chunk_coords);
    let target = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(out)?;
    let header = layout.header();
    target.write_all_at(&header, 0)?;
    let result: io::Result<()> = (0..shape.chunks()).into_par_iter().try_for_each(|chunk| {
        let (start, end) = shape.range(chunk);
        let mut values = Vec::new();
        vector::read_chunk(&aggregate, shape, chunk, &mut values)?;
        let mut consumed = 0usize;
        let mut base_values = Vec::new();
        for run in layout.runs(start, end) {
            base_values.resize(run.len as usize, 0.0);
            base.read_run(&run, &mut base_values)?;
            let spec = &layout.tensors[run.tensor];
            let mut bytes = Vec::with_capacity(run.len as usize * spec.dtype.width() as usize);
            for (k, b) in base_values.iter().enumerate() {
                let next = b + encoding.decode(Fp::new(values[consumed + k]), total_weight);
                match spec.dtype {
                    Dtype::F32 => bytes.extend_from_slice(&(next as f32).to_le_bytes()),
                    Dtype::F64 => bytes.extend_from_slice(&next.to_le_bytes()),
                }
            }
            consumed += run.len as usize;
            let at = base.tensor_start(run.tensor) + run.tensor_offset * spec.dtype.width();
            target.write_all_at(&bytes, at)?;
        }
        Ok(())
    });
    result?;
    target.sync_data()?;
    Ok(())
}

/// Checks every plain coordinate is in range. Only meaningful without masks.
pub fn in_range(path: &Path, shape: Shape, plain: u64, bound: u64) -> io::Result<bool> {
    let file = File::open(path)?;
    let chunk_ok = |chunk: u64| -> io::Result<bool> {
        let mut buf = Vec::new();
        vector::read_chunk(&file, shape, chunk, &mut buf)?;
        let start = shape.range(chunk).0;
        Ok(buf.iter().enumerate().all(|(i, v)| {
            start + i as u64 >= plain || Fp::new(*v).centered().unsigned_abs() <= bound
        }))
    };
    (0..shape.chunks())
        .into_par_iter()
        .map(chunk_ok)
        .try_reduce(|| true, |a, b| Ok(a && b))
}

/// Writes a model from tensors already in memory. Used to create the first one.
pub fn write_model(
    path: &Path,
    layout: &Layout,
    tensors: &[TensorView<'_>],
) -> Result<(), ModelError> {
    if tensors.len() != layout.tensors.len() {
        return Err(ModelError::Mismatch("wrong tensor count".into()));
    }
    let mut bytes = layout.header();
    for (view, spec) in tensors.iter().zip(&layout.tensors) {
        if view.len() as u64 != spec.coords() {
            return Err(ModelError::Mismatch(format!(
                "wrong size for {}",
                spec.name
            )));
        }
        for i in 0..view.len() {
            match spec.dtype {
                Dtype::F32 => bytes.extend_from_slice(&(view.get(i) as f32).to_le_bytes()),
                Dtype::F64 => bytes.extend_from_slice(&view.get(i).to_le_bytes()),
            }
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::{Digest, challenge, combine, fingerprint_file};

    fn layout() -> Layout {
        Layout {
            tensors: vec![
                TensorSpec {
                    name: "w".into(),
                    dtype: Dtype::F32,
                    shape: vec![3, 4],
                },
                TensorSpec {
                    name: "b".into(),
                    dtype: Dtype::F64,
                    shape: vec![5],
                },
            ],
        }
    }

    const ENCODING: Encoding = Encoding {
        frac_bits: 24,
        clip: 128.0,
    };

    fn encode_site(
        dir: &Path,
        name: &str,
        base: &Path,
        local: (&[f32], &[f64]),
        weight: u64,
        seeds: &[[u8; 32]],
    ) -> std::path::PathBuf {
        let out = dir.join(name);
        encode_update(&EncodeRequest {
            layout: &layout(),
            encoding: ENCODING,
            chunk_coords: 4,
            base,
            local: &[TensorView::F32(local.0), TensorView::F64(local.1)],
            weight,
            pad: [Fp::new(11), Fp::new(13)],
            seeds,
            out: &out,
        })
        .unwrap();
        out
    }

    #[test]
    fn weighted_mean_survives_masking_to_quantisation() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let base = d.join("base");
        let w0 = [0.5f32; 12];
        let b0 = [-1.0f64; 5];
        write_model(
            &base,
            &layout(),
            &[TensorView::F32(&w0), TensorView::F64(&b0)],
        )
        .unwrap();

        let seed_a = [1u8; 32];
        let seed_b = [2u8; 32];
        let a = encode_site(d, "a", &base, (&[1.5; 12], &[1.0; 5]), 1, &[seed_a]);
        let b = encode_site(d, "b", &base, (&[0.5; 12], &[-1.0; 5]), 3, &[seed_b]);

        let shape = layout().shape(4);
        let (sum, mask, agg) = (d.join("sum"), d.join("mask"), d.join("agg"));
        combine(&sum, shape, &[&a, &b], &[]).unwrap();
        vector::build(&mask, shape, |chunk, buf, _| {
            vector::apply_mask(&seed_a, chunk, buf);
            vector::apply_mask(&seed_b, chunk, buf);
            Ok(())
        })
        .unwrap();
        combine(&agg, shape, &[&sum], &[&mask]).unwrap();

        let next = d.join("next");
        apply_aggregate(&layout(), ENCODING, 4, &base, &agg, 4, &next).unwrap();
        let tensors = read_all(&next);
        for v in &tensors[..12] {
            assert!((v - 0.75).abs() < 1e-6, "{v}");
        }
        for v in &tensors[12..] {
            assert!((v - (-0.5)).abs() < 1e-6, "{v}");
        }
    }

    fn read_all(path: &Path) -> Vec<f64> {
        let model = ModelFile::open(path, &layout()).unwrap();
        let mut out = vec![0.0; 17];
        for run in layout().runs(0, 17) {
            let start = if run.tensor == 0 { 0 } else { 12 };
            let lo = start + run.tensor_offset as usize;
            model
                .read_run(&run, &mut out[lo..lo + run.len as usize])
                .unwrap();
        }
        out
    }

    #[test]
    fn every_coordinate_gets_its_own_weighted_mean_wherever_chunks_fall() {
        let base_w: Vec<f32> = (0..12).map(|i| i as f32 * 0.25).collect();
        let base_b: Vec<f64> = (0..5).map(|i| -(i as f64)).collect();
        let shift = |by: f64| -> (Vec<f32>, Vec<f64>) {
            (
                base_w
                    .iter()
                    .enumerate()
                    .map(|(i, v)| *v + (by * i as f64) as f32)
                    .collect(),
                base_b
                    .iter()
                    .enumerate()
                    .map(|(i, v)| *v + by * (i as f64 + 1.0))
                    .collect(),
            )
        };
        let (a_w, a_b) = shift(0.5);
        let (c_w, c_b) = shift(-1.5);
        let (weight_a, weight_c) = (2u64, 5u64);
        let mean = |by_a: f64, by_c: f64| (2.0 * by_a + 5.0 * by_c) / 7.0;

        for chunk_coords in [1, 3, 5, 64] {
            let dir = tempfile::tempdir().unwrap();
            let d = dir.path();
            let base = d.join("base");
            write_model(
                &base,
                &layout(),
                &[TensorView::F32(&base_w), TensorView::F64(&base_b)],
            )
            .unwrap();
            let encode = |name: &str, w: &[f32], b: &[f64], weight: u64| {
                let out = d.join(name);
                encode_update(&EncodeRequest {
                    layout: &layout(),
                    encoding: ENCODING,
                    chunk_coords,
                    base: &base,
                    local: &[TensorView::F32(w), TensorView::F64(b)],
                    weight,
                    pad: [Fp::new(11), Fp::new(13)],
                    seeds: &[],
                    out: &out,
                })
                .unwrap();
                out
            };
            let a = encode("a", &a_w, &a_b, weight_a);
            let c = encode("c", &c_w, &c_b, weight_c);
            let shape = layout().shape(chunk_coords);
            let (sum, next) = (d.join("sum"), d.join("next"));
            combine(&sum, shape, &[&a, &c], &[]).unwrap();
            apply_aggregate(&layout(), ENCODING, chunk_coords, &base, &sum, 7, &next).unwrap();

            let values = read_all(&next);
            for i in 0..12 {
                let want = f64::from(base_w[i]) + mean(0.5, -1.5) * i as f64;
                assert!(
                    (values[i] - want).abs() < 1e-5,
                    "chunk {chunk_coords} w[{i}]: {}",
                    values[i]
                );
            }
            for i in 0..5 {
                let want = base_b[i] + mean(0.5, -1.5) * (i as f64 + 1.0);
                assert!(
                    (values[12 + i] - want).abs() < 1e-5,
                    "chunk {chunk_coords} b[{i}]: {}",
                    values[12 + i]
                );
            }
        }
    }

    #[test]
    fn clipping_is_exact_at_the_limit_and_keeps_the_sign() {
        let encoding = Encoding {
            frac_bits: 4,
            clip: 8.0,
        };
        assert_eq!(encoding.bound(3), 8 * 16 * 3);
        let mut clipped = 0;
        assert_eq!(encoding.encode(8.0, 3, &mut clipped).centered(), 384);
        assert_eq!(encoding.encode(-8.0, 3, &mut clipped).centered(), -384);
        assert_eq!(clipped, 0);
        assert_eq!(encoding.encode(9.0, 3, &mut clipped).centered(), 384);
        assert_eq!(encoding.encode(-9.0, 3, &mut clipped).centered(), -384);
        assert_eq!(clipped, 2);
        assert_eq!(encoding.encode(0.25, 2, &mut clipped).centered(), 8);
        assert!((encoding.decode(Fp::from_i64(-384), 3) - (-8.0)).abs() < 1e-12);
    }

    #[test]
    fn fingerprint_of_a_masked_update_is_hidden_by_the_pad() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let base = d.join("base");
        write_model(
            &base,
            &layout(),
            &[TensorView::F32(&[0.0; 12]), TensorView::F64(&[0.0; 5])],
        )
        .unwrap();
        let shape = layout().shape(4);
        let rho = challenge(&Digest::of(b"c"));

        let with_pad = |name: &str, pad: [Fp; 2]| {
            let out = d.join(name);
            encode_update(&EncodeRequest {
                layout: &layout(),
                encoding: ENCODING,
                chunk_coords: 4,
                base: &base,
                local: &[TensorView::F32(&[1.0; 12]), TensorView::F64(&[1.0; 5])],
                weight: 1,
                pad,
                seeds: &[],
                out: &out,
            })
            .unwrap();
            fingerprint_file(&out, shape, rho).unwrap()
        };
        let f1 = with_pad("p1", [Fp::new(1), Fp::new(2)]);
        let f2 = with_pad("p2", [Fp::new(3), Fp::new(4)]);
        assert_ne!(f1, f2);
    }

    #[test]
    fn clipping_counts_and_clamps() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let base = d.join("base");
        write_model(
            &base,
            &layout(),
            &[TensorView::F32(&[0.0; 12]), TensorView::F64(&[0.0; 5])],
        )
        .unwrap();
        let out = d.join("o");
        let mut w = [0.0f32; 12];
        w[0] = 1e9;
        w[1] = f32::NAN;
        let encoded = encode_update(&EncodeRequest {
            layout: &layout(),
            encoding: ENCODING,
            chunk_coords: 4,
            base: &base,
            local: &[TensorView::F32(&w), TensorView::F64(&[0.0; 5])],
            weight: 1,
            pad: [Fp::ZERO; 2],
            seeds: &[],
            out: &out,
        })
        .unwrap();
        assert_eq!(encoded.clipped, 2);
        let shape = layout().shape(4);
        assert!(in_range(&out, shape, 17, ENCODING.bound(1)).unwrap());
    }

    fn wide() -> Layout {
        Layout {
            tensors: vec![TensorSpec {
                name: "w".into(),
                dtype: Dtype::F64,
                shape: vec![20_000],
            }],
        }
    }

    const QUIET: Privacy = Privacy {
        clip: 1 << 18,
        noise: 8,
        colluding: 0,
        budget: u64::MAX,
        epsilon: 1.0,
        delta: 1e-5,
    };

    /// Encodes `local` against a zero model and returns the plain coordinates.
    fn private_values(local: &[f64], privacy: &Privacy, seeds: &[[u8; 32]]) -> Vec<i64> {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let (base, out) = (d.join("base"), d.join("out"));
        write_model(&base, &wide(), &[TensorView::F64(&vec![0.0; 20_000])]).unwrap();
        let shape = wide().shape(4096);
        encode_private(&PrivateRequest {
            layout: &wide(),
            frac_bits: 18,
            chunk_coords: 4096,
            base: &base,
            local: &[TensorView::F64(local)],
            privacy,
            pad: [Fp::new(11), Fp::new(13)],
            seeds,
            out: &out,
        })
        .unwrap();
        let plain = d.join("plain");
        if seeds.is_empty() {
            std::fs::copy(&out, &plain).unwrap();
        } else {
            let mask = d.join("mask");
            vector::build(&mask, shape, |chunk, buf, _| {
                for seed in seeds {
                    vector::apply_mask(seed, chunk, buf);
                }
                Ok(())
            })
            .unwrap();
            combine(&plain, shape, &[&out], &[&mask]).unwrap();
        }
        let file = File::open(&plain).unwrap();
        let mut all = Vec::new();
        for chunk in 0..shape.chunks() {
            let mut buf = Vec::new();
            vector::read_chunk(&file, shape, chunk, &mut buf).unwrap();
            all.extend(buf.iter().map(|v| Fp::new(*v).centered()));
        }
        assert_eq!(&all[20_000..], &[11, 13], "the pad follows the model");
        all.truncate(20_000);
        all
    }

    fn norm(values: &[i64]) -> f64 {
        values
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn a_private_update_is_clipped_to_the_norm_and_keeps_its_direction() {
        let local: Vec<f64> = (0..20_000)
            .map(|i| if i % 2 == 0 { 3.0 } else { -1.0 })
            .collect();
        let values = private_values(&local, &QUIET, &[]);
        // The clip is one model unit, 2^18 on the grid. Noise of variance 8
        // over 20,000 coordinates adds a few hundred to the norm at most.
        let clip = (1u64 << 18) as f64;
        assert!(
            (norm(&values) / clip - 1.0).abs() < 0.01,
            "{}",
            norm(&values)
        );
        let unit = clip / norm(&local.iter().map(|v| *v as i64).collect::<Vec<_>>());
        for (got, want) in values.iter().zip(&local).take(200) {
            assert!((*got as f64 - want * unit).abs() < 40.0, "{got} for {want}");
        }
    }

    #[test]
    fn a_small_update_is_not_scaled_and_rounding_has_no_bias() {
        // 0.3 of a grid step each: far inside the clip, and never a whole number.
        let step = 0.3 / (1u64 << 18) as f64;
        let values = private_values(&vec![step; 20_000], &QUIET, &[]);
        let mean = values.iter().sum::<i64>() as f64 / 20_000.0;
        assert!((mean - 0.3).abs() < 0.08, "{mean}");
        let spread = values
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / 20_000.0;
        // Noise of variance 8 plus the rounding coin's 0.3 * 0.7.
        assert!((spread - 8.21).abs() < 0.5, "{spread}");
    }

    #[test]
    fn the_noise_has_the_variance_the_plan_names() {
        let loud = Privacy {
            noise: 1 << 30,
            ..QUIET
        };
        let values = private_values(&vec![0.0; 20_000], &loud, &[]);
        let spread = values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 20_000.0;
        assert!(
            (spread / (1u64 << 30) as f64 - 1.0).abs() < 0.05,
            "{spread}"
        );
        let again = private_values(&vec![0.0; 20_000], &loud, &[]);
        assert_ne!(values, again, "noise must be fresh every time");
    }

    #[test]
    fn masks_come_off_a_private_update() {
        let values = private_values(&vec![0.0; 20_000], &QUIET, &[[7u8; 32], [9u8; 32]]);
        assert!(
            values.iter().all(|v| v.abs() < 60),
            "masks left something behind"
        );
        assert!(values.iter().any(|v| *v != 0));
    }

    #[test]
    fn values_that_are_not_numbers_count_as_zero() {
        let mut local = vec![0.0; 20_000];
        (local[0], local[1], local[2]) = (f64::NAN, f64::INFINITY, 0.5);
        let values = private_values(&local, &QUIET, &[]);
        assert!(values[0].abs() < 60 && values[1].abs() < 60);
        assert!((values[2] - (1 << 17)).abs() < 60, "{}", values[2]);
    }

    #[test]
    fn the_rounded_norm_never_exceeds_what_the_accounting_assumes() {
        // With no noise to speak of, the vector that comes out is the rounded one.
        let local: Vec<f64> = (0..20_000).map(|i| (i as f64).sin() * 1e6).collect();
        let values = private_values(&local, &QUIET, &[]);
        let limit = QUIET.sensitivity(20_000) as f64;
        assert!(
            norm(&values) < limit + 500.0,
            "{} over {limit}",
            norm(&values)
        );
    }

    #[test]
    fn in_range_catches_oversized_coordinates_but_ignores_the_pad() {
        let dir = tempfile::tempdir().unwrap();
        let shape = layout().shape(4);
        let path = dir.path().join("v");
        vector::build(&path, shape, |chunk, buf, _| {
            let (start, _) = shape.range(chunk);
            for (i, v) in buf.iter_mut().enumerate() {
                *v = if start + i as u64 >= 17 { 1 << 62 } else { 5 };
            }
            Ok(())
        })
        .unwrap();
        assert!(in_range(&path, shape, 17, 5).unwrap());
        assert!(!in_range(&path, shape, 17, 4).unwrap());
    }

    #[test]
    fn model_file_opens_as_valid_safetensors_framing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m");
        write_model(
            &path,
            &layout(),
            &[TensorView::F32(&[1.0; 12]), TensorView::F64(&[2.0; 5])],
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!((len + 8) % 8, 0);
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
        assert_eq!(header["w"]["data_offsets"], serde_json::json!([0, 48]));
        assert_eq!(header["b"]["data_offsets"], serde_json::json!([48, 88]));
        assert_eq!(bytes.len(), 8 + len + 88);
    }

    #[test]
    fn wrong_tensor_size_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m");
        let err = write_model(
            &path,
            &layout(),
            &[TensorView::F32(&[1.0; 11]), TensorView::F64(&[2.0; 5])],
        );
        assert!(matches!(err, Err(ModelError::Mismatch(_))));
    }
}
