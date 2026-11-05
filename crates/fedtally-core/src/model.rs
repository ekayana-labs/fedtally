//! Tensor layouts, fixed-point encoding and the model file.
//!
//! A model file is a plain safetensors file with tensors in layout order, so
//! the same model always has the same bytes and the same digest.

use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::field::Fp;
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

/// Writes weight * (local - base) plus masks, one chunk at a time.
pub fn encode_update(req: &EncodeRequest<'_>) -> Result<Encoded, ModelError> {
    let layout = req.layout;
    if req.local.len() != layout.tensors.len() {
        return Err(ModelError::Mismatch("wrong tensor count".into()));
    }
    for (view, spec) in req.local.iter().zip(&layout.tensors) {
        if view.len() as u64 != spec.coords() {
            return Err(ModelError::Mismatch(format!(
                "wrong size for {}",
                spec.name
            )));
        }
    }
    let base = ModelFile::open(req.base, layout)?;
    let shape = layout.shape(req.chunk_coords);
    let plain = layout.coords();
    let clipped = std::sync::atomic::AtomicU64::new(0);
    let manifest = vector::build(req.out, shape, |chunk, buf, _| {
        let (start, end) = shape.range(chunk);
        let mut n = 0usize;
        let mut count = 0;
        let mut base_values = Vec::new();
        for run in layout.runs(start, end) {
            base_values.resize(run.len as usize, 0.0);
            base.read_run(&run, &mut base_values)?;
            let view = req.local[run.tensor];
            for (k, b) in base_values.iter().enumerate() {
                let local = view.get(run.tensor_offset as usize + k);
                buf[n] = req
                    .encoding
                    .encode(local - b, req.weight, &mut count)
                    .value();
                n += 1;
            }
        }
        for coord in start.max(plain)..end {
            buf[n] = req.pad[(coord - plain) as usize].value();
            n += 1;
        }
        for seed in req.seeds {
            vector::apply_mask(seed, chunk, buf);
        }
        clipped.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    })?;
    Ok(Encoded {
        manifest,
        clipped: clipped.into_inner(),
    })
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
