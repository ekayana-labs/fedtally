//! Times the per-round data path: `cargo run --release --example hotpath -p fedtally-core`.
//!
//! The first block is one chunk held in memory. The second is whole vectors on
//! disk with every core, as a node runs them. Pass a size in MiB of float32
//! model to change the second block; the default is 64.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use fedtally_core::field::{Fp, P};
use fedtally_core::model::{self, Dtype, EncodeRequest, Encoding, Layout, TensorSpec, TensorView};
use fedtally_core::vector::{self, Digest};

const CHUNK_COORDS: usize = 1 << 17;

/// Best of `repeats`, in MiB of vector per second.
fn rate(bytes: usize, repeats: usize, mut run: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..repeats {
        let started = Instant::now();
        run();
        best = best.min(started.elapsed().as_secs_f64());
    }
    bytes as f64 / (1 << 20) as f64 / best
}

fn line(name: &str, rate: f64) {
    println!("{name:<34} {rate:>9.0} MiB/s");
}

fn values(seed: u8) -> Vec<u64> {
    let mut out = vec![0u64; CHUNK_COORDS];
    vector::apply_mask(&[seed; 32], 0, &mut out);
    out
}

fn kernels() {
    let bytes = CHUNK_COORDS * 8;
    let (a, b) = (values(1), values(2));
    let rho = vector::challenge(&Digest::of(b"hotpath"));

    let mut acc = a.clone();
    line(
        "field add",
        rate(bytes, 200, || {
            for (x, y) in acc.iter_mut().zip(&b) {
                *x = (Fp::new(*x) + Fp::new(*y)).value();
            }
            black_box(&mut acc);
        }),
    );
    let mut acc = a.clone();
    line(
        "mask expand and add",
        rate(bytes, 200, || {
            vector::apply_mask(&[7; 32], 3, &mut acc);
            black_box(&mut acc);
        }),
    );
    line(
        "fingerprint",
        rate(bytes, 200, || {
            black_box(vector::fingerprint_chunk(black_box(&a), rho));
        }),
    );
    line(
        "merkle chunk root",
        rate(bytes, 200, || {
            black_box(vector::digest_chunk(black_box(&a)));
        }),
    );
    line(
        "blake3 digest",
        rate(bytes, 200, || {
            black_box(Digest::of(vector::chunk_bytes(black_box(&a))));
        }),
    );
    assert!(acc.iter().all(|v| *v < P));
}

fn files(mib: usize) -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let at = |name: &str| dir.path().join(name);
    let count = mib << 18;
    let layout = Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![count as u64],
        }],
    };
    let encoding = Encoding {
        frac_bits: 24,
        clip: 128.0,
    };
    let shape = layout.shape(CHUNK_COORDS as u64);
    let bytes = shape.coords as usize * 8;

    let base: Vec<f32> = (0..count).map(|i| (i % 1000) as f32 * 1e-3).collect();
    let local: Vec<f32> = base.iter().map(|v| v * 1.001).collect();
    model::write_model(&at("base"), &layout, &[TensorView::F32(&base)])?;

    let encode = |out: &Path, seeds: &[[u8; 32]]| {
        model::encode_update(&EncodeRequest {
            layout: &layout,
            encoding,
            chunk_coords: CHUNK_COORDS as u64,
            base: &at("base"),
            local: &[TensorView::F32(&local)],
            weight: 1,
            pad: [Fp::ONE, Fp::ONE],
            seeds,
            out,
        })
    };
    let run = |name: &str, repeats: usize, work: &mut dyn FnMut()| {
        line(name, rate(bytes, repeats, work));
    };

    run("encode, no mask", 3, &mut || {
        encode(&at("plain"), &[]).unwrap();
    });
    run("encode, one mask", 3, &mut || {
        encode(&at("a"), &[[1; 32]]).unwrap();
    });
    encode(&at("b"), &[[2; 32]])?;
    encode(&at("c"), &[[3; 32]])?;
    encode(&at("d"), &[[4; 32]])?;
    let seeds = [[1u8; 32], [2; 32], [3; 32], [4; 32]];
    run("mask sum of 4", 3, &mut || {
        vector::build(&at("mask"), shape, |chunk, buf, _| {
            for seed in &seeds {
                vector::apply_mask(seed, chunk, buf);
            }
            Ok(())
        })
        .unwrap();
    });
    let (a, b, c, d) = (at("a"), at("b"), at("c"), at("d"));
    let inputs: [&Path; 4] = [&a, &b, &c, &d];
    run("combine 4", 3, &mut || {
        vector::combine(&at("sum"), shape, &inputs, &[]).unwrap();
    });
    run("combine 1 - 1", 3, &mut || {
        vector::combine(&at("aggregate"), shape, &[&at("sum")], &[&at("mask")]).unwrap();
    });
    run("manifest of a file", 3, &mut || {
        vector::manifest_of(&a, shape).unwrap();
    });
    let rho = vector::challenge(&Digest::of(b"hotpath"));
    run("fingerprint a file", 3, &mut || {
        vector::fingerprint_file(&a, shape, rho).unwrap();
    });
    run("fingerprint a mask", 3, &mut || {
        vector::fingerprint_mask(&[1; 32], shape, rho).unwrap();
    });
    run("range check", 3, &mut || {
        model::in_range(&at("plain"), shape, count as u64, encoding.bound(1)).unwrap();
    });
    run("apply aggregate", 3, &mut || {
        model::apply_aggregate(
            &layout,
            encoding,
            CHUNK_COORDS as u64,
            &at("base"),
            &at("aggregate"),
            4,
            &at("next"),
        )
        .unwrap();
    });
    run("blake3 of a file", 3, &mut || {
        Digest::file(&a).unwrap();
    });
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mib = match std::env::args().nth(1) {
        Some(text) => text.parse()?,
        None => 64,
    };
    println!("one chunk of {CHUNK_COORDS} coordinates in memory");
    kernels();
    println!("\nall cores, a {mib} MiB float32 model on disk");
    files(mib)
}
