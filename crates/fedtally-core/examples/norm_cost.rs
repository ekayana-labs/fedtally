//! Times the norm proof at a few sizes: `cargo run --release --example norm_cost`.

use std::time::Instant;

use fedtally_core::field::Fp;
use fedtally_core::model::PAD;
use fedtally_core::norm::{self, Challenge, Params};
use fedtally_core::privacy::Privacy;
use fedtally_core::vector::{self, Digest, Shape};

fn main() -> std::io::Result<()> {
    let terms = Privacy {
        clip: 1 << 16,
        noise: 64,
        colluding: 0,
        budget: u64::MAX,
        epsilon: 1.0,
        delta: 1e-5,
    };
    let dir = tempfile::tempdir()?;
    println!("coords,proof_bytes,share_bytes,prove_s,aggregator_s,helper_s,decide_s");
    for coords in [1u64 << 16, 1 << 20, 1 << 24, 1 << 26] {
        let shape = Shape {
            coords: coords + PAD,
            chunk_coords: 1 << 20,
        };
        let seed = [7u8; 32];
        let path = dir.path().join("masked");
        let manifest = vector::build(&path, shape, |chunk, buf, _| {
            let (start, _) = shape.range(chunk);
            for (k, slot) in buf.iter_mut().enumerate() {
                let at = (start + k as u64) as i64;
                *slot = Fp::from_i64(at % 17 - 8).value();
            }
            vector::apply_mask(&seed, chunk, buf);
            Ok(())
        })?;
        let params = Params::new(&terms, shape).expect("bound fits the field");
        let joint = norm::joint(&Digest::of(b"plan"), 1, 0, &manifest.root, &[]);
        let challenge = Challenge::derive(&Digest::of(b"head"), 0);

        let clock = Instant::now();
        let proof = norm::prove(&params, &joint, &path, &[seed])?.expect("in bounds");
        let prove = clock.elapsed();
        let clock = Instant::now();
        let leader =
            norm::leader_share(&params, &joint, &challenge, &path, &proof)?.expect("well formed");
        let aggregator = clock.elapsed();
        let clock = Instant::now();
        let helper = norm::helper_share(&params, &joint, &challenge, &seed)?;
        let helping = clock.elapsed();
        let clock = Instant::now();
        assert!(norm::decide(&params, &[&leader, &helper]));
        println!(
            "{coords},{},{},{:.3},{:.3},{:.3},{:.6}",
            proof.len(),
            leader.len(),
            prove.as_secs_f64(),
            aggregator.as_secs_f64(),
            helping.as_secs_f64(),
            clock.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
