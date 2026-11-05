//! A proof that a masked update is short.
//!
//! A site's update x is shared between the aggregator, which holds x plus the
//! masks, and the helpers, which hold the masks. Nobody sees x, so nobody can
//! measure it. The site proves to all of them at once that the L2 norm of x is
//! at most `norm`: each party turns its share and a share of the proof into a
//! short message, and the sum of those messages decides.
//!
//! The proof is a fully linear one in the sense of Boneh et al. (CRYPTO 2019),
//! arranged as in PINE (Rothblum, Omri, Chen, Talwar, 2023):
//!
//! - The squares of x add up to v in the field, from one polynomial identity
//!   checked at a point the site could not predict.
//! - v lies between 0 and norm^2, from the bits of v and of norm^2 - v.
//! - No wraparound: `REPS` random sums of x with signs in {-1, 0, 1}, each
//!   shown to be small by its bits. A vector whose squares overflow the field
//!   passes one such test with probability at most 1/2 (PINE, Prop. 3.5).
//!
//! The first two parts need norm^2 to sit far below the field size. With
//! the noise of a private run that holds for small models only. Above it the
//! proof keeps the third part alone, which still holds a site to nine times
//! the interval's half width: a longer vector passes one test with probability
//! below 1/2, by the Berry Esseen argument PINE uses for wraparound. `enforced`
//! says which bound a plan gets.
//!
//! It is sound against a site that colludes with nobody who checks it. A
//! helper holds a share nobody else can recompute, so a site and a helper
//! together can pass anything.

use std::fs::File;
use std::io;
use std::path::Path;

use rand_chacha::ChaCha12Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;
use thiserror::Error;

use crate::field::{Fp, Fp2, P};
use crate::model::PAD;
use crate::privacy::{Privacy, ceil_sqrt};
use crate::vector::{self, Digest, Shape};

/// Wraparound tests per proof. A bad vector passes all of them with
/// probability 2^-64, and a site gets one try per vector it commits to.
pub const REPS: usize = 64;

/// An honest test value lies within SPREAD * norm except with probability
/// 2 * exp(-SPREAD^2).
const SPREAD: u64 = 7;

/// PINE's wraparound bound needs p >= 81 * a^2 * norm^2, and here a is below
/// 2 * SPREAD. Past this limit the squares can't be summed exactly.
const NORM_SQUARE_LIMIT: u128 = (P / (81 * 4 * SPREAD * SPREAD)) as u128;

/// The test interval must be this small a part of the field for a long
/// vector to fail a test half the time.
const INTERVAL_SHARE: u64 = 100_000;

/// A vector longer than this many half intervals fails a test half the time.
const REACH: u64 = 9;

const WORD: usize = size_of::<u64>();
const FP2_BYTES: usize = 2 * WORD;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NormError {
    #[error("the norm bound is too large for the field")]
    TooLarge,
    #[error("malformed {0}")]
    Malformed(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    shape: Shape,
    plain: u64,
    /// The bound proven, in grid units.
    pub norm: u64,
    half: u64,
    /// Whether the squares are summed and the bound is exactly `norm`.
    exact: bool,
    range_bits: u32,
    square_bits: u32,
    size: usize,
    bit_size: usize,
}

fn ceil_sqrt_wide(n: u128) -> u128 {
    let mut r = (n as f64).sqrt() as u128;
    while r * r < n {
        r += 1;
    }
    while r > 0 && (r - 1) * (r - 1) >= n {
        r -= 1;
    }
    r
}

impl Params {
    /// The bound covers the clipped update and the noise the site must add.
    /// The noise term holds except with probability e^-85 (Laurent and
    /// Massart, for a vector with variance proxy `noise` per coordinate).
    pub fn new(privacy: &Privacy, shape: Shape) -> Result<Params, NormError> {
        let plain = shape.coords.saturating_sub(PAD);
        if plain == 0 {
            return Err(NormError::TooLarge);
        }
        let spread = u128::from(privacy.noise) * (2 * u128::from(plain) + 256);
        let norm = u128::from(privacy.sensitivity(plain)) + ceil_sqrt_wide(spread);
        let norm = u64::try_from(norm).map_err(|_| NormError::TooLarge)?;
        let half = SPREAD
            .checked_mul(norm)
            .and_then(u64::checked_next_power_of_two)
            .filter(|half| half.checked_mul(INTERVAL_SHARE).is_some_and(|v| v <= P))
            .ok_or(NormError::TooLarge)?;
        let square = u128::from(norm).pow(2);
        let mut params = Params {
            shape,
            plain,
            norm,
            half,
            exact: square <= NORM_SQUARE_LIMIT,
            range_bits: half.trailing_zeros() + 1,
            square_bits: u128::BITS - square.leading_zeros(),
            size: (ceil_sqrt(plain / 2) + 1).next_power_of_two().max(2) as usize,
            bit_size: 0,
        };
        params.bit_size = (params.bits() + 1).next_power_of_two();
        Ok(params)
    }

    /// The longest update a dishonest site can get past the proof, in grid
    /// units: `norm` itself where the squares can be summed, and otherwise
    /// what the interval tests alone exclude.
    pub fn enforced(&self) -> u64 {
        if self.exact {
            self.norm
        } else {
            REACH * self.half
        }
    }

    fn wire_len(&self) -> usize {
        self.size - 1
    }

    fn covered(&self, chunk: u64) -> usize {
        let (start, end) = self.shape.range(chunk);
        end.min(self.plain).saturating_sub(start) as usize
    }

    fn wires_in(&self, chunk: u64) -> usize {
        if self.exact {
            self.covered(chunk).div_ceil(self.wire_len())
        } else {
            0
        }
    }

    fn wires(&self) -> usize {
        (0..self.shape.chunks()).map(|c| self.wires_in(c)).sum()
    }

    fn bits(&self) -> usize {
        let squares = if self.exact { 2 * self.square_bits } else { 0 };
        REPS * self.range_bits as usize + squares as usize
    }

    fn square_coefficients(&self) -> usize {
        if self.exact { 2 * self.size } else { 0 }
    }

    fn aux_len(&self) -> usize {
        self.wires() + 1 + self.bits() + self.square_coefficients() + 2 * self.bit_size
    }

    /// Bytes of the proof share a site uploads.
    pub fn proof_bytes(&self) -> usize {
        self.aux_len() * WORD
    }

    /// Bytes of one party's message about one site.
    pub fn share_bytes(&self) -> usize {
        (self.wires() + 4) * FP2_BYTES
    }
}

/// What the wraparound vectors are drawn from. It names every share of the
/// vector, so the site fixes the vector before it learns them.
pub fn joint(plan: &Digest, round: u32, site: u32, masked: &Digest, sealed: &[Digest]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("fedtally v1 norm joint");
    hasher.update(&plan.0);
    hasher.update(&round.to_le_bytes());
    hasher.update(&site.to_le_bytes());
    hasher.update(&masked.0);
    for digest in sealed {
        hasher.update(&digest.0);
    }
    *hasher.finalize().as_bytes()
}

/// The points a proof is checked at, from a log head the site could not have
/// known when it committed to the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Challenge {
    wire: Fp2,
    bit: Fp2,
    mix: Fp2,
}

impl Challenge {
    pub fn derive(head: &Digest, site: u32) -> Challenge {
        let mut reader = blake3::Hasher::new_derive_key("fedtally v1 norm challenge")
            .update(&head.0)
            .update(&site.to_le_bytes())
            .finalize_xof();
        let mut limb = move || loop {
            let mut bytes = [0u8; WORD];
            reader.fill(&mut bytes);
            let v = u64::from_le_bytes(bytes);
            if v < P {
                return Fp::new(v);
            }
        };
        // Outside the base field, so never on an evaluation domain.
        let mut point = move || {
            let a = limb();
            loop {
                let b = limb();
                if b != Fp::ZERO {
                    return Fp2 { a, b };
                }
            }
        };
        Challenge {
            wire: point(),
            bit: point(),
            mix: point(),
        }
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

fn inverse(v: Fp) -> Fp {
    v.pow(P - 2)
}

fn inverse2(v: Fp2) -> Fp2 {
    let d = inverse(v.a * v.a - Fp::new(7) * v.b * v.b);
    Fp2 {
        a: v.a * d,
        b: -(v.b * d),
    }
}

fn scale(v: Fp2, by: Fp) -> Fp2 {
    Fp2 {
        a: v.a * by,
        b: v.b * by,
    }
}

fn reduce(v: u128) -> Fp {
    Fp::new((v % u128::from(P)) as u64)
}

/// The n-th roots of unity, n a power of two.
struct Domain {
    n: usize,
    /// The first n/2 powers of the root of unity.
    twiddles: Vec<Fp>,
    n_inv: Fp,
}

impl Domain {
    fn new(n: usize) -> Domain {
        debug_assert!(n.is_power_of_two() && n >= 2);
        let root = Fp::new(7).pow((P - 1) / n as u64);
        let mut twiddles = Vec::with_capacity(n / 2);
        let mut w = Fp::ONE;
        for _ in 0..n / 2 {
            twiddles.push(w);
            w = w * root;
        }
        Domain {
            n,
            twiddles,
            n_inv: inverse(Fp::new(n as u64)),
        }
    }

    fn root(&self) -> Fp {
        self.twiddles.get(1).copied().unwrap_or(-Fp::ONE)
    }

    /// Coefficients to values on the domain, in place.
    fn evaluate(&self, a: &mut [Fp]) {
        let n = self.n;
        debug_assert_eq!(a.len(), n);
        let shift = usize::BITS - n.trailing_zeros();
        for i in 0..n {
            let j = i.reverse_bits() >> shift;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            for block in a.chunks_exact_mut(len) {
                let (lo, hi) = block.split_at_mut(len / 2);
                for (k, (x, y)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let t = *y * self.twiddles[k * step];
                    *y = *x - t;
                    *x = *x + t;
                }
            }
            len *= 2;
        }
    }

    /// Values on the domain to coefficients, in place.
    fn interpolate(&self, a: &mut [Fp]) {
        self.evaluate(a);
        a[1..].reverse();
        for v in a.iter_mut() {
            *v = *v * self.n_inv;
        }
    }

    /// The Lagrange basis of the domain at `at`: a polynomial of degree below
    /// n takes there the sum of basis[i] times its value at the i-th point.
    fn basis(&self, at: Fp2) -> Vec<Fp2> {
        let common = scale(at.pow(self.n as u64) - Fp2::ONE, self.n_inv);
        let root = self.root();
        let mut w = Fp::ONE;
        (0..self.n)
            .map(|_| {
                let term = scale(common * inverse2(at - Fp2 { a: w, b: Fp::ZERO }), w);
                w = w * root;
                term
            })
            .collect()
    }
}

fn horner(coefficients: &[Fp], at: Fp2) -> Fp2 {
    coefficients
        .iter()
        .rev()
        .fold(Fp2::ZERO, |acc, c| (acc * at).add_base(*c))
}

/// The slices of an auxiliary share: wire seeds, the bit wire, and the
/// coefficients of the two proof polynomials.
struct Aux<'a> {
    seeds: &'a [Fp],
    bits: &'a [Fp],
    squares: &'a [Fp],
    products: &'a [Fp],
}

impl Params {
    fn split<'a>(&self, aux: &'a [Fp]) -> Aux<'a> {
        let (seeds, rest) = aux.split_at(self.wires());
        let (bits, rest) = rest.split_at(1 + self.bits());
        let (squares, products) = rest.split_at(self.square_coefficients());
        Aux {
            seeds,
            bits,
            squares,
            products,
        }
    }
}

fn helper_aux(params: &Params, seed: &[u8; 32]) -> Vec<Fp> {
    let key = blake3::derive_key("fedtally v1 norm share", seed);
    let mut rng = ChaCha12Rng::from_seed(key);
    (0..params.aux_len()).map(|_| uniform(&mut rng)).collect()
}

fn mask_chunk(seed: &[u8; 32], shape: Shape, chunk: u64, buf: &mut Vec<u64>) {
    buf.clear();
    buf.resize(shape.chunk_bytes(chunk) / WORD, 0);
    vector::apply_mask(seed, chunk, buf);
}

/// The two words that pick z for one coordinate: bit k of the first adds the
/// coordinate to test k, bit k of the second subtracts it.
fn signs(joint: &[u8; 32], chunk: u64) -> ChaCha12Rng {
    let mut rng = ChaCha12Rng::from_seed(*joint);
    rng.set_stream(chunk);
    rng
}

struct Scan {
    wires: Vec<Fp2>,
    tests: [Fp; REPS],
}

/// A party's share of every wire at the challenge point and of every test
/// sum, from its share of the vector.
fn scan<S>(params: &Params, joint: &[u8; 32], basis: &[Fp2], share: S) -> io::Result<Scan>
where
    S: Fn(u64, &mut Vec<u64>) -> io::Result<()> + Sync,
{
    let per_chunk: io::Result<Vec<(Vec<Fp2>, [Fp; REPS])>> = (0..params.shape.chunks())
        .into_par_iter()
        .map(|chunk| {
            let mut buf = Vec::new();
            share(chunk, &mut buf)?;
            let values = &buf[..params.covered(chunk)];
            let wired = if params.exact { values } else { &[] };
            let wires = wired
                .chunks(params.wire_len())
                .map(|wire| {
                    wire.iter()
                        .zip(&basis[1..])
                        .fold(Fp2::ZERO, |acc, (v, b)| acc + scale(*b, Fp::new(*v)))
                })
                .collect();
            let mut rng = signs(joint, chunk);
            let (mut plus, mut minus) = ([0u128; REPS], [0u128; REPS]);
            for v in values {
                let (add, sub) = (rng.next_u64(), rng.next_u64());
                let v = u128::from(*v);
                for k in 0..REPS {
                    plus[k] += v * u128::from((add >> k) & 1);
                    minus[k] += v * u128::from((sub >> k) & 1);
                }
            }
            let tests = std::array::from_fn(|k| reduce(plus[k]) - reduce(minus[k]));
            Ok((wires, tests))
        })
        .collect();
    let mut out = Scan {
        wires: Vec::with_capacity(params.wires()),
        tests: [Fp::ZERO; REPS],
    };
    for (wires, tests) in per_chunk? {
        out.wires.extend(wires);
        for (acc, t) in out.tests.iter_mut().zip(tests) {
            *acc = *acc + t;
        }
    }
    Ok(out)
}

fn weighted(bits: &[Fp]) -> Fp {
    bits.iter()
        .rev()
        .fold(Fp::ZERO, |acc, bit| acc + acc + *bit)
}

/// One party's message: its share of each wire, of the bit wire and of both
/// proof polynomials at the challenge, and of everything that must be zero.
fn message(
    params: &Params,
    challenge: &Challenge,
    scan: Scan,
    aux: &[Fp],
    leader: bool,
    negate: bool,
) -> Vec<u8> {
    let aux = params.split(aux);
    let sign = |v: Fp2| if negate { Fp2::ZERO - v } else { v };
    let sign_base = |v: Fp| if negate { -v } else { v };

    let wire_domain = Domain::new(params.size);
    let seed_weight = wire_domain.basis(challenge.wire)[0];
    let wires = scan
        .wires
        .into_iter()
        .zip(aux.seeds)
        .map(|(x, seed)| sign(x) + scale(seed_weight, *seed));

    let bit_domain = Domain::new(params.bit_size);
    let bit_wire = bit_domain
        .basis(challenge.bit)
        .iter()
        .zip(aux.bits)
        .fold(Fp2::ZERO, |acc, (b, v)| acc + scale(*b, *v));

    let mut folded: Vec<Fp> = (0..params.bit_size)
        .map(|i| aux.products[i] + aux.products[i + params.bit_size])
        .collect();
    bit_domain.evaluate(&mut folded);

    let bits = &aux.bits[1..];
    let (ranges, squares) = bits.split_at(REPS * params.range_bits as usize);
    let (low, high) = squares.split_at(squares.len() / 2);
    let constant = |v: u64| if leader { Fp::new(v) } else { Fp::ZERO };

    let mut zero = Fp2::ZERO;
    let mut power = Fp2::ONE;
    let mut mix = |v: Fp| {
        zero = zero + scale(power, v);
        power = power * challenge.mix;
    };
    for (test, range) in scan
        .tests
        .iter()
        .zip(ranges.chunks(params.range_bits as usize))
    {
        mix(sign_base(*test) + constant(params.half) - weighted(range));
    }
    if params.exact {
        let n = Fp::new(params.size as u64);
        let total: Fp = aux.squares.iter().fold(Fp::ZERO, |acc, c| acc + *c);
        let square_sum = n * (aux.squares[0] + aux.squares[params.size]) - total;
        mix(weighted(low) + weighted(high) - constant(params.norm * params.norm));
        mix(square_sum - weighted(low));
    }
    for value in &folded[1..=params.bits()] {
        mix(*value);
    }

    let tail = [
        bit_wire,
        horner(aux.squares, challenge.wire),
        horner(aux.products, challenge.bit),
        zero,
    ];
    let mut out = Vec::with_capacity(params.share_bytes());
    for v in wires.chain(tail) {
        out.extend_from_slice(&v.a.value().to_le_bytes());
        out.extend_from_slice(&v.b.value().to_le_bytes());
    }
    out
}

fn parse_fp(bytes: &[u8], what: &'static str) -> Result<Vec<Fp>, NormError> {
    bytes
        .as_chunks::<WORD>()
        .0
        .iter()
        .map(|word| {
            let v = u64::from_le_bytes(*word);
            (v < P)
                .then(|| Fp::new(v))
                .ok_or(NormError::Malformed(what))
        })
        .collect()
}

/// The aggregator's message about one site, from the masked vector and the
/// proof share the site uploaded.
pub fn leader_share(
    params: &Params,
    joint: &[u8; 32],
    challenge: &Challenge,
    masked: &Path,
    proof: &[u8],
) -> io::Result<Result<Vec<u8>, NormError>> {
    if proof.len() != params.proof_bytes() {
        return Ok(Err(NormError::Malformed("proof")));
    }
    let aux = match parse_fp(proof, "proof") {
        Ok(aux) => aux,
        Err(e) => return Ok(Err(e)),
    };
    let file = File::open(masked)?;
    let basis = Domain::new(params.size).basis(challenge.wire);
    let scan = scan(params, joint, &basis, |chunk, buf| {
        vector::read_chunk(&file, params.shape, chunk, buf)
    })?;
    Ok(Ok(message(params, challenge, scan, &aux, true, false)))
}

/// A helper's message about one site, from the seed the site sealed to it.
pub fn helper_share(
    params: &Params,
    joint: &[u8; 32],
    challenge: &Challenge,
    seed: &[u8; 32],
) -> io::Result<Vec<u8>> {
    let basis = Domain::new(params.size).basis(challenge.wire);
    let scan = scan(params, joint, &basis, |chunk, buf| {
        mask_chunk(seed, params.shape, chunk, buf);
        Ok(())
    })?;
    let aux = helper_aux(params, seed);
    Ok(message(params, challenge, scan, &aux, false, true))
}

/// Whether the messages about one site add up to a valid proof. The caller
/// passes one from the aggregator and one from each helper.
pub fn decide(params: &Params, shares: &[&[u8]]) -> bool {
    if shares.is_empty() {
        return false;
    }
    let mut sum = vec![Fp2::ZERO; params.wires() + 4];
    for share in shares {
        if share.len() != params.share_bytes() {
            return false;
        }
        let Ok(limbs) = parse_fp(share, "share") else {
            return false;
        };
        for (acc, [a, b]) in sum.iter_mut().zip(limbs.as_chunks::<2>().0) {
            *acc = *acc + Fp2 { a: *a, b: *b };
        }
    }
    let (wires, tail) = sum.split_at(params.wires());
    let [bit_wire, squares, products, zero] = [tail[0], tail[1], tail[2], tail[3]];
    let wire_squares = wires.iter().fold(Fp2::ZERO, |acc, w| acc + *w * *w);
    zero == Fp2::ZERO && squares == wire_squares && products == bit_wire * (bit_wire - Fp2::ONE)
}

struct Witness {
    seeds: Vec<Fp>,
    /// The sum of the squared wires on the doubled domain, split into the domain and its coset.
    even: Vec<Fp>,
    odd: Vec<Fp>,
    squares: u128,
    tests: [i128; REPS],
}

fn witness(
    params: &Params,
    joint: &[u8; 32],
    masked: &Path,
    seeds: &[[u8; 32]],
) -> io::Result<Witness> {
    let file = File::open(masked)?;
    let n = params.size;
    let domain = Domain::new(n);
    let shift = Fp::new(7).pow((P - 1) / (2 * n as u64));
    let mut shifts = Vec::with_capacity(n);
    let mut s = Fp::ONE;
    for _ in 0..n {
        shifts.push(s);
        s = s * shift;
    }
    let blind = vector_seed();

    let per_chunk: io::Result<Vec<Witness>> = (0..params.shape.chunks())
        .into_par_iter()
        .map(|chunk| {
            let (mut buf, mut mask) = (Vec::new(), Vec::new());
            vector::read_chunk(&file, params.shape, chunk, &mut buf)?;
            for seed in seeds {
                mask_chunk(seed, params.shape, chunk, &mut mask);
                for (v, m) in buf.iter_mut().zip(&mask) {
                    *v = (Fp::new(*v) - Fp::new(*m)).value();
                }
            }
            let values = &buf[..params.covered(chunk)];

            let mut out = Witness {
                seeds: Vec::with_capacity(params.wires_in(chunk)),
                even: vec![Fp::ZERO; n],
                odd: vec![Fp::ZERO; n],
                squares: 0,
                tests: [0; REPS],
            };
            let mut rng = signs(joint, chunk);
            for v in values {
                let (add, sub) = (rng.next_u64(), rng.next_u64());
                let v = i128::from(Fp::new(*v).centered());
                out.squares = out.squares.saturating_add((v * v) as u128);
                for k in 0..REPS {
                    let z = ((add >> k) & 1) as i128 - ((sub >> k) & 1) as i128;
                    out.tests[k] += z * v;
                }
            }

            let mut blinds = ChaCha12Rng::from_seed(blind);
            blinds.set_stream(chunk);
            let mut wire = vec![Fp::ZERO; n];
            let wired = if params.exact { values } else { &[] };
            for part in wired.chunks(params.wire_len()) {
                let seed = uniform(&mut blinds);
                out.seeds.push(seed);
                wire.fill(Fp::ZERO);
                wire[0] = seed;
                for (slot, v) in wire[1..].iter_mut().zip(part) {
                    *slot = Fp::new(*v);
                }
                for (acc, v) in out.even.iter_mut().zip(&wire) {
                    *acc = *acc + *v * *v;
                }
                domain.interpolate(&mut wire);
                for (c, s) in wire.iter_mut().zip(&shifts) {
                    *c = *c * *s;
                }
                domain.evaluate(&mut wire);
                for (acc, v) in out.odd.iter_mut().zip(&wire) {
                    *acc = *acc + *v * *v;
                }
            }
            Ok(out)
        })
        .collect();

    let mut total = Witness {
        seeds: Vec::with_capacity(params.wires()),
        even: vec![Fp::ZERO; n],
        odd: vec![Fp::ZERO; n],
        squares: 0,
        tests: [0; REPS],
    };
    for part in per_chunk? {
        total.seeds.extend(part.seeds);
        for (acc, v) in total.even.iter_mut().zip(&part.even) {
            *acc = *acc + *v;
        }
        for (acc, v) in total.odd.iter_mut().zip(&part.odd) {
            *acc = *acc + *v;
        }
        total.squares = total.squares.saturating_add(part.squares);
        for (acc, v) in total.tests.iter_mut().zip(part.tests) {
            *acc += v;
        }
    }
    Ok(total)
}

fn vector_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed).expect("the operating system has no randomness");
    seed
}

fn push_bits(out: &mut Vec<Fp>, value: u128, count: u32) {
    out.extend((0..count).map(|j| Fp::new(((value >> j) & 1) as u64)));
}

fn assemble(params: &Params, witness: Witness, seeds: &[[u8; 32]]) -> Vec<u8> {
    let mut squares = Vec::with_capacity(params.square_coefficients());
    if params.exact {
        for (even, odd) in witness.even.iter().zip(&witness.odd) {
            squares.extend([*even, *odd]);
        }
        Domain::new(2 * params.size).interpolate(&mut squares);
    }

    let mut bits = Vec::with_capacity(1 + params.bits());
    let mut blinds = ChaCha12Rng::from_seed(vector_seed());
    bits.push(uniform(&mut blinds));
    for test in witness.tests {
        push_bits(
            &mut bits,
            (test + i128::from(params.half)) as u128,
            params.range_bits,
        );
    }
    if params.exact {
        let bound = u128::from(params.norm).pow(2);
        push_bits(&mut bits, witness.squares, params.square_bits);
        push_bits(
            &mut bits,
            bound.wrapping_sub(witness.squares),
            params.square_bits,
        );
    }

    let m = params.bit_size;
    let mut products = vec![Fp::ZERO; 2 * m];
    products[..bits.len()].copy_from_slice(&bits);
    Domain::new(m).interpolate(&mut products[..m]);
    let doubled = Domain::new(2 * m);
    doubled.evaluate(&mut products);
    for v in products.iter_mut() {
        *v = *v * (*v - Fp::ONE);
    }
    doubled.interpolate(&mut products);

    let mut aux = witness.seeds;
    aux.extend(bits);
    aux.extend(squares);
    aux.extend(products);
    for seed in seeds {
        for (v, share) in aux.iter_mut().zip(helper_aux(params, seed)) {
            *v = *v - share;
        }
    }
    bytemuck::cast_slice(&aux.iter().map(|v| v.value()).collect::<Vec<u64>>()).to_vec()
}

/// Whether the messages in each party's bundle for the site at `position`
/// add up to a valid proof. A bundle holds one message per screened site.
pub fn accepted(params: &Params, position: usize, leader: &[u8], helpers: &[Vec<u8>]) -> bool {
    let size = params.share_bytes();
    let shares: Option<Vec<&[u8]>> = std::iter::once(leader)
        .chain(helpers.iter().map(Vec::as_slice))
        .map(|bundle| bundle.get(position * size..(position + 1) * size))
        .collect();
    shares.is_some_and(|shares| decide(params, &shares))
}

/// What a site sends if it skips its own check. Nothing honest calls this:
/// it is how tests and attack benchmarks play a site that breaks the bound.
pub fn prove_unchecked(
    params: &Params,
    joint: &[u8; 32],
    masked: &Path,
    seeds: &[[u8; 32]],
) -> io::Result<Vec<u8>> {
    Ok(assemble(
        params,
        witness(params, joint, masked, seeds)?,
        seeds,
    ))
}

/// The proof share a site uploads next to its masked vector, or `None` when
/// the vector is out of bounds or failed a wraparound test. An honest site
/// gets `None` with negligible probability, and then encodes again.
pub fn prove(
    params: &Params,
    joint: &[u8; 32],
    masked: &Path,
    seeds: &[[u8; 32]],
) -> io::Result<Option<Vec<u8>>> {
    let witness = witness(params, joint, masked, seeds)?;
    let half = i128::from(params.half);
    let short = witness.squares <= u128::from(params.norm).pow(2);
    let inside = witness.tests.iter().all(|t| (-half..half).contains(t));
    Ok((short && inside).then(|| assemble(params, witness, seeds)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::Privacy;

    fn privacy(clip: u64, noise: u64) -> Privacy {
        Privacy {
            clip,
            noise,
            colluding: 0,
            budget: u64::MAX,
            epsilon: 1.0,
            delta: 1e-5,
        }
    }

    struct Case {
        _dir: tempfile::TempDir,
        path: std::path::PathBuf,
        params: Params,
        seeds: Vec<[u8; 32]>,
        joint: [u8; 32],
        challenge: Challenge,
    }

    fn case(values: &[i64], chunk_coords: u64, helpers: u8, terms: &Privacy) -> Case {
        let shape = Shape {
            coords: values.len() as u64 + PAD,
            chunk_coords,
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("masked");
        let seeds: Vec<[u8; 32]> = (0..helpers).map(|h| [h + 1; 32]).collect();
        let manifest = vector::build(&path, shape, |chunk, buf, _| {
            let (start, _) = shape.range(chunk);
            for (k, slot) in buf.iter_mut().enumerate() {
                let at = start as usize + k;
                *slot = values
                    .get(at)
                    .map_or(0xABCD + at as u64, |v| Fp::from_i64(*v).value());
            }
            for seed in &seeds {
                vector::apply_mask(seed, chunk, buf);
            }
            Ok(())
        })
        .unwrap();
        let sealed: Vec<Digest> = seeds.iter().map(|s| Digest::of(s)).collect();
        Case {
            params: Params::new(terms, shape).unwrap(),
            joint: joint(&Digest::of(b"plan"), 3, 1, &manifest.root, &sealed),
            challenge: Challenge::derive(&Digest::of(b"head"), 1),
            path,
            seeds,
            _dir: dir,
        }
    }

    impl Case {
        fn shares(&self, proof: &[u8]) -> Vec<Vec<u8>> {
            let leader = leader_share(
                &self.params,
                &self.joint,
                &self.challenge,
                &self.path,
                proof,
            )
            .unwrap()
            .unwrap();
            let mut shares = vec![leader];
            for seed in &self.seeds {
                shares
                    .push(helper_share(&self.params, &self.joint, &self.challenge, seed).unwrap());
            }
            shares
        }

        fn accepts(&self, proof: &[u8]) -> bool {
            let shares = self.shares(proof);
            let refs: Vec<&[u8]> = shares.iter().map(Vec::as_slice).collect();
            decide(&self.params, &refs)
        }

        fn forced(&self) -> Vec<u8> {
            prove_unchecked(&self.params, &self.joint, &self.path, &self.seeds).unwrap()
        }
    }

    fn ramp(len: usize, scale: i64) -> Vec<i64> {
        (0..len as i64).map(|i| (i % 7 - 3) * scale).collect()
    }

    #[test]
    fn transforms_invert_and_match_direct_evaluation() {
        let domain = Domain::new(16);
        let coefficients: Vec<Fp> = (1..=16u64).map(|v| Fp::new(v * v + 5)).collect();
        let mut values = coefficients.clone();
        domain.evaluate(&mut values);
        let root = domain.root();
        for (i, v) in values.iter().enumerate() {
            let at = Fp2 {
                a: root.pow(i as u64),
                b: Fp::ZERO,
            };
            assert_eq!(horner(&coefficients, at).a, *v);
        }
        let mut back = values.clone();
        domain.interpolate(&mut back);
        assert_eq!(back, coefficients);

        let at = Fp2 {
            a: Fp::new(12345),
            b: Fp::new(678),
        };
        let through_basis = domain
            .basis(at)
            .iter()
            .zip(&values)
            .fold(Fp2::ZERO, |acc, (b, v)| acc + scale(*b, *v));
        assert_eq!(through_basis, horner(&coefficients, at));
        assert_eq!(inverse2(at) * at, Fp2::ONE);
    }

    #[test]
    fn an_honest_vector_is_accepted() {
        let terms = privacy(40, 8);
        for (len, chunk, helpers) in [(1, 128, 1), (200, 128, 1), (1000, 256, 2), (3000, 1024, 3)] {
            let case = case(&ramp(len, 1), chunk, helpers, &terms);
            let proof = prove(&case.params, &case.joint, &case.path, &case.seeds)
                .unwrap()
                .expect("in bounds");
            assert_eq!(proof.len(), case.params.proof_bytes());
            assert!(case.accepts(&proof), "len {len}");
        }
    }

    #[test]
    fn a_vector_at_the_bound_passes_and_one_past_it_does_not() {
        let terms = privacy(10, 8);
        let shape = Shape {
            coords: 4 + PAD,
            chunk_coords: 128,
        };
        let norm = Params::new(&terms, shape).unwrap().norm as i64;
        let at = case(&[norm, 0, 0, 0], 128, 1, &terms);
        let proof = prove(&at.params, &at.joint, &at.path, &at.seeds).unwrap();
        assert!(at.accepts(&proof.expect("at the bound")));

        let past = case(&[norm, 1, 0, 0], 128, 1, &terms);
        assert!(
            prove(&past.params, &past.joint, &past.path, &past.seeds)
                .unwrap()
                .is_none()
        );
        assert!(!past.accepts(&past.forced()));
    }

    #[test]
    fn a_long_vector_is_refused_however_it_is_proven() {
        let terms = privacy(40, 8);
        let case = case(&ramp(500, 1000), 256, 1, &terms);
        assert!(
            prove(&case.params, &case.joint, &case.path, &case.seeds)
                .unwrap()
                .is_none()
        );
        assert!(!case.accepts(&case.forced()));
    }

    #[test]
    fn squares_that_wrap_the_field_are_refused() {
        // 2^32 squared is 2^32 - 1 in the field, and with a second coordinate
        // of the right size the squares add to a small value there.
        let terms = privacy(40, 8);
        let big = 1i64 << 32;
        let mut values = ramp(300, 0);
        values[7] = big;
        values[8] = big;
        let case = case(&values, 128, 1, &terms);
        let in_field =
            Fp::from_i64(big) * Fp::from_i64(big) + Fp::from_i64(big) * Fp::from_i64(big);
        assert!(in_field.value() > u64::from(u32::MAX));
        assert!(!case.accepts(&case.forced()));

        let mut values = ramp(300, 0);
        values[0] = (P / 2) as i64;
        values[1] = (P / 2) as i64 - 5;
        let huge = self::case(&values, 128, 1, &terms);
        assert!(!huge.accepts(&huge.forced()));
    }

    #[test]
    fn any_change_to_a_proof_or_a_share_is_refused() {
        let terms = privacy(40, 8);
        let case = case(&ramp(700, 1), 256, 2, &terms);
        let proof = prove(&case.params, &case.joint, &case.path, &case.seeds)
            .unwrap()
            .unwrap();
        let words = proof.len() / WORD;
        for word in [0, case.params.wires(), words / 2, words - 1] {
            let mut edited = proof.clone();
            edited[word * WORD] ^= 1;
            let outcome = leader_share(
                &case.params,
                &case.joint,
                &case.challenge,
                &case.path,
                &edited,
            )
            .unwrap();
            if outcome.is_ok() {
                assert!(!case.accepts(&edited), "word {word}");
            }
        }

        let shares = case.shares(&proof);
        for index in 0..shares.len() {
            for at in [0, shares[index].len() - 1] {
                let mut edited = shares.clone();
                edited[index][at] ^= 1;
                let refs: Vec<&[u8]> = edited.iter().map(Vec::as_slice).collect();
                assert!(!decide(&case.params, &refs));
            }
        }
        let refs: Vec<&[u8]> = shares[..2].iter().map(Vec::as_slice).collect();
        assert!(!decide(&case.params, &refs));
    }

    #[test]
    fn malformed_input_is_refused_without_panicking() {
        let terms = privacy(40, 8);
        let case = case(&ramp(100, 1), 128, 1, &terms);
        let proof = prove(&case.params, &case.joint, &case.path, &case.seeds)
            .unwrap()
            .unwrap();
        let share = |proof: &[u8]| {
            leader_share(
                &case.params,
                &case.joint,
                &case.challenge,
                &case.path,
                proof,
            )
            .unwrap()
        };
        assert_eq!(
            share(&proof[..proof.len() - 1]),
            Err(NormError::Malformed("proof"))
        );
        let mut out_of_range = proof.clone();
        out_of_range[..WORD].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(share(&out_of_range), Err(NormError::Malformed("proof")));
        assert!(!decide(&case.params, &[&[0u8; 3]]));
        assert!(!decide(&case.params, &[]));
    }

    #[test]
    fn a_proof_is_tied_to_its_vector_and_its_challenge() {
        let terms = privacy(40, 8);
        let case = case(&ramp(400, 1), 128, 1, &terms);
        let proof = prove(&case.params, &case.joint, &case.path, &case.seeds)
            .unwrap()
            .unwrap();
        let other = Case {
            joint: joint(&Digest::of(b"plan"), 4, 1, &Digest::of(b"x"), &[]),
            ..case
        };
        assert!(!other.accepts(&proof));
    }

    #[test]
    fn two_proofs_of_one_vector_share_nothing_visible() {
        let terms = privacy(40, 8);
        let case = case(&ramp(400, 1), 128, 1, &terms);
        let prove = || {
            prove(&case.params, &case.joint, &case.path, &case.seeds)
                .unwrap()
                .unwrap()
        };
        let (a, b) = (case.shares(&prove()), case.shares(&prove()));
        let wires = case.params.wires() * FP2_BYTES;
        assert_ne!(a[0][..wires], b[0][..wires]);
        assert_ne!(
            a[0][wires..wires + FP2_BYTES],
            b[0][wires..wires + FP2_BYTES]
        );
    }

    #[test]
    fn the_bound_is_exact_while_its_square_fits_and_looser_above() {
        let shape = Shape {
            coords: 1_000_000 + PAD,
            chunk_coords: 1 << 17,
        };
        let small = Params::new(&privacy(1 << 20, 100), shape).unwrap();
        assert_eq!(small.enforced(), small.norm);

        let noisy = Params::new(&privacy(1 << 18, 1 << 32), shape).unwrap();
        assert!(noisy.norm > 90_000_000);
        assert!(noisy.enforced() >= 9 * 7 * noisy.norm);
        assert!(noisy.enforced() < 9 * 14 * noisy.norm);
        assert_eq!(noisy.wires(), 0);

        let vast = Shape {
            coords: 1 << 40,
            chunk_coords: 1 << 17,
        };
        assert_eq!(
            Params::new(&privacy(1 << 32, 1 << 50), vast),
            Err(NormError::TooLarge)
        );
    }

    /// Variance 2^40 over a few thousand coordinates: too much for the exact bound.
    fn loose() -> Privacy {
        privacy(1 << 18, 1 << 40)
    }

    #[test]
    fn above_the_exact_range_an_honest_vector_still_passes() {
        let terms = loose();
        for (len, chunk, helpers) in [(2000, 256, 1), (5000, 1024, 2)] {
            let case = case(&ramp(len, 30_000), chunk, helpers, &terms);
            assert_ne!(case.params.enforced(), case.params.norm);
            let proof = prove(&case.params, &case.joint, &case.path, &case.seeds)
                .unwrap()
                .expect("in bounds");
            assert_eq!(proof.len(), case.params.proof_bytes());
            assert!(case.accepts(&proof));
        }
    }

    #[test]
    fn above_the_exact_range_only_a_much_longer_vector_is_refused() {
        let terms = loose();
        let norm = Params::new(
            &terms,
            Shape {
                coords: 2000 + PAD,
                chunk_coords: 256,
            },
        )
        .unwrap()
        .norm as i64;
        // Twice the bound: the site's own check refuses it, the proof does not.
        let twice = case(&ramp(2000, norm / 40), 256, 1, &terms);
        assert!(
            prove(&twice.params, &twice.joint, &twice.path, &twice.seeds)
                .unwrap()
                .is_none()
        );
        assert!(twice.accepts(&twice.forced()));

        let far = case(&ramp(2000, 20 * norm), 256, 1, &terms);
        assert!(!far.accepts(&far.forced()));

        let mut values = ramp(2000, 0);
        values[3] = (P / 2) as i64 - 9;
        let wrapped = self::case(&values, 256, 1, &terms);
        assert!(!wrapped.accepts(&wrapped.forced()));
    }
}
