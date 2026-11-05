//! Differential privacy for a whole site, from noise each site adds itself.
//!
//! A site clips its update to an L2 bound, rounds it onto the integer grid at
//! random and adds discrete Gaussian noise, all before masking. Everyone else
//! only sees sums, and the noise in a sum is the noise of every site in it.
//!
//! The run's budget is one number, the zCDP parameter rho, kept as an integer
//! so every replica computes the same value. Epsilon and delta are derived from
//! it for people and take no part in the rules.

use rand_core::RngCore;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::Encoding;

/// Rho is counted in units of 2^-40.
pub const RHO_BITS: u32 = 40;
/// A sum of discrete Gaussians is not quite a discrete Gaussian. The gap adds
/// tau * d to rho, where tau < 10 * n * exp(-pi^2 * variance) for n sites (Kairouz, Liu
/// and Steinke 2021, Theorem 1). From this variance up, and within the limits
/// below, that is under 2^-50, far inside one unit.
pub const MIN_NOISE: u64 = 8;
pub const MAX_NOISE: u64 = 1 << 50;
pub const MAX_CLIP: u64 = 1 << 32;
pub const MAX_COORDS: u64 = 1 << 40;
pub const MAX_SITES: usize = 1 << 20;
/// Added to every round: one unit for that gap, one for the division.
const SLACK: u64 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Privacy {
    /// The largest L2 norm a site's update may have, in grid units.
    pub clip: u64,
    /// The variance of the noise one site adds to each coordinate, in grid units.
    pub noise: u64,
    /// Sites in a cohort assumed to give their noise away.
    pub colluding: u32,
    /// What the run may spend, as rho in units of 2^-40.
    pub budget: u64,
    /// The target the budget was chosen for. Only `budget` is enforced.
    pub epsilon: f64,
    pub delta: f64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PrivacyError {
    #[error("clip must be between 1 and 2^32 grid units")]
    Clip,
    #[error("noise variance must be between {MIN_NOISE} and 2^50 grid units")]
    Noise,
    #[error("the cohort must be larger than the number of sites assumed to collude")]
    Colluding,
    #[error("the budget is too small for {0} rounds")]
    Budget(u32),
    #[error("epsilon must be positive and delta between 0 and 1")]
    Target,
    #[error("private runs take at most 2^40 coordinates and 2^20 sites")]
    Size,
}

pub fn ceil_sqrt(n: u64) -> u64 {
    let mut r = (n as f64).sqrt() as u64;
    while u128::from(r) * u128::from(r) < u128::from(n) {
        r += 1;
    }
    while r > 0 && u128::from(r - 1) * u128::from(r - 1) >= u128::from(n) {
        r -= 1;
    }
    r
}

impl Privacy {
    /// How far one site can move a sum, as an L2 norm in grid units. Rounding
    /// moves each of `coords` coordinates by less than one unit, and the last
    /// unit covers floating point error in the clip.
    pub fn sensitivity(&self, coords: u64) -> u64 {
        self.clip + ceil_sqrt(coords) + 1
    }

    /// What one round over `cohort` sites costs, in units of 2^-40.
    pub fn round_cost(&self, coords: u64, cohort: u32) -> Option<u64> {
        let honest = cohort.checked_sub(self.colluding).filter(|h| *h > 0)?;
        let s = u128::from(self.sensitivity(coords));
        let cost = ((s * s) << RHO_BITS) / (2 * u128::from(self.noise) * u128::from(honest));
        u64::try_from(cost).ok()?.checked_add(SLACK)
    }

    pub fn validate(&self, coords: u64, sites: usize, min_cohort: u32) -> Result<(), PrivacyError> {
        if coords > MAX_COORDS || sites > MAX_SITES {
            return Err(PrivacyError::Size);
        }
        if self.clip == 0 || self.clip > MAX_CLIP {
            return Err(PrivacyError::Clip);
        }
        if !(MIN_NOISE..=MAX_NOISE).contains(&self.noise) {
            return Err(PrivacyError::Noise);
        }
        if self.colluding >= min_cohort {
            return Err(PrivacyError::Colluding);
        }
        if !(self.epsilon.is_finite() && self.epsilon > 0.0 && self.delta > 0.0 && self.delta < 1.0)
        {
            return Err(PrivacyError::Target);
        }
        // Within these limits a sum over every site stays far inside the
        // field: under 2^53 for the updates, and noise a thousand standard
        // deviations out is still below 2^46.
        match self.round_cost(coords, min_cohort) {
            Some(cost) if cost <= self.budget => Ok(()),
            _ => Err(PrivacyError::Budget(1)),
        }
    }

    /// Picks the noise and budget for a target, spending evenly over `rounds`
    /// at the smallest cohort the plan allows.
    pub fn for_target(
        target: &Target,
        coords: u64,
        rounds: u32,
        min_cohort: u32,
    ) -> Result<Privacy, PrivacyError> {
        let Target {
            epsilon,
            delta,
            clip,
            colluding,
        } = *target;
        if !(epsilon.is_finite() && epsilon > 0.0 && delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::Target);
        }
        let honest = min_cohort
            .checked_sub(colluding)
            .filter(|h| *h > 0)
            .ok_or(PrivacyError::Colluding)?;
        let budget = budget_for(epsilon, delta);
        let per_round = budget / u64::from(rounds.max(1));
        if per_round <= SLACK {
            return Err(PrivacyError::Budget(rounds));
        }
        let mut privacy = Privacy {
            clip,
            noise: 0,
            colluding,
            budget,
            epsilon,
            delta,
        };
        let s = u128::from(privacy.sensitivity(coords));
        let need =
            ((s * s) << RHO_BITS) / (2 * u128::from(honest) * u128::from(per_round - SLACK)) + 1;
        privacy.noise = u64::try_from(need)
            .ok()
            .filter(|n| *n < MAX_NOISE)
            .ok_or(PrivacyError::Budget(rounds))?
            .max(MIN_NOISE);
        while privacy
            .round_cost(coords, min_cohort)
            .is_none_or(|cost| cost > per_round)
        {
            privacy.noise += 1;
        }
        Ok(privacy)
    }

    /// The standard deviation of the noise in a sum over `cohort` sites,
    /// relative to the clip. The usual noise multiplier.
    pub fn noise_multiplier(&self, cohort: u32) -> f64 {
        let honest = cohort.saturating_sub(self.colluding) as f64;
        (honest * self.noise as f64).sqrt() / self.clip as f64
    }
}

/// What a sponsor asks for. `clip` is in grid units.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub epsilon: f64,
    pub delta: f64,
    pub clip: u64,
    pub colluding: u32,
}

/// A target in the sponsor's own terms. `clip` is an L2 norm in model units.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub epsilon: f64,
    pub delta: f64,
    pub clip: f64,
    #[serde(default)]
    pub colluding: u32,
}

impl Request {
    /// The grid and the plan terms that meet this request.
    pub fn terms(
        &self,
        coords: u64,
        rounds: u32,
        min_cohort: u32,
    ) -> Result<(Encoding, Privacy), PrivacyError> {
        let (frac_bits, clip) = grid_for(self.clip, coords).ok_or(PrivacyError::Clip)?;
        let target = Target {
            epsilon: self.epsilon,
            delta: self.delta,
            clip,
            colluding: self.colluding,
        };
        let privacy = Privacy::for_target(&target, coords, rounds, min_cohort)?;
        let encoding = Encoding {
            frac_bits,
            clip: self.clip,
        };
        Ok((encoding, privacy))
    }
}

/// The fixed-point precision for a clip given in model units, and that clip in
/// grid units. The grid is fine enough that rounding adds under 2% to the
/// sensitivity, and no finer, which keeps the noise within the sampler's range.
pub fn grid_for(clip: f64, coords: u64) -> Option<(u32, u64)> {
    if !(clip.is_finite() && clip > 0.0) {
        return None;
    }
    let want = (64 * ceil_sqrt(coords)).max(1 << 18) as f64;
    let frac_bits = (want / clip).log2().ceil().clamp(1.0, 48.0) as u32;
    let units = (clip * (1u64 << frac_bits) as f64).round();
    (units >= 1.0 && units <= MAX_CLIP as f64).then_some((frac_bits, units as u64))
}

/// The largest budget, in units of 2^-40 of rho, that stays within `epsilon`
/// at `delta`.
pub fn budget_for(epsilon: f64, delta: f64) -> u64 {
    (rho_for(epsilon, delta) * (1u64 << RHO_BITS) as f64).floor() as u64
}

pub fn rho_units(units: u64) -> f64 {
    units as f64 / (1u64 << RHO_BITS) as f64
}

/// The epsilon that rho-zCDP gives at `delta`. This is the conversion of
/// Canonne, Kamath and Steinke (2020, Proposition 12), minimised over the order.
pub fn epsilon(rho: f64, delta: f64) -> f64 {
    if rho <= 0.0 {
        return 0.0;
    }
    let at = |log_order_minus_one: f64| {
        let alpha = 1.0 + log_order_minus_one.exp();
        alpha * rho + (-1.0 / alpha).ln_1p() - (delta.ln() + alpha.ln()) / (alpha - 1.0)
    };
    let (mut lo, mut hi) = (-30.0f64, 30.0f64);
    for _ in 0..200 {
        let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if at(a) < at(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    let simple = rho + 2.0 * (rho * (1.0 / delta).ln()).sqrt();
    at((lo + hi) / 2.0).min(simple).max(0.0)
}

/// The largest rho that stays within `epsilon` at `delta`.
pub fn rho_for(target: f64, delta: f64) -> f64 {
    let mut hi = 1.0f64;
    while epsilon(hi, delta) < target {
        hi *= 2.0;
    }
    let mut lo = 0.0f64;
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if epsilon(mid, delta) <= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

fn below(rng: &mut impl RngCore, n: u64) -> u64 {
    if n <= 1 {
        return 0;
    }
    let mask = u64::MAX >> (n - 1).leading_zeros();
    loop {
        let v = rng.next_u64() & mask;
        if v < n {
            return v;
        }
    }
}

fn below_wide(rng: &mut impl RngCore, n: u128) -> u128 {
    if let Ok(small) = u64::try_from(n) {
        return below(rng, small).into();
    }
    let mask = u128::MAX >> (n - 1).leading_zeros();
    loop {
        let v = (u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())) & mask;
        if v < n {
            return v;
        }
    }
}

/// True with probability exp(-num/den), for num <= den.
fn exp_minus_fraction(rng: &mut impl RngCore, num: u128, den: u128) -> bool {
    let mut k = 1u64;
    // Each step succeeds with probability (num/den)/k, drawn as two coins.
    while below(rng, k) == 0 && below_wide(rng, den) < num {
        k += 1;
    }
    k % 2 == 1
}

/// True with probability exp(-num/den).
fn exp_minus(rng: &mut impl RngCore, num: u128, den: u128) -> bool {
    for _ in 0..num / den {
        if !exp_minus_fraction(rng, 1, 1) {
            return false;
        }
    }
    exp_minus_fraction(rng, num % den, den)
}

/// Exact samples from the discrete Gaussian, by the algorithm of Canonne,
/// Kamath and Steinke (2020). It uses integers only: floating point samplers
/// leak through their rounding.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    variance: u64,
    scale: u64,
}

impl Sampler {
    pub fn new(variance: u64) -> Sampler {
        let variance = variance.clamp(1, MAX_NOISE);
        let mut scale = ceil_sqrt(variance);
        if scale * scale == variance {
            scale += 1;
        }
        Sampler { variance, scale }
    }

    /// A discrete Laplace sample with scale `self.scale`, as sign and size.
    fn laplace(&self, rng: &mut impl RngCore) -> (bool, u64) {
        let t = self.scale;
        loop {
            let u = below(rng, t);
            if !exp_minus_fraction(rng, u.into(), t.into()) {
                continue;
            }
            let mut v = 0u64;
            while exp_minus_fraction(rng, 1, 1) {
                v += 1;
            }
            let Some(size) = v.checked_mul(t).and_then(|x| x.checked_add(u)) else {
                continue;
            };
            let negative = rng.next_u32() & 1 == 1;
            if negative && size == 0 {
                continue;
            }
            return (negative, size);
        }
    }

    pub fn sample(&self, rng: &mut impl RngCore) -> i64 {
        let (variance, t) = (u128::from(self.variance), u128::from(self.scale));
        let den = 2 * variance * t * t;
        loop {
            let (negative, size) = self.laplace(rng);
            let gap = (u128::from(size) * t).abs_diff(variance);
            let (Some(num), Ok(value)) = (gap.checked_mul(gap), i64::try_from(size)) else {
                continue;
            };
            if exp_minus(rng, num, den) {
                return if negative { -value } else { value };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn rng(seed: u64) -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(seed)
    }

    #[test]
    fn square_roots_round_up() {
        for (n, want) in [(0, 0), (1, 1), (2, 2), (4, 2), (5, 3), (9, 3), (10, 4)] {
            assert_eq!(ceil_sqrt(n), want, "{n}");
        }
        assert_eq!(ceil_sqrt(u64::MAX), 1 << 32);
        assert_eq!(ceil_sqrt((1 << 40) + 1), (1 << 20) + 1);
    }

    #[test]
    fn uniform_draws_cover_the_range_evenly() {
        let mut r = rng(1);
        let mut seen = [0u32; 7];
        for _ in 0..70_000 {
            seen[below(&mut r, 7) as usize] += 1;
        }
        for count in seen {
            assert!((9_500..10_500).contains(&count), "{seen:?}");
        }
        assert_eq!(below(&mut r, 1), 0);
        let big = (1u128 << 100) + 12345;
        for _ in 0..1000 {
            assert!(below_wide(&mut r, big) < big);
        }
    }

    #[test]
    fn the_exponential_coin_has_the_right_bias() {
        let mut r = rng(2);
        let trials = 200_000;
        for (num, den) in [(0u128, 1u128), (1, 2), (1, 1), (5, 2), (7, 1)] {
            let hits = (0..trials).filter(|_| exp_minus(&mut r, num, den)).count();
            let want = (-(num as f64) / den as f64).exp();
            let got = hits as f64 / trials as f64;
            assert!((got - want).abs() < 0.005, "{num}/{den}: {got} vs {want}");
        }
    }

    #[test]
    fn small_variances_match_the_exact_distribution() {
        for variance in [1u64, 2, 9] {
            let sampler = Sampler::new(variance);
            let mut r = rng(variance);
            let trials = 400_000usize;
            let mut counts = std::collections::BTreeMap::<i64, usize>::new();
            for _ in 0..trials {
                *counts.entry(sampler.sample(&mut r)).or_default() += 1;
            }
            let weight = |x: i64| (-((x * x) as f64) / (2.0 * variance as f64)).exp();
            let total: f64 = (-200..=200).map(weight).sum();
            let mut chi = 0.0;
            let mut cells = 0;
            for x in -8..=8i64 {
                let expected = trials as f64 * weight(x) / total;
                if expected < 20.0 {
                    continue;
                }
                let got = *counts.get(&x).unwrap_or(&0) as f64;
                chi += (got - expected).powi(2) / expected;
                cells += 1;
            }
            // Far above the 99.9th percentile for this many cells.
            assert!(
                chi < 3.0 * cells as f64 + 20.0,
                "variance {variance}: chi {chi}"
            );
        }
    }

    #[test]
    fn large_variances_have_the_right_mean_and_spread() {
        for variance in [1u64 << 20, (1 << 40) + 12345, MAX_NOISE] {
            let sampler = Sampler::new(variance);
            let mut r = rng(variance);
            let trials = 100_000;
            let samples: Vec<f64> = (0..trials).map(|_| sampler.sample(&mut r) as f64).collect();
            let sigma = (variance as f64).sqrt();
            let mean = samples.iter().sum::<f64>() / trials as f64;
            let spread = samples.iter().map(|s| s * s).sum::<f64>() / trials as f64;
            assert!(
                mean.abs() < 0.02 * sigma,
                "variance {variance}: mean {mean}"
            );
            assert!(
                (spread / variance as f64 - 1.0).abs() < 0.02,
                "variance {variance}: spread {spread}"
            );
            let beyond_three = samples.iter().filter(|s| s.abs() > 3.0 * sigma).count();
            assert!((120..420).contains(&beyond_three), "{beyond_three}");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_noise() {
        let sampler = Sampler::new(1 << 30);
        let draw = |seed| -> Vec<i64> {
            let mut r = rng(seed);
            (0..50).map(|_| sampler.sample(&mut r)).collect()
        };
        assert_eq!(draw(5), draw(5));
        assert_ne!(draw(5), draw(6));
    }

    fn terms() -> Privacy {
        Privacy {
            clip: 1 << 18,
            noise: 1 << 34,
            colluding: 0,
            budget: 1 << RHO_BITS,
            epsilon: 8.0,
            delta: 1e-5,
        }
    }

    #[test]
    fn a_round_costs_the_gaussian_rho_plus_slack() {
        let p = terms();
        // clip 2^18, no rounding room: s = 2^18 + 1 + 1.
        let s = (1u128 << 18) + 2;
        let want = ((s * s) << 40) / (2 * (1u128 << 34) * 4) + 2;
        assert_eq!(p.round_cost(1, 4), Some(want as u64));
        // About 1/(2 z^2), with noise multiplier z = sqrt(4 * 2^34) / 2^18 = 1.
        let rho = rho_units(p.round_cost(1, 4).unwrap());
        assert!((rho - 0.5).abs() < 1e-4, "{rho}");
        assert!((p.noise_multiplier(4) - 1.0).abs() < 1e-12);
        assert!((p.noise_multiplier(16) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn fewer_honest_sites_cost_more_and_none_is_refused() {
        let mut p = terms();
        let all = p.round_cost(1000, 8).unwrap();
        p.colluding = 4;
        let half = p.round_cost(1000, 8).unwrap();
        assert!((half as f64 / all as f64 - 2.0).abs() < 1e-6);
        assert_eq!(p.round_cost(1000, 4), None);
        assert_eq!(p.round_cost(1000, 3), None);
    }

    #[test]
    fn more_coordinates_leave_more_room_for_rounding() {
        let p = terms();
        assert_eq!(p.sensitivity(0), (1 << 18) + 1);
        assert_eq!(p.sensitivity(1 << 20), (1 << 18) + (1 << 10) + 1);
        assert!(p.round_cost(1 << 20, 4) > p.round_cost(1, 4));
    }

    #[test]
    fn validation_names_what_is_wrong() {
        let ok = terms();
        ok.validate(1000, 8, 4).unwrap();
        let with = |edit: fn(&mut Privacy)| {
            let mut p = terms();
            edit(&mut p);
            p.validate(1000, 8, 4)
        };
        assert_eq!(with(|p| p.clip = 0), Err(PrivacyError::Clip));
        assert_eq!(with(|p| p.clip = MAX_CLIP + 1), Err(PrivacyError::Clip));
        assert_eq!(with(|p| p.noise = MIN_NOISE - 1), Err(PrivacyError::Noise));
        assert_eq!(with(|p| p.noise = MAX_NOISE + 1), Err(PrivacyError::Noise));
        assert_eq!(with(|p| p.colluding = 4), Err(PrivacyError::Colluding));
        assert_eq!(with(|p| p.budget = 10), Err(PrivacyError::Budget(1)));
        assert_eq!(with(|p| p.delta = 1.0), Err(PrivacyError::Target));
        assert_eq!(with(|p| p.epsilon = f64::NAN), Err(PrivacyError::Target));
        let mut edge = terms();
        (edge.clip, edge.noise, edge.budget) = (1, MIN_NOISE, u64::MAX);
        edge.validate(1000, 8, 4).unwrap();
        let mut wide = terms();
        (wide.clip, wide.noise, wide.budget) = (MAX_CLIP, MAX_NOISE, u64::MAX);
        wide.validate(1000, MAX_SITES, 4).unwrap();
        wide.validate(MAX_COORDS, 8, 4).unwrap();
        assert_eq!(wide.validate(MAX_COORDS + 1, 8, 4), Err(PrivacyError::Size));
        assert_eq!(ok.validate(1000, MAX_SITES + 1, 4), Err(PrivacyError::Size));
    }

    #[test]
    fn the_grid_keeps_rounding_small_whatever_the_clip() {
        assert_eq!(grid_for(1.0, 1000), Some((18, 1 << 18)));
        assert_eq!(grid_for(0.5, 1000), Some((19, 1 << 18)));
        // 2^28 coordinates need a clip of at least 64 * 2^14 units.
        assert_eq!(grid_for(1.0, 1 << 28), Some((20, 1 << 20)));
        let (bits, units) = grid_for(3.7, 50_000).unwrap();
        assert_eq!(units, (3.7 * (1u64 << bits) as f64).round() as u64);
        assert!((1 << 18..1 << 19).contains(&units));
        let (bits, units) = grid_for(1e-9, 1000).unwrap();
        assert_eq!(bits, 48);
        assert_eq!(units, 281_475);
        assert_eq!(grid_for(1e6, 1000), Some((1, 2_000_000)));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-20, 1e12] {
            assert_eq!(grid_for(bad, 1000), None, "{bad}");
        }
    }

    #[test]
    fn a_request_in_model_units_becomes_plan_terms() {
        let request = Request {
            epsilon: 3.0,
            delta: 1e-5,
            clip: 0.5,
            colluding: 0,
        };
        let (encoding, privacy) = request.terms(1000, 10, 3).unwrap();
        assert_eq!((encoding.frac_bits, encoding.clip), (19, 0.5));
        assert_eq!(privacy.clip, 1 << 18);
        privacy.validate(1000, 4, 3).unwrap();
        assert!(privacy.round_cost(1000, 3).unwrap() * 10 <= privacy.budget);
        let bad = Request {
            clip: -1.0,
            ..request
        };
        assert_eq!(bad.terms(1000, 10, 3), Err(PrivacyError::Clip));
    }

    #[test]
    fn epsilon_matches_known_values_and_inverts() {
        // rho = 0.5 is the Gaussian mechanism with noise multiplier 1. Its
        // exact epsilon at delta 1e-5 is about 4.37, and zCDP gives a little more.
        let e = epsilon(0.5, 1e-5);
        assert!((4.3..5.4).contains(&e), "{e}");
        assert!(e < 0.5 + 2.0 * (0.5f64 * (1e5f64).ln()).sqrt());
        assert_eq!(epsilon(0.0, 1e-5), 0.0);
        assert!(epsilon(0.1, 1e-5) < epsilon(0.2, 1e-5));
        assert!(epsilon(0.1, 1e-6) > epsilon(0.1, 1e-5));
        // Worked from the same formula in the sources: Canonne, Kamath and
        // Steinke give (1, 1e-7) for rho = 0.02, and 5.22 at rho = 0.5, 1e-6.
        assert!(
            (epsilon(0.5, 1e-6) - 5.22).abs() < 0.01,
            "{}",
            epsilon(0.5, 1e-6)
        );
        let example = epsilon(0.02, 1e-7);
        assert!((0.97..=1.0).contains(&example), "{example}");
        for target in [0.5, 1.0, 3.0, 8.0] {
            let rho = rho_for(target, 1e-5);
            assert!((epsilon(rho, 1e-5) - target).abs() < 1e-6, "{target}");
        }
    }

    #[test]
    fn a_target_becomes_noise_that_fits_its_budget() {
        let target = Target {
            epsilon: 3.0,
            delta: 1e-5,
            clip: 1 << 18,
            colluding: 1,
        };
        let p = Privacy::for_target(&target, 100_000, 20, 4).unwrap();
        p.validate(100_000, 8, 4).unwrap();
        let cost = p.round_cost(100_000, 4).unwrap();
        assert!(cost * 20 <= p.budget);
        // One step less noise would overspend.
        let mut less = p;
        less.noise -= 1;
        assert!(less.round_cost(100_000, 4).unwrap() * 20 > p.budget - 20 * SLACK);
        assert!((epsilon(rho_units(p.budget), 1e-5) - 3.0).abs() < 1e-3);
        // A larger cohort than the minimum only spends less.
        assert!(p.round_cost(100_000, 8).unwrap() < cost);

        assert_eq!(
            Privacy::for_target(&target, 100_000, u32::MAX, 4),
            Err(PrivacyError::Budget(u32::MAX))
        );
        assert_eq!(
            Privacy::for_target(&target, 100_000, 20, 1),
            Err(PrivacyError::Colluding)
        );
    }
}
