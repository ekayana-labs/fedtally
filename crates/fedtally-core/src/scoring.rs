//! Rewards that follow contribution, for groups of sites.
//!
//! A score for one site is a statistic about that site alone, which is what
//! site level privacy bounds. So a scoring run splits each round's cohort
//! into groups, releases each group's sum, and scores the groups. A group's
//! sum carries the noise of its members only, and the privacy accounting
//! charges the round at the size of its smallest group.
//!
//! A group's score is how far its sum points along the sum of all the others.
//! Its own length doesn't count, so neither its noise nor a long random update
//! earns anything, and anyone holding the released sums can recompute it.
//!
//! Two groups always tie, since each scores the same inner product against the
//! other. With no reference data, two groups can't show which of them is right,
//! so it takes three to tell them apart.

use std::fs::File;
use std::io;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::field::Fp;
use crate::vector::{self, Digest, Shape};

/// Shares of a round's reward add up to this.
pub const WHOLE: u32 = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scoring {
    /// The fewest sites a group may hold.
    pub group: u32,
}

impl Scoring {
    pub fn count(&self, cohort: usize) -> usize {
        (cohort / self.group.max(1) as usize).max(1)
    }

    /// The size of the smallest group in a cohort, never below `group` once
    /// the cohort is at least that large.
    pub fn smallest(&self, cohort: usize) -> usize {
        cohort / self.count(cohort)
    }

    /// Cohort positions by group. The split comes from the head of the log
    /// after the round closed, so nobody knows it when contributing.
    pub fn groups(&self, close_head: &Digest, cohort: usize) -> Vec<Vec<usize>> {
        let mut reader = blake3::Hasher::new_derive_key("fedtally v1 groups")
            .update(&close_head.0)
            .finalize_xof();
        let mut order: Vec<usize> = (0..cohort).collect();
        for i in (1..cohort).rev() {
            let mut bytes = [0u8; 8];
            reader.fill(&mut bytes);
            order.swap(i, (u64::from_le_bytes(bytes) % (i as u64 + 1)) as usize);
        }
        let count = self.count(cohort);
        let (size, larger) = (cohort / count, cohort % count);
        let mut rest = order.as_slice();
        (0..count)
            .map(|g| {
                let (group, tail) = rest.split_at(size + usize::from(g < larger));
                rest = tail;
                let mut group = group.to_vec();
                group.sort_unstable();
                group
            })
            .collect()
    }
}

/// The inner product of a group's sum with the sum of every other group, over
/// the first `plain` coordinates, reading each as the integer nearest zero.
pub fn alignment(group: &Path, total: &Path, shape: Shape, plain: u64) -> io::Result<i128> {
    let (group, total) = (File::open(group)?, File::open(total)?);
    (0..shape.chunks())
        .into_par_iter()
        .map(|chunk| {
            let (mut own, mut all) = (Vec::new(), Vec::new());
            vector::read_chunk(&group, shape, chunk, &mut own)?;
            vector::read_chunk(&total, shape, chunk, &mut all)?;
            let (start, end) = shape.range(chunk);
            let covered = end.min(plain).saturating_sub(start) as usize;
            Ok(own[..covered]
                .iter()
                .zip(&all[..covered])
                .map(|(g, a)| {
                    let (g, a) = (Fp::new(*g), Fp::new(*a));
                    i128::from(g.centered()) * i128::from((a - g).centered())
                })
                .sum::<i128>())
        })
        .try_reduce(|| 0, |a, b| Ok(a + b))
}

/// Each group's share of the round's reward, out of `WHOLE`. With nothing to
/// tell the groups apart, they share equally.
pub fn shares(alignments: &[i128]) -> Vec<u32> {
    let scores: Vec<u128> = alignments
        .iter()
        .map(|a| (*a).max(0).unsigned_abs())
        .collect();
    let total = scores.iter().fold(0u128, |sum, s| sum.saturating_add(*s));
    let mut shares: Vec<u32> = if total > 0 {
        // Scaled down first so the product below can't overflow.
        let shift = (u128::BITS - total.leading_zeros()).saturating_sub(96);
        let total = (total >> shift).max(1);
        scores
            .iter()
            .map(|s| (u128::from(WHOLE) * (s >> shift) / total) as u32)
            .collect()
    } else {
        vec![WHOLE / scores.len().max(1) as u32; scores.len()]
    };
    let given: u32 = shares.iter().sum();
    let best = (0..shares.len()).max_by_key(|g| (scores[*g], std::cmp::Reverse(*g)));
    if let Some(best) = best {
        shares[best] += WHOLE.saturating_sub(given);
    }
    shares
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_cover_the_cohort_once_and_are_never_too_small() {
        for group in 2..6u32 {
            let scoring = Scoring { group };
            for cohort in group as usize..40 {
                let groups = scoring.groups(&Digest::of(b"head"), cohort);
                assert_eq!(groups.len(), scoring.count(cohort));
                let mut all: Vec<usize> = groups.concat();
                all.sort_unstable();
                assert_eq!(all, (0..cohort).collect::<Vec<_>>());
                let smallest = groups.iter().map(Vec::len).min().unwrap();
                assert_eq!(smallest, scoring.smallest(cohort));
                assert!(smallest >= group as usize);
                assert!(groups.iter().all(|g| g.len() <= smallest + 1));
            }
        }
    }

    #[test]
    fn the_split_follows_the_head() {
        let scoring = Scoring { group: 3 };
        let a = scoring.groups(&Digest::of(b"one"), 12);
        assert_eq!(a, scoring.groups(&Digest::of(b"one"), 12));
        assert_ne!(a, scoring.groups(&Digest::of(b"two"), 12));
    }

    #[test]
    fn shares_follow_alignment_and_add_up() {
        let split = shares(&[120, 60, -10]);
        assert_eq!(split.iter().sum::<u32>(), WHOLE);
        assert_eq!(split[2], 0);
        assert!(split[0] > split[1] && split[1] > 0);
        assert_eq!(split[0], WHOLE - split[1]);

        assert_eq!(shares(&[-4, -3]), [500_000, 500_000]);
        let thirds = shares(&[0; 3]);
        assert_eq!(thirds.iter().sum::<u32>(), WHOLE);
        assert!(thirds.iter().all(|s| s.abs_diff(WHOLE / 3) <= 1));
        let vast = shares(&[i128::MAX / 4, i128::MAX / 4]);
        assert_eq!(vast.iter().sum::<u32>(), WHOLE);
        assert!(vast[0].abs_diff(vast[1]) <= 2);
    }

    #[test]
    fn a_groups_own_length_earns_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let shape = Shape {
            coords: 256,
            chunk_coords: 128,
        };
        let path = dir.path().join("long");
        vector::build(&path, shape, |chunk, buf, _| {
            let (start, _) = shape.range(chunk);
            for (k, slot) in buf.iter_mut().enumerate() {
                let sign = if (start as usize + k).is_multiple_of(2) {
                    1
                } else {
                    -1
                };
                *slot = Fp::from_i64(sign * 9_000).value();
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(alignment(&path, &path, shape, 256).unwrap(), 0);
    }

    #[test]
    fn alignment_reads_signed_coordinates_and_skips_the_pad() {
        let dir = tempfile::tempdir().unwrap();
        let shape = Shape {
            coords: 300,
            chunk_coords: 128,
        };
        let write = |name: &str, value: fn(i64) -> i64| {
            let path = dir.path().join(name);
            vector::build(&path, shape, |chunk, buf, _| {
                let (start, _) = shape.range(chunk);
                for (k, slot) in buf.iter_mut().enumerate() {
                    *slot = Fp::from_i64(value(start as i64 + k as i64)).value();
                }
                Ok(())
            })
            .unwrap();
            path
        };
        let group = write("group", |i| i - 100);
        let total = write("total", |i| i - 100 + if i % 2 == 0 { 3 } else { -3 });
        let want: i128 = (0..298)
            .map(|i| i128::from((i - 100) * if i % 2 == 0 { 3 } else { -3 }))
            .sum();
        assert_eq!(alignment(&group, &total, shape, 298).unwrap(), want);
    }
}
