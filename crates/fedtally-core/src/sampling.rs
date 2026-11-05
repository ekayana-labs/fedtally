//! Choosing who is asked into a round, in a way anyone can recompute.
//!
//! Each round has a beacon, derived from the plan, the round number and the
//! model the round starts from. Every site draws a ticket from it, and the
//! sites with the smallest tickets are invited. SHA-256 is used so a program
//! on a ledger can check the same draw.
//!
//! After the first round the model carries noise or updates from honest
//! sites, which an aggregator cannot predict before the round that produced
//! it has closed. The first round's beacon is fixed by the plan alone.

use sha2::{Digest as _, Sha256};

use crate::vector::Digest;

pub type Ticket = [u8; 32];

pub fn beacon(plan: &Digest, round: u32, model: &Digest) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"fedtally v1 beacon");
    hasher.update(plan.0);
    hasher.update(round.to_le_bytes());
    hasher.update(model.0);
    hasher.finalize().into()
}

pub fn ticket(beacon: &[u8; 32], site: u32) -> Ticket {
    let mut hasher = Sha256::new();
    hasher.update(b"fedtally v1 ticket");
    hasher.update(beacon);
    hasher.update(site.to_le_bytes());
    hasher.finalize().into()
}

/// The `size` sites with the smallest tickets, in ascending site order, and
/// the largest ticket among them.
pub fn invited(beacon: &[u8; 32], sites: usize, size: usize) -> (Vec<usize>, Ticket) {
    let mut drawn: Vec<(Ticket, usize)> =
        (0..sites).map(|s| (ticket(beacon, s as u32), s)).collect();
    drawn.sort_unstable();
    drawn.truncate(size);
    let threshold = drawn.last().map_or([0; 32], |(ticket, _)| *ticket);
    let mut chosen: Vec<usize> = drawn.into_iter().map(|(_, site)| site).collect();
    chosen.sort_unstable();
    (chosen, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Digest {
        let mut bytes = [0u8; 32];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        Digest(bytes)
    }

    #[test]
    fn the_draw_matches_the_vectors_shared_with_the_ledger() {
        let b = beacon(&plan(), 3, &Digest([7; 32]));
        assert_eq!(
            hex::encode(b),
            "ceca54fea8e95b4f6e7a32645347900387e511946c06cfa23f46dab22702a892"
        );
        let tickets: Vec<String> = (0..4).map(|s| hex::encode(ticket(&b, s))).collect();
        assert_eq!(
            tickets,
            [
                "86bfe57064f76aca7fe63c538a83180403864e7fc52896aba88461d4cd1972a8",
                "4338762cb6b74497a040b4e90df29821c52140b80b456f7021494209a7fd2a61",
                "65cb3cd5b6d47bf5e99d613407526f600801950a0e35c452bfe931fc471a3ba2",
                "00e856ccbe31ca24de4504aad3a7c4d5c4346d7540961f4d6e3774cf471b85fd",
            ]
        );
        // Smallest first: site 3, then 1, then 2, then 0.
        assert_eq!(invited(&b, 4, 2).0, [1, 3]);
        assert_eq!(invited(&b, 4, 3), (vec![1, 2, 3], ticket(&b, 2)));
        assert_eq!(invited(&b, 4, 4).0, [0, 1, 2, 3]);
        assert_eq!(invited(&b, 4, 9).0, [0, 1, 2, 3]);
        assert!(invited(&b, 4, 0).0.is_empty());
    }

    #[test]
    fn exactly_the_invited_sites_are_at_or_below_the_threshold() {
        for round in 1..40 {
            let b = beacon(&plan(), round, &Digest([round as u8; 32]));
            let (chosen, threshold) = invited(&b, 50, 20);
            assert_eq!(chosen.len(), 20);
            for site in 0..50 {
                assert_eq!(ticket(&b, site as u32) <= threshold, chosen.contains(&site));
            }
        }
    }

    #[test]
    fn every_input_changes_the_beacon() {
        let base = beacon(&plan(), 1, &Digest([1; 32]));
        assert_ne!(base, beacon(&plan(), 2, &Digest([1; 32])));
        assert_ne!(base, beacon(&plan(), 1, &Digest([2; 32])));
        assert_ne!(base, beacon(&Digest([9; 32]), 1, &Digest([1; 32])));
    }

    #[test]
    fn sites_are_invited_about_equally_often() {
        let mut times = [0u32; 10];
        for round in 0..2000 {
            let b = beacon(&plan(), round, &Digest([3; 32]));
            for site in invited(&b, 10, 4).0 {
                times[site] += 1;
            }
        }
        for count in times {
            assert!((700..900).contains(&count), "{times:?}");
        }
    }
}
