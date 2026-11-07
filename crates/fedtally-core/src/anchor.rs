//! What a run posts to a ledger: a few digests per round, worked out from the
//! log. The ledger program in `ledger/` holds the other half of these rules.

use pq_ucan::did::Did;

use crate::keys::key_of;
use crate::merkle::{self, Hash};
use crate::plan::Plan;
use crate::replay::Replay;
use crate::sampling::{self, Ticket};
use crate::vector::{Digest, LEAF_COORDS};

/// The terms of a plan that a ledger enforces alongside the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Terms {
    pub plan: Digest,
    pub sites_root: Hash,
    pub sites: u16,
    pub helpers: u8,
    /// Coordinates in a vector, pad included, and so `leaves` in its tree.
    pub coords: u64,
    pub leaves: u64,
    pub rounds: u32,
    pub min_cohort: u16,
    /// Zero when every site is invited to every round.
    pub sample: u16,
    /// Zero noise means the plan has no privacy terms.
    pub sensitivity: u64,
    pub noise: u64,
    pub colluding: u16,
    pub budget: u64,
    pub model: Digest,
    /// Rounds that may be given up for a silent site.
    pub retries: u32,
    /// The fewest helpers the run may carry on with, or 0 if it needs them all.
    pub min_helpers: u8,
    /// The fewest sites a scored group may hold, or 0 if shares are equal.
    pub score_group: u16,
}

/// What the aggregator posts once a round's cohort is fixed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Committed {
    pub round: u32,
    pub beacon: [u8; 32],
    /// Site indexes, ascending.
    pub cohort: Vec<usize>,
    pub inputs_root: Hash,
    pub close_head: Digest,
    /// The largest ticket among the invited sites, when the plan samples.
    pub threshold: Ticket,
}

/// What the aggregator posts once a round is sealed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Posted {
    pub round: u32,
    pub mask_roots: Vec<Digest>,
    pub aggregate_root: Digest,
    pub next_model: Digest,
    pub seal_head: Digest,
    /// What each cohort member is owed in a scored round, in cohort order, and
    /// the root over it. Empty and zero otherwise.
    pub shares: Vec<SiteShare>,
    pub shares_root: Hash,
}

/// A cohort member's part of a scored round's reward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiteShare {
    pub site: u16,
    /// Its group's share of the reward, in millionths.
    pub share: u32,
    /// How many sites its group holds, who split that share alike.
    pub members: u16,
}

impl SiteShare {
    fn leaf(&self) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[..2].copy_from_slice(&self.site.to_le_bytes());
        bytes[2..6].copy_from_slice(&self.share.to_le_bytes());
        bytes[6..].copy_from_slice(&self.members.to_le_bytes());
        bytes
    }
}

fn shares_tree(shares: &[SiteShare]) -> Vec<Hash> {
    shares.iter().map(|s| merkle::leaf(&s.leaf())).collect()
}

impl Posted {
    /// A site's part of the round and the path that shows it in `shares_root`.
    pub fn share_of(&self, site: u16) -> Option<(SiteShare, Vec<Hash>)> {
        let position = self.shares.iter().position(|s| s.site == site)?;
        let path = merkle::path(shares_tree(&self.shares), position);
        Some((self.shares[position], path))
    }
}

/// The Solana address of a party: its Ed25519 key. A party with another kind
/// of key has no address and cannot take part on a ledger.
pub fn wallet(did: &str) -> Result<[u8; 32], String> {
    let did = Did::parse(did).map_err(|e| e.to_string())?;
    key_of(&did)?.as_bytes().try_into().map_err(|_| {
        format!(
            "{} has no Solana address: it is not an Ed25519 key",
            did.as_str()
        )
    })
}

fn tree_over(items: impl Iterator<Item = [u8; 32]>) -> Vec<Hash> {
    items.map(|item| merkle::leaf(&item)).collect()
}

/// The root over the cohort's vector roots, in cohort order.
pub fn inputs_root(vector_roots: &[Digest]) -> Hash {
    merkle::root(tree_over(vector_roots.iter().map(|r| r.0)))
}

/// The path that shows the vector root at `position` is in `inputs_root`.
pub fn inputs_path(vector_roots: &[Digest], position: usize) -> Vec<Hash> {
    merkle::path(tree_over(vector_roots.iter().map(|r| r.0)), position)
}

impl Plan {
    /// The terms a ledger needs, if every site can hold an account on one.
    pub fn ledger_terms(&self, plan_id: Digest) -> Result<Terms, String> {
        let wallets = self
            .sites
            .iter()
            .map(|site| wallet(site))
            .collect::<Result<Vec<_>, _>>()?;
        let shape = self.shape();
        if !shape.aligned() {
            return Err(
                "chunks must be 128 * 2^k coordinates for single leaves to be proven".into(),
            );
        }
        let narrow = |what: &str, value: usize| {
            u16::try_from(value).map_err(|_| format!("too many {what} for a ledger"))
        };
        let privacy = self.privacy;
        Ok(Terms {
            plan: plan_id,
            sites_root: merkle::root(tree_over(wallets.into_iter())),
            sites: narrow("sites", self.sites.len())?,
            helpers: u8::try_from(self.helpers.len()).map_err(|_| "too many helpers")?,
            coords: shape.coords,
            leaves: shape.coords.div_ceil(LEAF_COORDS),
            rounds: self.rounds,
            min_cohort: narrow("sites", self.min_cohort as usize)?,
            sample: narrow("sites", self.sample.unwrap_or(0) as usize)?,
            sensitivity: privacy.map_or(0, |p| p.sensitivity(self.layout.coords())),
            noise: privacy.map_or(0, |p| p.noise),
            colluding: narrow("sites", privacy.map_or(0, |p| p.colluding) as usize)?,
            budget: privacy.map_or(0, |p| p.budget),
            model: self.init_model,
            retries: self.retries,
            min_helpers: u8::try_from(self.min_helpers).map_err(|_| "too many helpers")?,
            score_group: narrow("sites", self.scoring.map_or(0, |s| s.group) as usize)?,
        })
    }
}

impl Replay {
    /// The commitment for a round whose cohort is fixed.
    pub fn committed(&self, round: u32) -> Option<Committed> {
        let plan = self.plan()?;
        let state = self.rounds.get(round.checked_sub(1)? as usize)?;
        if state.close_head == Digest::default() {
            return None;
        }
        let roots: Vec<Digest> = state
            .included
            .iter()
            .map(|site| state.contributions[site].masked)
            .collect();
        let beacon = sampling::beacon(&self.plan_id, state.step, &state.base);
        let threshold = plan.sample.map_or([0; 32], |size| {
            sampling::invited(&beacon, plan.sites.len(), size as usize).1
        });
        Some(Committed {
            round,
            beacon,
            cohort: state.included.clone(),
            inputs_root: inputs_root(&roots),
            close_head: state.close_head,
            threshold,
        })
    }

    /// What follows once the round is sealed.
    pub fn posted(&self, round: u32) -> Option<Posted> {
        let plan = self.plan()?;
        let state = self.rounds.get(round.checked_sub(1)? as usize)?;
        let (outcome, seal) = (state.outcome.as_ref()?, state.seal.as_ref()?);
        let mut shares = Vec::new();
        for (group, members) in plan
            .groups(&state.close_head, state.included.len())
            .iter()
            .enumerate()
        {
            for position in members {
                shares.push((
                    *position,
                    SiteShare {
                        site: u16::try_from(state.included[*position]).ok()?,
                        share: *seal.shares.get(group)?,
                        members: u16::try_from(members.len()).ok()?,
                    },
                ));
            }
        }
        shares.sort_unstable_by_key(|(position, _)| *position);
        let shares: Vec<SiteShare> = shares.into_iter().map(|(_, share)| share).collect();
        let shares_root = match shares.is_empty() {
            true => [0; 32],
            false => merkle::root(shares_tree(&shares)),
        };
        Some(Posted {
            round,
            mask_roots: state.masks.values().copied().collect(),
            aggregate_root: outcome.aggregate,
            next_model: outcome.model,
            seal_head: seal.head,
            shares,
            shares_root,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{Identity, SigAlg};

    #[test]
    fn a_party_with_an_ed25519_key_has_the_same_address_under_either_name() {
        let key = Identity::generate(SigAlg::Ed25519).unwrap();
        let by_key = wallet(&key.did_string()).unwrap();
        let bio = key.on_registry("devnet").unwrap();
        assert_eq!(wallet(&bio.did_string()).unwrap(), by_key);
        let subject = bio.did_string().rsplit(':').next().unwrap().to_owned();
        assert_eq!(bs58::encode(by_key).into_string(), subject);

        let pq = Identity::generate(SigAlg::MlDsa65).unwrap();
        assert!(wallet(&pq.did_string()).unwrap_err().contains("Ed25519"));
    }

    #[test]
    fn every_cohort_member_can_be_shown_in_the_inputs_root() {
        let roots: Vec<Digest> = (0..7u8).map(|i| Digest([i; 32])).collect();
        let top = inputs_root(&roots);
        for (position, root) in roots.iter().enumerate() {
            let path = inputs_path(&roots, position);
            let leaf = merkle::leaf(&root.0);
            assert_eq!(merkle::climb(leaf, position as u64, 7, &path), Some(top));
        }
        assert_ne!(inputs_root(&roots[..6]), top);
    }

    #[test]
    fn every_share_can_be_shown_in_the_shares_root() {
        let shares: Vec<SiteShare> = [0u16, 3, 4, 7, 9]
            .iter()
            .map(|site| SiteShare {
                site: *site,
                share: 200_000,
                members: 2,
            })
            .collect();
        let posted = Posted {
            round: 1,
            mask_roots: Vec::new(),
            aggregate_root: Digest::default(),
            next_model: Digest::default(),
            seal_head: Digest::default(),
            shares_root: merkle::root(shares_tree(&shares)),
            shares: shares.clone(),
        };
        for (position, share) in shares.iter().enumerate() {
            let (found, path) = posted.share_of(share.site).unwrap();
            assert_eq!(found, *share);
            let leaf = merkle::leaf(&share.leaf());
            assert_eq!(
                merkle::climb(leaf, position as u64, 5, &path),
                Some(posted.shares_root)
            );
        }
        assert!(posted.share_of(1).is_none());
    }
}
