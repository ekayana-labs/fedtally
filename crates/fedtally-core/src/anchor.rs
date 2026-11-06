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
        if self.retries > 0 {
            return Err("a plan that retries cannot be anchored yet".into());
        }
        if self.scoring.is_some() {
            // The program charges a round at its cohort and pays equal shares.
            return Err("a scoring plan cannot be anchored yet".into());
        }
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
        let state = self.rounds.get(round.checked_sub(1)? as usize)?;
        let (outcome, seal) = (state.outcome.as_ref()?, state.seal.as_ref()?);
        Some(Posted {
            round,
            mask_roots: state.masks.values().copied().collect(),
            aggregate_root: outcome.aggregate,
            next_model: outcome.model,
            seal_head: seal.head,
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
}
