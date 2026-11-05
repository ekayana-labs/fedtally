//! The plan: everything a run commits to before it starts.

use std::collections::BTreeSet;

use pq_ucan::did::Did;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::keys::{SealPublic, verify_seal_proof};
use crate::model::{Encoding, Layout};
use crate::vector::{Digest, Shape};

const SUM_BITS_LIMIT: u32 = 62;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregatorSpec {
    pub did: String,
    /// BLAKE3 digest of the aggregator's TLS certificate.
    pub tls_pin: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HelperSpec {
    pub did: String,
    pub seal: SealPublic,
    /// The helper's own signature over `seal`.
    pub seal_proof: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timeouts {
    pub join_secs: u64,
    pub collect_secs: u64,
    pub attest_secs: u64,
    pub ack_secs: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub version: u32,
    pub nonce: String,
    pub purpose: String,
    pub sponsor: String,
    pub aggregator: AggregatorSpec,
    pub helpers: Vec<HelperSpec>,
    pub sites: Vec<String>,
    pub layout: Layout,
    pub init_model: Digest,
    pub rounds: u32,
    pub min_cohort: u32,
    pub chunk_coords: u64,
    pub encoding: Encoding,
    pub max_weight: u64,
    /// How many other sites' masked vectors each site re-checks per round.
    pub audits: u32,
    pub timeouts: Timeouts,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlanError {
    #[error("unsupported plan version {0}")]
    Version(u32),
    #[error("plan needs at least one round")]
    NoRounds,
    #[error("plan needs at least one site")]
    NoSites,
    #[error("a DID appears in more than one role or twice: {0}")]
    DuplicateParty(String),
    #[error("bad DID {0}")]
    BadDid(String),
    #[error("helper {0} did not sign the sealing key listed for it")]
    BadSealProof(String),
    #[error("min_cohort must be between {low} and the site count, not {got}")]
    MinCohort { low: u32, got: u32 },
    #[error("chunk_coords must be positive")]
    ChunkSize,
    #[error("layout has no coordinates or repeats a name")]
    Layout,
    #[error("clip and frac_bits must be positive and finite")]
    Encoding,
    #[error("max_weight must be positive")]
    MaxWeight,
    #[error("sums need {0} bits but the field holds {SUM_BITS_LIMIT}")]
    FieldOverflow(u32),
}

impl Plan {
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.version != 1 {
            return Err(PlanError::Version(self.version));
        }
        if self.rounds == 0 {
            return Err(PlanError::NoRounds);
        }
        if self.sites.is_empty() {
            return Err(PlanError::NoSites);
        }
        let mut seen = BTreeSet::new();
        let everyone = self
            .sites
            .iter()
            .chain(self.helpers.iter().map(|h| &h.did))
            .chain([&self.aggregator.did]);
        for did in everyone {
            if !seen.insert(did) {
                return Err(PlanError::DuplicateParty(did.clone()));
            }
        }
        for did in seen.iter().copied().chain([&self.sponsor]) {
            Did::parse(did)
                .ok()
                .filter(|d| d.key().is_ok())
                .ok_or_else(|| PlanError::BadDid(did.clone()))?;
        }
        for helper in &self.helpers {
            let did = Did::parse(&helper.did).map_err(|_| PlanError::BadDid(helper.did.clone()))?;
            verify_seal_proof(&did, &helper.seal, &helper.seal_proof)
                .map_err(|_| PlanError::BadSealProof(helper.did.clone()))?;
        }
        // One site's masked sum would just be that site's update.
        let low = if self.helpers.is_empty() { 1 } else { 2 };
        if self.min_cohort < low || self.min_cohort as usize > self.sites.len() {
            return Err(PlanError::MinCohort {
                low,
                got: self.min_cohort,
            });
        }
        if self.chunk_coords == 0 {
            return Err(PlanError::ChunkSize);
        }
        let names: BTreeSet<_> = self.layout.tensors.iter().map(|t| &t.name).collect();
        if self.layout.coords() == 0 || names.len() != self.layout.tensors.len() {
            return Err(PlanError::Layout);
        }
        if self.encoding.frac_bits == 0
            || self.encoding.frac_bits > 48
            || !self.encoding.clip.is_finite()
            || self.encoding.clip <= 0.0
        {
            return Err(PlanError::Encoding);
        }
        if self.max_weight == 0 {
            return Err(PlanError::MaxWeight);
        }
        let total = self.max_weight.saturating_mul(self.sites.len() as u64);
        let bits = self.encoding.sum_bits(total);
        if bits > SUM_BITS_LIMIT {
            return Err(PlanError::FieldOverflow(bits));
        }
        Ok(())
    }

    pub fn shape(&self) -> Shape {
        self.layout.shape(self.chunk_coords)
    }

    pub fn site_index(&self, did: &str) -> Option<usize> {
        self.sites.iter().position(|s| s == did)
    }

    pub fn helper_index(&self, did: &str) -> Option<usize> {
        self.helpers.iter().position(|h| h.did == did)
    }

    /// True when the aggregator sees masked vectors only.
    pub fn is_masked(&self) -> bool {
        !self.helpers.is_empty()
    }

    /// The cohort positions the site at `position` audits this round. The
    /// shifts come from the challenge, so nobody knows them when committing,
    /// and a site is never assigned to itself.
    pub fn audit_targets(&self, challenge: &Digest, position: usize, cohort: usize) -> Vec<usize> {
        if !self.is_masked() || cohort < 2 {
            return Vec::new();
        }
        let span = cohort as u64 - 1;
        let bytes: [u8; 8] = challenge.0[..8].try_into().unwrap_or_default();
        let start = u64::from_le_bytes(bytes);
        (0..u64::from(self.audits).min(span))
            .map(|k| (position + 1 + (start.wrapping_add(k) % span) as usize) % cohort)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{Identity, SigAlg};
    use crate::model::{Dtype, TensorSpec};

    fn did() -> String {
        Identity::generate(SigAlg::Ed25519).unwrap().did_string()
    }

    fn plan(audits: u32) -> Plan {
        let helper = Identity::generate(SigAlg::Ed25519).unwrap();
        Plan {
            version: 1,
            nonce: "n".into(),
            purpose: "p".into(),
            sponsor: did(),
            aggregator: AggregatorSpec {
                did: did(),
                tls_pin: Digest::of(b"c"),
            },
            helpers: vec![HelperSpec {
                did: helper.did_string(),
                seal: helper.seal_public().clone(),
                seal_proof: helper.seal_proof().unwrap(),
            }],
            sites: (0..4).map(|_| did()).collect(),
            layout: Layout {
                tensors: vec![TensorSpec {
                    name: "w".into(),
                    dtype: Dtype::F32,
                    shape: vec![4],
                }],
            },
            init_model: Digest::of(b"m"),
            rounds: 1,
            min_cohort: 2,
            chunk_coords: 2,
            encoding: Encoding {
                frac_bits: 24,
                clip: 128.0,
            },
            max_weight: 10,
            audits,
            timeouts: Timeouts {
                join_secs: 1,
                collect_secs: 1,
                attest_secs: 1,
                ack_secs: 1,
            },
        }
    }

    #[test]
    fn a_helper_paired_with_someone_elses_key_is_refused() {
        let mut swapped = plan(1);
        swapped.validate().unwrap();
        swapped.helpers[0].seal = Identity::generate(SigAlg::Ed25519)
            .unwrap()
            .seal_public()
            .clone();
        assert!(matches!(
            swapped.validate(),
            Err(PlanError::BadSealProof(_))
        ));
    }

    #[test]
    fn only_the_assigned_auditor_may_read_a_vector() {
        let (one, challenge) = (plan(1), Digest::of(b"c"));
        for position in 0..4 {
            let target = one.audit_targets(&challenge, position, 4)[0];
            for candidate in 0..4 {
                let allowed = one
                    .audit_targets(&challenge, position, 4)
                    .contains(&candidate);
                assert_eq!(allowed, candidate == target);
            }
        }
    }

    #[test]
    fn every_site_is_audited_and_none_audits_itself() {
        for seed in 0..50u32 {
            let challenge = Digest::of(&seed.to_le_bytes());
            let one = plan(1);
            let mut audited = vec![0; 5];
            for position in 0..5 {
                let targets = one.audit_targets(&challenge, position, 5);
                assert_eq!(targets.len(), 1);
                assert_ne!(targets[0], position);
                audited[targets[0]] += 1;
            }
            assert_eq!(audited, vec![1; 5]);
        }
    }

    #[test]
    fn audits_stay_distinct_and_are_capped_by_the_cohort() {
        for seed in 0..200u32 {
            let challenge = Digest::of(&seed.to_le_bytes());
            let mut targets = plan(9).audit_targets(&challenge, 2, 4);
            targets.sort_unstable();
            assert_eq!(targets, vec![0, 1, 3]);
            let mut two = plan(2).audit_targets(&challenge, 0, 5);
            two.sort_unstable();
            two.dedup();
            assert_eq!(two.len(), 2);
            assert!(!two.contains(&0));
        }
        assert!(plan(0).audit_targets(&Digest::of(b"c"), 0, 4).is_empty());
    }

    #[test]
    fn the_smallest_cohort_that_can_audit_is_two() {
        let challenge = Digest::of(b"c");
        assert_eq!(plan(1).audit_targets(&challenge, 0, 2), vec![1]);
        assert_eq!(plan(1).audit_targets(&challenge, 1, 2), vec![0]);
        assert!(plan(1).audit_targets(&challenge, 0, 1).is_empty());
        assert!(plan(1).audit_targets(&challenge, 0, 0).is_empty());
    }

    #[test]
    fn the_field_limit_is_inclusive() {
        // 24 fractional bits and a clip of 128 take 31 bits. Four sites at this
        // weight total just under 2^31, which takes the other 31.
        let mut at_limit = plan(1);
        at_limit.max_weight = (1 << 31) / 4 - 1;
        at_limit.validate().unwrap();
        let mut over = plan(1);
        over.max_weight = (1 << 31) / 4;
        assert_eq!(over.validate(), Err(PlanError::FieldOverflow(63)));
    }

    #[test]
    fn plain_runs_have_no_audits() {
        let mut plain = plan(1);
        plain.helpers.clear();
        assert!(plain.audit_targets(&Digest::of(b"c"), 0, 4).is_empty());
    }
}
