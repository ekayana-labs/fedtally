//! The plan: everything a run commits to before it starts.

use std::collections::BTreeSet;

use pq_ucan::did::Did;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::credential::Credential;
use crate::keys::{SealPublic, key_of, verify_seal_proof};
use crate::model::{Encoding, Layout};
use crate::norm::{NormError, Params};
use crate::policy::Policy;
use crate::privacy::{Privacy, PrivacyError};
use crate::scoring::Scoring;
use crate::vector::{Digest, Shape};

const SUM_BITS_LIMIT: u32 = 62;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregatorSpec {
    pub did: String,
    /// BLAKE3 digest of the aggregator's TLS certificate.
    pub tls_pin: Digest,
    /// Credentials about the aggregator, for a plan whose policy asks for them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credentials: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HelperSpec {
    pub did: String,
    pub seal: SealPublic,
    /// The helper's own signature over `seal`.
    pub seal_proof: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credentials: Vec<String>,
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
    /// Noise every site adds, and what the run may spend. Absent in a run
    /// that makes no privacy promise about the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy: Option<Privacy>,
    /// The credentials every party must hold, and what the data is for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission: Option<Policy>,
    /// How many sites each round invites, drawn afresh from a beacon. Absent
    /// when every site is asked into every round.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<u32>,
    /// Whether each site proves, before a round closes, that its update is
    /// no longer than the privacy terms allow.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bounded: bool,
    /// Split each round's cohort into groups, release each group's sum, and
    /// score the groups for rewards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoring: Option<Scoring>,
    /// How many times a round may be given up and done again without a site
    /// that went silent before its seal. Each attempt publishes another sum
    /// over nearly the same updates, and pays for it.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub retries: u32,
    /// The fewest helpers a run may carry on with, if it may lose any. A lost
    /// helper is dropped and the round redone without it, and the aggregator
    /// alone decides one is lost, so privacy needs fewer than this many helpers
    /// to collude. Absent, every helper is needed and one honest helper is enough.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub min_helpers: u32,
    /// Apply each round's average update through Adam on the server, rather than
    /// adding it to the model. Everyone works this out for themselves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_adam: Option<ServerAdam>,
}

/// FedAdam's settings (Reddi et al., Adaptive Federated Optimization).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerAdam {
    pub rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub tau: f64,
}

impl ServerAdam {
    fn is_valid(&self) -> bool {
        let beta = |b: f64| (0.0..1.0).contains(&b);
        self.rate.is_finite()
            && self.rate > 0.0
            && beta(self.beta1)
            && beta(self.beta2)
            && self.tau.is_finite()
            && self.tau > 0.0
    }
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlanError {
    #[error("unsupported plan version {0}")]
    Version(u32),
    #[error("plan needs at least one round")]
    NoRounds,
    #[error("server_adam needs a positive rate and tau, and betas in [0, 1)")]
    ServerAdam,
    #[error("a plan's rounds and redos must fit in a 32 bit round number")]
    TooManyRounds,
    #[error("a plan with no noise cannot retry: two sums would give a site's update away")]
    RetriesOpen,
    #[error("a scoring plan cannot retry")]
    RetriesScoring,
    #[error("a round already at the minimum cohort has nobody to retry without")]
    RetriesCohort,
    #[error("the privacy budget cannot cover every round and every redo")]
    RetriesBudget,
    #[error("min_helpers must be at least 2 and fewer than the plan's helpers")]
    MinHelpers,
    #[error("a plan that can lose a helper cannot also retry, score or prove norms yet")]
    MinHelpersWith,
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
    #[error("privacy: {0}")]
    Privacy(#[from] PrivacyError),
    #[error("a private run needs a helper, or the aggregator reads each noisy update")]
    PrivacyUnmasked,
    #[error("a private run counts every site the same, so max_weight must be 1")]
    PrivacyWeights,
    #[error("admission policy: {0}")]
    Policy(String),
    #[error("a credential listed for {0} is not valid or not about it")]
    Credential(String),
    #[error("a round must invite at least min_cohort sites and no more than there are")]
    Sample,
    #[error("norm proofs take their bound from the privacy terms, and the plan has none")]
    BoundedOpen,
    #[error("norm proofs: {0}")]
    Bounded(#[from] NormError),
    #[error("scoring releases group sums, which only a masked run keeps apart")]
    ScoringUnmasked,
    #[error("a scored group needs at least two sites and no more than min_cohort")]
    ScoringGroup,
}

impl Plan {
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.version != 1 {
            return Err(PlanError::Version(self.version));
        }
        if self.rounds == 0 {
            return Err(PlanError::NoRounds);
        }
        if self.attempts() > u64::from(u32::MAX) {
            return Err(PlanError::TooManyRounds);
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
        // The same key under two names, did:key and did:bio, is one party.
        let mut keys = BTreeSet::new();
        for did in everyone {
            let key = Did::parse(did).ok().and_then(|d| key_of(&d).ok());
            let fresh = key.is_none_or(|k| keys.insert(k.as_bytes().to_vec()));
            if !seen.insert(did) || !fresh {
                return Err(PlanError::DuplicateParty(did.clone()));
            }
        }
        for did in seen.iter().copied().chain([&self.sponsor]) {
            Did::parse(did)
                .ok()
                .filter(|d| key_of(d).is_ok())
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
        if let Some(privacy) = &self.privacy {
            if self.helpers.is_empty() {
                return Err(PlanError::PrivacyUnmasked);
            }
            if self.max_weight != 1 {
                return Err(PlanError::PrivacyWeights);
            }
            privacy.validate(self.layout.coords(), self.sites.len(), self.fewest())?;
        }
        if let Some(scoring) = &self.scoring {
            if self.helpers.is_empty() {
                return Err(PlanError::ScoringUnmasked);
            }
            if scoring.group < 2 || scoring.group > self.min_cohort {
                return Err(PlanError::ScoringGroup);
            }
        }
        if self.bounded {
            let privacy = self.privacy.as_ref().ok_or(PlanError::BoundedOpen)?;
            Params::new(privacy, self.layout.shape(self.chunk_coords))?;
        }
        if let Some(policy) = &self.admission {
            policy.validate().map_err(PlanError::Policy)?;
        }
        for (did, credentials) in self.operators() {
            let about_it = |text: &String| Credential::parse(text).is_ok_and(|c| c.subject == did);
            if !credentials.iter().all(about_it) {
                return Err(PlanError::Credential(did.to_owned()));
            }
        }
        if let Some(size) = self.sample
            && (size < self.min_cohort || size as usize > self.sites.len())
        {
            return Err(PlanError::Sample);
        }
        if self.server_adam.is_some_and(|adam| !adam.is_valid()) {
            return Err(PlanError::ServerAdam);
        }
        if self.retries > 0 {
            self.can_retry()?;
        }
        if self.min_helpers > 0 {
            self.can_lose_helpers()?;
        }
        Ok(())
    }

    fn can_retry(&self) -> Result<(), PlanError> {
        let privacy = self.privacy.as_ref().ok_or(PlanError::RetriesOpen)?;
        if self.scoring.is_some() {
            return Err(PlanError::RetriesScoring);
        }
        let drawn = self.sample.unwrap_or(self.sites.len() as u32);
        if drawn <= self.min_cohort {
            return Err(PlanError::RetriesCohort);
        }
        self.budget_covers_redos(privacy)
    }

    fn can_lose_helpers(&self) -> Result<(), PlanError> {
        if self.min_helpers < 2 || self.min_helpers as usize >= self.helpers.len() {
            return Err(PlanError::MinHelpers);
        }
        if self.retries > 0 || self.scoring.is_some() || self.bounded {
            return Err(PlanError::MinHelpersWith);
        }
        match &self.privacy {
            Some(privacy) => self.budget_covers_redos(privacy),
            None => Ok(()),
        }
    }

    /// Using up every redo must not be able to cut a run short.
    fn budget_covers_redos(&self, privacy: &Privacy) -> Result<(), PlanError> {
        let dearest = self.dearest_round().ok_or(PlanError::RetriesBudget)?;
        if u128::from(self.attempts()) * u128::from(dearest) > u128::from(privacy.budget) {
            return Err(PlanError::RetriesBudget);
        }
        Ok(())
    }

    /// How many helpers the run may lose.
    pub fn removals(&self) -> u32 {
        match self.min_helpers {
            0 => 0,
            floor => (self.helpers.len() as u32).saturating_sub(floor),
        }
    }

    /// How many rounds can close, and be paid for, in the whole run.
    pub fn attempts(&self) -> u64 {
        u64::from(self.rounds) + u64::from(self.retries) + u64::from(self.removals())
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

    /// The aggregator and each helper, with the credentials the plan lists for them.
    pub fn operators(&self) -> impl Iterator<Item = (&str, &[String])> {
        std::iter::once((
            self.aggregator.did.as_str(),
            self.aggregator.credentials.as_slice(),
        ))
        .chain(
            self.helpers
                .iter()
                .map(|h| (h.did.as_str(), h.credentials.as_slice())),
        )
    }

    /// True for the sponsor, the aggregator, a helper or a site.
    pub fn is_party(&self, did: &str) -> bool {
        did == self.sponsor
            || did == self.aggregator.did
            || self.helper_index(did).is_some()
            || self.site_index(did).is_some()
    }

    /// The fewest sites whose noise can stand behind one released sum: a
    /// whole cohort, or one group of it in a scoring run.
    pub fn fewest(&self) -> u32 {
        self.scoring.map_or(self.min_cohort, |s| s.group)
    }

    fn cost_over(&self, sites: usize) -> Option<u64> {
        match &self.privacy {
            Some(privacy) => privacy.round_cost(self.layout.coords(), sites as u32),
            None => Some(0),
        }
    }

    /// What a round over `cohort` sites spends, in units of 2^-40 of rho. A
    /// scoring run pays for its smallest group, since a site's data moves
    /// one group's sum only.
    pub fn round_cost(&self, cohort: usize) -> Option<u64> {
        self.cost_over(self.scoring.map_or(cohort, |s| s.smallest(cohort)))
    }

    /// The most a round can spend, whatever cohort closes it.
    pub fn dearest_round(&self) -> Option<u64> {
        self.cost_over(self.fewest() as usize)
    }

    /// The cohort positions of each scored group, once a round has closed.
    pub fn groups(&self, close_head: &Digest, cohort: usize) -> Vec<Vec<usize>> {
        self.scoring
            .map(|s| s.groups(close_head, cohort))
            .unwrap_or_default()
    }

    /// What a norm proof in this run is checked against.
    pub fn norm(&self) -> Option<Params> {
        let privacy = self.privacy.as_ref().filter(|_| self.bounded)?;
        Params::new(privacy, self.layout.shape(self.chunk_coords)).ok()
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
                credentials: Vec::new(),
            },
            helpers: vec![HelperSpec {
                did: helper.did_string(),
                seal: helper.seal_public().clone(),
                seal_proof: helper.seal_proof().unwrap(),
                credentials: Vec::new(),
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
            privacy: None,
            admission: None,
            sample: None,
            bounded: false,
            scoring: None,
            retries: 0,
            min_helpers: 0,
            server_adam: None,
        }
    }

    fn private() -> Plan {
        let mut plan = plan(1);
        plan.max_weight = 1;
        plan.privacy = Some(Privacy {
            clip: 1 << 18,
            noise: 1 << 34,
            colluding: 0,
            budget: 4 << 40,
            epsilon: 8.0,
            delta: 1e-5,
        });
        plan
    }

    #[test]
    fn a_private_plan_needs_helpers_equal_weights_and_sound_terms() {
        private().validate().unwrap();
        assert_eq!(
            private().round_cost(4),
            private().privacy.unwrap().round_cost(4, 4)
        );
        assert_eq!(plan(1).round_cost(4), Some(0));

        let mut plain = private();
        plain.helpers.clear();
        plain.min_cohort = 1;
        assert_eq!(plain.validate(), Err(PlanError::PrivacyUnmasked));

        let mut weighted = private();
        weighted.max_weight = 2;
        assert_eq!(weighted.validate(), Err(PlanError::PrivacyWeights));

        let mut colluding = private();
        colluding.privacy.as_mut().unwrap().colluding = 2;
        assert_eq!(
            colluding.validate(),
            Err(PlanError::Privacy(PrivacyError::Colluding))
        );
    }

    #[test]
    fn a_plan_without_privacy_keeps_its_old_encoding() {
        let plain = plan(1);
        let text = serde_json::to_string(&plain).unwrap();
        assert!(!text.contains("privacy"));
        assert_eq!(serde_json::from_str::<Plan>(&text).unwrap(), plain);
        let with = private();
        let text = serde_json::to_string(&with).unwrap();
        let back: Plan = serde_json::from_str(&text).unwrap();
        assert_eq!(back, with);
        assert_eq!(serde_json::to_string(&back).unwrap(), text);
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
