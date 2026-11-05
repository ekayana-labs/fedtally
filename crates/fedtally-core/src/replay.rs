//! The rules of a run, as a fold over its log.
//!
//! The aggregator, each helper, each site and an offline auditor all run this
//! same code. Nobody holds round state that the log can't rebuild. A record is
//! either applied whole or refused with the state unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pq_ucan::did::Did;
use pq_ucan::time::Timestamp;
use thiserror::Error;

use crate::admission::{self, Admission};
use crate::field::Fp2;
use crate::log::{GENESIS, chain};
use crate::plan::Plan;
use crate::record::{Body, Included, Record, decode_token};
use crate::vector::Digest;

#[derive(Debug, Error)]
pub enum Fault {
    #[error("record: {0}")]
    Record(#[from] crate::record::RecordError),
    #[error("the first record must be a plan signed by its sponsor")]
    NoPlan,
    #[error("invalid plan: {0}")]
    Plan(#[from] crate::plan::PlanError),
    #[error("{kind} signed by {signer}, who may not sign it")]
    WrongSigner { kind: &'static str, signer: String },
    #[error("{0} names another run")]
    WrongPlan(&'static str),
    #[error("{kind} does not fit phase {phase:?}")]
    WrongPhase { kind: &'static str, phase: Phase },
    #[error("{0} does not chain to the current head")]
    BrokenChain(&'static str),
    #[error("{0} is for round {1}, the current round is {2}")]
    WrongRound(&'static str, u32, u32),
    #[error("{0} repeated by {1}")]
    Duplicate(&'static str, String),
    #[error("site {0} refused: {1}")]
    Admission(String, String),
    #[error("site {0} has not joined")]
    NotJoined(String),
    #[error("contribution from {site}: {why}")]
    BadContribution { site: String, why: String },
    #[error("close: {0}")]
    BadClose(String),
    #[error("open: base is {got:?}, the current model is {want:?}")]
    WrongBase { got: Digest, want: Digest },
    #[error("outcome claims total weight {got}, the included weights add to {want}")]
    WrongWeight { got: u64, want: u64 },
    #[error("{0} names the wrong challenge")]
    WrongChallenge(&'static str),
    #[error("{0} has the wrong number of entries")]
    WrongLength(&'static str),
    #[error("helper {0}: its site shares do not add up to its total")]
    HelperTotal(String),
    #[error("seal: the aggregator's per-site fingerprints do not add up to its sum")]
    SealSum,
    #[error("seal: sum minus masks does not equal the aggregate")]
    SealAggregate,
    #[error("seal: site {0} attested a different masked fingerprint")]
    SiteMasked(String),
    #[error("seal: site {0} has not attested")]
    Unattested(String),
    #[error("site {0} has not acknowledged the last round")]
    Unacknowledged(String),
    #[error("seal: helper {0} has not attested")]
    HelperMissing(String),
    #[error("ack names a different head")]
    WrongHead,
    #[error("finish before the last round")]
    EarlyFinish,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Empty,
    Joining,
    Collecting,
    Masking,
    Attesting,
    Acking,
    Done,
    Aborted,
}

#[derive(Clone, Debug)]
pub struct Contribution {
    pub hash: Digest,
    pub masked: Digest,
    pub weight: u64,
    pub sealed: Vec<Digest>,
}

#[derive(Clone, Debug)]
pub struct SiteAttest {
    pub masked: Fp2,
}

#[derive(Clone, Debug)]
pub struct HelperAttest {
    pub total: Fp2,
    pub per_site: Vec<Fp2>,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub total_weight: u64,
    pub aggregate: Digest,
    pub model: Digest,
    pub challenge: Digest,
}

#[derive(Clone, Debug)]
pub struct Seal {
    pub per_site: Vec<Fp2>,
    pub sum: Fp2,
    pub aggregate: Fp2,
    pub head: Digest,
}

#[derive(Clone, Debug, Default)]
pub struct RoundState {
    pub round: u32,
    pub base: Digest,
    pub contributions: BTreeMap<usize, Contribution>,
    /// Site indexes in the round, ascending, once closed.
    pub included: Vec<usize>,
    pub sum: Digest,
    pub close_head: Digest,
    pub masks: BTreeMap<usize, Digest>,
    pub outcome: Option<Outcome>,
    pub site_attests: BTreeMap<usize, SiteAttest>,
    pub helper_attests: BTreeMap<usize, HelperAttest>,
    pub seal: Option<Seal>,
    pub acks: BTreeSet<usize>,
}

#[derive(Clone, Debug)]
pub struct Joined {
    pub at: i64,
    pub admission: Admission,
}

#[derive(Clone)]
pub struct Replay {
    pub plan: Option<Arc<Plan>>,
    pub plan_id: Digest,
    pub head: Digest,
    pub entries: u64,
    pub phase: Phase,
    pub model: Digest,
    pub joined: BTreeMap<usize, Joined>,
    pub rounds: Vec<RoundState>,
    pub abort: Option<String>,
}

impl Default for Replay {
    fn default() -> Self {
        Replay::new()
    }
}

fn kind(body: &Body) -> &'static str {
    match body {
        Body::Plan(_) => "plan",
        Body::Join { .. } => "join",
        Body::Open { .. } => "open",
        Body::Contribute { .. } => "contribute",
        Body::Close { .. } => "close",
        Body::MaskSum { .. } => "mask_sum",
        Body::Outcome { .. } => "outcome",
        Body::Attest { .. } => "attest",
        Body::HelperAttest { .. } => "helper_attest",
        Body::Seal { .. } => "seal",
        Body::Ack { .. } => "ack",
        Body::Abort { .. } => "abort",
        Body::Finish { .. } => "finish",
    }
}

impl Replay {
    pub fn new() -> Replay {
        Replay {
            plan: None,
            plan_id: GENESIS,
            head: GENESIS,
            entries: 0,
            phase: Phase::Empty,
            model: GENESIS,
            joined: BTreeMap::new(),
            rounds: Vec::new(),
            abort: None,
        }
    }

    pub fn from_lines<'a>(lines: impl IntoIterator<Item = &'a [u8]>) -> Result<Replay, Fault> {
        let mut replay = Replay::new();
        for line in lines {
            replay.apply_line(line)?;
        }
        Ok(replay)
    }

    pub fn plan(&self) -> Option<&Plan> {
        self.plan.as_deref()
    }

    pub fn current(&self) -> Option<&RoundState> {
        self.rounds.last()
    }

    pub fn round_number(&self) -> u32 {
        self.current().map_or(0, |r| r.round)
    }

    pub fn apply_line(&mut self, line: &[u8]) -> Result<Record, Fault> {
        let record = Record::parse(line)?;
        self.apply(&record)?;
        Ok(record)
    }

    pub fn apply(&mut self, record: &Record) -> Result<(), Fault> {
        let head = chain(&self.head, &record.line);
        match &record.body {
            Body::Plan(plan) => self.on_plan(record, plan)?,
            body => {
                let plan = self.plan.clone().ok_or(Fault::NoPlan)?;
                self.on_body(record, body, &plan, head)?;
            }
        }
        self.head = head;
        self.entries += 1;
        Ok(())
    }

    fn on_plan(&mut self, record: &Record, plan: &Plan) -> Result<(), Fault> {
        if self.phase != Phase::Empty || record.signer != plan.sponsor {
            return Err(Fault::NoPlan);
        }
        plan.validate()?;
        self.plan_id = record.body_hash;
        self.model = plan.init_model;
        self.plan = Some(Arc::new(plan.clone()));
        self.phase = Phase::Joining;
        Ok(())
    }

    /// A run moves on only when every site in the last round has signed the
    /// same head. So a site shown a forked log never sees the next round.
    fn all_acknowledged(&self, plan: &Plan) -> Result<(), Fault> {
        let Some(state) = self.current() else {
            return Ok(());
        };
        match state.included.iter().find(|s| !state.acks.contains(s)) {
            Some(site) => Err(Fault::Unacknowledged(plan.sites[*site].clone())),
            None => Ok(()),
        }
    }

    fn wants(&self, phases: &[Phase], body: &Body) -> Result<(), Fault> {
        if phases.contains(&self.phase) {
            Ok(())
        } else {
            Err(Fault::WrongPhase {
                kind: kind(body),
                phase: self.phase,
            })
        }
    }

    fn on_body(
        &mut self,
        record: &Record,
        body: &Body,
        plan: &Plan,
        head_after: Digest,
    ) -> Result<(), Fault> {
        let name = kind(body);
        let signer = record.signer.as_str();
        let wrong_signer = || Fault::WrongSigner {
            kind: name,
            signer: signer.to_owned(),
        };
        let from_aggregator = || {
            if signer == plan.aggregator.did {
                Ok(())
            } else {
                Err(wrong_signer())
            }
        };
        let from_site = || plan.site_index(signer).ok_or_else(wrong_signer);
        let from_helper = || plan.helper_index(signer).ok_or_else(wrong_signer);
        let same_run = |id: &Digest| {
            if *id == self.plan_id {
                Ok(())
            } else {
                Err(Fault::WrongPlan(name))
            }
        };
        let chained = |prev: &Digest| {
            if *prev == self.head {
                Ok(())
            } else {
                Err(Fault::BrokenChain(name))
            }
        };
        let this_round = |round: u32| {
            let current = self.round_number();
            if round == current {
                Ok(())
            } else {
                Err(Fault::WrongRound(name, round, current))
            }
        };

        match body {
            Body::Plan(_) => Err(Fault::NoPlan),

            Body::Join {
                plan: id,
                at,
                invocation,
                proofs,
            } => {
                let site = from_site()?;
                same_run(id)?;
                self.wants(&[Phase::Joining, Phase::Collecting, Phase::Acking], body)?;
                if self.joined.contains_key(&site) {
                    return Err(Fault::Duplicate(name, signer.to_owned()));
                }
                let refused = |why: String| Fault::Admission(signer.to_owned(), why);
                let token = decode_token(invocation).map_err(|e| refused(e.to_string()))?;
                let chain_tokens = proofs
                    .iter()
                    .map(|p| decode_token(p))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| refused(e.to_string()))?;
                let did = Did::parse(signer).map_err(|e| refused(e.to_string()))?;
                let now = Timestamp::from_unix(*at).map_err(|e| refused(e.to_string()))?;
                let admission = admission::check(&did, id, plan, &token, &chain_tokens, now, None)
                    .map_err(|e| refused(e.to_string()))?;
                self.joined.insert(site, Joined { at: *at, admission });
                Ok(())
            }

            Body::Open {
                plan: id,
                round,
                prev,
                base,
            } => {
                from_aggregator()?;
                same_run(id)?;
                chained(prev)?;
                self.wants(&[Phase::Joining, Phase::Acking], body)?;
                self.all_acknowledged(plan)?;
                let next = self.round_number() + 1;
                if *round != next || *round > plan.rounds {
                    return Err(Fault::WrongRound(name, *round, next));
                }
                if *base != self.model {
                    return Err(Fault::WrongBase {
                        got: *base,
                        want: self.model,
                    });
                }
                self.rounds.push(RoundState {
                    round: *round,
                    base: *base,
                    ..RoundState::default()
                });
                self.phase = Phase::Collecting;
                Ok(())
            }

            Body::Contribute {
                plan: id,
                round,
                base,
                masked,
                weight,
                sealed,
            } => {
                let site = from_site()?;
                same_run(id)?;
                self.wants(&[Phase::Collecting], body)?;
                this_round(*round)?;
                if !self.joined.contains_key(&site) {
                    return Err(Fault::NotJoined(signer.to_owned()));
                }
                let bad = |why: String| Fault::BadContribution {
                    site: signer.to_owned(),
                    why,
                };
                if *weight == 0 || *weight > plan.max_weight {
                    return Err(bad(format!(
                        "weight {weight} outside 1..={}",
                        plan.max_weight
                    )));
                }
                if *base != self.model {
                    return Err(bad("trained on another model".to_owned()));
                }
                if sealed.len() != plan.helpers.len() {
                    return Err(bad("wrong number of sealed seeds".to_owned()));
                }
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                if state.contributions.contains_key(&site) {
                    return Err(Fault::Duplicate(name, signer.to_owned()));
                }
                state.contributions.insert(
                    site,
                    Contribution {
                        hash: record.body_hash,
                        masked: *masked,
                        weight: *weight,
                        sealed: sealed.clone(),
                    },
                );
                Ok(())
            }

            Body::Close {
                plan: id,
                round,
                prev,
                included,
                sum,
            } => {
                from_aggregator()?;
                same_run(id)?;
                chained(prev)?;
                self.wants(&[Phase::Collecting], body)?;
                this_round(*round)?;
                let state = self.current().ok_or(Fault::NoPlan)?;
                let sites = check_included(plan, state, included)?;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                state.included = sites;
                state.sum = *sum;
                state.close_head = head_after;
                self.phase = Phase::Masking;
                Ok(())
            }

            Body::MaskSum {
                plan: id,
                round,
                close,
                mask,
            } => {
                let helper = from_helper()?;
                same_run(id)?;
                self.wants(&[Phase::Masking], body)?;
                this_round(*round)?;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                if *close != state.close_head {
                    return Err(Fault::BrokenChain(name));
                }
                if state.masks.contains_key(&helper) {
                    return Err(Fault::Duplicate(name, signer.to_owned()));
                }
                state.masks.insert(helper, *mask);
                Ok(())
            }

            Body::Outcome {
                plan: id,
                round,
                prev,
                total_weight,
                aggregate,
                model,
            } => {
                from_aggregator()?;
                same_run(id)?;
                chained(prev)?;
                self.wants(&[Phase::Masking], body)?;
                this_round(*round)?;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                if let Some(missing) =
                    (0..plan.helpers.len()).find(|h| !state.masks.contains_key(h))
                {
                    return Err(Fault::HelperMissing(plan.helpers[missing].did.clone()));
                }
                let want: u64 = state
                    .included
                    .iter()
                    .map(|s| state.contributions[s].weight)
                    .sum();
                if *total_weight != want {
                    return Err(Fault::WrongWeight {
                        got: *total_weight,
                        want,
                    });
                }
                state.outcome = Some(Outcome {
                    total_weight: *total_weight,
                    aggregate: *aggregate,
                    model: *model,
                    challenge: head_after,
                });
                self.phase = Phase::Attesting;
                Ok(())
            }

            Body::Attest {
                plan: id,
                round,
                challenge,
                masked,
            } => {
                let site = from_site()?;
                same_run(id)?;
                self.wants(&[Phase::Attesting], body)?;
                this_round(*round)?;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                if !state.included.contains(&site) {
                    return Err(Fault::WrongSigner {
                        kind: name,
                        signer: signer.to_owned(),
                    });
                }
                if state.outcome.as_ref().map(|o| o.challenge) != Some(*challenge) {
                    return Err(Fault::WrongChallenge(name));
                }
                // A site checks the helpers' shares before it attests, so they come first.
                if let Some(missing) =
                    (0..plan.helpers.len()).find(|h| !state.helper_attests.contains_key(h))
                {
                    return Err(Fault::HelperMissing(plan.helpers[missing].did.clone()));
                }
                if state.site_attests.contains_key(&site) {
                    return Err(Fault::Duplicate(name, signer.to_owned()));
                }
                state
                    .site_attests
                    .insert(site, SiteAttest { masked: *masked });
                Ok(())
            }

            Body::HelperAttest {
                plan: id,
                round,
                challenge,
                total,
                per_site,
            } => {
                let helper = from_helper()?;
                same_run(id)?;
                self.wants(&[Phase::Attesting], body)?;
                this_round(*round)?;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                if state.outcome.as_ref().map(|o| o.challenge) != Some(*challenge) {
                    return Err(Fault::WrongChallenge(name));
                }
                if per_site.len() != state.included.len() {
                    return Err(Fault::WrongLength(name));
                }
                if per_site.iter().copied().sum::<Fp2>() != *total {
                    return Err(Fault::HelperTotal(signer.to_owned()));
                }
                if state.helper_attests.contains_key(&helper) {
                    return Err(Fault::Duplicate(name, signer.to_owned()));
                }
                state.helper_attests.insert(
                    helper,
                    HelperAttest {
                        total: *total,
                        per_site: per_site.clone(),
                    },
                );
                Ok(())
            }

            Body::Seal {
                plan: id,
                round,
                prev,
                per_site,
                sum,
                aggregate,
            } => {
                from_aggregator()?;
                same_run(id)?;
                chained(prev)?;
                self.wants(&[Phase::Attesting], body)?;
                this_round(*round)?;
                let state = self.current().ok_or(Fault::NoPlan)?;
                check_seal(plan, state, per_site, *sum, *aggregate)?;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                state.seal = Some(Seal {
                    per_site: per_site.clone(),
                    sum: *sum,
                    aggregate: *aggregate,
                    head: head_after,
                });
                let model = state
                    .outcome
                    .as_ref()
                    .map(|o| o.model)
                    .ok_or(Fault::NoPlan)?;
                self.model = model;
                self.phase = Phase::Acking;
                Ok(())
            }

            Body::Ack {
                plan: id,
                round,
                head,
            } => {
                let site = from_site()?;
                same_run(id)?;
                self.wants(&[Phase::Acking], body)?;
                this_round(*round)?;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                if state.seal.as_ref().map(|s| s.head) != Some(*head) {
                    return Err(Fault::WrongHead);
                }
                if !state.included.contains(&site) {
                    return Err(Fault::WrongSigner {
                        kind: name,
                        signer: signer.to_owned(),
                    });
                }
                if !state.acks.insert(site) {
                    return Err(Fault::Duplicate(name, signer.to_owned()));
                }
                Ok(())
            }

            Body::Abort {
                plan: id, reason, ..
            } => {
                if signer != plan.aggregator.did && plan.helper_index(signer).is_none() {
                    return Err(wrong_signer());
                }
                same_run(id)?;
                if matches!(self.phase, Phase::Done | Phase::Aborted) {
                    return Err(Fault::WrongPhase {
                        kind: name,
                        phase: self.phase,
                    });
                }
                self.abort = Some(reason.clone());
                self.phase = Phase::Aborted;
                Ok(())
            }

            Body::Finish { plan: id, prev } => {
                from_aggregator()?;
                same_run(id)?;
                chained(prev)?;
                self.wants(&[Phase::Acking], body)?;
                self.all_acknowledged(plan)?;
                if self.round_number() != plan.rounds {
                    return Err(Fault::EarlyFinish);
                }
                self.phase = Phase::Done;
                Ok(())
            }
        }
    }
}

fn check_included(
    plan: &Plan,
    state: &RoundState,
    included: &[Included],
) -> Result<Vec<usize>, Fault> {
    if included.len() < plan.min_cohort as usize {
        return Err(Fault::BadClose(format!(
            "{} sites, the plan needs {}",
            included.len(),
            plan.min_cohort
        )));
    }
    let mut sites = Vec::with_capacity(included.len());
    for item in included {
        let site = plan
            .site_index(&item.site)
            .ok_or_else(|| Fault::BadClose(format!("{} is not in the plan", item.site)))?;
        if sites.last().is_some_and(|last| *last >= site) {
            return Err(Fault::BadClose("sites out of order or repeated".to_owned()));
        }
        let contribution = state
            .contributions
            .get(&site)
            .ok_or_else(|| Fault::BadClose(format!("{} did not contribute", item.site)))?;
        if contribution.hash != item.contribution {
            return Err(Fault::BadClose(format!(
                "{} is listed with a different contribution",
                item.site
            )));
        }
        sites.push(site);
    }
    Ok(sites)
}

fn check_seal(
    plan: &Plan,
    state: &RoundState,
    per_site: &[Fp2],
    sum: Fp2,
    aggregate: Fp2,
) -> Result<(), Fault> {
    if per_site.len() != state.included.len() {
        return Err(Fault::WrongLength("seal"));
    }
    if per_site.iter().copied().sum::<Fp2>() != sum {
        return Err(Fault::SealSum);
    }
    let mut masks = Fp2::ZERO;
    for (h, helper) in plan.helpers.iter().enumerate() {
        let attest = state
            .helper_attests
            .get(&h)
            .ok_or_else(|| Fault::HelperMissing(helper.did.clone()))?;
        masks = masks + attest.total;
    }
    if sum - masks != aggregate {
        return Err(Fault::SealAggregate);
    }
    // Every fingerprint must be backed by the site it belongs to. If one could
    // be left to the aggregator, it could solve for that value and make the
    // equations hold for any aggregate at all.
    for (pos, site) in state.included.iter().enumerate() {
        let attest = state
            .site_attests
            .get(site)
            .ok_or_else(|| Fault::Unattested(plan.sites[*site].clone()))?;
        if attest.masked != per_site[pos] {
            return Err(Fault::SiteMasked(plan.sites[*site].clone()));
        }
    }
    Ok(())
}
