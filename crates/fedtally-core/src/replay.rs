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
use crate::credential::Credential;
use crate::field::Fp2;
use crate::log::{GENESIS, chain};
use crate::plan::Plan;
use crate::policy::{Check, Holder, Policy, Rule};
use crate::record::{Body, Included, Record, decode_token};
use crate::sampling;
use crate::scoring;
use crate::status::StatusList;
use crate::vector::Digest;

/// More than any policy can ask of one site.
const MAX_CREDENTIALS: usize = 64;

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
    #[error("screen: {0}")]
    BadScreen(String),
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
    #[error("round {round} would spend {cost}, and only {left} of the privacy budget is left")]
    OverBudget { round: u32, cost: u64, left: u64 },
    #[error("status list: {0}")]
    Status(String),
    #[error("seal: the groups' shares must add up to the whole reward")]
    Shares,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Empty,
    Joining,
    Collecting,
    Screening,
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
    pub proof: Option<Digest>,
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
    pub groups: Vec<Digest>,
}

#[derive(Clone, Debug)]
pub struct Seal {
    pub per_site: Vec<Fp2>,
    pub sum: Fp2,
    pub aggregate: Fp2,
    pub head: Digest,
    pub shares: Vec<u32>,
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
    /// Site indexes whose norm proofs were put up for checking, ascending.
    pub screened: Vec<usize>,
    pub screen_head: Digest,
    /// Each helper's messages about the screened proofs.
    pub checks: BTreeMap<usize, Digest>,
    /// The aggregator's messages about them.
    pub shares: Option<Digest>,
    pub masks: BTreeMap<usize, Digest>,
    /// Each helper's mask totals for the scored groups.
    pub group_masks: BTreeMap<usize, Vec<Digest>>,
    pub outcome: Option<Outcome>,
    pub site_attests: BTreeMap<usize, SiteAttest>,
    pub helper_attests: BTreeMap<usize, HelperAttest>,
    pub seal: Option<Seal>,
    pub acks: BTreeSet<usize>,
    /// What closing the round spent of the privacy budget.
    pub cost: u64,
    /// The sites drawn for this round, where the plan samples.
    pub invited: Option<Vec<usize>>,
    /// The sites that could have been in the round when it closed.
    pub expected: Vec<usize>,
    /// The sealed round this one starts from, 0 for the initial model.
    pub from: u32,
    /// Which model this round would make. Numbers count attempts, steps don't.
    pub step: u32,
}

#[derive(Clone, Debug)]
pub struct Joined {
    pub at: i64,
    pub admission: Admission,
    pub dataset: Option<String>,
    pub credentials: Vec<Credential>,
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
    /// Privacy budget spent so far, in units of 2^-40 of rho.
    pub spent: u64,
    /// The newest revocation list from each issuer, by list id.
    pub lists: BTreeMap<(String, String), StatusList>,
    /// Sites whose credentials no longer meet the policy, and why.
    pub revoked: BTreeMap<usize, String>,
    /// The credentials the plan lists for the aggregator and each helper.
    pub operators: Vec<Vec<Credential>>,
    /// Round numbers given up so far. Each one is an attempt that made no model.
    pub retried: u32,
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
        Body::Screen { .. } => "screen",
        Body::Check { .. } => "check",
        Body::Close { .. } => "close",
        Body::MaskSum { .. } => "mask_sum",
        Body::Outcome { .. } => "outcome",
        Body::Attest { .. } => "attest",
        Body::HelperAttest { .. } => "helper_attest",
        Body::Seal { .. } => "seal",
        Body::Ack { .. } => "ack",
        Body::Abort { .. } => "abort",
        Body::Status { .. } => "status",
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
            spent: 0,
            lists: BTreeMap::new(),
            revoked: BTreeMap::new(),
            operators: Vec::new(),
            retried: 0,
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
        self.operators = plan
            .operators()
            .map(|(did, held)| {
                held.iter()
                    .map(|text| Credential::parse(text))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| Fault::Admission(did.to_owned(), e.to_string()))
            })
            .collect::<Result<_, _>>()?;
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

    /// The newest list an issuer has published under an id. Lists are kept
    /// per issuer, so one issuer can't take another's id.
    pub fn list(&self, issuer: &str, id: &str) -> Option<&StatusList> {
        self.lists.get(&(issuer.to_owned(), id.to_owned()))
    }

    fn is_revoked(&self, credential: &Credential) -> bool {
        credential.status.as_ref().is_some_and(|entry| {
            self.list(&credential.issuer, &entry.list)
                .is_some_and(|list| list.is_set(entry.index))
        })
    }

    /// A credential that can be revoked only counts once its issuer's list is
    /// in the log, so every replica judges it against the same list.
    fn unlisted<'a>(&self, held: &'a [Credential]) -> Option<&'a Credential> {
        held.iter().find(|c| {
            c.status
                .as_ref()
                .is_some_and(|entry| self.list(&c.issuer, &entry.list).is_none())
        })
    }

    fn meets(
        &self,
        policy: &Policy,
        rule: &Rule,
        holder: Holder<'_>,
        held: &[Credential],
        now: i64,
    ) -> Result<(), String> {
        if let Some(credential) = self.unlisted(held) {
            return Err(format!(
                "the status list of {} is not in the log",
                credential.issuer
            ));
        }
        Check {
            purpose: &policy.purpose,
            holder,
            credentials: held,
            now,
            revoked: &|c| self.is_revoked(c),
        }
        .meets(rule)
    }

    /// The first operator that does not meet the policy at `now`, if any.
    fn operators_meet(&self, plan: &Plan, policy: &Policy, now: i64) -> Result<(), String> {
        let Some(rule) = &policy.operators else {
            return Ok(());
        };
        for ((did, _), held) in plan.operators().zip(&self.operators) {
            let holder = Holder { did, dataset: None };
            self.meets(policy, rule, holder, held, now)
                .map_err(|why| format!("{did}: {why}"))?;
        }
        Ok(())
    }

    /// Whether a site would be admitted at `now` on the credentials it holds,
    /// judged against the lists in this log. A plan without a policy admits.
    pub fn site_meets(
        &self,
        plan: &Plan,
        did: &str,
        dataset: Option<&str>,
        held: &[Credential],
        now: i64,
    ) -> Result<(), String> {
        let Some(policy) = &plan.admission else {
            return Ok(());
        };
        let holder = Holder { did, dataset };
        self.meets(policy, &policy.sites, holder, held, now)?;
        self.operators_meet(plan, policy, now)
    }

    /// The sites that may take part in the current round as things stand.
    pub fn expected(&self, plan: &Plan) -> Vec<usize> {
        let invited = self.current().and_then(|r| r.invited.clone());
        invited
            .unwrap_or_else(|| (0..plan.sites.len()).collect())
            .into_iter()
            .filter(|site| !self.revoked.contains_key(site))
            .collect()
    }

    /// A round's cost, if the budget still covers it.
    fn affordable(&self, plan: &Plan, round: u32, cost: Option<u64>) -> Result<u64, Fault> {
        let left = plan
            .privacy
            .as_ref()
            .map_or(u64::MAX, |p| p.budget.saturating_sub(self.spent));
        match cost {
            Some(cost) if cost <= left => Ok(cost),
            cost => Err(Fault::OverBudget {
                round,
                cost: cost.unwrap_or(u64::MAX),
                left,
            }),
        }
    }

    fn still_admitted(&self, plan: &Plan, sites: &[usize]) -> Result<(), Fault> {
        match sites.iter().find(|s| self.revoked.contains_key(s)) {
            Some(site) => Err(Fault::BadClose(format!(
                "{} is no longer admitted",
                plan.sites[*site]
            ))),
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
                dataset,
                credentials,
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
                let held = match &plan.admission {
                    None if dataset.is_some() || !credentials.is_empty() => {
                        return Err(refused("this plan asks for no credentials".into()));
                    }
                    None => Vec::new(),
                    Some(policy) => {
                        // A site's word for the time can't be earlier than a
                        // list already in the log, or an expired credential
                        // could be backdated past its revocation.
                        if self.lists.values().any(|list| *at < list.issued) {
                            return Err(refused(
                                "it gives a time before a revocation list in the log".into(),
                            ));
                        }
                        let taken = self
                            .joined
                            .values()
                            .any(|other| dataset.is_some() && other.dataset == *dataset);
                        if taken {
                            return Err(refused("another site trains on that dataset".into()));
                        }
                        if credentials.len() > MAX_CREDENTIALS {
                            return Err(refused("too many credentials".into()));
                        }
                        let held = credentials
                            .iter()
                            .map(|text| Credential::parse(text))
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| refused(e.to_string()))?;
                        let holder = Holder {
                            did: signer,
                            dataset: dataset.as_deref(),
                        };
                        self.meets(policy, &policy.sites, holder, &held, *at)
                            .map_err(refused)?;
                        self.operators_meet(plan, policy, *at).map_err(refused)?;
                        held
                    }
                };
                self.joined.insert(
                    site,
                    Joined {
                        at: *at,
                        admission,
                        dataset: dataset.clone(),
                        credentials: held,
                    },
                );
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
                if *round != next || *round > plan.rounds + self.retried {
                    return Err(Fault::WrongRound(name, *round, next));
                }
                if *base != self.model {
                    return Err(Fault::WrongBase {
                        got: *base,
                        want: self.model,
                    });
                }
                // If the dearest round fits, whatever Close picks fits too.
                self.affordable(plan, *round, plan.dearest_round())?;
                let step = *round - self.retried;
                let invited = plan.sample.map(|size| {
                    let beacon = sampling::beacon(&self.plan_id, step, &self.model);
                    sampling::invited(&beacon, plan.sites.len(), size as usize).0
                });
                self.rounds.push(RoundState {
                    round: *round,
                    base: *base,
                    invited,
                    from: next - 1,
                    step,
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
                proof,
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
                if let Some(why) = self.revoked.get(&site) {
                    return Err(bad(format!("it is no longer admitted: {why}")));
                }
                let invited = self.current().and_then(|r| r.invited.as_ref());
                if invited.is_some_and(|sites| !sites.contains(&site)) {
                    return Err(bad("it was not drawn for this round".to_owned()));
                }
                if sealed.len() != plan.helpers.len() {
                    return Err(bad("wrong number of sealed seeds".to_owned()));
                }
                if proof.is_some() != plan.bounded {
                    return Err(bad("a norm proof exactly when the plan asks".to_owned()));
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
                        proof: *proof,
                    },
                );
                Ok(())
            }

            Body::Screen {
                plan: id,
                round,
                prev,
                candidates,
            } => {
                from_aggregator()?;
                same_run(id)?;
                chained(prev)?;
                self.wants(&[Phase::Collecting], body)?;
                this_round(*round)?;
                let screen = |fault| match fault {
                    Fault::BadClose(why) => Fault::BadScreen(why),
                    other => other,
                };
                if !plan.bounded {
                    return Err(Fault::BadScreen("the plan asks for no proofs".to_owned()));
                }
                let state = self.current().ok_or(Fault::NoPlan)?;
                let sites = check_included(plan, state, candidates).map_err(screen)?;
                self.still_admitted(plan, &sites).map_err(screen)?;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                state.screened = sites;
                state.screen_head = head_after;
                self.phase = Phase::Screening;
                Ok(())
            }

            Body::Check {
                plan: id,
                round,
                screen,
                shares,
            } => {
                let helper = from_helper()?;
                same_run(id)?;
                self.wants(&[Phase::Screening], body)?;
                this_round(*round)?;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                if *screen != state.screen_head {
                    return Err(Fault::BrokenChain(name));
                }
                if state.checks.contains_key(&helper) {
                    return Err(Fault::Duplicate(name, signer.to_owned()));
                }
                state.checks.insert(helper, *shares);
                Ok(())
            }

            Body::Close {
                plan: id,
                round,
                prev,
                included,
                sum,
                shares,
            } => {
                from_aggregator()?;
                same_run(id)?;
                chained(prev)?;
                let after = if plan.bounded {
                    Phase::Screening
                } else {
                    Phase::Collecting
                };
                self.wants(&[after], body)?;
                this_round(*round)?;
                let state = self.current().ok_or(Fault::NoPlan)?;
                let sites = check_included(plan, state, included)?;
                self.still_admitted(plan, &sites)?;
                if shares.is_some() != plan.bounded {
                    return Err(Fault::BadClose(
                        "the aggregator's proof messages exactly when the plan asks".to_owned(),
                    ));
                }
                if plan.bounded {
                    if let Some(missing) =
                        (0..plan.helpers.len()).find(|h| !state.checks.contains_key(h))
                    {
                        return Err(Fault::HelperMissing(plan.helpers[missing].did.clone()));
                    }
                    if let Some(site) = sites.iter().find(|s| !state.screened.contains(s)) {
                        return Err(Fault::BadClose(format!(
                            "{} was not screened",
                            plan.sites[*site]
                        )));
                    }
                }
                let expected = self.expected(plan);
                // The sum is as good as released once the cohort is fixed, so
                // the round is paid for here, whether or not it gets sealed.
                let cost = self.affordable(plan, *round, plan.round_cost(sites.len()))?;
                self.spent += cost;
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                state.included = sites;
                state.sum = *sum;
                state.shares = *shares;
                state.close_head = head_after;
                state.cost = cost;
                state.expected = expected;
                self.phase = Phase::Masking;
                Ok(())
            }

            Body::MaskSum {
                plan: id,
                round,
                close,
                mask,
                groups,
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
                if groups.len() != group_count(plan, state) {
                    return Err(Fault::WrongLength(name));
                }
                state.group_masks.insert(helper, groups.clone());
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
                groups,
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
                if groups.len() != group_count(plan, state) {
                    return Err(Fault::WrongLength(name));
                }
                state.outcome = Some(Outcome {
                    total_weight: *total_weight,
                    aggregate: *aggregate,
                    model: *model,
                    challenge: head_after,
                    groups: groups.clone(),
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
                shares,
            } => {
                from_aggregator()?;
                same_run(id)?;
                chained(prev)?;
                self.wants(&[Phase::Attesting], body)?;
                this_round(*round)?;
                let state = self.current().ok_or(Fault::NoPlan)?;
                check_seal(plan, state, per_site, *sum, *aggregate)?;
                if shares.len() != group_count(plan, state) {
                    return Err(Fault::WrongLength(name));
                }
                let whole = shares.iter().try_fold(0u32, |sum, s| sum.checked_add(*s));
                if !shares.is_empty() && whole != Some(scoring::WHOLE) {
                    return Err(Fault::Shares);
                }
                let state = self.rounds.last_mut().ok_or(Fault::NoPlan)?;
                state.seal = Some(Seal {
                    per_site: per_site.clone(),
                    sum: *sum,
                    aggregate: *aggregate,
                    head: head_after,
                    shares: shares.clone(),
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

            Body::Status { plan: id, list } => {
                if !plan.is_party(signer) {
                    return Err(wrong_signer());
                }
                same_run(id)?;
                if matches!(self.phase, Phase::Empty | Phase::Done | Phase::Aborted) {
                    return Err(Fault::WrongPhase {
                        kind: name,
                        phase: self.phase,
                    });
                }
                let policy = plan
                    .admission
                    .as_ref()
                    .ok_or_else(|| Fault::Status("this plan asks for no credentials".into()))?;
                let list = StatusList::parse(list).map_err(|e| Fault::Status(e.to_string()))?;
                if !policy.relies_on(&list.issuer) {
                    return Err(Fault::Status(
                        "no rule in this plan accepts that issuer".into(),
                    ));
                }
                if let Some(old) = self.list(&list.issuer, &list.id) {
                    if !list.keeps(old) {
                        return Err(Fault::Status("a revocation cannot be undone".into()));
                    }
                    // An issuer may revoke without changing the list's date.
                    let adds = !old.keeps(&list);
                    if list.issued < old.issued || (list.issued == old.issued && !adds) {
                        return Err(Fault::Status("the log already holds a newer list".into()));
                    }
                }
                self.lists
                    .insert((list.issuer.clone(), list.id.clone()), list);

                let fallen: Vec<(usize, String)> = self
                    .joined
                    .iter()
                    .filter(|(site, _)| !self.revoked.contains_key(site))
                    .filter_map(|(site, joined)| {
                        let holder = Holder {
                            did: &plan.sites[*site],
                            dataset: joined.dataset.as_deref(),
                        };
                        self.meets(
                            policy,
                            &policy.sites,
                            holder,
                            &joined.credentials,
                            joined.at,
                        )
                        .err()
                        .map(|why| (*site, why))
                    })
                    .collect();
                self.revoked.extend(fallen);
                // An operator that loses its standing ends the run for everyone.
                let first_join = self.joined.values().map(|j| j.at).min();
                if let Some(at) = first_join
                    && let Err(why) = self.operators_meet(plan, policy, at)
                {
                    self.abort = Some(why);
                    self.phase = Phase::Aborted;
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
                if self.round_number() != plan.rounds + self.retried {
                    return Err(Fault::EarlyFinish);
                }
                self.phase = Phase::Done;
                Ok(())
            }
        }
    }
}

fn group_count(plan: &Plan, state: &RoundState) -> usize {
    plan.scoring.map_or(0, |s| s.count(state.included.len()))
}

impl RoundState {
    /// The fingerprint each scored group's sum must have, worked out from
    /// values every member site and helper has signed. `None` before the seal.
    pub fn group_prints(&self, plan: &Plan) -> Option<Vec<Fp2>> {
        let seal = self.seal.as_ref()?;
        let over = |values: &[Fp2], group: &[usize]| -> Option<Fp2> {
            group.iter().map(|p| values.get(*p).copied()).sum()
        };
        plan.groups(&self.close_head, self.included.len())
            .iter()
            .map(|group| {
                let mut print = over(&seal.per_site, group)?;
                for helper in 0..plan.helpers.len() {
                    print = print - over(&self.helper_attests.get(&helper)?.per_site, group)?;
                }
                Some(print)
            })
            .collect()
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
