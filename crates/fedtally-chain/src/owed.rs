//! What a party owes the ledger next, decided from the log and the ledger's
//! accounts alone. Nothing here reads or sends anything.

use std::collections::{BTreeMap, BTreeSet};

use fedtally_ledger::client::ProofPart;
#[cfg(test)]
use fedtally_ledger::state::ROUND_VOID;
use fedtally_ledger::state::{
    CLAIM_WINDOWS, LEAF_LEN, ROUND_COLLECTING, ROUND_COMMITTED, ROUND_FINAL, ROUND_FRAUD,
    ROUND_POSTED, SITE_REVOKED, SITE_WITHDRAWN, TASK_ABORTED, TASK_FINISHED,
};
use thiserror::Error;

use crate::ledger::{
    BufferView, DemandView, DisputeView, EnrollmentView, RoundView, StatusView, TaskView,
};
use crate::run::{Ending, RoundLog, RunView};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Party {
    Publisher,
    Aggregator,
    Site(u16),
    Watcher,
    Issuer,
}

/// The accounts a decision looks at, read at one slot. A party only fills in
/// what its role reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub slot: u64,
    pub task: Option<TaskView>,
    /// What a task account needs to stay rent exempt.
    pub task_rent: u64,
    pub rounds: BTreeMap<u32, RoundView>,
    pub enrollments: BTreeMap<u16, EnrollmentView>,
    /// The party's own dispute over the task's latest round.
    pub dispute: Option<DisputeView>,
    pub buffer: Option<BufferView>,
    /// The open demands on the task: all of them for the aggregator, its own
    /// for a watcher.
    pub demands: Vec<DemandView>,
    /// An issuer's lists on the ledger, by number.
    pub lists: BTreeMap<u32, StatusView>,
    /// The slot at which this party first saw the log over.
    pub log_ended: Option<u64>,
}

/// A leaf a watcher could not fetch and asks for on the ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wanted {
    pub part: ProofPart,
    pub leaf: u64,
    /// A demand for it has been sent.
    pub opened: bool,
    /// Its answer has gone into the watcher's copy of the vector.
    pub taken: bool,
}

/// What an issuer has signed, as the log has it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SignedList {
    pub bits: u32,
    pub set: Vec<u32>,
}

/// What a watcher found when it added up a round's vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Sound,
    /// The first leaf where the aggregate is not the sum.
    Wrong(u64),
    /// The vectors behind the ledger's roots could not be had.
    Unprovable,
}

/// What a watcher keeps between steps. All of it can be worked out again.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Notes {
    pub verdicts: BTreeMap<u32, Verdict>,
    /// Rounds this watcher has already taken to a dispute.
    pub disputed: BTreeSet<u32>,
    /// The slot each site's status lists were last applied at, since the log
    /// revoked it. An issuer's list may reach the ledger later than the check.
    pub checked: BTreeMap<u16, u64>,
    pub wanted: BTreeMap<u32, Vec<Wanted>>,
    /// What this issuer's lists say, by ledger number.
    pub signed: BTreeMap<u32, SignedList>,
}

/// Slots between two checks of one revoked site.
const RECHECK_SLOTS: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Differs {
    Beacon,
    Cohort,
    InputsRoot,
    CloseHead,
    MaskRoots,
    AggregateRoot,
    NextModel,
    SealHead,
    SharesRoot,
}

impl std::fmt::Display for Differs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Differs::Beacon => "beacon",
            Differs::Cohort => "cohort",
            Differs::InputsRoot => "inputs root",
            Differs::CloseHead => "close head",
            Differs::MaskRoots => "mask roots",
            Differs::AggregateRoot => "aggregate root",
            Differs::NextModel => "next model",
            Differs::SealHead => "seal head",
            Differs::SharesRoot => "shares root",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum Divergence {
    #[error("the task's {0} is not the plan's")]
    Terms(&'static str),
    #[error("round {round}: the ledger's {what} is not the log's")]
    Round { round: u32, what: Differs },
    #[error("the ledger is at round {0}, which the log has not reached")]
    Ahead(u32),
    #[error("site {0} is enrolled with another wallet")]
    Enrollment(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    /// The task is not on the ledger yet.
    Task,
    /// The log has not got there yet.
    Log,
    Enrollment(u16),
    Acks,
    Dispute,
    /// The challenge window, which ends at this slot.
    Window(u64),
    /// The run goes on and nothing is due.
    Run,
    /// Rounds are left that can't be closed yet: shares not claimed, or a
    /// dispute still open.
    Claims,
    /// The ledger's round is not the one this site acknowledged in its log.
    Refusing {
        round: u32,
        what: Differs,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Publish,
    WithdrawPublisher,
    Back,
    OpenRound(u32),
    CommitRoot(u32),
    PostAggregate(u32),
    Settle(u32),
    Abort,
    WithdrawAggregator,
    Enroll,
    Ack(u32),
    Claim(u32),
    WithdrawSite,
    /// Fetch the round's vectors and add them up.
    Examine(u32),
    OpenBuffer(u32),
    OpenDispute {
        round: u32,
        leaf: u64,
        /// The answered demand that lets this dispute open past the window.
        after: Option<ProofPart>,
    },
    OpenDemand {
        round: u32,
        part: ProofPart,
        leaf: u64,
    },
    TakeAnswer {
        round: u32,
        part: ProofPart,
        leaf: u64,
    },
    CloseDemand {
        round: u32,
        part: ProofPart,
        leaf: u64,
    },
    /// The aggregator publishes a leaf someone demanded.
    AnswerDemand {
        round: u32,
        part: ProofPart,
        leaf: u64,
        address: [u8; 32],
    },
    CloseRound(u32),
    /// Gives up a committed round the log gave up.
    VoidRound {
        round: u32,
        remove_helper: bool,
    },
    InitList {
        list: u32,
        bits: u32,
    },
    SetBit {
        list: u32,
        bit: u32,
    },
    GiveLeaf {
        round: u32,
        leaf: u64,
        part: ProofPart,
    },
    CloseDispute(u32),
    CloseBuffer,
    RevokeCheck(u16),
    Wait(Wait),
    Done,
}

/// Where the ledger's round and the log's differ, in what the round's state
/// says the ledger holds.
pub fn differences(round: &RoundView, log: &RoundLog) -> Vec<Differs> {
    let mut found = Vec::new();
    let mut check = |same: bool, what: Differs| {
        if !same {
            found.push(what);
        }
    };
    if let Some(committed) = &log.committed {
        check(round.beacon == committed.beacon, Differs::Beacon);
        if round.state != ROUND_COLLECTING {
            let cohort: Vec<u16> = committed.cohort.iter().map(|s| *s as u16).collect();
            check(round.cohort.members() == cohort, Differs::Cohort);
            check(
                round.inputs_root == committed.inputs_root,
                Differs::InputsRoot,
            );
            check(
                round.close_head == committed.close_head.0,
                Differs::CloseHead,
            );
        }
    }
    if let Some(posted) = &log.posted
        && !matches!(round.state, ROUND_COLLECTING | ROUND_COMMITTED)
    {
        let masks: Vec<[u8; 32]> = posted.mask_roots.iter().map(|m| m.0).collect();
        check(round.mask_roots == masks, Differs::MaskRoots);
        check(
            round.aggregate_root == posted.aggregate_root.0,
            Differs::AggregateRoot,
        );
        check(round.next_model == posted.next_model.0, Differs::NextModel);
        check(round.seal_head == posted.seal_head.0, Differs::SealHead);
        check(round.shares_root == posted.shares_root, Differs::SharesRoot);
    }
    found
}

/// The first term of the task that is not the plan's. A party stakes on
/// nothing else.
pub fn terms_differ(task: &TaskView, run: &RunView) -> Option<&'static str> {
    let terms = &run.terms;
    let checks = [
        (task.publisher == run.publisher, "publisher"),
        (task.aggregator == run.aggregator, "aggregator"),
        (task.plan_id == terms.plan.0, "plan"),
        (task.sites_root == terms.sites_root, "sites root"),
        (task.sites == terms.sites, "number of sites"),
        (task.helpers == terms.helpers, "number of helpers"),
        (task.coords == terms.coords, "vector length"),
        (task.rounds == terms.rounds, "number of rounds"),
        (task.min_cohort == terms.min_cohort, "minimum cohort"),
        (task.sample == terms.sample, "sample"),
        (task.sensitivity == terms.sensitivity, "sensitivity"),
        (task.noise == terms.noise, "noise"),
        (task.colluding == terms.colluding, "colluding sites"),
        (task.budget == terms.budget, "budget"),
        (task.retries == terms.retries, "retries"),
        (task.min_helpers == terms.min_helpers, "minimum helpers"),
        (task.score_group == terms.score_group, "scored group size"),
    ];
    checks.iter().find(|(same, _)| !same).map(|(_, what)| *what)
}

fn depth(leaves: u64) -> u32 {
    leaves.max(1).next_power_of_two().trailing_zeros()
}

/// Room for the longest proof a dispute over this run can need: a full leaf,
/// a vector root and the two paths.
pub fn proof_capacity(run: &RunView) -> u32 {
    let hashes = 1 + depth(u64::from(run.terms.sites)) + depth(run.terms.leaves);
    LEAF_LEN as u32 + 32 * hashes
}

fn window_end(task: &TaskView, round: &RoundView) -> u64 {
    round.posted_slot.saturating_add(task.challenge_slots)
}

fn disputable(task: &TaskView, round: &RoundView, slot: u64) -> bool {
    round.state == ROUND_POSTED && (round.disputes > 0 || slot < window_end(task, round))
}

/// Whether `round_close` takes this round now.
fn closable(task: &TaskView, number: u32, round: &RoundView, slot: u64) -> bool {
    let released = !task.is_open() && task.stake_held == 0;
    if number >= task.round && !released {
        return false;
    }
    match round.state {
        ROUND_FINAL => {
            let window = task.challenge_slots.saturating_mul(CLAIM_WINDOWS);
            round.claimed == round.cohort || slot >= round.settled_slot.saturating_add(window)
        }
        _ => round.disputes == 0,
    }
}

fn known<'a>(
    task: &'a Option<TaskView>,
    run: &RunView,
) -> Result<Option<&'a TaskView>, Divergence> {
    match task {
        None => Ok(None),
        Some(task) => match terms_differ(task, run) {
            Some(what) => Err(Divergence::Terms(what)),
            None => Ok(Some(task)),
        },
    }
}

fn first_difference(number: u32, round: &RoundView, log: &RoundLog) -> Result<(), Divergence> {
    match differences(round, log).first() {
        Some(what) => Err(Divergence::Round {
            round: number,
            what: *what,
        }),
        None => Ok(()),
    }
}

fn publisher(run: &RunView, ledger: &Snapshot) -> Result<Step, Divergence> {
    let Some(task) = known(&ledger.task, run)? else {
        return Ok(Step::Publish);
    };
    if task.is_open() {
        return Ok(Step::Wait(Wait::Run));
    }
    let locked = ledger
        .task_rent
        .saturating_add(task.owed)
        .saturating_add(task.stake_held);
    if task.lamports > locked {
        return Ok(Step::WithdrawPublisher);
    }
    if task.owed == 0 {
        return Ok(Step::Done);
    }
    // What sites never claim comes back once their rounds are closed. The
    // aggregator gets the rent, so it would do this too, but it may be gone.
    let expired = ledger
        .rounds
        .iter()
        .find(|(number, round)| closable(task, **number, round, ledger.slot));
    Ok(match expired {
        Some((number, _)) => Step::CloseRound(*number),
        None => Step::Wait(Wait::Claims),
    })
}

fn aggregator(run: &RunView, ledger: &Snapshot) -> Result<Step, Divergence> {
    let Some(task) = known(&ledger.task, run)? else {
        return Ok(Step::Wait(Wait::Task));
    };
    let latest = ledger.rounds.get(&task.round);
    let capacity = proof_capacity(run);
    let unanswered = ledger
        .demands
        .iter()
        .filter(|demand| {
            !demand.answered
                && ledger.slot <= demand.deadline
                && ledger
                    .rounds
                    .get(&demand.round)
                    .is_some_and(|round| round.state == ROUND_POSTED)
        })
        .min_by_key(|demand| demand.deadline);
    if let Some(demand) = unanswered {
        return Ok(match ledger.buffer {
            None => Step::OpenBuffer(capacity),
            Some(buffer) if buffer.capacity < capacity => Step::CloseBuffer,
            Some(_) => Step::AnswerDemand {
                round: demand.round,
                part: demand.part,
                leaf: demand.leaf,
                address: demand.address,
            },
        });
    }
    if ledger.buffer.is_some() {
        return Ok(Step::CloseBuffer);
    }
    let spent = ledger
        .rounds
        .iter()
        .find(|(number, round)| closable(task, **number, round, ledger.slot));
    if let Some((number, _)) = spent {
        return Ok(Step::CloseRound(*number));
    }
    match task.state {
        TASK_FINISHED if task.stake_held > 0 => return Ok(Step::WithdrawAggregator),
        TASK_ABORTED if task.stake_held > 0 && !task.fraud => {
            return Ok(match latest {
                Some(round) if disputable(task, round, ledger.slot) => match round.disputes {
                    0 => Step::Wait(Wait::Window(window_end(task, round))),
                    _ => Step::Wait(Wait::Dispute),
                },
                _ => Step::WithdrawAggregator,
            });
        }
        TASK_FINISHED | TASK_ABORTED => {
            return Ok(match ledger.rounds.is_empty() {
                true => Step::Done,
                false => Step::Wait(Wait::Claims),
            });
        }
        _ => {}
    }
    if !task.backed {
        return Ok(Step::Back);
    }
    // Nothing more comes from a log that was aborted, so the task is too.
    let behind = match run.ending {
        Ending::Aborted => Step::Abort,
        _ => Step::Wait(Wait::Log),
    };
    if task.settled == task.round {
        let next = task.round + 1;
        let closed = run.round(next).is_some_and(|log| log.committed.is_some());
        return Ok(match closed {
            true => Step::OpenRound(next),
            false => behind,
        });
    }
    let number = task.round;
    let (Some(round), Some(log)) = (latest, run.round(number)) else {
        return match latest {
            Some(_) => Err(Divergence::Ahead(number)),
            None => Ok(Step::Wait(Wait::Task)),
        };
    };
    first_difference(number, round, log)?;
    Ok(match round.state {
        ROUND_COLLECTING => match &log.committed {
            None => behind,
            Some(committed) => {
                let missing = committed
                    .cohort
                    .iter()
                    .map(|site| *site as u16)
                    .find(|site| !task.enrolled.has(*site));
                let short = committed
                    .cohort
                    .iter()
                    .any(|site| !log.acks.contains(&(*site as u16)));
                match missing {
                    Some(_) if never_coming(task, run, ledger, short) => Step::Abort,
                    Some(site) => Step::Wait(Wait::Enrollment(site)),
                    None => Step::CommitRoot(number),
                }
            }
        },
        ROUND_COMMITTED if log.voided => Step::VoidRound {
            round: number,
            remove_helper: log.removed_helper,
        },
        ROUND_COMMITTED => match (&log.committed, &log.posted) {
            (None, _) => return Err(Divergence::Ahead(number)),
            (_, None) => behind,
            (_, Some(_)) => Step::PostAggregate(number),
        },
        ROUND_POSTED if log.posted.is_none() => return Err(Divergence::Ahead(number)),
        ROUND_POSTED if round.disputes > 0 => Step::Wait(Wait::Dispute),
        ROUND_POSTED if round.acked != round.cohort => {
            let short = round.cohort.members().iter().any(|s| !log.acks.contains(s));
            match never_coming(task, run, ledger, short) {
                true => Step::Abort,
                false => Step::Wait(Wait::Acks),
            }
        }
        ROUND_POSTED if ledger.slot < window_end(task, round) => {
            Step::Wait(Wait::Window(window_end(task, round)))
        }
        ROUND_POSTED => Step::Settle(number),
        _ => Step::Done,
    })
}

/// Whether what a round is waiting for from its sites can no longer arrive. A
/// site acts here only on what it did in its log, so an aborted log that is
/// `short` of acks will never bring them. Once a log is over, a site still quiet
/// after `stall_slots` more is gone.
fn never_coming(task: &TaskView, run: &RunView, ledger: &Snapshot, short: bool) -> bool {
    let quiet = task.last_change.saturating_add(task.stall_slots);
    let waited = ledger.slot > quiet
        && ledger
            .log_ended
            .is_some_and(|at| ledger.slot > at.saturating_add(task.stall_slots));
    match run.ending {
        Ending::Running => false,
        Ending::Finished => waited,
        Ending::Aborted => short || waited,
    }
}

fn site(me: u16, wallet: &[u8; 32], run: &RunView, ledger: &Snapshot) -> Result<Step, Divergence> {
    let Some(task) = known(&ledger.task, run)? else {
        return Ok(Step::Wait(Wait::Task));
    };
    let Some(enrollment) = ledger.enrollments.get(&me) else {
        return Ok(match (task.is_open(), run.joins.contains_key(&me)) {
            (false, _) => Step::Done,
            (true, true) => Step::Enroll,
            (true, false) => Step::Wait(Wait::Log),
        });
    };
    if &enrollment.wallet != wallet {
        return Err(Divergence::Enrollment(me));
    }
    let mut waiting = Wait::Run;
    for (number, round) in &ledger.rounds {
        if !round.cohort.has(me) {
            continue;
        }
        match round.state {
            ROUND_FINAL if !round.claimed.has(me) => return Ok(Step::Claim(*number)),
            ROUND_POSTED if !round.acked.has(me) => {
                // A site acknowledges here only what it acknowledged in its
                // own log, and only if the ledger holds that very round.
                let checked = run.round(*number).filter(|log| {
                    log.committed.is_some() && log.posted.is_some() && log.acks.contains(&me)
                });
                waiting = match checked.map(|log| differences(round, log)) {
                    None => Wait::Log,
                    Some(found) => match found.first() {
                        None => return Ok(Step::Ack(*number)),
                        Some(what) => Wait::Refusing {
                            round: *number,
                            what: *what,
                        },
                    },
                };
            }
            _ => {}
        }
    }
    let released = !task.is_open() || enrollment.status == SITE_REVOKED;
    Ok(match enrollment.status {
        SITE_WITHDRAWN if !task.is_open() => Step::Done,
        SITE_WITHDRAWN => Step::Wait(waiting),
        _ if released => Step::WithdrawSite,
        _ => Step::Wait(waiting),
    })
}

fn next_part(dispute: &DisputeView, cohort: usize, helpers: u8) -> Option<ProofPart> {
    let input = (0..cohort as u16).find(|position| !dispute.seen_inputs.has(*position));
    let mask = (0..helpers).find(|helper| dispute.seen_masks & (1 << helper) == 0);
    input
        .map(ProofPart::Input)
        .or(mask.map(ProofPart::Mask))
        .or((!dispute.aggregate_given).then_some(ProofPart::Aggregate))
}

fn watcher(run: &RunView, ledger: &Snapshot, notes: &Notes) -> Result<Step, Divergence> {
    let Some(task) = known(&ledger.task, run)? else {
        return Ok(Step::Wait(Wait::Task));
    };
    let revoked = run.revoked.iter().find(|site| {
        task.enrolled.has(**site)
            && !task.revoked.has(**site)
            && notes
                .checked
                .get(site)
                .is_none_or(|slot| ledger.slot >= slot + RECHECK_SLOTS)
    });
    if let Some(site) = revoked {
        return Ok(Step::RevokeCheck(*site));
    }

    let number = task.round;
    let round = ledger.rounds.get(&number);
    let capacity = proof_capacity(run);
    let roomy = |buffer: &BufferView| buffer.capacity >= capacity;
    if let (Some(dispute), Some(round)) = (&ledger.dispute, round) {
        let part = next_part(dispute, round.cohort.len(), task.active_helpers());
        let over = round.state == ROUND_FRAUD || ledger.slot > dispute.deadline;
        return Ok(match (part, ledger.buffer) {
            (None, _) => Step::CloseDispute(number),
            (Some(_), _) if over => Step::CloseDispute(number),
            (Some(_), None) => Step::OpenBuffer(capacity),
            (Some(_), Some(buffer)) if !roomy(&buffer) => Step::CloseBuffer,
            (Some(part), Some(_)) => Step::GiveLeaf {
                round: number,
                leaf: dispute.leaf_index,
                part,
            },
        });
    }

    let posted = round.filter(|round| round.state == ROUND_POSTED);
    let open = posted.filter(|round| ledger.slot < window_end(task, round));
    if let Some(step) = demanding(number, posted, open.is_some(), ledger, notes) {
        return Ok(step);
    }
    let answered = |leaf: Option<u64>| {
        ledger.demands.iter().find(|demand| {
            demand.round == number
                && demand.answered
                && ledger.slot <= demand.deadline
                && leaf.is_none_or(|leaf| demand.leaf == leaf)
        })
    };
    let verdict = notes.verdicts.get(&number).copied();
    let live = posted.is_some() && (open.is_some() || answered(None).is_some());
    if live && !notes.disputed.contains(&number) {
        match (verdict, ledger.buffer) {
            (None, _) => return Ok(Step::Examine(number)),
            (Some(Verdict::Wrong(leaf)), buffer)
                if open.is_some() || answered(Some(leaf)).is_some() =>
            {
                match buffer {
                    None => return Ok(Step::OpenBuffer(capacity)),
                    Some(buffer) if !roomy(&buffer) => return Ok(Step::CloseBuffer),
                    Some(_) => {
                        return Ok(Step::OpenDispute {
                            round: number,
                            leaf,
                            after: match open {
                                Some(_) => None,
                                None => answered(Some(leaf)).map(|demand| demand.part),
                            },
                        });
                    }
                }
            }
            (Some(_), _) => {}
        }
    }
    if let Some(demand) = ledger.demands.iter().find(|demand| demand.answered) {
        return Ok(Step::CloseDemand {
            round: demand.round,
            part: demand.part,
            leaf: demand.leaf,
        });
    }
    if ledger.buffer.is_some() {
        return Ok(Step::CloseBuffer);
    }
    Ok(match (task.is_open(), open) {
        (false, _) => Step::Done,
        (true, Some(round)) => Step::Wait(Wait::Window(window_end(task, round))),
        (true, None) => Step::Wait(Wait::Run),
    })
}

/// What a watcher does about the leaves it had to ask for on the ledger.
fn demanding(
    number: u32,
    posted: Option<&RoundView>,
    open: bool,
    ledger: &Snapshot,
    notes: &Notes,
) -> Option<Step> {
    // A leaf proven wrong needs no more data, and the window may be closing.
    let settled = matches!(notes.verdicts.get(&number), Some(Verdict::Wrong(_)));
    let mut pending = false;
    for demand in ledger.demands.iter().filter(|demand| !demand.answered) {
        if ledger.slot > demand.deadline || (demand.round == number && posted.is_none()) {
            return Some(Step::CloseDemand {
                round: demand.round,
                part: demand.part,
                leaf: demand.leaf,
            });
        }
        pending = true;
    }
    for wanted in notes.wanted.get(&number).into_iter().flatten() {
        let (part, leaf) = (wanted.part, wanted.leaf);
        let demand = ledger
            .demands
            .iter()
            .find(|demand| demand.round == number && demand.part == part && demand.leaf == leaf);
        let round = number;
        match demand {
            None if !wanted.opened && !settled && open => {
                return Some(Step::OpenDemand { round, part, leaf });
            }
            Some(demand) if demand.answered && !wanted.taken => {
                return Some(Step::TakeAnswer { round, part, leaf });
            }
            _ => {}
        }
    }
    (pending && !settled).then_some(Step::Wait(Wait::Dispute))
}

fn issuer(run: &RunView, ledger: &Snapshot, notes: &Notes) -> Step {
    for (list, signed) in &notes.signed {
        let Some(held) = ledger.lists.get(list) else {
            return Step::InitList {
                list: *list,
                bits: signed.bits,
            };
        };
        let missing = signed.set.iter().find(|bit| !held.set.contains(bit));
        if let Some(bit) = missing {
            return Step::SetBit {
                list: *list,
                bit: *bit,
            };
        }
    }
    match run.ending {
        Ending::Running => Step::Wait(Wait::Run),
        _ => Step::Done,
    }
}

/// The one thing `party` owes next. `wallet` is its own address.
pub fn owed(
    party: Party,
    wallet: &[u8; 32],
    run: &RunView,
    ledger: &Snapshot,
    notes: &Notes,
) -> Result<Step, Divergence> {
    match party {
        Party::Publisher => publisher(run, ledger),
        Party::Aggregator => aggregator(run, ledger),
        Party::Site(me) => site(me, wallet, run, ledger),
        Party::Watcher => watcher(run, ledger, notes),
        Party::Issuer => Ok(issuer(run, ledger, notes)),
    }
}

#[cfg(test)]
mod tests {
    use fedtally_core::anchor::{Committed, Posted, Terms};
    use fedtally_core::vector::Digest;
    use fedtally_ledger::state::{SITE_ACTIVE, TASK_OPEN};

    use super::*;

    const ME: [u8; 32] = [0xaa; 32];
    const WINDOW: u64 = 50;

    fn run(sites: u16, rounds: u32) -> RunView {
        RunView {
            terms: Terms {
                plan: Digest([1; 32]),
                sites_root: [2; 32],
                sites,
                helpers: 1,
                coords: 300,
                leaves: 3,
                rounds,
                min_cohort: 2,
                sample: 0,
                sensitivity: 0,
                noise: 0,
                colluding: 0,
                budget: 0,
                model: Digest([3; 32]),
                retries: 0,
                min_helpers: 0,
                score_group: 0,
            },
            publisher: [4; 32],
            aggregator: [5; 32],
            wallets: (0..sites).map(|site| [site as u8 + 100; 32]).collect(),
            joins: BTreeMap::new(),
            status_refs: BTreeMap::new(),
            rounds: Vec::new(),
            revoked: BTreeSet::new(),
            ending: Ending::Running,
        }
    }

    fn task(run: &RunView) -> TaskView {
        TaskView {
            lamports: 1_000,
            state: TASK_OPEN,
            backed: true,
            helpers: run.terms.helpers,
            sites: run.terms.sites,
            min_cohort: run.terms.min_cohort,
            rounds: run.terms.rounds,
            publisher: run.publisher,
            aggregator: run.aggregator,
            plan_id: run.terms.plan.0,
            sites_root: run.terms.sites_root,
            coords: run.terms.coords,
            stake_held: 500,
            challenge_slots: WINDOW,
            enrolled: (0..run.terms.sites).collect(),
            ..TaskView::default()
        }
    }

    /// A round the log has sealed and every member acknowledged.
    fn sealed(number: u32, cohort: &[usize]) -> RoundLog {
        let tag = number as u8;
        RoundLog {
            committed: Some(Committed {
                round: number,
                beacon: [tag; 32],
                cohort: cohort.to_vec(),
                inputs_root: [tag + 10; 32],
                close_head: Digest([tag + 20; 32]),
                threshold: [0; 32],
            }),
            posted: Some(Posted {
                round: number,
                mask_roots: vec![Digest([tag + 30; 32])],
                aggregate_root: Digest([tag + 40; 32]),
                next_model: Digest([tag + 50; 32]),
                seal_head: Digest([tag + 60; 32]),
                shares: Vec::new(),
                shares_root: [0; 32],
            }),
            vectors: cohort
                .iter()
                .map(|site| (*site as u16, Digest([*site as u8; 32])))
                .collect(),
            masks: vec![Digest([tag + 30; 32])],
            active: vec![0],
            aggregate: Some(Digest([tag + 40; 32])),
            acks: cohort.iter().map(|site| *site as u16).collect(),
            voided: false,
            removed_helper: false,
        }
    }

    /// The ledger's round as an honest aggregator posts `log`, up to `state`.
    fn mirrored(log: &RoundLog, state: u8, posted_slot: u64) -> RoundView {
        let committed = log.committed.as_ref().unwrap();
        let posted = log.posted.as_ref().unwrap();
        let mut round = RoundView {
            state,
            beacon: committed.beacon,
            ..RoundView::default()
        };
        if state != ROUND_COLLECTING {
            round.cohort = committed.cohort.iter().map(|s| *s as u16).collect();
            round.inputs_root = committed.inputs_root;
            round.close_head = committed.close_head.0;
        }
        if !matches!(state, ROUND_COLLECTING | ROUND_COMMITTED) {
            round.mask_roots = posted.mask_roots.iter().map(|m| m.0).collect();
            round.aggregate_root = posted.aggregate_root.0;
            round.next_model = posted.next_model.0;
            round.seal_head = posted.seal_head.0;
            round.shares_root = posted.shares_root;
            round.posted_slot = posted_slot;
        }
        round
    }

    fn snapshot(task: TaskView, slot: u64) -> Snapshot {
        Snapshot {
            slot,
            task: Some(task),
            ..Snapshot::default()
        }
    }

    fn decide(party: Party, run: &RunView, ledger: &Snapshot) -> Result<Step, Divergence> {
        owed(party, &ME, run, ledger, &Notes::default())
    }

    #[test]
    fn a_publisher_publishes_once_and_takes_back_what_is_left() {
        let run = run(3, 2);
        let empty = Snapshot::default();
        assert_eq!(decide(Party::Publisher, &run, &empty), Ok(Step::Publish));

        let mut ledger = snapshot(task(&run), 10);
        ledger.task_rent = 300;
        let waiting = Ok(Step::Wait(Wait::Run));
        assert_eq!(decide(Party::Publisher, &run, &ledger), waiting);

        let held = ledger.task.as_mut().unwrap();
        held.state = TASK_FINISHED;
        held.owed = 100;
        let withdraw = Ok(Step::WithdrawPublisher);
        assert_eq!(decide(Party::Publisher, &run, &ledger), withdraw);
        // Rent, unclaimed rewards and the aggregator's stake are not the publisher's.
        ledger.task.as_mut().unwrap().lamports = 300 + 100 + 500;
        let waiting = Ok(Step::Wait(Wait::Claims));
        assert_eq!(decide(Party::Publisher, &run, &ledger), waiting);
        ledger.task.as_mut().unwrap().owed = 0;
        ledger.task.as_mut().unwrap().lamports = 300 + 500;
        assert_eq!(decide(Party::Publisher, &run, &ledger), Ok(Step::Done));
    }

    #[test]
    fn nobody_acts_on_a_task_with_other_terms() {
        let run = run(3, 2);
        let mut held = task(&run);
        held.rounds = 9;
        let ledger = snapshot(held, 0);
        let parties = [
            Party::Publisher,
            Party::Aggregator,
            Party::Site(0),
            Party::Watcher,
        ];
        for party in parties {
            let refused = decide(party, &run, &ledger);
            let terms = Err(Divergence::Terms("number of rounds"));
            assert_eq!(refused, terms, "{party:?}");
        }
        let mut held = task(&run);
        held.aggregator = [9; 32];
        let refused = decide(Party::Site(0), &run, &snapshot(held, 0));
        assert_eq!(refused, Err(Divergence::Terms("aggregator")));
    }

    #[test]
    fn an_aggregator_takes_a_round_from_open_to_settled() {
        let mut run = run(3, 2);
        let aggregator = |run: &RunView, ledger: &Snapshot| decide(Party::Aggregator, run, ledger);
        let nothing = Snapshot::default();
        assert_eq!(aggregator(&run, &nothing), Ok(Step::Wait(Wait::Task)));

        let mut held = task(&run);
        held.backed = false;
        assert_eq!(aggregator(&run, &snapshot(held, 0)), Ok(Step::Back));

        // The ledger opens a round only once the log has fixed its cohort.
        let mut ledger = snapshot(task(&run), 0);
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Wait(Wait::Log)));
        let log = sealed(1, &[0, 2]);
        run.rounds.push(RoundLog {
            committed: None,
            posted: None,
            ..log.clone()
        });
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Wait(Wait::Log)));
        run.rounds[0].committed = log.committed.clone();
        assert_eq!(aggregator(&run, &ledger), Ok(Step::OpenRound(1)));

        ledger.task.as_mut().unwrap().round = 1;
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Wait(Wait::Task)));
        ledger.rounds.insert(1, mirrored(&log, ROUND_COLLECTING, 0));
        ledger.task.as_mut().unwrap().enrolled = [0, 1].into_iter().collect();
        let enrollment = Ok(Step::Wait(Wait::Enrollment(2)));
        assert_eq!(aggregator(&run, &ledger), enrollment);
        ledger.task.as_mut().unwrap().enrolled = [0, 1, 2].into_iter().collect();
        assert_eq!(aggregator(&run, &ledger), Ok(Step::CommitRoot(1)));

        ledger.rounds.insert(1, mirrored(&log, ROUND_COMMITTED, 0));
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Wait(Wait::Log)));
        run.rounds[0] = log.clone();
        assert_eq!(aggregator(&run, &ledger), Ok(Step::PostAggregate(1)));

        ledger.rounds.insert(1, mirrored(&log, ROUND_POSTED, 100));
        ledger.slot = 100 + WINDOW;
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Wait(Wait::Acks)));
        ledger.rounds.get_mut(&1).unwrap().acked = [0, 2].into_iter().collect();
        ledger.slot = 100 + WINDOW - 1;
        let end = Wait::Window(100 + WINDOW);
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Wait(end)));
        ledger.slot = 100 + WINDOW;
        ledger.rounds.get_mut(&1).unwrap().disputes = 1;
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Wait(Wait::Dispute)));
        ledger.rounds.get_mut(&1).unwrap().disputes = 0;
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Settle(1)));

        ledger.task.as_mut().unwrap().settled = 1;
        ledger.rounds.insert(1, mirrored(&log, ROUND_FINAL, 100));
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Wait(Wait::Log)));
        run.rounds.push(sealed(2, &[0, 1, 2]));
        assert_eq!(aggregator(&run, &ledger), Ok(Step::OpenRound(2)));

        let held = ledger.task.as_mut().unwrap();
        (held.round, held.settled, held.state) = (2, 2, TASK_FINISHED);
        assert_eq!(aggregator(&run, &ledger), Ok(Step::WithdrawAggregator));
        ledger.task.as_mut().unwrap().stake_held = 0;
        let unclaimed = Ok(Step::Wait(Wait::Claims));
        assert_eq!(aggregator(&run, &ledger), unclaimed);
        ledger.rounds.clear();
        assert_eq!(aggregator(&run, &ledger), Ok(Step::Done));
    }

    #[test]
    fn an_aggregator_stops_when_the_ledger_is_not_its_log() {
        let mut run = run(3, 2);
        let log = sealed(1, &[0, 1, 2]);
        run.rounds.push(log.clone());
        let mut held = task(&run);
        held.round = 1;
        let mut ledger = snapshot(held, 0);

        let mut round = mirrored(&log, ROUND_COLLECTING, 0);
        round.beacon = [0xee; 32];
        ledger.rounds.insert(1, round);
        let differs = |what| Err(Divergence::Round { round: 1, what });
        let found = decide(Party::Aggregator, &run, &ledger);
        assert_eq!(found, differs(Differs::Beacon));

        let mut round = mirrored(&log, ROUND_COMMITTED, 0);
        round.inputs_root = [0xee; 32];
        ledger.rounds.insert(1, round);
        let found = decide(Party::Aggregator, &run, &ledger);
        assert_eq!(found, differs(Differs::InputsRoot));

        ledger.rounds.insert(1, mirrored(&log, ROUND_POSTED, 0));
        run.rounds.clear();
        let found = decide(Party::Aggregator, &run, &ledger);
        assert_eq!(found, Err(Divergence::Ahead(1)));
    }

    #[test]
    fn an_aborted_log_aborts_the_task_and_the_stake_waits_out_the_window() {
        let mut run = run(3, 2);
        let log = sealed(1, &[0, 1, 2]);
        run.rounds.push(log.clone());
        run.ending = Ending::Aborted;
        let mut held = task(&run);
        (held.round, held.settled) = (1, 1);
        let mut ledger = snapshot(held, 500);
        ledger.rounds.insert(1, mirrored(&log, ROUND_FINAL, 100));
        let aggregator = |ledger: &Snapshot| decide(Party::Aggregator, &run, ledger);
        assert_eq!(aggregator(&ledger), Ok(Step::Abort));

        // A round the log sealed is still settled before anything is aborted.
        ledger.task.as_mut().unwrap().settled = 0;
        let mut posted = mirrored(&log, ROUND_POSTED, 100);
        posted.acked = posted.cohort;
        ledger.rounds.insert(1, posted);
        assert_eq!(aggregator(&ledger), Ok(Step::Settle(1)));

        ledger.task.as_mut().unwrap().state = TASK_ABORTED;
        ledger.slot = 120;
        let end = Wait::Window(100 + WINDOW);
        assert_eq!(aggregator(&ledger), Ok(Step::Wait(end)));
        ledger.slot = 100 + WINDOW;
        assert_eq!(aggregator(&ledger), Ok(Step::WithdrawAggregator));
        ledger.rounds.get_mut(&1).unwrap().disputes = 1;
        assert_eq!(aggregator(&ledger), Ok(Step::Wait(Wait::Dispute)));
        ledger.task.as_mut().unwrap().fraud = true;
        let holding = Ok(Step::Wait(Wait::Claims));
        ledger.task.as_mut().unwrap().stake_held = 0;
        assert_eq!(aggregator(&ledger), holding);
        ledger.rounds.get_mut(&1).unwrap().disputes = 0;
        assert_eq!(aggregator(&ledger), Ok(Step::CloseRound(1)));
    }

    fn enrolled(wallet: [u8; 32]) -> EnrollmentView {
        EnrollmentView {
            status: SITE_ACTIVE,
            wallet,
            status_lists: Vec::new(),
        }
    }

    #[test]
    fn a_site_enrolls_acknowledges_claims_and_withdraws() {
        let mut run = run(3, 2);
        let site = |run: &RunView, ledger: &Snapshot| decide(Party::Site(1), run, ledger);
        let mut ledger = snapshot(task(&run), 0);
        assert_eq!(site(&run, &ledger), Ok(Step::Wait(Wait::Log)));
        run.joins.insert(1, Digest([7; 32]));
        assert_eq!(site(&run, &ledger), Ok(Step::Enroll));

        ledger.enrollments.insert(1, enrolled([9; 32]));
        assert_eq!(site(&run, &ledger), Err(Divergence::Enrollment(1)));
        ledger.enrollments.insert(1, enrolled(ME));
        assert_eq!(site(&run, &ledger), Ok(Step::Wait(Wait::Run)));

        // Round 1 leaves this site out, round 2 has it.
        let (first, second) = (sealed(1, &[0, 2]), sealed(2, &[0, 1]));
        run.rounds.push(first.clone());
        ledger.rounds.insert(1, mirrored(&first, ROUND_POSTED, 10));
        assert_eq!(site(&run, &ledger), Ok(Step::Wait(Wait::Run)));

        ledger.rounds.insert(1, mirrored(&first, ROUND_FINAL, 10));
        ledger.rounds.insert(2, mirrored(&second, ROUND_POSTED, 90));
        assert_eq!(site(&run, &ledger), Ok(Step::Wait(Wait::Log)));
        run.rounds.push(RoundLog {
            acks: [0].into_iter().collect(),
            ..second.clone()
        });
        let early = site(&run, &ledger);
        assert_eq!(early, Ok(Step::Wait(Wait::Log)), "acked before its log did");
        run.rounds[1] = second.clone();
        assert_eq!(site(&run, &ledger), Ok(Step::Ack(2)));

        ledger.rounds.get_mut(&2).unwrap().acked = [0, 1].into_iter().collect();
        assert_eq!(site(&run, &ledger), Ok(Step::Wait(Wait::Run)));
        ledger.rounds.get_mut(&2).unwrap().state = ROUND_FINAL;
        assert_eq!(site(&run, &ledger), Ok(Step::Claim(2)));
        ledger.rounds.get_mut(&2).unwrap().claimed = [1].into_iter().collect();
        assert_eq!(site(&run, &ledger), Ok(Step::Wait(Wait::Run)));

        ledger.task.as_mut().unwrap().state = TASK_FINISHED;
        assert_eq!(site(&run, &ledger), Ok(Step::WithdrawSite));
        ledger.enrollments.get_mut(&1).unwrap().status = SITE_WITHDRAWN;
        assert_eq!(site(&run, &ledger), Ok(Step::Done));
    }

    #[test]
    fn a_site_refuses_a_round_that_is_not_the_one_in_its_log() {
        let mut run = run(3, 1);
        let log = sealed(1, &[0, 1, 2]);
        run.rounds.push(log.clone());
        let mut ledger = snapshot(task(&run), 0);
        ledger.enrollments.insert(0, enrolled(ME));
        let mut round = mirrored(&log, ROUND_POSTED, 10);
        round.aggregate_root = [0xee; 32];
        ledger.rounds.insert(1, round);
        let refusing = Ok(Step::Wait(Wait::Refusing {
            round: 1,
            what: Differs::AggregateRoot,
        }));
        let site = |ledger: &Snapshot| decide(Party::Site(0), &run, ledger);
        assert_eq!(site(&ledger), refusing);

        // Its stake comes back all the same once the task is over or it is revoked.
        ledger.enrollments.get_mut(&0).unwrap().status = SITE_REVOKED;
        assert_eq!(site(&ledger), Ok(Step::WithdrawSite));
        ledger.enrollments.get_mut(&0).unwrap().status = SITE_WITHDRAWN;
        assert_eq!(site(&ledger), refusing);
        ledger.task.as_mut().unwrap().state = TASK_ABORTED;
        assert_eq!(site(&ledger), Ok(Step::Done));
    }

    #[test]
    fn a_site_whose_log_never_acknowledged_still_gets_its_stake_back() {
        let mut run = run(3, 1);
        let log = sealed(1, &[0, 1, 2]);
        run.rounds.push(RoundLog {
            acks: BTreeSet::new(),
            ..log.clone()
        });
        let mut ledger = snapshot(task(&run), 0);
        ledger.enrollments.insert(0, enrolled(ME));
        ledger.rounds.insert(1, mirrored(&log, ROUND_POSTED, 10));
        let site = |ledger: &Snapshot| decide(Party::Site(0), &run, ledger);
        assert_eq!(site(&ledger), Ok(Step::Wait(Wait::Log)));
        ledger.task.as_mut().unwrap().state = TASK_ABORTED;
        assert_eq!(site(&ledger), Ok(Step::WithdrawSite));

        let mut gone = snapshot(task(&run), 0);
        gone.task.as_mut().unwrap().state = TASK_ABORTED;
        assert_eq!(site(&gone), Ok(Step::Done));
    }

    fn demand(part: ProofPart, leaf: u64, answered: bool, deadline: u64) -> DemandView {
        DemandView {
            address: [9; 32],
            challenger: ME,
            round: 1,
            part,
            leaf,
            answered,
            deadline,
            proof: Vec::new(),
        }
    }

    #[test]
    fn a_watcher_demands_what_it_cannot_fetch_and_takes_each_answer() {
        let (run, mut ledger) = watched();
        let part = ProofPart::Aggregate;
        let mut notes = Notes::default();
        let want = Wanted {
            part,
            leaf: 1,
            opened: false,
            taken: false,
        };
        notes.wanted.insert(1, vec![want]);
        let watcher =
            |ledger: &Snapshot, notes: &Notes| owed(Party::Watcher, &ME, &run, ledger, notes);
        let open = Step::OpenDemand {
            round: 1,
            part,
            leaf: 1,
        };
        assert_eq!(watcher(&ledger, &notes), Ok(open));

        notes.wanted.get_mut(&1).unwrap()[0].opened = true;
        ledger.demands = vec![demand(part, 1, false, 200)];
        assert_eq!(watcher(&ledger, &notes), Ok(Step::Wait(Wait::Dispute)));

        let close = Step::CloseDemand {
            round: 1,
            part,
            leaf: 1,
        };
        ledger.slot = 201;
        assert_eq!(watcher(&ledger, &notes), Ok(close));
        ledger.slot = 110;

        ledger.demands = vec![demand(part, 1, true, 200)];
        let take = Step::TakeAnswer {
            round: 1,
            part,
            leaf: 1,
        };
        assert_eq!(watcher(&ledger, &notes), Ok(take));

        notes.wanted.get_mut(&1).unwrap()[0].taken = true;
        assert_eq!(watcher(&ledger, &notes), Ok(Step::Examine(1)));

        notes.verdicts.insert(1, Verdict::Sound);
        assert_eq!(watcher(&ledger, &notes), Ok(close));
    }

    #[test]
    fn an_answered_demand_carries_a_dispute_past_the_window() {
        let (run, mut ledger) = watched();
        let part = ProofPart::Input(0);
        let capacity = proof_capacity(&run);
        ledger.slot = 100 + WINDOW + 5;
        ledger.buffer = Some(BufferView { capacity });
        ledger.demands = vec![demand(part, 2, true, 100 + 2 * WINDOW)];
        let mut notes = with(Verdict::Wrong(2));
        let step = owed(Party::Watcher, &ME, &run, &ledger, &notes);
        assert_eq!(
            step,
            Ok(Step::OpenDispute {
                round: 1,
                leaf: 2,
                after: Some(part),
            })
        );
        // Wrong at a leaf nobody demanded, past the window: too late to prove.
        notes.verdicts.insert(1, Verdict::Wrong(1));
        let step = owed(Party::Watcher, &ME, &run, &ledger, &notes);
        assert_eq!(
            step,
            Ok(Step::CloseDemand {
                round: 1,
                part,
                leaf: 2,
            })
        );
    }

    #[test]
    fn the_aggregator_answers_demands_first_and_cleans_up_after() {
        let (run, mut ledger) = watched();
        let part = ProofPart::Mask(0);
        ledger.demands = vec![demand(part, 0, false, 200)];
        let notes = Notes::default();
        let aggregator =
            |ledger: &Snapshot| owed(Party::Aggregator, &[5; 32], &run, ledger, &notes);
        let capacity = proof_capacity(&run);
        assert_eq!(aggregator(&ledger), Ok(Step::OpenBuffer(capacity)));
        ledger.buffer = Some(BufferView { capacity: 1 });
        assert_eq!(aggregator(&ledger), Ok(Step::CloseBuffer));
        ledger.buffer = Some(BufferView { capacity });
        let answer = Step::AnswerDemand {
            round: 1,
            part,
            leaf: 0,
            address: [9; 32],
        };
        assert_eq!(aggregator(&ledger), Ok(answer));
        ledger.demands[0].answered = true;
        assert_eq!(aggregator(&ledger), Ok(Step::CloseBuffer));
    }

    #[test]
    fn a_paid_out_round_is_closed_unless_it_is_the_latest() {
        let (run, mut ledger) = watched();
        ledger.buffer = None;
        let log = run.round(1).unwrap().clone();
        let mut paid = mirrored(&log, ROUND_FINAL, 50);
        paid.claimed = paid.cohort;
        ledger.rounds.insert(1, paid.clone());
        let notes = Notes::default();
        let task_at = |round| {
            let mut ledger = ledger.clone();
            ledger.task.as_mut().unwrap().round = round;
            ledger.rounds.insert(round, paid.clone());
            ledger
        };
        let step = |ledger: &Snapshot| owed(Party::Aggregator, &[5; 32], &run, ledger, &notes);
        assert_eq!(step(&task_at(2)), Ok(Step::CloseRound(1)));
        assert_ne!(step(&task_at(1)), Ok(Step::CloseRound(1)));
    }

    #[test]
    fn an_issuer_mirrors_its_lists_and_stops_with_the_run() {
        let (mut run, mut ledger) = watched();
        let mut notes = Notes::default();
        notes.signed.insert(
            7,
            SignedList {
                bits: 64,
                set: vec![3, 9],
            },
        );
        let init = Step::InitList { list: 7, bits: 64 };
        let issuer = |run: &RunView, ledger: &Snapshot, notes: &Notes| {
            owed(Party::Issuer, &ME, run, ledger, notes)
        };
        assert_eq!(issuer(&run, &ledger, &notes), Ok(init));
        let mut held = StatusView {
            bits: 64,
            set: vec![3],
        };
        ledger.lists.insert(7, held.clone());
        let next = Step::SetBit { list: 7, bit: 9 };
        assert_eq!(issuer(&run, &ledger, &notes), Ok(next));
        held.set.push(9);
        ledger.lists.insert(7, held);
        assert_eq!(issuer(&run, &ledger, &notes), Ok(Step::Wait(Wait::Run)));
        run.ending = Ending::Finished;
        assert_eq!(issuer(&run, &ledger, &notes), Ok(Step::Done));
    }

    fn watched() -> (RunView, Snapshot) {
        let mut run = run(3, 2);
        let log = sealed(1, &[0, 2]);
        run.rounds.push(log.clone());
        let mut held = task(&run);
        held.round = 1;
        let mut ledger = snapshot(held, 110);
        ledger.rounds.insert(1, mirrored(&log, ROUND_POSTED, 100));
        (run, ledger)
    }

    fn with(verdict: Verdict) -> Notes {
        Notes {
            verdicts: [(1, verdict)].into_iter().collect(),
            ..Notes::default()
        }
    }

    #[test]
    fn a_watcher_examines_each_posted_round_while_it_can_be_disputed() {
        let (run, mut ledger) = watched();
        let watcher =
            |ledger: &Snapshot, notes: &Notes| owed(Party::Watcher, &ME, &run, ledger, notes);
        let fresh = Notes::default();
        assert_eq!(watcher(&ledger, &fresh), Ok(Step::Examine(1)));
        let end = Ok(Step::Wait(Wait::Window(100 + WINDOW)));
        assert_eq!(watcher(&ledger, &with(Verdict::Sound)), end);
        assert_eq!(watcher(&ledger, &with(Verdict::Unprovable)), end);

        ledger.slot = 100 + WINDOW;
        let idle = Ok(Step::Wait(Wait::Run));
        assert_eq!(watcher(&ledger, &fresh), idle);
        assert_eq!(watcher(&ledger, &with(Verdict::Wrong(2))), idle);

        ledger.slot = 110;
        ledger.rounds.get_mut(&1).unwrap().state = ROUND_COMMITTED;
        assert_eq!(watcher(&ledger, &fresh), idle);
        ledger.task.as_mut().unwrap().state = TASK_FINISHED;
        assert_eq!(watcher(&ledger, &fresh), Ok(Step::Done));
    }

    #[test]
    fn a_watcher_takes_a_wrong_leaf_through_the_whole_dispute() {
        let (run, mut ledger) = watched();
        let capacity = proof_capacity(&run);
        assert_eq!(capacity, 1024 + 32 * (1 + 2 + 2));
        let mut notes = with(Verdict::Wrong(2));
        let watcher =
            |ledger: &Snapshot, notes: &Notes| owed(Party::Watcher, &ME, &run, ledger, notes);

        assert_eq!(watcher(&ledger, &notes), Ok(Step::OpenBuffer(capacity)));
        ledger.buffer = Some(BufferView { capacity: 100 });
        assert_eq!(watcher(&ledger, &notes), Ok(Step::CloseBuffer));
        ledger.buffer = Some(BufferView { capacity });
        let open = Step::OpenDispute {
            round: 1,
            leaf: 2,
            after: None,
        };
        assert_eq!(watcher(&ledger, &notes), Ok(open));

        notes.disputed.insert(1);
        let mut dispute = DisputeView {
            leaf_index: 2,
            deadline: 110 + WINDOW,
            ..DisputeView::default()
        };
        let mut given = Vec::new();
        loop {
            ledger.dispute = Some(dispute.clone());
            let step = watcher(&ledger, &notes).unwrap();
            let Step::GiveLeaf {
                round: 1,
                leaf: 2,
                part,
            } = step
            else {
                assert_eq!(step, Step::CloseDispute(1));
                break;
            };
            given.push(part);
            match part {
                ProofPart::Input(position) => {
                    let mut seen = dispute.seen_inputs.members();
                    seen.push(position);
                    dispute.seen_inputs = seen.into_iter().collect();
                }
                ProofPart::Mask(helper) => dispute.seen_masks |= 1 << helper,
                ProofPart::Aggregate => dispute.aggregate_given = true,
            }
        }
        let parts = [
            ProofPart::Input(0),
            ProofPart::Input(1),
            ProofPart::Mask(0),
            ProofPart::Aggregate,
        ];
        assert_eq!(given, parts);

        // The dispute is decided: the round is a fraud and the task is over.
        ledger.dispute = None;
        ledger.rounds.get_mut(&1).unwrap().state = ROUND_FRAUD;
        ledger.task.as_mut().unwrap().state = TASK_ABORTED;
        assert_eq!(watcher(&ledger, &notes), Ok(Step::CloseBuffer));
        ledger.buffer = None;
        assert_eq!(watcher(&ledger, &notes), Ok(Step::Done));
    }

    #[test]
    fn a_dispute_asks_only_for_the_masks_a_round_has() {
        let (mut run, mut ledger) = watched();
        run.terms.helpers = 3;
        let task = ledger.task.as_mut().unwrap();
        (task.helpers, task.removed_helpers) = (3, 1);
        ledger.buffer = Some(BufferView {
            capacity: proof_capacity(&run),
        });
        let mut notes = with(Verdict::Wrong(2));
        notes.disputed.insert(1);
        let mut dispute = DisputeView {
            leaf_index: 2,
            deadline: 110 + WINDOW,
            seen_inputs: [0u16, 1].into_iter().collect(),
            ..DisputeView::default()
        };
        let mut given = Vec::new();
        for seen_masks in 0..=2u8 {
            dispute.seen_masks = (1 << seen_masks) - 1;
            ledger.dispute = Some(dispute.clone());
            let step = owed(Party::Watcher, &ME, &run, &ledger, &notes);
            let Ok(Step::GiveLeaf { part, .. }) = step else {
                panic!("{step:?}");
            };
            given.push(part);
        }
        let parts = [ProofPart::Mask(0), ProofPart::Mask(1), ProofPart::Aggregate];
        assert_eq!(given, parts);
    }

    #[test]
    fn a_dispute_that_cannot_go_on_is_closed() {
        let (run, mut ledger) = watched();
        let notes = with(Verdict::Wrong(0));
        let watcher = |ledger: &Snapshot| owed(Party::Watcher, &ME, &run, ledger, &notes);
        ledger.buffer = Some(BufferView { capacity: 4_000 });
        ledger.dispute = Some(DisputeView {
            deadline: 150,
            ..DisputeView::default()
        });
        let leaf = Step::GiveLeaf {
            round: 1,
            leaf: 0,
            part: ProofPart::Input(0),
        };
        assert_eq!(watcher(&ledger), Ok(leaf));
        ledger.buffer = None;
        let buffer = Step::OpenBuffer(proof_capacity(&run));
        assert_eq!(watcher(&ledger), Ok(buffer));

        ledger.slot = 151;
        assert_eq!(watcher(&ledger), Ok(Step::CloseDispute(1)));
        ledger.slot = 120;
        ledger.rounds.get_mut(&1).unwrap().state = ROUND_FRAUD;
        assert_eq!(watcher(&ledger), Ok(Step::CloseDispute(1)));
    }

    #[test]
    fn a_round_is_disputed_once() {
        let (run, mut ledger) = watched();
        ledger.buffer = Some(BufferView { capacity: 4_000 });
        let mut notes = with(Verdict::Wrong(0));
        notes.disputed.insert(1);
        let step = owed(Party::Watcher, &ME, &run, &ledger, &notes);
        assert_eq!(step, Ok(Step::CloseBuffer));
    }

    #[test]
    fn a_revocation_in_the_log_is_taken_to_the_ledger_until_it_lands() {
        let (mut run, mut ledger) = watched();
        run.revoked.insert(2);
        let mut notes = with(Verdict::Sound);
        let step = owed(Party::Watcher, &ME, &run, &ledger, &notes);
        assert_eq!(step, Ok(Step::RevokeCheck(2)));
        notes.checked.insert(2, ledger.slot);
        let waiting = Ok(Step::Wait(Wait::Window(100 + WINDOW)));
        assert_eq!(owed(Party::Watcher, &ME, &run, &ledger, &notes), waiting);

        // The issuer's list may have come in since, so it is tried again.
        ledger.slot += RECHECK_SLOTS;
        let again = owed(Party::Watcher, &ME, &run, &ledger, &notes);
        assert_eq!(again, Ok(Step::RevokeCheck(2)));
        ledger.slot -= RECHECK_SLOTS;

        notes.checked.clear();
        ledger.task.as_mut().unwrap().revoked = [2].into_iter().collect();
        assert_eq!(owed(Party::Watcher, &ME, &run, &ledger, &notes), waiting);
    }

    #[test]
    fn differences_cover_what_the_rounds_state_says_is_posted() {
        let log = sealed(1, &[0, 1]);
        for state in [ROUND_COLLECTING, ROUND_COMMITTED, ROUND_POSTED, ROUND_FINAL] {
            assert_eq!(differences(&mirrored(&log, state, 5), &log), []);
        }
        let mut round = mirrored(&log, ROUND_POSTED, 5);
        round.cohort = [0, 2].into_iter().collect();
        round.mask_roots = vec![[0; 32]];
        round.seal_head = [0; 32];
        round.shares_root = [9; 32];
        let found = differences(&round, &log);
        let all = [
            Differs::Cohort,
            Differs::MaskRoots,
            Differs::SealHead,
            Differs::SharesRoot,
        ];
        assert_eq!(found, all);
        // A committed round holds nothing yet where the log has the result.
        round.state = ROUND_COMMITTED;
        assert_eq!(differences(&round, &log), [Differs::Cohort]);
        assert_eq!(differences(&round, &RoundLog::default()), []);
    }

    #[test]
    fn a_leaf_proven_wrong_stops_the_demands_and_goes_to_dispute() {
        let mut run = run(2, 1);
        run.rounds = vec![sealed(1, &[0, 1])];
        let mut held = task(&run);
        held.round = 1;
        let mut ledger = snapshot(held, 10);
        ledger
            .rounds
            .insert(1, mirrored(&run.rounds[0], ROUND_POSTED, 5));
        ledger.buffer = Some(BufferView {
            capacity: proof_capacity(&run),
        });
        ledger.demands = vec![DemandView {
            round: 1,
            part: ProofPart::Input(0),
            leaf: 2,
            answered: false,
            deadline: 100,
            address: [0; 32],
            challenger: ME,
            proof: Vec::new(),
        }];
        let want = |leaf| Wanted {
            part: ProofPart::Input(0),
            leaf,
            opened: false,
            taken: false,
        };
        let mut notes = Notes::default();
        notes.wanted.insert(1, vec![want(2), want(1)]);

        // Still looking: another leaf is asked for.
        let asking = owed(Party::Watcher, &ME, &run, &ledger, &notes);
        let another = Step::OpenDemand {
            round: 1,
            part: ProofPart::Input(0),
            leaf: 1,
        };
        assert_eq!(asking, Ok(another));

        // Found: the demand still open doesn't hold up the dispute.
        notes.verdicts.insert(1, Verdict::Wrong(0));
        let found = owed(Party::Watcher, &ME, &run, &ledger, &notes);
        let dispute = Step::OpenDispute {
            round: 1,
            leaf: 0,
            after: None,
        };
        assert_eq!(found, Ok(dispute));
    }

    #[test]
    fn a_log_that_is_over_stops_the_wait_for_acks_that_cannot_come() {
        let mut run = run(3, 2);
        run.rounds.push(sealed(1, &[0, 1, 2]));
        let mut held = task(&run);
        (held.round, held.stall_slots, held.last_change) = (1, 1_000, 100);
        let mut ledger = snapshot(held, 200);
        let mut posted = mirrored(&run.rounds[0], ROUND_POSTED, 100);
        posted.acked = [0, 1].into_iter().collect();
        ledger.rounds.insert(1, posted);
        let waiting = Ok(Step::Wait(Wait::Acks));

        // Every ack is in the log, so the last one may still arrive here.
        assert_eq!(decide(Party::Aggregator, &run, &ledger), waiting);
        run.ending = Ending::Finished;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), waiting);
        run.ending = Ending::Aborted;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), waiting);

        // Quiet for as long as the ledger lets anyone stop a task isn't enough:
        // a site can only ack here once the ack is in its log.
        ledger.slot = 5_000;
        ledger.log_ended = Some(4_500);
        assert_eq!(decide(Party::Aggregator, &run, &ledger), waiting);
        ledger.slot = 5_501;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), Ok(Step::Abort));
        run.ending = Ending::Finished;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), Ok(Step::Abort));
        // A run still going may yet catch up.
        run.ending = Ending::Running;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), waiting);

        // The log was aborted with this ack missing: it can never come.
        ledger.slot = 200;
        run.rounds[0].acks.remove(&2);
        run.ending = Ending::Aborted;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), Ok(Step::Abort));

        // A dispute is settled first.
        ledger.rounds.get_mut(&1).unwrap().disputes = 1;
        let dispute = Ok(Step::Wait(Wait::Dispute));
        assert_eq!(decide(Party::Aggregator, &run, &ledger), dispute);
    }

    #[test]
    fn a_log_that_is_over_stops_the_wait_for_an_enrollment_that_cannot_come() {
        let mut run = run(3, 2);
        run.rounds.push(sealed(1, &[0, 1, 2]));
        let mut held = task(&run);
        (held.round, held.stall_slots, held.last_change) = (1, 1_000, 100);
        held.enrolled = [0, 1].into_iter().collect();
        let mut ledger = snapshot(held, 200);
        let collecting = mirrored(&run.rounds[0], ROUND_COLLECTING, 0);
        ledger.rounds.insert(1, collecting);
        let waiting = Ok(Step::Wait(Wait::Enrollment(2)));

        assert_eq!(decide(Party::Aggregator, &run, &ledger), waiting);
        run.ending = Ending::Finished;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), waiting);
        ledger.log_ended = Some(200);
        ledger.slot = 1_201;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), Ok(Step::Abort));
        run.ending = Ending::Running;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), waiting);

        ledger.slot = 200;
        ledger.log_ended = Some(200);
        run.rounds[0].acks.remove(&2);
        run.ending = Ending::Aborted;
        assert_eq!(decide(Party::Aggregator, &run, &ledger), Ok(Step::Abort));
    }

    #[test]
    fn a_round_the_log_gave_up_is_voided_on_the_ledger_and_the_next_opens() {
        let mut run = run(3, 2);
        let shaped = sealed(1, &[0, 1, 2]);
        let mut given_up = shaped.clone();
        given_up.posted = None;
        given_up.voided = true;
        given_up.removed_helper = true;
        run.rounds.push(given_up.clone());
        run.rounds.push(sealed(2, &[0, 1, 2]));
        let mut held = task(&run);
        held.round = 1;
        let mut ledger = snapshot(held, 200);
        let aggregator = |ledger: &Snapshot| decide(Party::Aggregator, &run, ledger);

        // The charge on the ledger follows the log's, so it commits first.
        ledger
            .rounds
            .insert(1, mirrored(&shaped, ROUND_COLLECTING, 0));
        assert_eq!(aggregator(&ledger), Ok(Step::CommitRoot(1)));

        ledger
            .rounds
            .insert(1, mirrored(&shaped, ROUND_COMMITTED, 0));
        let void = Step::VoidRound {
            round: 1,
            remove_helper: true,
        };
        assert_eq!(aggregator(&ledger), Ok(void));

        let held = ledger.task.as_mut().unwrap();
        (held.settled, held.voided, held.removed_helpers) = (1, 1, 1);
        ledger.rounds.insert(1, mirrored(&shaped, ROUND_VOID, 0));
        assert_eq!(aggregator(&ledger), Ok(Step::OpenRound(2)));

        // A site owes nothing for it.
        let me = run.wallets[0];
        ledger.enrollments.insert(0, enrolled(me));
        let site = owed(Party::Site(0), &me, &run, &ledger, &Notes::default());
        assert!(!matches!(
            site,
            Ok(Step::Ack(1)) | Ok(Step::Claim(1)) | Err(_)
        ));
    }

    #[test]
    fn a_demand_is_not_opened_once_the_window_has_closed() {
        let mut run = run(2, 1);
        run.rounds = vec![sealed(1, &[0, 1])];
        let mut held = task(&run);
        held.round = 1;
        let mut ledger = snapshot(held, 10);
        ledger
            .rounds
            .insert(1, mirrored(&run.rounds[0], ROUND_POSTED, 5));
        let (part, leaf) = (ProofPart::Aggregate, 0);
        let mut notes = Notes::default();
        notes.wanted.insert(
            1,
            vec![Wanted {
                part,
                leaf,
                opened: false,
                taken: false,
            }],
        );
        let round = 1;

        ledger.slot = 5 + WINDOW - 1;
        let asking = Ok(Step::OpenDemand { round, part, leaf });
        assert_eq!(owed(Party::Watcher, &ME, &run, &ledger, &notes), asking);

        ledger.slot = 5 + WINDOW;
        assert_ne!(owed(Party::Watcher, &ME, &run, &ledger, &notes), asking);
    }

    #[test]
    fn a_share_nobody_claims_expires_and_the_last_round_closes_at_the_end() {
        let (run, mut ledger) = watched();
        ledger.rounds.clear();
        ledger.task_rent = 0;
        let log = run.round(1).unwrap().clone();
        let mut unclaimed = mirrored(&log, ROUND_FINAL, 50);
        unclaimed.settled_slot = 100;
        ledger.rounds.insert(1, unclaimed);
        let deadline = 100 + WINDOW * CLAIM_WINDOWS;
        let held = ledger.task.as_mut().unwrap();
        (held.round, held.settled, held.owed) = (2, 1, 40);
        held.lamports = 40;
        let at = |slot, state, stake| {
            let mut ledger = ledger.clone();
            ledger.slot = slot;
            let held = ledger.task.as_mut().unwrap();
            (held.state, held.stake_held) = (state, stake);
            ledger
        };
        let notes = Notes::default();
        let by = |party, ledger: &Snapshot| owed(party, &[5; 32], &run, ledger, &notes);
        let close = Ok(Step::CloseRound(1));

        assert_ne!(
            by(Party::Aggregator, &at(deadline - 1, TASK_OPEN, 500)),
            close
        );
        assert_eq!(by(Party::Aggregator, &at(deadline, TASK_OPEN, 500)), close);

        // Over, but a share is still in time: nobody can finish yet.
        let early = at(deadline - 1, TASK_FINISHED, 0);
        let waiting = Ok(Step::Wait(Wait::Claims));
        assert_eq!(by(Party::Aggregator, &early), waiting);
        assert_eq!(by(Party::Publisher, &early), waiting);
        // The publisher closes it itself if the aggregator is gone.
        assert_eq!(by(Party::Publisher, &at(deadline, TASK_FINISHED, 0)), close);

        // The latest round waits for the stake, and for its disputes.
        let mut last = at(10, TASK_ABORTED, 500);
        last.task.as_mut().unwrap().round = 1;
        last.task.as_mut().unwrap().owed = 0;
        last.rounds.insert(1, mirrored(&log, ROUND_COMMITTED, 50));
        assert_eq!(by(Party::Aggregator, &last), Ok(Step::WithdrawAggregator));
        last.task.as_mut().unwrap().stake_held = 0;
        assert_eq!(by(Party::Aggregator, &last), close);
        last.rounds.get_mut(&1).unwrap().disputes = 1;
        assert_eq!(by(Party::Aggregator, &last), waiting);
        last.rounds.clear();
        assert_eq!(by(Party::Aggregator, &last), Ok(Step::Done));
    }
}
