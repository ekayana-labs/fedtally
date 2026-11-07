//! The loop each role runs: read the log, read the ledger, send what is owed.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use fedtally_core::keys::Identity;
use fedtally_core::merkle;
use fedtally_ledger::client::{self, ProofPart, TaskTerms};
use fedtally_ledger::state::{
    DEMAND_DISCRIMINATOR, DemandLayout, MAX_STATUS_BITS, ROUND_FINAL, ROUND_VOID, TASK_SPACE,
};
use fedtally_node::NodeError;
use fedtally_node::observer::{self, ObserverConfig};

use crate::error::ChainError;
use crate::ledger::{self, Ledger, Sink};
use crate::owed::{self, Notes, Party, SignedList, Snapshot, Step, Verdict, Wait, Wanted};
use crate::rpc::{Rpc, RpcError};
use crate::run::{Ending, Run, RunView, list_number};
use crate::tx::Wallet;
use crate::watch::Vault;

const REJECTIONS: u32 = 5;
const RETRY_FIRST: Duration = Duration::from_millis(250);
const RETRY_MAX: Duration = Duration::from_secs(10);
const CONFIRM_EVERY: Duration = Duration::from_millis(150);
/// Leaves a watcher asks for at a time.
const DEMANDS_AT_ONCE: usize = 16;

/// What a publisher sets beyond the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stakes {
    pub site_stake: u64,
    pub aggregator_stake: u64,
    pub reward_per_round: u64,
    pub dispute_bond: u64,
    pub challenge_slots: u64,
    pub stall_slots: u64,
}

/// Where a watcher follows the run and keeps what it fetches.
#[derive(Clone)]
pub struct Watching {
    pub addr: String,
    pub dir: PathBuf,
    pub runtime: tokio::runtime::Handle,
}

#[derive(Clone)]
pub enum Role {
    Publisher(Stakes),
    Aggregator,
    Site,
    Watcher(Watching),
    /// Mirrors this issuer's revocation lists from the log onto the ledger.
    Issuer,
}

pub struct Config {
    pub role: Role,
    pub identity: Arc<Identity>,
    pub plan_line: Vec<u8>,
    /// The log this party's node keeps. A watcher keeps its own instead.
    pub log: PathBuf,
    pub rpc: String,
    pub program: [u8; 32],
    /// A priority fee, in micro-lamports per compute unit.
    pub price: Option<u64>,
    pub poll: Duration,
    /// How long the endpoint may keep failing before the anchor stops.
    pub give_up_after: Duration,
    pub sink: Option<Sink>,
}

pub struct Anchor {
    party: Party,
    stakes: Option<Stakes>,
    run: Run,
    ledger: Ledger,
    task: [u8; 32],
    task_rent: Option<u64>,
    program_seen: bool,
    /// The first round a site may still owe something for.
    cursor: u32,
    notes: Notes,
    vault: Option<Vault>,
    mirror: Option<tokio::task::JoinHandle<Result<(), NodeError>>>,
    held: BTreeSet<u32>,
    reported: BTreeSet<u32>,
    quiet_until: Instant,
    /// When to try fetching ahead again. Apart from `quiet_until`, or a failing
    /// fetch would keep a round from ever being examined.
    fetch_again: Instant,
    /// The slot this anchor first saw the log over at. A restart only lengthens
    /// what depends on it.
    log_ended: Option<u64>,
    said: Option<Step>,
    poll: Duration,
    give_up_after: Duration,
}

fn anchor_wallet(did: &str) -> Option<[u8; 32]> {
    fedtally_core::anchor::wallet(did).ok()
}

fn name(address: &[u8; 32]) -> String {
    bs58::encode(address).into_string()
}

fn task_terms(run: &RunView, stakes: &Stakes) -> TaskTerms {
    let terms = &run.terms;
    TaskTerms {
        aggregator: run.aggregator,
        plan_id: terms.plan.0,
        sites_root: terms.sites_root,
        model: terms.model.0,
        sites: terms.sites,
        helpers: terms.helpers,
        coords: terms.coords,
        rounds: terms.rounds,
        min_cohort: terms.min_cohort,
        sample: terms.sample,
        sensitivity: terms.sensitivity,
        noise: terms.noise,
        colluding: terms.colluding,
        budget: terms.budget,
        site_stake: stakes.site_stake,
        aggregator_stake: stakes.aggregator_stake,
        reward_per_round: stakes.reward_per_round,
        dispute_bond: stakes.dispute_bond,
        challenge_slots: stakes.challenge_slots,
        stall_slots: stakes.stall_slots,
        retries: terms.retries,
        min_helpers: terms.min_helpers,
        score_group: terms.score_group,
    }
}

/// A unit limit well above what each instruction was measured to take. One
/// that creates an account searches for its bump, at about 1,500 units a
/// miss. A sampling `commit_root` hashes once per site of the task.
fn ceiling(step: &Step, run: &RunView) -> u32 {
    match step {
        Step::Publish
        | Step::Enroll
        | Step::OpenRound(_)
        | Step::OpenDispute { .. }
        | Step::OpenDemand { .. }
        | Step::InitList { .. }
        | Step::OpenBuffer(_) => 40_000,
        Step::CommitRoot(_) if run.terms.sample > 0 => 20_000 + 200 * u32::from(run.terms.sites),
        Step::GiveLeaf { .. } | Step::AnswerDemand { .. } => 30_000,
        _ => 10_000,
    }
}

fn is_transient(error: &ChainError) -> bool {
    match error {
        ChainError::Rpc(RpcError::Rejected { .. }) => false,
        ChainError::Rpc(_) | ChainError::Expired => true,
        _ => false,
    }
}

impl Anchor {
    pub fn new(config: Config) -> Result<Anchor, ChainError> {
        let built = *fedtally_ledger::ID.as_array();
        if config.program != built {
            return Err(ChainError::Program {
                built: name(&built),
                asked: name(&config.program),
            });
        }
        let wallet = Wallet::new(config.identity.clone())?;
        let (log, watching) = match &config.role {
            Role::Watcher(watching) => (observer::log_path(&watching.dir), Some(watching.clone())),
            _ => (config.log.clone(), None),
        };
        let run = Run::open(&config.plan_line, &log)?;
        let view = run.view();
        let me = wallet.address();
        let (party, stakes) = match &config.role {
            Role::Publisher(_) if *me != view.publisher => {
                return Err(ChainError::NotTheParty("sponsor"));
            }
            Role::Aggregator if *me != view.aggregator => {
                return Err(ChainError::NotTheParty("aggregator"));
            }
            Role::Publisher(stakes) => (Party::Publisher, Some(*stakes)),
            Role::Aggregator => (Party::Aggregator, None),
            Role::Site => {
                let index = view.wallets.iter().position(|site| site == me);
                let index = index.ok_or(ChainError::NotTheParty("site"))?;
                (Party::Site(index as u16), None)
            }
            Role::Watcher(_) => (Party::Watcher, None),
            Role::Issuer => (Party::Issuer, None),
        };
        let task = client::task(&view.publisher, &view.terms.plan.0);

        let mut vault = None;
        let mut mirror = None;
        if let Some(watching) = watching {
            let source = ObserverConfig {
                identity: config.identity.clone(),
                plan_line: config.plan_line.clone(),
                addr: watching.addr,
                dir: watching.dir,
                give_up_after: config.give_up_after,
            };
            mirror = Some(watching.runtime.spawn(observer::mirror_log(source.clone())));
            vault = Some(Vault::new(source, Some(watching.runtime), run.shape()));
        }
        // The aggregator answers demands from its own spool, next to its log.
        if let (Role::Aggregator, Some(dir)) = (&config.role, config.log.parent()) {
            let source = ObserverConfig {
                identity: config.identity.clone(),
                plan_line: config.plan_line.clone(),
                addr: String::new(),
                dir: dir.to_owned(),
                give_up_after: config.give_up_after,
            };
            vault = Some(Vault::new(source, None, run.shape()));
        }
        Ok(Anchor {
            party,
            stakes,
            run,
            ledger: Ledger {
                rpc: Rpc::new(&config.rpc),
                wallet,
                price: config.price,
                confirm_every: CONFIRM_EVERY,
                sink: config.sink,
            },
            task,
            task_rent: None,
            program_seen: false,
            cursor: 1,
            notes: Notes::default(),
            vault,
            mirror,
            held: BTreeSet::new(),
            reported: BTreeSet::new(),
            quiet_until: Instant::now(),
            fetch_again: Instant::now(),
            log_ended: None,
            said: None,
            poll: config.poll,
            give_up_after: config.give_up_after,
        })
    }

    /// The address of the run's task.
    pub fn task(&self) -> &[u8; 32] {
        &self.task
    }

    fn me(&self) -> [u8; 32] {
        *self.ledger.wallet.address()
    }

    fn check_program(&mut self) -> Result<(), ChainError> {
        if self.program_seen {
            return Ok(());
        }
        let id = *fedtally_ledger::ID.as_array();
        let program = self.ledger.rpc.account(&id)?;
        if !program.is_some_and(|account| account.executable) {
            return Err(ChainError::NoProgram(name(&id)));
        }
        self.program_seen = true;
        Ok(())
    }

    /// Reads the accounts this party's next step depends on.
    pub fn snapshot(&mut self) -> Result<Snapshot, ChainError> {
        let rpc = &self.ledger.rpc;
        // The slot is read first, so it is never ahead of the accounts.
        let slot = rpc.slot()?;
        let held = rpc.account(&self.task)?;
        let task = ledger::task(held.as_ref());
        if held.is_some() && task.is_none() {
            return Err(ChainError::Account {
                address: name(&self.task),
                kind: "task",
            });
        }
        if self.run.view().ending != Ending::Running {
            self.log_ended.get_or_insert(slot);
        }
        let mut snapshot = Snapshot {
            slot,
            task,
            log_ended: self.log_ended,
            ..Snapshot::default()
        };
        let Some(task) = &snapshot.task else {
            return Ok(snapshot);
        };
        let helpers = task.active_helpers();
        let (latest, me) = (task.round, self.me());
        let round_at = |number: u32| client::round(&self.task, number);
        match self.party {
            Party::Publisher => {
                let rent = match self.task_rent {
                    Some(rent) => rent,
                    None => rpc.rent_exempt_minimum(TASK_SPACE)?,
                };
                self.task_rent = Some(rent);
                snapshot.task_rent = rent;
                if !task.is_open() && task.owed > 0 {
                    let numbers: Vec<u32> = (1..=latest).collect();
                    let addresses: Vec<[u8; 32]> =
                        numbers.iter().map(|number| round_at(*number)).collect();
                    let accounts = rpc.accounts(&addresses)?;
                    for (number, account) in numbers.into_iter().zip(accounts) {
                        let round = ledger::round(account.as_ref(), helpers);
                        snapshot.rounds.extend(round.map(|round| (number, round)));
                    }
                }
            }
            Party::Aggregator if latest > 0 => {
                let numbers: Vec<u32> = (self.cursor..=latest).collect();
                let mut addresses = vec![client::buffer(&me)];
                addresses.extend(numbers.iter().map(|number| round_at(*number)));
                let mut accounts = rpc.accounts(&addresses)?.into_iter();
                snapshot.buffer = ledger::buffer(accounts.next().flatten().as_ref());
                for (number, account) in numbers.into_iter().zip(accounts) {
                    let round = ledger::round(account.as_ref(), helpers);
                    snapshot.rounds.extend(round.map(|round| (number, round)));
                }
                while self.cursor < latest && !snapshot.rounds.contains_key(&self.cursor) {
                    self.cursor += 1;
                }
                let busy = snapshot.rounds.get(&latest).is_some_and(|r| r.disputes > 0);
                if busy {
                    snapshot.demands = self.open_demands(None)?;
                }
            }
            Party::Aggregator => {}
            Party::Site(index) => {
                let numbers: Vec<u32> = (self.cursor..=latest).collect();
                let mut addresses = vec![client::enrollment(&self.task, index)];
                addresses.extend(numbers.iter().map(|number| round_at(*number)));
                let mut accounts = rpc.accounts(&addresses)?.into_iter();
                let site = accounts.next().flatten();
                snapshot
                    .enrollments
                    .extend(ledger::enrollment(site.as_ref()).map(|site| (index, site)));
                for (number, account) in numbers.into_iter().zip(accounts) {
                    let round = ledger::round(account.as_ref(), helpers);
                    snapshot.rounds.extend(round.map(|round| (number, round)));
                }
                // A final round this site has nothing left to take from is
                // not read again.
                while self.cursor < latest {
                    let done = snapshot.rounds.get(&self.cursor).is_none_or(|round| {
                        round.state == ROUND_VOID
                            || round.state == ROUND_FINAL
                                && (!round.cohort.has(index) || round.claimed.has(index))
                    });
                    if !done {
                        break;
                    }
                    self.cursor += 1;
                }
            }
            Party::Watcher => {
                let revoked: Vec<u16> = self.run.view().revoked.iter().copied().collect();
                let mut addresses = vec![
                    round_at(latest),
                    client::dispute(&self.task, latest, &me),
                    client::buffer(&me),
                ];
                addresses.extend(revoked.iter().map(|s| client::enrollment(&self.task, *s)));
                let accounts = rpc.accounts(&addresses)?;
                let round = ledger::round(accounts[0].as_ref(), helpers);
                snapshot.rounds.extend(round.map(|round| (latest, round)));
                snapshot.dispute = ledger::dispute(accounts[1].as_ref());
                snapshot.buffer = ledger::buffer(accounts[2].as_ref());
                for (site, account) in revoked.into_iter().zip(&accounts[3..]) {
                    let enrolled = ledger::enrollment(account.as_ref());
                    snapshot
                        .enrollments
                        .extend(enrolled.map(|view| (site, view)));
                }
                snapshot.demands = self.open_demands(Some(&me))?;
            }
            Party::Issuer => {
                let numbers: Vec<u32> = self.notes.signed.keys().copied().collect();
                let addresses: Vec<[u8; 32]> = numbers
                    .iter()
                    .map(|list| client::status_list(&me, *list))
                    .collect();
                for (list, account) in numbers.into_iter().zip(rpc.accounts(&addresses)?) {
                    let held = ledger::status(account.as_ref());
                    snapshot.lists.extend(held.map(|held| (list, held)));
                }
            }
        }
        Ok(snapshot)
    }

    /// The demands open on the task, or only those of one challenger.
    fn open_demands(
        &self,
        challenger: Option<&[u8; 32]>,
    ) -> Result<Vec<ledger::DemandView>, ChainError> {
        let mut matching: Vec<(usize, &[u8])> = vec![
            (0, &DEMAND_DISCRIMINATOR),
            (std::mem::offset_of!(DemandLayout, task), &self.task),
        ];
        if let Some(challenger) = challenger {
            matching.push((std::mem::offset_of!(DemandLayout, challenger), challenger));
        }
        let program = *fedtally_ledger::ID.as_array();
        let found = self.ledger.rpc.program_accounts(&program, &matching)?;
        Ok(found
            .iter()
            .filter_map(|(address, account)| ledger::demand(*address, account))
            .collect())
    }

    /// What the log says this issuer has revoked, as the ledger can hold it.
    fn read_signed(&mut self) {
        let me = self.me();
        self.notes.signed = self
            .run
            .replay()
            .lists
            .values()
            .filter(|list| anchor_wallet(&list.issuer).is_some_and(|wallet| wallet == me))
            .map(|list| {
                let set = (0..MAX_STATUS_BITS)
                    .filter(|bit| list.is_set(u64::from(*bit)))
                    .collect();
                let signed = SignedList {
                    bits: MAX_STATUS_BITS,
                    set,
                };
                (list_number(&list.id), signed)
            })
            .collect();
    }

    /// A round on the ledger is settled for good, so its vectors can go.
    fn prune(&mut self, number: u32) {
        if let Some(vault) = &mut self.vault {
            vault.forget(number);
            let _ = std::fs::remove_dir_all(vault.source.dir.join(format!("round-{number}")));
        }
    }

    /// Keeps the watcher's replica going and its copy of the vectors current.
    fn watch(&mut self) -> Result<(), ChainError> {
        if self
            .mirror
            .as_ref()
            .is_some_and(|mirror| mirror.is_finished())
            && let (Some(mirror), Some(runtime)) = (
                self.mirror.take(),
                self.vault.as_ref().and_then(|vault| vault.runtime.as_ref()),
            )
        {
            match runtime.block_on(mirror) {
                Ok(Ok(())) => {}
                Ok(Err(e @ (NodeError::Disconnected(_) | NodeError::Io(_)))) => {
                    tracing::warn!("the log can no longer be followed: {e}");
                }
                Ok(Err(e)) => return Err(e.into()),
                Err(e) => return Err(NodeError::Join(e.to_string()).into()),
            }
        }
        self.run.refresh()?;
        let Some(vault) = &mut self.vault else {
            return Ok(());
        };
        if Instant::now() < self.fetch_again {
            return Ok(());
        }
        // Only the newest rounds: an aggregator drops a round's vectors once
        // the next one is under way, unless it keeps its spool.
        for (i, log) in self.run.view().rounds.iter().enumerate().rev().take(2) {
            let number = i as u32 + 1;
            if self.held.contains(&number) || log.voided {
                continue;
            }
            match vault.prefetch(number, log) {
                Ok(true) => {
                    tracing::info!("round {number}: holding its vectors");
                    self.held.insert(number);
                }
                Ok(false) => {}
                Err(e) => {
                    tracing::debug!("round {number}: its vectors are not to be had: {e}");
                    self.fetch_again = Instant::now() + self.poll.max(RETRY_FIRST) * 4;
                }
            }
        }
        Ok(())
    }

    fn examine(&mut self, number: u32, snapshot: &Snapshot) -> Result<(), ChainError> {
        let (Some(round), Some(vault)) = (snapshot.rounds.get(&number), &mut self.vault) else {
            return Ok(());
        };
        let empty = Default::default();
        let log = self.run.view().round(number).unwrap_or(&empty);
        if self.reported.insert(number) {
            for what in owed::differences(round, log) {
                tracing::error!("round {number}: the ledger's {what} is not the log's");
            }
        }
        match vault.examine(number, round, log) {
            Ok(Some(verdict)) => {
                match verdict {
                    Verdict::Sound => tracing::info!("round {number}: the aggregate is the sum"),
                    Verdict::Wrong(leaf) => tracing::error!(
                        "round {number}: the aggregate is not the sum at leaf {leaf}"
                    ),
                    Verdict::Unprovable => tracing::error!(
                        "round {number}: the ledger's inputs root is not over the vectors in the log, so nothing can be proven"
                    ),
                }
                self.notes.verdicts.insert(number, verdict);
            }
            Ok(None) => self.quiet_until = Instant::now() + self.poll,
            Err(e) => {
                tracing::warn!(
                    "round {number}: the vectors behind the ledger's roots cannot be had: {e}"
                );
                self.quiet_until = Instant::now() + self.poll.max(RETRY_FIRST) * 4;
                match vault.recover(number, round, log) {
                    Ok(Some(leaf)) => {
                        tracing::error!(
                            "round {number}: the aggregate is not the sum at leaf {leaf}, seen in the parts that could be had"
                        );
                        self.notes.verdicts.insert(number, Verdict::Wrong(leaf));
                        return Ok(());
                    }
                    Ok(None) => {}
                    Err(e) => tracing::warn!("round {number}: nothing recovered: {e}"),
                }
                // Half the window goes to asking nicely. What's still refused
                // then is asked for on the ledger.
                let Some(task) = &snapshot.task else {
                    return Ok(());
                };
                let (window, slot) = (task.challenge_slots, snapshot.slot);
                let halfway = round.posted_slot + window / 2;
                let end = round.posted_slot + window;
                if slot >= end {
                    tracing::error!("round {number}: nothing more can be asked for");
                    self.notes.verdicts.insert(number, Verdict::Unprovable);
                } else if slot >= halfway {
                    let wanted = self.notes.wanted.entry(number).or_default();
                    let asked = vault.missing(number, round, log, DEMANDS_AT_ONCE);
                    for (part, leaf) in asked {
                        if !wanted.iter().any(|w| w.part == part && w.leaf == leaf) {
                            tracing::warn!(
                                "round {number}: demanding leaf {leaf} of {part:?} on the ledger"
                            );
                            wanted.push(Wanted {
                                part,
                                leaf,
                                opened: false,
                                taken: false,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn perform(&mut self, step: Step, snapshot: &Snapshot) -> Result<(), ChainError> {
        let (me, task) = (self.me(), self.task);
        let view = self.run.view();
        let ceiling = ceiling(&step, view);
        let latest = snapshot.task.as_ref().map_or(0, |task| task.round);
        let behind = || ChainError::Divergence(owed::Divergence::Ahead(latest));
        let index = match self.party {
            Party::Site(index) => Ok(index),
            _ => Err(ChainError::NotTheParty("site")),
        };
        let (what, round, ix) = match step {
            Step::Wait(_) | Step::Done => return Ok(()),
            Step::Examine(number) => return self.examine(number, snapshot),
            Step::Publish => {
                let stakes = self.stakes.ok_or(ChainError::NotTheParty("sponsor"))?;
                let terms = task_terms(view, &stakes);
                ("publish", None, client::publish(&me, &terms))
            }
            Step::WithdrawPublisher => {
                let ix = client::withdraw_publisher(&me, &task);
                ("withdraw (publisher)", None, ix)
            }
            Step::Back => ("back", None, client::back(&me, &task)),
            Step::OpenRound(number) => {
                let ix = client::open_round(&me, &task, number);
                ("open_round", Some(number), ix)
            }
            Step::CommitRoot(number) => {
                let log = view.round(number).and_then(|log| log.committed.as_ref());
                let committed = log.ok_or_else(behind)?;
                let cohort = client::site_bitmap(committed.cohort.iter().map(|s| *s as u16));
                let ix = client::commit_root(
                    &me,
                    &task,
                    number,
                    &cohort,
                    &committed.inputs_root,
                    &committed.close_head.0,
                    &committed.threshold,
                );
                ("commit_root", Some(number), ix)
            }
            Step::PostAggregate(number) => {
                let log = view.round(number).and_then(|log| log.posted.as_ref());
                let posted = log.ok_or_else(behind)?;
                let masks: Vec<[u8; 32]> = posted.mask_roots.iter().map(|m| m.0).collect();
                let ix = client::post_aggregate(
                    &me,
                    &task,
                    number,
                    &masks,
                    &posted.aggregate_root.0,
                    &posted.next_model.0,
                    &posted.seal_head.0,
                    &posted.shares_root,
                );
                ("post_aggregate", Some(number), ix)
            }
            Step::Settle(number) => ("settle", Some(number), client::settle(&task, number)),
            Step::Abort => ("abort", None, client::abort(&me, &task, latest)),
            Step::WithdrawAggregator => {
                let ix = client::withdraw_aggregator(&me, &task, latest);
                ("withdraw (aggregator)", None, ix)
            }
            Step::Enroll => {
                let index = index?;
                let leaves = view.wallets.iter().map(|w| merkle::leaf(w)).collect();
                let path = merkle::path(leaves, index as usize);
                let admission = view.joins.get(&index).ok_or_else(behind)?;
                let none = Vec::new();
                let status = view.status_refs.get(&index).unwrap_or(&none);
                let ix = client::enroll(&me, &task, index, &path, &admission.0, status);
                ("enroll", None, ix)
            }
            Step::Ack(number) => {
                let log = view.round(number).and_then(|log| log.posted.as_ref());
                let head = log.ok_or_else(behind)?.seal_head.0;
                let ix = client::ack(&me, &task, index?, number, &head);
                ("ack", Some(number), ix)
            }
            Step::Claim(number) => {
                let ix = client::claim(&me, &task, index?, number);
                ("claim", Some(number), ix)
            }
            Step::WithdrawSite => {
                let ix = client::withdraw_site(&me, &task, index?);
                ("withdraw (site)", None, ix)
            }
            Step::OpenBuffer(capacity) => ("buffer_init", None, client::buffer_init(&me, capacity)),
            Step::CloseBuffer => ("buffer_close", None, client::buffer_close(&me)),
            Step::OpenDispute { round, leaf, after } => {
                let ix = match after {
                    Some(part) => client::dispute_open_after_demand(&me, &task, round, part, leaf),
                    None => client::dispute_open(&me, &task, round, leaf),
                };
                self.ledger.send("dispute_open", Some(round), ix, ceiling)?;
                self.notes.disputed.insert(round);
                return Ok(());
            }
            Step::GiveLeaf { round, leaf, part } => {
                let (Some(held), Some(vault)) = (snapshot.rounds.get(&round), &mut self.vault)
                else {
                    return Ok(());
                };
                let empty = Default::default();
                let log = view.round(round).unwrap_or(&empty);
                let proof = vault.proof(round, held, log, leaf, part)?;
                for write in client::upload(&me, &proof) {
                    self.ledger
                        .send("buffer_write", Some(round), write, ceiling)?;
                }
                let what = match part {
                    ProofPart::Input(_) => "dispute_leaf (input)",
                    ProofPart::Mask(_) => "dispute_leaf (mask)",
                    ProofPart::Aggregate => "dispute_leaf (aggregate)",
                };
                let ix = client::dispute_leaf(&me, &task, round, part);
                (what, Some(round), ix)
            }
            Step::OpenDemand { round, part, leaf } => {
                let ix = client::demand_open(&me, &task, round, part, leaf);
                self.ledger.send("demand_open", Some(round), ix, ceiling)?;
                let wanted = self.notes.wanted.entry(round).or_default();
                for one in wanted.iter_mut() {
                    one.opened |= one.part == part && one.leaf == leaf;
                }
                return Ok(());
            }
            Step::TakeAnswer { round, part, leaf } => {
                let found = snapshot
                    .demands
                    .iter()
                    .find(|d| d.round == round && d.part == part && d.leaf == leaf);
                if let (Some(demand), Some(held), Some(vault)) =
                    (found, snapshot.rounds.get(&round), &mut self.vault)
                {
                    let empty = Default::default();
                    let log = view.round(round).unwrap_or(&empty);
                    vault.patch(round, held, log, part, leaf, &demand.proof)?;
                }
                let wanted = self.notes.wanted.entry(round).or_default();
                for one in wanted.iter_mut() {
                    one.taken |= one.part == part && one.leaf == leaf;
                }
                if !matches!(self.notes.verdicts.get(&round), Some(Verdict::Wrong(_))) {
                    self.notes.verdicts.remove(&round);
                }
                self.quiet_until = Instant::now();
                return Ok(());
            }
            Step::CloseDemand { round, part, leaf } => {
                let address = client::demand(&task, &me, round, part, leaf);
                let ix = client::demand_close(&task, round, address, &me, true);
                ("demand_close", Some(round), ix)
            }
            Step::AnswerDemand {
                round,
                part,
                leaf,
                address,
            } => {
                let (Some(held), Some(vault)) = (snapshot.rounds.get(&round), &mut self.vault)
                else {
                    return Ok(());
                };
                let empty = Default::default();
                let log = view.round(round).unwrap_or(&empty);
                let proof = match vault.proof(round, held, log, leaf, part) {
                    Ok(proof) => proof,
                    Err(e) => {
                        tracing::error!("round {round}: cannot answer for leaf {leaf}: {e}");
                        std::thread::sleep(self.poll);
                        return Ok(());
                    }
                };
                for write in client::upload(&me, &proof) {
                    self.ledger
                        .send("buffer_write", Some(round), write, ceiling)?;
                }
                let ix = client::demand_answer(&me, &task, round, address);
                ("demand_answer", Some(round), ix)
            }
            Step::VoidRound {
                round: number,
                remove_helper,
            } => {
                let ix = client::void_round(&view.aggregator, &task, number, remove_helper);
                ("void_round", Some(number), ix)
            }
            Step::CloseRound(number) => {
                let ix = client::round_close(&view.aggregator, &task, number);
                ("round_close", Some(number), ix)
            }
            Step::InitList { list, bits } => {
                ("status_init", None, client::status_init(&me, list, bits))
            }
            Step::SetBit { list, bit } => ("revoke", None, client::revoke(&me, list, bit)),
            Step::CloseDispute(number) => {
                let ix = client::dispute_close(&task, number, &me, &view.aggregator);
                ("dispute_close", Some(number), ix)
            }
            Step::RevokeCheck(site) => {
                let lists = snapshot.enrollments.get(&site);
                let lists = lists
                    .map(|site| site.status_lists.clone())
                    .unwrap_or_default();
                let ix = client::revoke_check(&task, site, &lists);
                self.ledger.send("revoke_check", None, ix, ceiling)?;
                self.notes.checked.insert(site, snapshot.slot);
                return Ok(());
            }
        };
        self.ledger.send(what, round, ix, ceiling)?;
        if let Step::Settle(number) | Step::VoidRound { round: number, .. } = step {
            self.prune(number);
        }
        Ok(())
    }

    fn say(&mut self, step: Step) {
        if self.said == Some(step) {
            return;
        }
        self.said = Some(step);
        match step {
            Step::Wait(Wait::Refusing { round, what }) => tracing::error!(
                "round {round}: not acknowledging, the ledger's {what} is not the one in this site's log"
            ),
            Step::Wait(wait) => tracing::info!("waiting: {wait:?}"),
            Step::Done => tracing::info!("nothing more is owed"),
            step => tracing::info!("sending {step:?}"),
        }
    }

    /// Works out what is owed and sends it. Returns what that was.
    pub fn step(&mut self) -> Result<Step, ChainError> {
        self.check_program()?;
        match self.party {
            Party::Watcher => self.watch()?,
            Party::Issuer => {
                self.run.refresh()?;
                self.read_signed();
            }
            _ => {
                self.run.refresh()?;
            }
        }
        let snapshot = self.snapshot()?;
        let me = self.me();
        let mut step = owed::owed(self.party, &me, self.run.view(), &snapshot, &self.notes)?;
        if matches!(step, Step::Examine(_)) && Instant::now() < self.quiet_until {
            step = Step::Wait(Wait::Log);
        }
        self.say(step);
        self.perform(step, &snapshot)?;
        Ok(step)
    }

    /// Runs until nothing more is owed, `stop` is set, or something is wrong
    /// that waiting will not fix.
    pub fn run(&mut self, stop: &AtomicBool) -> Result<(), ChainError> {
        let mut failing: Option<(Instant, Duration)> = None;
        let mut rejected = 0;
        while !stop.load(Ordering::Relaxed) {
            match self.step() {
                Ok(Step::Done) => return Ok(()),
                Ok(Step::Wait(_)) => {
                    (failing, rejected) = (None, 0);
                    std::thread::sleep(self.poll);
                }
                Ok(_) => (failing, rejected) = (None, 0),
                Err(e) if is_transient(&e) => {
                    let (since, wait) = failing.unwrap_or((Instant::now(), RETRY_FIRST));
                    if since.elapsed() > self.give_up_after {
                        return Err(ChainError::GaveUp(e.to_string()));
                    }
                    tracing::warn!("trying again in {wait:?}: {e}");
                    std::thread::sleep(wait);
                    failing = Some((since, (wait * 2).min(RETRY_MAX)));
                }
                // The ledger may have moved between the read and the send,
                // so a refusal is worth another look before it counts.
                Err(ChainError::Rpc(RpcError::Rejected { reason, logs })) => {
                    rejected += 1;
                    tracing::warn!("refused: {reason}");
                    if rejected >= REJECTIONS {
                        return Err(ChainError::Stuck {
                            step: format!("{:?}", self.said),
                            tries: rejected,
                            last: format!("{reason}\n{}", logs.join("\n")),
                        });
                    }
                    std::thread::sleep(self.poll.max(RETRY_FIRST) * rejected);
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

impl Drop for Anchor {
    fn drop(&mut self) {
        if let Some(mirror) = &self.mirror {
            mirror.abort();
        }
    }
}
