//! The loop each role runs: read the log, read the ledger, send what is owed.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use fedtally_core::keys::Identity;
use fedtally_core::merkle;
use fedtally_ledger::client::{self, ProofPart, TaskTerms};
use fedtally_ledger::state::{ROUND_FINAL, TASK_SPACE};
use fedtally_node::NodeError;
use fedtally_node::observer::{self, ObserverConfig};

use crate::error::ChainError;
use crate::ledger::{self, Ledger, Sink};
use crate::owed::{self, Notes, Party, Snapshot, Step, Verdict, Wait};
use crate::rpc::{Rpc, RpcError};
use crate::run::{Run, RunView};
use crate::tx::Wallet;
use crate::watch::Vault;

const REJECTIONS: u32 = 5;
const RETRY_FIRST: Duration = Duration::from_millis(250);
const RETRY_MAX: Duration = Duration::from_secs(10);
const CONFIRM_EVERY: Duration = Duration::from_millis(150);

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
    said: Option<Step>,
    poll: Duration,
    give_up_after: Duration,
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
        | Step::OpenBuffer(_) => 40_000,
        Step::CommitRoot(_) if run.terms.sample > 0 => 20_000 + 200 * u32::from(run.terms.sites),
        Step::GiveLeaf { .. } => 30_000,
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
            vault = Some(Vault::new(source, watching.runtime, run.shape()));
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
        let mut snapshot = Snapshot {
            slot,
            task,
            ..Snapshot::default()
        };
        let Some(task) = &snapshot.task else {
            return Ok(snapshot);
        };
        let (helpers, latest, me) = (task.helpers, task.round, self.me());
        let round_at = |number: u32| client::round(&self.task, number);
        match self.party {
            Party::Publisher => {
                let rent = match self.task_rent {
                    Some(rent) => rent,
                    None => rpc.rent_exempt_minimum(TASK_SPACE)?,
                };
                self.task_rent = Some(rent);
                snapshot.task_rent = rent;
            }
            Party::Aggregator if latest > 0 => {
                let round = rpc.account(&round_at(latest))?;
                let round = ledger::round(round.as_ref(), helpers);
                snapshot.rounds.extend(round.map(|round| (latest, round)));
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
                while let Some(round) = snapshot.rounds.get(&self.cursor) {
                    let done = round.state == ROUND_FINAL
                        && (!round.cohort.has(index) || round.claimed.has(index));
                    if !done {
                        break;
                    }
                    self.cursor += 1;
                }
            }
            Party::Watcher => {
                let revoked: Vec<u16> = self
                    .run
                    .view()
                    .revoked
                    .iter()
                    .filter(|site| !self.notes.checked.contains(site))
                    .copied()
                    .collect();
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
            }
        }
        Ok(snapshot)
    }

    /// Keeps the watcher's replica going and its copy of the vectors current.
    fn watch(&mut self) -> Result<(), ChainError> {
        if self
            .mirror
            .as_ref()
            .is_some_and(|mirror| mirror.is_finished())
            && let (Some(mirror), Some(vault)) = (self.mirror.take(), &self.vault)
        {
            match vault.runtime.block_on(mirror) {
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
        if Instant::now() < self.quiet_until {
            return Ok(());
        }
        // Only the newest rounds: an aggregator drops a round's vectors once
        // the next one is under way, unless it keeps its spool.
        for (i, log) in self.run.view().rounds.iter().enumerate().rev().take(2) {
            let number = i as u32 + 1;
            if self.held.contains(&number) {
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
                    self.quiet_until = Instant::now() + self.poll.max(RETRY_FIRST) * 4;
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
            Step::OpenDispute { round, leaf } => {
                let ix = client::dispute_open(&me, &task, round, leaf);
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
                self.notes.checked.insert(site);
                return Ok(());
            }
        };
        self.ledger.send(what, round, ix, ceiling)
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
