//! The ledger's accounts as plain values, and a sender that waits for each
//! transaction to be confirmed.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fedtally_ledger::bitmap::{self, SITE_BITMAP_LEN, SiteBitmap};
use fedtally_ledger::client::{self, ProofPart};
use fedtally_ledger::instructions::dispute_leaf::{PART_AGGREGATE, PART_INPUT, PART_MASK};
use fedtally_ledger::state::{
    Buffer, Demand, Dispute, Enrollment, MAX_SITES, Round, StatusList, TASK_OPEN, Task,
};

use crate::error::ChainError;
use crate::rpc::{Account, Rpc, RpcError};
use crate::tx::{self, Budget, Wallet};

/// A set of sites, as the ledger keeps one: a bit per site index.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Sites(pub SiteBitmap);

impl Default for Sites {
    fn default() -> Sites {
        Sites([0; SITE_BITMAP_LEN])
    }
}

impl std::fmt::Debug for Sites {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.members()).finish()
    }
}

impl Sites {
    pub fn has(&self, site: u16) -> bool {
        (site as usize) < MAX_SITES && bitmap::get(&self.0, site as usize)
    }

    /// The sites in the set, ascending.
    pub fn members(&self) -> Vec<u16> {
        (0..MAX_SITES as u16).filter(|s| self.has(*s)).collect()
    }

    pub fn len(&self) -> usize {
        bitmap::count(&self.0) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl FromIterator<u16> for Sites {
    fn from_iter<I: IntoIterator<Item = u16>>(sites: I) -> Sites {
        Sites(client::site_bitmap(
            sites.into_iter().filter(|s| (*s as usize) < MAX_SITES),
        ))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TaskView {
    pub lamports: u64,
    pub state: u8,
    pub backed: bool,
    pub fraud: bool,
    pub helpers: u8,
    pub sites: u16,
    pub min_cohort: u16,
    pub sample: u16,
    pub colluding: u16,
    pub rounds: u32,
    pub round: u32,
    pub settled: u32,
    pub publisher: [u8; 32],
    pub aggregator: [u8; 32],
    pub plan_id: [u8; 32],
    /// The model of the last settled round.
    pub model: [u8; 32],
    pub sites_root: [u8; 32],
    pub coords: u64,
    pub sensitivity: u64,
    pub noise: u64,
    pub budget: u64,
    pub stake_held: u64,
    pub owed: u64,
    pub challenge_slots: u64,
    pub stall_slots: u64,
    pub last_change: u64,
    pub spent: u64,
    pub retries: u32,
    pub voided: u32,
    pub min_helpers: u8,
    pub removed_helpers: u8,
    pub enrolled: Sites,
    pub revoked: Sites,
}

impl TaskView {
    pub fn is_open(&self) -> bool {
        self.state == TASK_OPEN
    }

    /// The helpers whose masks a round has.
    pub fn active_helpers(&self) -> u8 {
        self.helpers.saturating_sub(self.removed_helpers)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RoundView {
    pub state: u8,
    pub disputes: u32,
    pub beacon: [u8; 32],
    pub cohort: Sites,
    pub inputs_root: [u8; 32],
    pub close_head: [u8; 32],
    /// One root per helper of the task.
    pub mask_roots: Vec<[u8; 32]>,
    pub aggregate_root: [u8; 32],
    pub next_model: [u8; 32],
    pub seal_head: [u8; 32],
    pub posted_slot: u64,
    pub settled_slot: u64,
    pub acked: Sites,
    pub claimed: Sites,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnrollmentView {
    pub status: u8,
    pub wallet: [u8; 32],
    /// The status lists the enrollment names, in its order.
    pub status_lists: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DisputeView {
    pub leaf_index: u64,
    pub deadline: u64,
    /// The cohort positions whose input leaf has been given.
    pub seen_inputs: Sites,
    pub seen_masks: u8,
    pub aggregate_given: bool,
}

/// A demand for one leaf, with the proof that answered it, if one has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemandView {
    pub address: [u8; 32],
    pub challenger: [u8; 32],
    pub round: u32,
    pub part: ProofPart,
    pub leaf: u64,
    pub answered: bool,
    pub deadline: u64,
    pub proof: Vec<u8>,
}

/// An issuer's list on the ledger.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StatusView {
    pub bits: u32,
    pub set: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferView {
    pub capacity: u32,
}

fn program() -> [u8; 32] {
    *fedtally_ledger::ID.as_array()
}

/// Accounts of another owner are someone else's: a program address can hold
/// lamports before the program creates anything there.
fn ours(account: Option<&Account>) -> Option<&Account> {
    account.filter(|account| account.owner == program())
}

pub fn task(account: Option<&Account>) -> Option<TaskView> {
    let account = ours(account)?;
    let task = Task::parse(account.data.as_slice()).ok()?;
    Some(TaskView {
        lamports: account.lamports,
        state: task.state(),
        backed: task.backed(),
        fraud: task.fraud(),
        helpers: task.helpers(),
        sites: task.sites(),
        min_cohort: task.min_cohort(),
        sample: task.sample(),
        colluding: task.colluding(),
        rounds: task.rounds(),
        round: task.round(),
        settled: task.settled(),
        publisher: *task.publisher(),
        aggregator: *task.aggregator(),
        plan_id: *task.plan_id(),
        model: *task.model(),
        sites_root: *task.sites_root(),
        coords: task.coords(),
        sensitivity: task.sensitivity(),
        noise: task.noise(),
        budget: task.budget(),
        stake_held: task.stake_held(),
        owed: task.owed(),
        challenge_slots: task.challenge_slots(),
        stall_slots: task.stall_slots(),
        last_change: task.last_change(),
        spent: task.spent(),
        retries: task.retries(),
        voided: task.voided(),
        min_helpers: task.min_helpers(),
        removed_helpers: task.removed_helpers(),
        enrolled: Sites(*task.enrolled()),
        revoked: Sites(*task.revoked()),
    })
}

pub fn round(account: Option<&Account>, helpers: u8) -> Option<RoundView> {
    let round = Round::parse(ours(account)?.data.as_slice()).ok()?;
    Some(RoundView {
        state: round.state(),
        disputes: round.disputes(),
        beacon: *round.beacon(),
        cohort: Sites(*round.cohort()),
        inputs_root: *round.inputs_root(),
        close_head: *round.close_head(),
        mask_roots: (0..helpers as usize)
            .map(|helper| *round.mask_root(helper))
            .collect(),
        aggregate_root: *round.aggregate_root(),
        next_model: *round.next_model(),
        seal_head: *round.seal_head(),
        posted_slot: round.posted_slot(),
        settled_slot: round.settled_slot(),
        acked: Sites(*round.acked()),
        claimed: Sites(*round.claimed()),
    })
}

pub fn enrollment(account: Option<&Account>) -> Option<EnrollmentView> {
    let site = Enrollment::parse(ours(account)?.data.as_slice()).ok()?;
    Some(EnrollmentView {
        status: site.status(),
        wallet: *site.wallet(),
        status_lists: (0..site.status_ref_count() as usize)
            .map(|i| *site.status_ref(i).0)
            .collect(),
    })
}

pub fn dispute(account: Option<&Account>) -> Option<DisputeView> {
    let dispute = Dispute::parse(ours(account)?.data.as_slice()).ok()?;
    Some(DisputeView {
        leaf_index: dispute.leaf_index(),
        deadline: dispute.deadline(),
        seen_inputs: Sites(*dispute.seen_inputs()),
        seen_masks: dispute.seen_masks(),
        aggregate_given: dispute.aggregate_given(),
    })
}

pub fn demand(address: [u8; 32], account: &Account) -> Option<DemandView> {
    let demand = Demand::parse(ours(Some(account))?.data.as_slice()).ok()?;
    let part = match (demand.kind(), demand.position()) {
        (PART_INPUT, position) => ProofPart::Input(position),
        (PART_MASK, helper) => ProofPart::Mask(u8::try_from(helper).ok()?),
        (PART_AGGREGATE, _) => ProofPart::Aggregate,
        _ => return None,
    };
    Some(DemandView {
        address,
        challenger: *demand.challenger(),
        round: demand.round(),
        part,
        leaf: demand.leaf_index(),
        answered: demand.answered(),
        deadline: demand.deadline(),
        proof: demand.proof().to_vec(),
    })
}

pub fn status(account: Option<&Account>) -> Option<StatusView> {
    let list = StatusList::parse(ours(account)?.data.as_slice()).ok()?;
    Some(StatusView {
        bits: list.bits(),
        set: (0..list.bits()).filter(|bit| list.is_set(*bit)).collect(),
    })
}

pub fn buffer(account: Option<&Account>) -> Option<BufferView> {
    let buffer = Buffer::parse(ours(account)?.data.as_slice()).ok()?;
    Some(BufferView {
        capacity: buffer.capacity() as u32,
    })
}

/// A transaction that went through.
#[derive(Clone, Debug)]
pub struct Sent {
    pub what: &'static str,
    pub round: Option<u32>,
    pub by: [u8; 32],
    pub signature: String,
    pub budget: Budget,
    pub sent: Instant,
    pub confirmed: Instant,
}

pub type Sink = Arc<dyn Fn(Sent) + Send + Sync>;

pub struct Ledger {
    pub rpc: Rpc,
    pub wallet: Wallet,
    /// The priority fee to offer, in micro-lamports per compute unit.
    pub price: Option<u64>,
    pub confirm_every: Duration,
    pub sink: Option<Sink>,
}

/// The runtime's limit for a transaction that asks for none.
const DEFAULT_UNITS: u32 = 200_000;
const MAX_UNITS: u32 = 1_400_000;

impl Ledger {
    /// Asks for a unit limit only where it matters: when the instruction may
    /// need more than the default, or when units are paid for.
    pub fn budget(&self, ceiling: u32) -> Budget {
        let limited = self.price.is_some() || ceiling > DEFAULT_UNITS;
        Budget {
            units: limited.then_some(ceiling.min(MAX_UNITS)),
            price: self.price,
        }
    }

    /// Sends one instruction and waits until it is confirmed.
    pub fn send(
        &self,
        what: &'static str,
        round: Option<u32>,
        ix: client::Instruction,
        ceiling: u32,
    ) -> Result<(), ChainError> {
        let budget = self.budget(ceiling);
        let blockhash = self.rpc.latest_blockhash()?;
        let signed = tx::sign(&self.wallet, ix, budget, blockhash.hash)?;
        let sent = Instant::now();
        let signature = self.rpc.send_transaction(&signed.wire)?;
        loop {
            match self.rpc.confirmed(&signature)? {
                Some(Ok(())) => break,
                Some(Err(reason)) => {
                    let logs = Vec::new();
                    return Err(RpcError::Rejected { reason, logs }.into());
                }
                None if self.rpc.block_height()? > blockhash.last_valid_height => {
                    return Err(ChainError::Expired);
                }
                None => std::thread::sleep(self.confirm_every),
            }
        }
        tracing::info!("{what} confirmed: {signature}");
        if let Some(sink) = &self.sink {
            sink(Sent {
                what,
                round,
                by: *self.wallet.address(),
                signature,
                budget,
                sent,
                confirmed: Instant::now(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use fedtally_ledger::state::{ROUND_POSTED, ROUND_SPACE, TASK_SPACE};

    use super::*;

    fn held(data: Vec<u8>) -> Account {
        Account {
            lamports: 5,
            owner: program(),
            executable: false,
            data,
        }
    }

    #[test]
    fn a_round_reads_back_what_the_program_wrote() {
        let mut data = vec![0u8; ROUND_SPACE];
        let mut view = Round::init(data.as_mut_slice()).unwrap();
        view.set_state(ROUND_POSTED);
        view.set_posted_slot(77);
        *view.cohort_mut() = client::site_bitmap([1, 9, 700]);
        *view.aggregate_root_mut() = [4; 32];
        view.mask_roots_mut()[32..64].copy_from_slice(&[6; 32]);
        let account = held(data);

        let round = round(Some(&account), 2).unwrap();
        assert_eq!(round.state, ROUND_POSTED);
        assert_eq!(round.posted_slot, 77);
        assert_eq!(round.cohort.members(), [1, 9, 700]);
        assert_eq!(round.cohort, [700, 1, 9].into_iter().collect());
        assert_eq!(round.mask_roots, [[0; 32], [6; 32]]);
        assert_eq!(round.aggregate_root, [4; 32]);
        assert!(task(Some(&account)).is_none(), "a round was read as a task");
    }

    #[test]
    fn only_this_programs_accounts_are_read() {
        let mut data = vec![0u8; TASK_SPACE];
        Task::init(data.as_mut_slice()).unwrap().set_rounds(3);
        let mut account = held(data);
        assert_eq!(task(Some(&account)).unwrap().rounds, 3);
        account.owner = [0; 32];
        assert!(task(Some(&account)).is_none());
        assert!(task(None).is_none());
        account.owner = program();
        account.data.truncate(100);
        assert!(task(Some(&account)).is_none());
    }

    #[test]
    fn a_site_past_the_bitmap_is_not_a_member() {
        let sites: Sites = [3, 5000].into_iter().collect();
        assert_eq!(sites.members(), [3]);
        assert!(!sites.has(5000));
        assert_eq!(sites.len(), 1);
    }
}
