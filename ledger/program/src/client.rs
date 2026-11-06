//! Instruction builders and account addresses for clients, available off
//! chain.
//!
//! Each builder returns an [`Instruction`] holding the program ID, the
//! account metas in the order the program reads them and the encoded
//! arguments. Addresses are plain 32 byte arrays, so an instruction converts
//! into the types of any Solana SDK generation. The builders only encode.
//!
//! Accounts decode through the views in [`crate::state`], which work on any
//! byte slice:
//!
//! ```
//! use fedtally_ledger::{client, state::Task};
//!
//! let publisher = [7u8; 32];
//! let terms = client::TaskTerms { sites: 4, min_cohort: 2, rounds: 1, ..Default::default() };
//! let ix = client::publish(&publisher, &terms);
//! assert_eq!(ix.data[..8], fedtally_ledger::ix::PUBLISH);
//! assert_eq!(ix.accounts[1].address, client::task(&publisher, &terms.plan_id));
//!
//! let account_data = vec![0u8; 561];
//! assert!(Task::parse(account_data.as_slice()).is_err());
//! ```

use std::vec::Vec;

use crate::{
    bitmap::{self, SiteBitmap, SITE_BITMAP_LEN},
    hash::Digest,
    instructions::{
        dispute_leaf::{PART_AGGREGATE, PART_INPUT, PART_MASK},
        withdraw::{ROLE_AGGREGATOR, ROLE_PUBLISHER, ROLE_SITE},
    },
    state::{
        demand_place, BUFFER_SEED, DEMAND_SEED, DISPUTE_SEED, ROUND_SEED, SITE_SEED, STATUS_SEED,
        TASK_SEED,
    },
};

/// The system program, `11111111111111111111111111111111`.
pub const SYSTEM_PROGRAM: [u8; 32] = [0; 32];

/// The chunk size [`upload`] writes, which keeps every `buffer_write`
/// transaction under the 1232 byte limit even with a separate fee payer.
pub const CHUNK_LEN: usize = 900;

/// An account an instruction reads, with its signer and writable roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccountMeta {
    pub address: [u8; 32],
    pub is_signer: bool,
    pub is_writable: bool,
}

impl AccountMeta {
    /// A writable account.
    pub const fn writable(address: [u8; 32], is_signer: bool) -> Self {
        Self {
            address,
            is_signer,
            is_writable: true,
        }
    }

    /// A read-only account.
    pub const fn readonly(address: [u8; 32], is_signer: bool) -> Self {
        Self {
            address,
            is_signer,
            is_writable: false,
        }
    }
}

/// One ledger instruction, ready to convert into an SDK instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub program_id: [u8; 32],
    pub accounts: Vec<AccountMeta>,
    pub data: Vec<u8>,
}

impl Instruction {
    fn new(discriminator: [u8; 8], accounts: Vec<AccountMeta>) -> Self {
        Self {
            program_id: *crate::ID.as_array(),
            accounts,
            data: discriminator.to_vec(),
        }
    }

    fn raw(mut self, value: &[u8]) -> Self {
        self.data.extend_from_slice(value);
        self
    }

    /// A u32 count, then the items back to back.
    fn items<T: AsRef<[u8]>>(mut self, items: &[T]) -> Self {
        self.data
            .extend_from_slice(&(items.len() as u32).to_le_bytes());
        for item in items {
            self.data.extend_from_slice(item.as_ref());
        }
        self
    }

    fn account(mut self, meta: AccountMeta) -> Self {
        self.accounts.push(meta);
        self
    }
}

fn find(seeds: &[&[u8]]) -> [u8; 32] {
    let (address, _) = pinocchio::Address::find_program_address(seeds, &crate::ID);
    *address.as_array()
}

/// The task a publisher has for a plan, `["task", publisher, plan_id]`.
pub fn task(publisher: &[u8; 32], plan_id: &[u8; 32]) -> [u8; 32] {
    find(&[TASK_SEED, publisher, plan_id])
}

/// The enrollment of site `index`, `["site", task, index u16 LE]`.
pub fn enrollment(task: &[u8; 32], index: u16) -> [u8; 32] {
    find(&[SITE_SEED, task, &index.to_le_bytes()])
}

/// Round `round` of a task, `["round", task, round u32 LE]`.
pub fn round(task: &[u8; 32], round: u32) -> [u8; 32] {
    find(&[ROUND_SEED, task, &round.to_le_bytes()])
}

/// A challenger's dispute over round `round`,
/// `["dispute", task, round u32 LE, challenger]`.
pub fn dispute(task: &[u8; 32], round: u32, challenger: &[u8; 32]) -> [u8; 32] {
    find(&[DISPUTE_SEED, task, &round.to_le_bytes(), challenger])
}

/// A challenger's demand for one leaf of one vector of round `round`.
pub fn demand(
    task: &[u8; 32],
    challenger: &[u8; 32],
    round: u32,
    part: ProofPart,
    leaf_index: u64,
) -> [u8; 32] {
    let (kind, position) = part.encoded();
    find(&[
        DEMAND_SEED,
        task,
        challenger,
        &demand_place(round, kind, position, leaf_index),
    ])
}

/// Status list `list` of an issuer, `["status", issuer, list u32 LE]`.
pub fn status_list(issuer: &[u8; 32], list: u32) -> [u8; 32] {
    find(&[STATUS_SEED, issuer, &list.to_le_bytes()])
}

/// The buffer of a wallet, `["buffer", owner]`.
pub fn buffer(owner: &[u8; 32]) -> [u8; 32] {
    find(&[BUFFER_SEED, owner])
}

/// What a publisher sets for a task, in the order `publish` encodes it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TaskTerms {
    pub aggregator: [u8; 32],
    pub plan_id: [u8; 32],
    pub sites_root: Digest,
    /// The digest of the initial model.
    pub model: Digest,
    pub sites: u16,
    pub helpers: u8,
    pub coords: u64,
    pub rounds: u32,
    pub min_cohort: u16,
    /// How many sites a round invites, or 0 to invite them all.
    pub sample: u16,
    pub sensitivity: u64,
    pub noise: u64,
    pub colluding: u16,
    pub budget: u64,
    pub site_stake: u64,
    pub aggregator_stake: u64,
    pub reward_per_round: u64,
    pub dispute_bond: u64,
    pub challenge_slots: u64,
    pub stall_slots: u64,
}

/// One status list bit that revokes an enrollment when it is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatusRef {
    pub list: [u8; 32],
    pub bit: u32,
}

/// A part of a dispute's proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofPart {
    /// The input of the member at this position of the cohort, counted in
    /// ascending site index.
    Input(u16),
    /// The mask of this helper.
    Mask(u8),
    Aggregate,
}

impl ProofPart {
    fn encoded(self) -> (u8, u16) {
        match self {
            ProofPart::Input(position) => (PART_INPUT, position),
            ProofPart::Mask(helper) => (PART_MASK, helper as u16),
            ProofPart::Aggregate => (PART_AGGREGATE, 0),
        }
    }
}

/// Publishes a task and deposits the reward of every round.
pub fn publish(publisher: &[u8; 32], terms: &TaskTerms) -> Instruction {
    Instruction::new(
        crate::ix::PUBLISH,
        std::vec![
            AccountMeta::writable(*publisher, true),
            AccountMeta::writable(task(publisher, &terms.plan_id), false),
            AccountMeta::readonly(SYSTEM_PROGRAM, false),
        ],
    )
    .raw(&terms.aggregator)
    .raw(&terms.plan_id)
    .raw(&terms.sites_root)
    .raw(&terms.model)
    .raw(&terms.sites.to_le_bytes())
    .raw(&[terms.helpers])
    .raw(&terms.coords.to_le_bytes())
    .raw(&terms.rounds.to_le_bytes())
    .raw(&terms.min_cohort.to_le_bytes())
    .raw(&terms.sample.to_le_bytes())
    .raw(&terms.sensitivity.to_le_bytes())
    .raw(&terms.noise.to_le_bytes())
    .raw(&terms.colluding.to_le_bytes())
    .raw(&terms.budget.to_le_bytes())
    .raw(&terms.site_stake.to_le_bytes())
    .raw(&terms.aggregator_stake.to_le_bytes())
    .raw(&terms.reward_per_round.to_le_bytes())
    .raw(&terms.dispute_bond.to_le_bytes())
    .raw(&terms.challenge_slots.to_le_bytes())
    .raw(&terms.stall_slots.to_le_bytes())
}

/// Stakes on a task as its aggregator.
pub fn back(aggregator: &[u8; 32], task: &[u8; 32]) -> Instruction {
    Instruction::new(
        crate::ix::BACK,
        std::vec![
            AccountMeta::writable(*aggregator, true),
            AccountMeta::writable(*task, false),
            AccountMeta::readonly(SYSTEM_PROGRAM, false),
        ],
    )
}

/// Enrolls `wallet` as site `index`, with its path in the sites root.
pub fn enroll(
    wallet: &[u8; 32],
    task: &[u8; 32],
    index: u16,
    path: &[Digest],
    admission: &Digest,
    status_refs: &[StatusRef],
) -> Instruction {
    let status_refs: Vec<[u8; 36]> = status_refs
        .iter()
        .map(|status_ref| {
            let mut entry = [0u8; 36];
            entry[..32].copy_from_slice(&status_ref.list);
            entry[32..].copy_from_slice(&status_ref.bit.to_le_bytes());
            entry
        })
        .collect();
    Instruction::new(
        crate::ix::ENROLL,
        std::vec![
            AccountMeta::writable(*wallet, true),
            AccountMeta::writable(*task, false),
            AccountMeta::writable(enrollment(task, index), false),
            AccountMeta::readonly(SYSTEM_PROGRAM, false),
        ],
    )
    .raw(&index.to_le_bytes())
    .items(path)
    .raw(admission)
    .items(&status_refs)
}

/// Opens round `number`, which has to be the task's next.
pub fn open_round(aggregator: &[u8; 32], task: &[u8; 32], number: u32) -> Instruction {
    Instruction::new(
        crate::ix::OPEN_ROUND,
        std::vec![
            AccountMeta::writable(*aggregator, true),
            AccountMeta::writable(*task, false),
            AccountMeta::writable(round(task, number), false),
            AccountMeta::readonly(SYSTEM_PROGRAM, false),
        ],
    )
}

fn aggregator_step(
    discriminator: [u8; 8],
    aggregator: &[u8; 32],
    task: &[u8; 32],
    number: u32,
) -> Instruction {
    Instruction::new(
        discriminator,
        std::vec![
            AccountMeta::readonly(*aggregator, true),
            AccountMeta::writable(*task, false),
            AccountMeta::writable(round(task, number), false),
        ],
    )
}

/// Commits the cohort and the root over its inputs. A task that does not
/// sample ignores `threshold`.
pub fn commit_root(
    aggregator: &[u8; 32],
    task: &[u8; 32],
    number: u32,
    cohort: &SiteBitmap,
    inputs_root: &Digest,
    close_head: &Digest,
    threshold: &Digest,
) -> Instruction {
    aggregator_step(crate::ix::COMMIT_ROOT, aggregator, task, number)
        .raw(cohort)
        .raw(inputs_root)
        .raw(close_head)
        .raw(threshold)
}

/// Posts the round's mask roots, one per helper, and its result.
pub fn post_aggregate(
    aggregator: &[u8; 32],
    task: &[u8; 32],
    number: u32,
    mask_roots: &[Digest],
    aggregate_root: &Digest,
    next_model: &Digest,
    seal_head: &Digest,
) -> Instruction {
    aggregator_step(crate::ix::POST_AGGREGATE, aggregator, task, number)
        .items(mask_roots)
        .raw(aggregate_root)
        .raw(next_model)
        .raw(seal_head)
}

/// Acknowledges round `number` as site `index`.
pub fn ack(
    wallet: &[u8; 32],
    task: &[u8; 32],
    index: u16,
    number: u32,
    seal_head: &Digest,
) -> Instruction {
    Instruction::new(
        crate::ix::ACK,
        std::vec![
            AccountMeta::readonly(*wallet, true),
            AccountMeta::readonly(enrollment(task, index), false),
            AccountMeta::writable(round(task, number), false),
        ],
    )
    .raw(&number.to_le_bytes())
    .raw(seal_head)
}

/// Settles round `number`.
pub fn settle(task: &[u8; 32], number: u32) -> Instruction {
    Instruction::new(
        crate::ix::SETTLE,
        std::vec![
            AccountMeta::writable(*task, false),
            AccountMeta::writable(round(task, number), false),
        ],
    )
}

/// Claims the share of site `index` in round `number`.
pub fn claim(wallet: &[u8; 32], task: &[u8; 32], index: u16, number: u32) -> Instruction {
    Instruction::new(
        crate::ix::CLAIM,
        std::vec![
            AccountMeta::writable(*wallet, true),
            AccountMeta::readonly(enrollment(task, index), false),
            AccountMeta::writable(*task, false),
            AccountMeta::writable(round(task, number), false),
        ],
    )
}

/// Opens a dispute over leaf `leaf_index` of round `number`.
pub fn dispute_open(
    challenger: &[u8; 32],
    task: &[u8; 32],
    number: u32,
    leaf_index: u64,
) -> Instruction {
    Instruction::new(
        crate::ix::DISPUTE_OPEN,
        std::vec![
            AccountMeta::writable(*challenger, true),
            AccountMeta::readonly(*task, false),
            AccountMeta::writable(round(task, number), false),
            AccountMeta::writable(dispute(task, number, challenger), false),
            AccountMeta::readonly(SYSTEM_PROGRAM, false),
        ],
    )
    .raw(&number.to_le_bytes())
    .raw(&leaf_index.to_le_bytes())
}

/// Gives one part of the proof, which the challenger's buffer holds.
pub fn dispute_leaf(
    challenger: &[u8; 32],
    task: &[u8; 32],
    number: u32,
    part: ProofPart,
) -> Instruction {
    let (kind, position) = part.encoded();
    Instruction::new(
        crate::ix::DISPUTE_LEAF,
        std::vec![
            AccountMeta::readonly(*challenger, true),
            AccountMeta::readonly(*task, false),
            AccountMeta::readonly(round(task, number), false),
            AccountMeta::writable(dispute(task, number, challenger), false),
            AccountMeta::readonly(buffer(challenger), false),
        ],
    )
    .raw(&[kind])
    .raw(&position.to_le_bytes())
}

/// Closes the dispute over round `number`.
pub fn dispute_close(
    task: &[u8; 32],
    number: u32,
    challenger: &[u8; 32],
    aggregator: &[u8; 32],
) -> Instruction {
    Instruction::new(
        crate::ix::DISPUTE_CLOSE,
        std::vec![
            AccountMeta::writable(*task, false),
            AccountMeta::writable(round(task, number), false),
            AccountMeta::writable(dispute(task, number, challenger), false),
            AccountMeta::writable(*challenger, false),
            AccountMeta::writable(*aggregator, false),
        ],
    )
}

/// Opens a dispute past the window, on the strength of an answered demand.
pub fn dispute_open_after_demand(
    challenger: &[u8; 32],
    task: &[u8; 32],
    number: u32,
    part: ProofPart,
    leaf_index: u64,
) -> Instruction {
    let mut ix = dispute_open(challenger, task, number, leaf_index);
    ix.accounts.push(AccountMeta::readonly(
        demand(task, challenger, number, part, leaf_index),
        false,
    ));
    ix
}

/// Demands one leaf of one vector of a posted round.
pub fn demand_open(
    challenger: &[u8; 32],
    task: &[u8; 32],
    number: u32,
    part: ProofPart,
    leaf_index: u64,
) -> Instruction {
    let (kind, position) = part.encoded();
    Instruction::new(
        crate::ix::DEMAND_OPEN,
        std::vec![
            AccountMeta::writable(*challenger, true),
            AccountMeta::readonly(*task, false),
            AccountMeta::writable(round(task, number), false),
            AccountMeta::writable(demand(task, challenger, number, part, leaf_index), false),
            AccountMeta::readonly(SYSTEM_PROGRAM, false),
        ],
    )
    .raw(&number.to_le_bytes())
    .raw(&[kind])
    .raw(&position.to_le_bytes())
    .raw(&leaf_index.to_le_bytes())
}

/// Answers a demand with the proof in the responder's buffer.
pub fn demand_answer(
    responder: &[u8; 32],
    task: &[u8; 32],
    number: u32,
    demand: [u8; 32],
) -> Instruction {
    Instruction::new(
        crate::ix::DEMAND_ANSWER,
        std::vec![
            AccountMeta::writable(*responder, true),
            AccountMeta::readonly(*task, false),
            AccountMeta::readonly(round(task, number), false),
            AccountMeta::writable(demand, false),
            AccountMeta::readonly(buffer(responder), false),
        ],
    )
}

/// Closes a demand. Before its deadline only the challenger's signature will do.
pub fn demand_close(
    task: &[u8; 32],
    number: u32,
    demand: [u8; 32],
    challenger: &[u8; 32],
    signed: bool,
) -> Instruction {
    Instruction::new(
        crate::ix::DEMAND_CLOSE,
        std::vec![
            AccountMeta::writable(*task, false),
            AccountMeta::writable(round(task, number), false),
            AccountMeta::writable(demand, false),
            AccountMeta::writable(*challenger, signed),
        ],
    )
}

/// Closes a final, fully claimed round and returns its rent to the aggregator.
pub fn round_close(aggregator: &[u8; 32], task: &[u8; 32], number: u32) -> Instruction {
    Instruction::new(
        crate::ix::ROUND_CLOSE,
        std::vec![
            AccountMeta::writable(*aggregator, false),
            AccountMeta::readonly(*task, false),
            AccountMeta::writable(round(task, number), false),
        ],
    )
}

/// Creates status list `list` of the issuer with `bits` bits.
pub fn status_init(issuer: &[u8; 32], list: u32, bits: u32) -> Instruction {
    Instruction::new(
        crate::ix::STATUS_INIT,
        std::vec![
            AccountMeta::writable(*issuer, true),
            AccountMeta::writable(status_list(issuer, list), false),
            AccountMeta::readonly(SYSTEM_PROGRAM, false),
        ],
    )
    .raw(&list.to_le_bytes())
    .raw(&bits.to_le_bytes())
}

/// Sets bit `bit` of the issuer's status list `list`.
pub fn revoke(issuer: &[u8; 32], list: u32, bit: u32) -> Instruction {
    Instruction::new(
        crate::ix::REVOKE,
        std::vec![
            AccountMeta::readonly(*issuer, true),
            AccountMeta::writable(status_list(issuer, list), false),
        ],
    )
    .raw(&bit.to_le_bytes())
}

/// Applies the status lists the enrollment of site `index` names, which are
/// passed in the order the enrollment names them.
pub fn revoke_check(task: &[u8; 32], index: u16, status_lists: &[[u8; 32]]) -> Instruction {
    status_lists.iter().fold(
        Instruction::new(
            crate::ix::REVOKE_CHECK,
            std::vec![
                AccountMeta::writable(*task, false),
                AccountMeta::writable(enrollment(task, index), false),
            ],
        ),
        |ix, list| ix.account(AccountMeta::readonly(*list, false)),
    )
}

/// The task's latest round as an extra account, when it has one.
fn with_latest_round(ix: Instruction, task: &[u8; 32], latest_round: u32) -> Instruction {
    match latest_round {
        0 => ix,
        number => ix.account(AccountMeta::readonly(round(task, number), false)),
    }
}

/// Aborts a task. `latest_round` is the task's `round`, 0 before any.
pub fn abort(caller: &[u8; 32], task: &[u8; 32], latest_round: u32) -> Instruction {
    let ix = Instruction::new(
        crate::ix::ABORT,
        std::vec![
            AccountMeta::readonly(*caller, true),
            AccountMeta::writable(*task, false),
        ],
    );
    with_latest_round(ix, task, latest_round)
}

fn withdraw(wallet: &[u8; 32], task: AccountMeta, role: u8) -> Instruction {
    Instruction::new(
        crate::ix::WITHDRAW,
        std::vec![AccountMeta::writable(*wallet, true), task],
    )
    .raw(&[role])
}

/// Takes back the stake of site `index`.
pub fn withdraw_site(wallet: &[u8; 32], task: &[u8; 32], index: u16) -> Instruction {
    withdraw(wallet, AccountMeta::readonly(*task, false), ROLE_SITE)
        .account(AccountMeta::writable(enrollment(task, index), false))
}

/// Takes back the aggregator's stake. `latest_round` is the task's `round`.
pub fn withdraw_aggregator(
    aggregator: &[u8; 32],
    task: &[u8; 32],
    latest_round: u32,
) -> Instruction {
    let ix = withdraw(
        aggregator,
        AccountMeta::writable(*task, false),
        ROLE_AGGREGATOR,
    );
    with_latest_round(ix, task, latest_round)
}

/// Takes what the task holds beyond its rent, the rewards owed to sites and
/// the aggregator's stake.
pub fn withdraw_publisher(publisher: &[u8; 32], task: &[u8; 32]) -> Instruction {
    withdraw(
        publisher,
        AccountMeta::writable(*task, false),
        ROLE_PUBLISHER,
    )
}

/// Creates the owner's buffer with room for `capacity` bytes.
pub fn buffer_init(owner: &[u8; 32], capacity: u32) -> Instruction {
    Instruction::new(
        crate::ix::BUFFER_INIT,
        std::vec![
            AccountMeta::writable(*owner, true),
            AccountMeta::writable(buffer(owner), false),
            AccountMeta::readonly(SYSTEM_PROGRAM, false),
        ],
    )
    .raw(&capacity.to_le_bytes())
}

/// Writes `chunk` at `offset`, which is 0 to start the buffer over or the
/// end of the previous write to continue it.
pub fn buffer_write(owner: &[u8; 32], offset: u32, chunk: &[u8]) -> Instruction {
    Instruction::new(
        crate::ix::BUFFER_WRITE,
        std::vec![
            AccountMeta::readonly(*owner, true),
            AccountMeta::writable(buffer(owner), false),
        ],
    )
    .raw(&offset.to_le_bytes())
    .raw(&(chunk.len() as u32).to_le_bytes())
    .raw(chunk)
}

/// Closes the owner's buffer and returns its rent.
pub fn buffer_close(owner: &[u8; 32]) -> Instruction {
    Instruction::new(
        crate::ix::BUFFER_CLOSE,
        std::vec![
            AccountMeta::writable(*owner, true),
            AccountMeta::writable(buffer(owner), false),
        ],
    )
}

/// The writes that replace the contents of the owner's buffer with `proof`,
/// in [`CHUNK_LEN`] chunks. Each goes in a transaction of its own, in order.
pub fn upload(owner: &[u8; 32], proof: &[u8]) -> Vec<Instruction> {
    proof
        .chunks(CHUNK_LEN)
        .enumerate()
        .map(|(i, chunk)| buffer_write(owner, (i * CHUNK_LEN) as u32, chunk))
        .collect()
}

/// The proof of an input leaf: the leaf, the member's vector root, that
/// root's path in the inputs root, and the leaf's path in the vector root.
pub fn input_proof(
    leaf: &[u8],
    vector_root: &Digest,
    inputs_path: &[Digest],
    leaf_path: &[Digest],
) -> Vec<u8> {
    let mut proof = leaf.to_vec();
    proof.extend_from_slice(vector_root);
    proof.extend(inputs_path.iter().flatten());
    proof.extend(leaf_path.iter().flatten());
    proof
}

/// The proof of a mask leaf or of the aggregate leaf: the leaf and its path
/// in the vector root the round posted.
pub fn leaf_proof(leaf: &[u8], leaf_path: &[Digest]) -> Vec<u8> {
    let mut proof = leaf.to_vec();
    proof.extend(leaf_path.iter().flatten());
    proof
}

/// The bitmap with the bits of `sites` set.
pub fn site_bitmap(sites: impl IntoIterator<Item = u16>) -> SiteBitmap {
    let mut bits = [0u8; SITE_BITMAP_LEN];
    for site in sites {
        bitmap::set(&mut bits, site as usize);
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_match_the_stored_derivation() {
        let task_address = task(&[8; 32], &[3; 32]);
        let mut data = std::vec![0u8; crate::state::ROUND_SPACE];
        let mut view = crate::state::Round::init(data.as_mut_slice()).unwrap();
        *view.task_mut() = task_address;
        view.set_round(5);
        let expected = round(&task_address, 5);
        let bump = (0..=255u8)
            .rev()
            .find(|bump| {
                view.set_bump(*bump);
                view.address().as_array() == &expected
            })
            .unwrap();
        view.set_bump(bump);
        assert_eq!(view.address().as_array(), &expected);
        assert_ne!(dispute(&task_address, 5, &[1; 32]), expected);
        assert_ne!(round(&task_address, 6), expected);
    }

    #[test]
    fn uploads_cover_the_proof_in_order() {
        let owner = [1; 32];
        let proof: Vec<u8> = (0..2000u32).map(|i| i as u8).collect();
        let writes = upload(&owner, &proof);
        assert_eq!(writes.len(), 3);
        let mut seen = Vec::new();
        for write in &writes {
            assert_eq!(write.data[..8], crate::ix::BUFFER_WRITE);
            let offset = u32::from_le_bytes(write.data[8..12].try_into().unwrap());
            let len = u32::from_le_bytes(write.data[12..16].try_into().unwrap());
            assert_eq!(offset as usize, seen.len());
            assert_eq!(len as usize, write.data.len() - 16);
            seen.extend_from_slice(&write.data[16..]);
        }
        assert_eq!(seen, proof);
    }

    #[test]
    fn proofs_lay_out_leaf_then_paths() {
        let leaf = [9u8; 16];
        let proof = input_proof(&leaf, &[1; 32], &[[2; 32]], &[[3; 32], [4; 32]]);
        assert_eq!(proof.len(), 16 + 32 * 4);
        assert_eq!(proof[16..48], [1; 32]);
        assert_eq!(proof[48..80], [2; 32]);
        assert_eq!(proof[112..], [4; 32]);
        assert_eq!(leaf_proof(&leaf, &[[3; 32]]).len(), 48);
        assert_eq!(site_bitmap([0, 9])[..2], [1, 2]);
    }
}
