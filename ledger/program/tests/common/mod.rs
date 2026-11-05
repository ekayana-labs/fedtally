//! What the test files share: a LiteSVM world that sends the client
//! module's instructions, a task driven through its rounds, and a reference
//! of the shared definitions written from the specification alone.
//!
//! Build the program first with
//! `cargo build-sbf --manifest-path program/Cargo.toml`.

#![allow(dead_code)]

pub mod reference;
pub mod vectors;

use std::path::PathBuf;

use fedtally_ledger::{
    client::{self, ProofPart, StatusRef, TaskTerms},
    error::LedgerError,
    state::{Dispute, Enrollment, Round, StatusList, Task},
};
use litesvm::LiteSVM;
use solana_compute_budget_interface::ComputeBudgetInstruction;
use solana_instruction::{AccountMeta, Instruction};
use solana_keypair::Keypair;
use solana_message::{Message, VersionedMessage};
use solana_pubkey::Pubkey;
use solana_signer::Signer;
use solana_transaction::versioned::VersionedTransaction;

pub type Hash = [u8; 32];

/// What every wallet starts with.
pub const WALLET: u64 = 10_000_000_000;

pub fn program_id() -> Pubkey {
    Pubkey::new_from_array(*fedtally_ledger::ID.as_array())
}

fn program_so() -> Vec<u8> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/deploy/fedtally_ledger.so");
    std::fs::read(&path).unwrap_or_else(|_| {
        panic!(
            "{} not found, run `cargo build-sbf --manifest-path program/Cargo.toml` first",
            path.display(),
        )
    })
}

pub fn sdk(ix: client::Instruction) -> Instruction {
    Instruction {
        program_id: Pubkey::new_from_array(ix.program_id),
        accounts: ix
            .accounts
            .into_iter()
            .map(|meta| AccountMeta {
                pubkey: Pubkey::new_from_array(meta.address),
                is_signer: meta.is_signer,
                is_writable: meta.is_writable,
            })
            .collect(),
        data: ix.data,
    }
}

pub fn key(wallet: &Keypair) -> [u8; 32] {
    wallet.pubkey().to_bytes()
}

/// Why a transaction is expected to fail.
#[derive(Clone, Copy, Debug)]
pub enum Refusal {
    Ledger(LedgerError),
    /// The name of a built-in program error, as the runtime prints it.
    Program(&'static str),
}

impl From<LedgerError> for Refusal {
    fn from(error: LedgerError) -> Self {
        Refusal::Ledger(error)
    }
}

impl From<&'static str> for Refusal {
    fn from(name: &'static str) -> Self {
        Refusal::Program(name)
    }
}

/// The lamports, owner and data of an account, or `None` if it does not
/// exist.
type Snapshot = Option<(u64, Pubkey, Vec<u8>)>;

pub struct World {
    pub svm: LiteSVM,
    /// Pays every transaction fee, so wallets move only by what the program
    /// moves.
    pub fees: Keypair,
    pub slot: u64,
    /// The number of the next wallet when wallets are derived from fixed
    /// seeds rather than drawn at random.
    fixed: Option<u32>,
}

impl World {
    pub fn new() -> Self {
        Self::with(Keypair::new(), None)
    }

    /// A world whose wallets are the same on every run, so that the bump
    /// searches of its program addresses cost the same every time.
    pub fn fixed() -> Self {
        Self::with(Keypair::new_from_array([0xfe; 32]), Some(0))
    }

    fn with(fees: Keypair, fixed: Option<u32>) -> Self {
        let mut svm = LiteSVM::new();
        svm.add_program(program_id(), &program_so()).unwrap();
        svm.airdrop(&fees.pubkey(), 1_000 * WALLET).unwrap();
        svm.warp_to_slot(0);
        Self {
            svm,
            fees,
            slot: 0,
            fixed,
        }
    }

    pub fn wallet(&mut self) -> Keypair {
        let wallet = match self.fixed.as_mut() {
            Some(next) => {
                *next += 1;
                Keypair::new_from_array(reference::sha(&[b"wallet", &next.to_le_bytes()]))
            }
            None => Keypair::new(),
        };
        self.svm.airdrop(&wallet.pubkey(), WALLET).unwrap();
        wallet
    }

    pub fn warp(&mut self, slots: u64) {
        self.slot += slots;
        self.svm.warp_to_slot(self.slot);
    }

    pub fn balance(&self, address: &[u8; 32]) -> u64 {
        self.svm
            .get_balance(&Pubkey::new_from_array(*address))
            .unwrap_or(0)
    }

    pub fn exists(&self, address: &[u8; 32]) -> bool {
        self.snapshot(&Pubkey::new_from_array(*address)).is_some()
    }

    pub fn data(&self, address: &[u8; 32]) -> Vec<u8> {
        let account = self
            .svm
            .get_account(&Pubkey::new_from_array(*address))
            .expect("account exists");
        assert_eq!(account.owner, program_id(), "account owner");
        account.data
    }

    pub fn rent(&self, len: usize) -> u64 {
        self.svm.minimum_balance_for_rent_exemption(len)
    }

    fn snapshot(&self, address: &Pubkey) -> Snapshot {
        self.svm
            .get_account(address)
            .filter(|account| account.lamports > 0)
            .map(|account| (account.lamports, account.owner, account.data))
    }

    fn transact(
        &mut self,
        ixs: &[Instruction],
        signers: &[&Keypair],
    ) -> Result<u64, (String, Vec<String>)> {
        // A fresh blockhash, so a repeated instruction is a new transaction.
        self.svm.expire_blockhash();
        let message = Message::new_with_blockhash(
            ixs,
            Some(&self.fees.pubkey()),
            &self.svm.latest_blockhash(),
        );
        let mut all = vec![&self.fees];
        for signer in signers {
            if all.iter().all(|known| known.pubkey() != signer.pubkey()) {
                all.push(signer);
            }
        }
        let tx = VersionedTransaction::try_new(VersionedMessage::Legacy(message), &all)
            .map_err(|e| (e.to_string(), Vec::new()))?;
        self.svm
            .send_transaction(tx)
            .map(|meta| meta.compute_units_consumed)
            .map_err(|e| (format!("{:?}", e.err), e.meta.logs))
    }

    pub fn send(&mut self, ix: client::Instruction, signers: &[&Keypair]) -> Result<u64, String> {
        self.transact(&[sdk(ix)], signers)
            .map_err(|(error, _)| error)
    }

    /// Sends an instruction that has to succeed and returns the compute
    /// units it took.
    pub fn run(&mut self, ix: client::Instruction, signers: &[&Keypair]) -> u64 {
        self.transact(&[sdk(ix)], signers)
            .unwrap_or_else(|(error, logs)| panic!("{error}\n{}", logs.join("\n")))
    }

    /// As [`World::run`], with a compute unit limit above the default.
    pub fn run_with_limit(
        &mut self,
        ix: client::Instruction,
        signers: &[&Keypair],
        units: u32,
    ) -> u64 {
        let limit = ComputeBudgetInstruction::set_compute_unit_limit(units);
        let total = self
            .transact(&[limit, sdk(ix)], signers)
            .unwrap_or_else(|(error, logs)| panic!("{error}\n{}", logs.join("\n")));
        // The compute budget program charges for its own instruction.
        total - 150
    }

    /// Sends an instruction that has to be refused for `why`, and checks
    /// that no account it names changed.
    pub fn refuse(
        &mut self,
        ix: client::Instruction,
        signers: &[&Keypair],
        why: impl Into<Refusal>,
    ) {
        let ix = sdk(ix);
        let addresses: Vec<Pubkey> = ix.accounts.iter().map(|meta| meta.pubkey).collect();
        let before: Vec<Snapshot> = addresses.iter().map(|a| self.snapshot(a)).collect();

        let why = why.into();
        let wanted = match why {
            Refusal::Ledger(error) => format!("Custom({})", error.code()),
            Refusal::Program(name) => name.to_string(),
        };
        match self.transact(&[ix], signers) {
            Ok(_) => panic!("expected {why:?}, but the transaction went through"),
            Err((error, _)) => assert!(error.contains(&wanted), "expected {why:?}, got {error}"),
        }

        let after: Vec<Snapshot> = addresses.iter().map(|a| self.snapshot(a)).collect();
        assert_eq!(before, after, "a refused transaction changed state");
    }

    pub fn task(&self, address: &[u8; 32]) -> Task<Vec<u8>> {
        Task::parse(self.data(address)).unwrap()
    }

    pub fn round(&self, task: &[u8; 32], number: u32) -> Round<Vec<u8>> {
        Round::parse(self.data(&client::round(task, number))).unwrap()
    }

    pub fn enrollment(&self, task: &[u8; 32], index: u16) -> Enrollment<Vec<u8>> {
        Enrollment::parse(self.data(&client::enrollment(task, index))).unwrap()
    }

    pub fn dispute(&self, task: &[u8; 32], number: u32, challenger: &Keypair) -> Dispute<Vec<u8>> {
        Dispute::parse(self.data(&client::dispute(task, number, &key(challenger)))).unwrap()
    }

    pub fn status_list(&self, issuer: &[u8; 32], list: u32) -> StatusList<Vec<u8>> {
        StatusList::parse(self.data(&client::status_list(issuer, list))).unwrap()
    }

    /// Replaces the contents of the owner's buffer, which must exist.
    pub fn upload(&mut self, owner: &Keypair, proof: &[u8]) {
        for write in client::upload(&key(owner), proof) {
            self.run(write, &[owner]);
        }
    }
}

/// What one round posts, beyond its cohort.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Posting {
    pub inputs_root: Hash,
    pub close_head: Hash,
    pub mask_roots: Vec<Hash>,
    pub aggregate_root: Hash,
    pub next_model: Hash,
    pub seal_head: Hash,
}

impl Posting {
    /// A posting whose digests only have to differ from round to round.
    pub fn filler(number: u32, helpers: u8) -> Self {
        let tag = |kind: u8| reference::sha(&[&[kind], &number.to_le_bytes()]);
        Self {
            inputs_root: tag(1),
            close_head: tag(2),
            mask_roots: (0..helpers).map(|helper| tag(10 + helper)).collect(),
            aggregate_root: tag(3),
            next_model: tag(4),
            seal_head: tag(5),
        }
    }
}

/// A task with its wallets, driven through the client module.
pub struct Run {
    pub world: World,
    pub publisher: Keypair,
    pub aggregator: Keypair,
    pub sites: Vec<Keypair>,
    pub terms: TaskTerms,
    pub task: [u8; 32],
}

impl Run {
    /// A world with a publisher, an aggregator and `sites` site wallets, and
    /// the terms of a task over them. Nothing is sent yet. The defaults are
    /// a task without privacy terms or sampling, one helper, two rounds and
    /// vectors of three leaves, the last one short.
    pub fn new(sites: usize, adjust: impl FnOnce(&mut TaskTerms)) -> Self {
        Self::in_world(World::new(), sites, adjust)
    }

    pub fn in_world(mut world: World, sites: usize, adjust: impl FnOnce(&mut TaskTerms)) -> Self {
        let publisher = world.wallet();
        let aggregator = world.wallet();
        let sites: Vec<Keypair> = (0..sites).map(|_| world.wallet()).collect();
        let wallets: Vec<Hash> = sites.iter().map(key).collect();
        let mut terms = TaskTerms {
            aggregator: key(&aggregator),
            plan_id: [0x51; 32],
            sites_root: reference::root(&wallets),
            model: [0xa0; 32],
            sites: sites.len() as u16,
            helpers: 1,
            coords: 296,
            rounds: 2,
            min_cohort: 2,
            sample: 0,
            sensitivity: 0,
            noise: 0,
            colluding: 0,
            budget: 0,
            site_stake: 2_000_000,
            aggregator_stake: 50_000_000,
            reward_per_round: 10_000_000,
            dispute_bond: 7_000_000,
            challenge_slots: 100,
            stall_slots: 1_000,
        };
        adjust(&mut terms);
        let task = client::task(&key(&publisher), &terms.plan_id);
        Self {
            world,
            publisher,
            aggregator,
            sites,
            terms,
            task,
        }
    }

    /// A published and backed task with every site enrolled.
    pub fn started(sites: usize, adjust: impl FnOnce(&mut TaskTerms)) -> Self {
        let mut run = Self::new(sites, adjust);
        run.publish();
        run.back();
        for index in 0..run.sites.len() as u16 {
            run.enroll(index);
        }
        run
    }

    pub fn publish(&mut self) {
        let ix = client::publish(&key(&self.publisher), &self.terms);
        self.world.run(ix, &[&self.publisher]);
    }

    pub fn back(&mut self) {
        let ix = client::back(&key(&self.aggregator), &self.task);
        self.world.run(ix, &[&self.aggregator]);
    }

    pub fn site_path(&self, index: u16) -> Vec<Hash> {
        let wallets: Vec<Hash> = self.sites.iter().map(key).collect();
        reference::path(&wallets, index as usize)
    }

    pub fn enroll_ix(&self, index: u16, status_refs: &[StatusRef]) -> client::Instruction {
        client::enroll(
            &key(&self.sites[index as usize]),
            &self.task,
            index,
            &self.site_path(index),
            &admission(index),
            status_refs,
        )
    }

    pub fn enroll(&mut self, index: u16) {
        self.enroll_with(index, &[]);
    }

    pub fn enroll_with(&mut self, index: u16, status_refs: &[StatusRef]) {
        let ix = self.enroll_ix(index, status_refs);
        self.world.run(ix, &[&self.sites[index as usize]]);
    }

    pub fn state(&self) -> Task<Vec<u8>> {
        self.world.task(&self.task)
    }

    pub fn round(&self, number: u32) -> Round<Vec<u8>> {
        self.world.round(&self.task, number)
    }

    pub fn open_round(&mut self) -> u32 {
        let number = self.state().round() + 1;
        let ix = client::open_round(&key(&self.aggregator), &self.task, number);
        self.world.run(ix, &[&self.aggregator]);
        number
    }

    pub fn commit_ix(
        &self,
        number: u32,
        cohort: &[u16],
        posting: &Posting,
        threshold: &Hash,
    ) -> client::Instruction {
        client::commit_root(
            &key(&self.aggregator),
            &self.task,
            number,
            &client::site_bitmap(cohort.iter().copied()),
            &posting.inputs_root,
            &posting.close_head,
            threshold,
        )
    }

    pub fn post_ix(&self, number: u32, posting: &Posting) -> client::Instruction {
        client::post_aggregate(
            &key(&self.aggregator),
            &self.task,
            number,
            &posting.mask_roots,
            &posting.aggregate_root,
            &posting.next_model,
            &posting.seal_head,
        )
    }

    /// The threshold that invites exactly the task's sample under the
    /// round's beacon, and the sites it invites. Without sampling the
    /// threshold is ignored and every site is invited.
    pub fn invitation(&self, number: u32) -> (Hash, Vec<u16>) {
        let sites = self.terms.sites;
        if self.terms.sample == 0 {
            return ([0; 32], (0..sites).collect());
        }
        let beacon = *self.round(number).beacon();
        reference::invitation(&beacon, sites, self.terms.sample)
    }

    pub fn commit(&mut self, number: u32, cohort: &[u16], posting: &Posting) {
        let (threshold, _) = self.invitation(number);
        let ix = self.commit_ix(number, cohort, posting, &threshold);
        self.world.run_with_limit(ix, &[&self.aggregator], 400_000);
    }

    pub fn post(&mut self, number: u32, posting: &Posting) {
        let ix = self.post_ix(number, posting);
        self.world.run(ix, &[&self.aggregator]);
    }

    pub fn ack_ix(&self, index: u16, number: u32, seal_head: &Hash) -> client::Instruction {
        let wallet = key(&self.sites[index as usize]);
        client::ack(&wallet, &self.task, index, number, seal_head)
    }

    pub fn ack(&mut self, index: u16, number: u32) {
        let seal_head = *self.round(number).seal_head();
        let ix = self.ack_ix(index, number, &seal_head);
        self.world.run(ix, &[&self.sites[index as usize]]);
    }

    pub fn settle(&mut self, number: u32) {
        self.world.run(client::settle(&self.task, number), &[]);
    }

    /// Opens a round and takes it to Posted with `cohort` and `posting`.
    pub fn post_round(&mut self, cohort: &[u16], posting: &Posting) -> u32 {
        let number = self.open_round();
        self.commit(number, cohort, posting);
        self.post(number, posting);
        number
    }

    /// Takes a whole round from open to Final with filler digests.
    pub fn settle_round(&mut self, cohort: &[u16]) -> u32 {
        let posting = Posting::filler(self.state().round() + 1, self.terms.helpers);
        self.settle_round_with(cohort, &posting)
    }

    pub fn settle_round_with(&mut self, cohort: &[u16], posting: &Posting) -> u32 {
        let number = self.post_round(cohort, posting);
        for index in cohort {
            self.ack(*index, number);
        }
        self.world.warp(self.terms.challenge_slots);
        self.settle(number);
        number
    }

    pub fn claim_ix(&self, index: u16, number: u32) -> client::Instruction {
        let wallet = key(&self.sites[index as usize]);
        client::claim(&wallet, &self.task, index, number)
    }

    pub fn claim(&mut self, index: u16, number: u32) {
        let ix = self.claim_ix(index, number);
        self.world.run(ix, &[&self.sites[index as usize]]);
    }

    pub fn abort_ix(&self, caller: &Keypair) -> client::Instruction {
        client::abort(&key(caller), &self.task, self.state().round())
    }

    pub fn withdraw_site_ix(&self, index: u16) -> client::Instruction {
        client::withdraw_site(&key(&self.sites[index as usize]), &self.task, index)
    }

    pub fn withdraw_aggregator_ix(&self) -> client::Instruction {
        client::withdraw_aggregator(&key(&self.aggregator), &self.task, self.state().round())
    }

    pub fn withdraw_publisher_ix(&self) -> client::Instruction {
        client::withdraw_publisher(&key(&self.publisher), &self.task)
    }

    /// A funded wallet with a buffer large enough for any proof here.
    pub fn challenger(&mut self) -> Keypair {
        let challenger = self.world.wallet();
        self.world.run(
            client::buffer_init(&key(&challenger), 4_000),
            &[&challenger],
        );
        challenger
    }

    pub fn dispute_open_ix(
        &self,
        challenger: &Keypair,
        number: u32,
        leaf_index: u64,
    ) -> client::Instruction {
        client::dispute_open(&key(challenger), &self.task, number, leaf_index)
    }

    pub fn dispute_open(&mut self, challenger: &Keypair, number: u32, leaf_index: u64) {
        let ix = self.dispute_open_ix(challenger, number, leaf_index);
        self.world.run(ix, &[challenger]);
    }

    /// Uploads `proof` to the challenger's buffer and gives it as `part`.
    pub fn give(
        &mut self,
        challenger: &Keypair,
        number: u32,
        part: ProofPart,
        proof: &[u8],
    ) -> u64 {
        self.world.upload(challenger, proof);
        let ix = self.dispute_leaf_ix(challenger, number, part);
        self.world.run(ix, &[challenger])
    }

    /// Gives every part of the proof for leaf `leaf_index`.
    pub fn give_all(
        &mut self,
        challenger: &Keypair,
        number: u32,
        leaf_index: usize,
        vectors: &vectors::Vectors,
    ) {
        for part in vectors.parts() {
            self.give(challenger, number, part, &vectors.proof(part, leaf_index));
        }
    }

    pub fn dispute_leaf_ix(
        &self,
        challenger: &Keypair,
        number: u32,
        part: ProofPart,
    ) -> client::Instruction {
        client::dispute_leaf(&key(challenger), &self.task, number, part)
    }

    pub fn dispute_close_ix(&self, challenger: &Keypair, number: u32) -> client::Instruction {
        client::dispute_close(&self.task, number, &key(challenger), &key(&self.aggregator))
    }
}

/// The admission digest the tests give site `index`.
pub fn admission(index: u16) -> Hash {
    reference::sha(&[b"join", &index.to_le_bytes()])
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(text: &str) -> Hash {
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}
