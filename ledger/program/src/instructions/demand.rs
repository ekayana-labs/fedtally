//! A demand that one leaf of one committed vector be published.
//!
//! A fraud proof needs the vectors, and the aggregator holds them. So anyone
//! refused a leaf during a round's challenge window can put up the task's bond
//! and demand it on the ledger. Anyone holding the leaf answers with the same
//! proof a dispute takes, and the leaf and its paths stay readable in the
//! demand account, for the challenger to carry on with.
//!
//! - Answered in time: a small fee pays the one who answered, for the
//!   transactions, and the rest returns to the challenger at the close. A fee
//!   near the cost means withholding earns nothing and demanding costs little.
//!   The challenger then has one more window to open a dispute over that leaf.
//! - Not answered by its deadline, which queues behind those of earlier
//!   demands on the round: the data is taken to be gone, which is as
//!   good as a wrong aggregate. The round becomes Fraud, the task is aborted,
//!   and the aggregator's stake goes to the challenger.
//!
//! A leaf reveals nothing a watcher couldn't already fetch: inputs are masked,
//! and a mask total only ever sums over the cohort.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::{
        dispute_leaf::{part, proof_len, proven_leaf},
        shared::*,
    },
    reader::{Args, Reader},
    state::*,
};

/// What an answer earns, in lamports: about the fees of its transactions.
pub const ANSWER_FEE: u64 = 50_000;

/// Slots between the deadlines of two demands on one round, enough for the
/// transactions of one answer. Without it a burst of demands could outrun an
/// honest aggregator, and whoever sent them would take its stake.
pub const ANSWER_SLOTS: u64 = 64;

/// The accounts are `[challenger, task, round, demand, system_program]`.
pub fn open(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [challenger, task_account, round_account, demand_account, system_program, ..] = accounts
    else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let mut r = Reader::<Args>::new(args);
    let number = r.u32()?;
    let kind = r.u8()?;
    let position = r.u16()?;
    let leaf_index = r.u64()?;
    r.finish()?;
    check_payer(challenger)?;
    check_writable(demand_account)?;
    check_system_program(system_program)?;

    let task_address = task_account.address().as_array();
    let slot = current_slot()?;
    let (bond, length, deadline) = {
        let task: Task<_> = load(task_account)?;
        let mut round: Round<_> = load_mut(round_account)?;
        require(
            round.task() == task_address && round.round() == number,
            LedgerError::AccountMismatch,
        )?;
        require(
            round.state() == ROUND_POSTED,
            LedgerError::InvalidRoundState,
        )?;
        require(
            slot < window_end(round.posted_slot(), task.challenge_slots())?,
            LedgerError::ChallengeWindowClosed,
        )?;
        require(leaf_index < task.leaves(), LedgerError::InvalidLeafIndex)?;
        let part = part(kind, position, task.active_helpers(), round.cohort_count())?;
        let length = proof_len(part, &task, &round, leaf_index)?;
        let disputes = round
            .disputes()
            .checked_add(1)
            .ok_or(ProgramError::ArithmeticOverflow)?;
        round.set_disputes(disputes);
        let deadline = window_end(slot, task.challenge_slots())?
            .max(window_end(round.demand_tail(), ANSWER_SLOTS)?);
        round.set_demand_tail(deadline);
        (task.dispute_bond(), length, deadline)
    };

    let place = demand_place(number, kind, position, leaf_index);
    let bump = create(
        challenger,
        demand_account,
        &[
            DEMAND_SEED,
            task_address,
            challenger.address().as_array(),
            &place,
        ],
        demand_space(length),
        bond,
    )?;

    let mut demand = Demand::init(demand_account.try_borrow_mut()?)?;
    demand.set_bump(bump);
    demand.set_kind(kind);
    demand.set_position(position);
    demand.set_round(number);
    demand.set_proof_len(length as u32);
    *demand.task_mut() = *task_address;
    *demand.challenger_mut() = *challenger.address().as_array();
    demand.set_leaf_index(leaf_index);
    demand.set_deadline(deadline);
    Ok(())
}

/// The accounts are `[responder, task, round, demand, buffer]`. The proof is in
/// the responder's buffer, as for a dispute.
pub fn answer(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [responder, task_account, round_account, demand_account, buffer_account, ..] = accounts
    else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    check_writable(responder)?;
    check_signer(responder)?;

    let slot = current_slot()?;
    let fee = {
        let task: Task<_> = load(task_account)?;
        let round: Round<_> = load(round_account)?;
        let buffer: Buffer<_> = load(buffer_account)?;
        let mut demand: Demand<_> = load_mut(demand_account)?;
        require(
            buffer.owner() == responder.address().as_array()
                && demand.task() == task_account.address().as_array()
                && round.task() == demand.task()
                && round.round() == demand.round(),
            LedgerError::AccountMismatch,
        )?;
        require(!demand.answered(), LedgerError::DemandAnswered)?;
        require(
            round.state() == ROUND_POSTED,
            LedgerError::InvalidRoundState,
        )?;
        require(slot <= demand.deadline(), LedgerError::DemandExpired)?;
        let part = part(
            demand.kind(),
            demand.position(),
            task.active_helpers(),
            round.cohort_count(),
        )?;
        let proof = buffer.contents();
        proven_leaf(part, &task, &round, demand.leaf_index(), proof)?;
        demand.proof_mut().copy_from_slice(proof);
        demand.set_answered(true);
        demand.set_deadline(window_end(slot, task.challenge_slots())?);
        task.dispute_bond().min(ANSWER_FEE)
    };
    pay(demand_account, responder, fee)
}

/// The accounts are `[task, round, demand, challenger]`. The challenger closes
/// an answered demand once it has what it wanted. Past its deadline anyone may
/// close a demand, and one still unanswered then costs the aggregator its stake.
pub fn close(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [task_account, round_account, demand_account, challenger, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    check_writable(challenger)?;

    let task_address = *task_account.address();
    let slot = current_slot()?;
    let forfeit = {
        let mut task: Task<_> = load_mut(task_account)?;
        let mut round: Round<_> = load_mut(round_account)?;
        let demand: Demand<_> = load_mut(demand_account)?;
        require(
            demand.task() == task_address.as_array()
                && round.task() == demand.task()
                && round.round() == demand.round()
                && challenger.address().as_array() == demand.challenger(),
            LedgerError::AccountMismatch,
        )?;
        let over = slot > demand.deadline() || round.state() != ROUND_POSTED;
        require(
            over || (demand.answered() && challenger.is_signer()),
            LedgerError::DemandNotOver,
        )?;
        let forfeit = if !demand.answered() && round.state() == ROUND_POSTED {
            round.set_state(ROUND_FRAUD);
            task.set_state(TASK_ABORTED);
            task.set_fraud(true);
            let stake = task.stake_held();
            task.set_stake_held(0);
            stake
        } else {
            0
        };
        let disputes = round
            .disputes()
            .checked_sub(1)
            .ok_or(ProgramError::InvalidAccountData)?;
        round.set_disputes(disputes);
        task.set_last_change(slot);
        forfeit
    };
    pay(task_account, challenger, forfeit)?;
    close_to(demand_account, challenger)
}
