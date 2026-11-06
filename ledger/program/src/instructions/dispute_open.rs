//! Open a dispute over one leaf of a posted round. The challenger signs,
//! pays the rent of the dispute account and puts up the task's bond. The
//! dispute is the challenger's own, so one challenger cannot keep another
//! out. From
//! here the challenger has one more challenge window to give, for that leaf,
//! every cohort member's input, every helper's mask and the aggregate.
//!
//! A challenger who had to demand that leaf may open past the window, with the
//! answered demand as the proof of it, since the answer may come late.
//!
//! The accounts are `[challenger, task, round, dispute, system_program]`, then
//! that demand if there is one.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    reader::{Args, Reader},
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [challenger, task_account, round_account, dispute_account, system_program, rest @ ..] =
        accounts
    else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let mut r = Reader::<Args>::new(args);
    let number = r.u32()?;
    let leaf_index = r.u64()?;
    r.finish()?;
    check_payer(challenger)?;
    check_writable(dispute_account)?;
    check_system_program(system_program)?;

    let task_address = task_account.address().as_array();
    let slot = current_slot()?;
    let (bond, leaf_coords, deadline) = {
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
        let in_window = slot < window_end(round.posted_slot(), task.challenge_slots())?;
        let answered = match rest.first() {
            Some(account) if !in_window => {
                let demand: Demand<_> = load(account)?;
                demand.answered()
                    && slot <= demand.deadline()
                    && demand.task() == task_address
                    && demand.round() == number
                    && demand.leaf_index() == leaf_index
                    && demand.challenger() == challenger.address().as_array()
            }
            _ => false,
        };
        require(in_window || answered, LedgerError::ChallengeWindowClosed)?;
        require(leaf_index < task.leaves(), LedgerError::InvalidLeafIndex)?;
        let disputes = round
            .disputes()
            .checked_add(1)
            .ok_or(ProgramError::ArithmeticOverflow)?;
        round.set_disputes(disputes);
        (
            task.dispute_bond(),
            task.leaf_coords(leaf_index),
            window_end(slot, task.challenge_slots())?,
        )
    };

    let bump = create(
        challenger,
        dispute_account,
        &[
            DISPUTE_SEED,
            task_address,
            &number.to_le_bytes(),
            challenger.address().as_array(),
        ],
        DISPUTE_SPACE,
        bond,
    )?;

    let mut dispute = Dispute::init(dispute_account.try_borrow_mut()?)?;
    dispute.set_bump(bump);
    dispute.set_round(number);
    dispute.set_leaf_coords(leaf_coords);
    *dispute.task_mut() = *task_address;
    *dispute.challenger_mut() = *challenger.address().as_array();
    dispute.set_leaf_index(leaf_index);
    dispute.set_deadline(deadline);
    Ok(())
}
