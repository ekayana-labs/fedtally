//! Settle a posted round once its whole cohort has acknowledged it and the
//! challenge window has passed without an open dispute. Anyone may call it.
//! The reward is split evenly and whatever does not divide stays with the
//! task for the publisher.
//!
//! The accounts are `[task, round]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [task_account, round_account, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    let task_address = *task_account.address();
    let mut task: Task<_> = load_mut(task_account)?;
    let mut round: Round<_> = load_mut(round_account)?;

    require(task.is_open(), LedgerError::TaskNotOpen)?;
    require(
        round.task() == task_address.as_array(),
        LedgerError::AccountMismatch,
    )?;
    require(
        round.state() == ROUND_POSTED,
        LedgerError::InvalidRoundState,
    )?;
    require(round.disputes() == 0, LedgerError::RoundDisputed)?;
    require(round.acked() == round.cohort(), LedgerError::AcksMissing)?;
    let slot = current_slot()?;
    require(
        slot >= window_end(round.posted_slot(), task.challenge_slots())?,
        LedgerError::ChallengeWindowOpen,
    )?;

    let members = round.cohort_count() as u64;
    let share = task
        .reward_per_round()
        .checked_div(members)
        .ok_or(ProgramError::InvalidAccountData)?;
    let owed = task
        .owed()
        .checked_add(share * members)
        .ok_or(ProgramError::ArithmeticOverflow)?;

    round.set_share(share);
    round.set_state(ROUND_FINAL);
    round.set_settled_slot(slot);
    *task.model_mut() = *round.next_model();
    task.set_owed(owed);
    task.set_settled(round.round());
    if round.round() == task.rounds() {
        task.set_state(TASK_FINISHED);
    }
    task.set_last_change(slot);
    Ok(())
}
