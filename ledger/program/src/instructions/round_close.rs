//! Close a round nothing will read again. Anyone may call it, and the rent
//! returns to the aggregator, who paid it.
//!
//! - A final round closes once every share is claimed, or once the time to
//!   claim has passed. What was not claimed stops being owed, so the publisher
//!   can take it back.
//! - The latest round closes only when the task is over and the aggregator's
//!   stake is gone, since a withdrawal reads it until then. Whatever state it
//!   stopped in, no dispute or demand may still be open on it.
//!
//! The accounts are `[aggregator, task, round]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [aggregator, task_account, round_account, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    check_writable(aggregator)?;
    let task_address = *task_account.address();
    {
        let mut task: Task<_> = load_mut(task_account)?;
        let round: Round<_> = load_mut(round_account)?;
        require(
            round.task() == task_address.as_array()
                && aggregator.address().as_array() == task.aggregator(),
            LedgerError::AccountMismatch,
        )?;
        let released = !task.is_open() && task.stake_held() == 0;
        require(
            round.round() < task.round() || released,
            LedgerError::LatestRound,
        )?;
        if round.state() == ROUND_FINAL {
            let claimed: u32 = round.claimed().iter().map(|byte| byte.count_ones()).sum();
            let unclaimed = (round.cohort_count() as u64).saturating_sub(claimed as u64);
            if unclaimed > 0 {
                let window = task
                    .challenge_slots()
                    .checked_mul(CLAIM_WINDOWS)
                    .ok_or(ProgramError::ArithmeticOverflow)?;
                require(
                    current_slot()? >= window_end(round.settled_slot(), window)?,
                    LedgerError::SharesUnclaimed,
                )?;
                let owed = task
                    .owed()
                    .checked_sub(round.share() * unclaimed)
                    .ok_or(ProgramError::InsufficientFunds)?;
                task.set_owed(owed);
            }
        } else {
            require(round.disputes() == 0, LedgerError::RoundDisputed)?;
        }
    }
    close_to(round_account, aggregator)
}
