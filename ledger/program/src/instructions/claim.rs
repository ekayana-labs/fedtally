//! Claim a site's share of a settled round. The share leaves the task for
//! the site's wallet. A site keeps this right after it is revoked or has
//! withdrawn its stake.
//!
//! The accounts are `[wallet, enrollment, task, round]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    bitmap,
    error::{require, LedgerError},
    instructions::shared::*,
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [wallet, enrollment, task_account, round_account, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    check_writable(wallet)?;

    let task_address = *task_account.address();
    let share = {
        let site: Enrollment<_> = load(enrollment)?;
        check_wallet(wallet, site.wallet())?;
        let mut task: Task<_> = load_mut(task_account)?;
        let mut round: Round<_> = load_mut(round_account)?;
        require(
            site.task() == task_address.as_array() && round.task() == task_address.as_array(),
            LedgerError::AccountMismatch,
        )?;
        require(round.state() == ROUND_FINAL, LedgerError::InvalidRoundState)?;

        let index = site.index() as usize;
        require(bitmap::get(round.cohort(), index), LedgerError::NotInCohort)?;
        require(
            !bitmap::get(round.claimed(), index),
            LedgerError::AlreadyClaimed,
        )?;
        bitmap::set(round.claimed_mut(), index);

        let share = round.share();
        let owed = task
            .owed()
            .checked_sub(share)
            .ok_or(ProgramError::InsufficientFunds)?;
        task.set_owed(owed);
        share
    };
    pay(task_account, wallet, share)
}
