//! Abort an open task. The aggregator and the publisher may at any time.
//! Anyone may once nothing has changed for longer than the task's stall
//! window, so a run nobody drives does not hold stakes forever. A round with
//! an open dispute has to see it closed first.
//!
//! The accounts are `[caller, task]`, then the latest round if there is one.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [caller, task_account, rest @ ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    check_signer(caller)?;
    let task_address = *task_account.address();
    let mut task: Task<_> = load_mut(task_account)?;
    require(task.is_open(), LedgerError::TaskNotOpen)?;
    if let Some(round) = latest_round(&task, &task_address, rest)? {
        require(round.disputes() == 0, LedgerError::RoundDisputed)?;
    }

    let slot = current_slot()?;
    let caller = caller.address().as_array();
    if caller != task.aggregator() && caller != task.publisher() {
        let stalled = task
            .last_change()
            .checked_add(task.stall_slots())
            .is_some_and(|quiet_until| slot > quiet_until);
        require(stalled, LedgerError::NotStalled)?;
    }
    task.set_state(TASK_ABORTED);
    task.set_last_change(slot);
    Ok(())
}
