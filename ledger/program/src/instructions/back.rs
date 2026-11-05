//! Stake on a task as its aggregator. It happens once, and no round opens
//! before it.
//!
//! The accounts are `[aggregator, task, system_program]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [aggregator, task_account, system_program, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    check_payer(aggregator)?;
    check_system_program(system_program)?;

    let stake = {
        let mut task: Task<_> = load_mut(task_account)?;
        check_wallet(aggregator, task.aggregator())?;
        require(task.is_open(), LedgerError::TaskNotOpen)?;
        require(!task.backed(), LedgerError::AlreadyBacked)?;
        let stake = task.aggregator_stake();
        task.set_backed(true);
        task.set_stake_held(stake);
        task.set_last_change(current_slot()?);
        stake
    };
    deposit(aggregator, task_account, stake)
}
