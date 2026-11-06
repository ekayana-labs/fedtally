//! Close a final round whose every share has been claimed. Anyone may call it,
//! and the rent returns to the aggregator, who paid it. The latest round stays,
//! since a withdrawal reads it.
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
    {
        let task: Task<_> = load(task_account)?;
        let round: Round<_> = load_mut(round_account)?;
        require(
            round.task() == task_account.address().as_array()
                && aggregator.address().as_array() == task.aggregator(),
            LedgerError::AccountMismatch,
        )?;
        require(round.state() == ROUND_FINAL, LedgerError::InvalidRoundState)?;
        require(round.round() < task.round(), LedgerError::LatestRound)?;
        require(
            round.claimed() == round.cohort(),
            LedgerError::SharesUnclaimed,
        )?;
    }
    close_to(round_account, aggregator)
}
