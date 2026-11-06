//! Gives up a round the log has given up: for a silent site, or with one helper
//! removed. Its budget stays spent and nobody is paid for it. The next round
//! starts from the same model. Only a committed round can be voided, since one
//! that has been posted can be acknowledged and settled, which would adopt a
//! model the log refused.

use pinocchio::{account::AccountView, error::ProgramError, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    reader::{Args, Reader},
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let mut r = Reader::<Args>::new(args);
    let remove_helper = r.u8()? != 0;
    r.finish()?;

    let (mut task, mut round) = aggregator_step(accounts)?;
    require(
        round.state() == ROUND_COMMITTED,
        LedgerError::InvalidRoundState,
    )?;
    let voided = task
        .voided()
        .checked_add(1)
        .ok_or(ProgramError::ArithmeticOverflow)?;
    let removed = task.removed_helpers();
    if remove_helper {
        require(
            task.min_helpers() > 0 && task.active_helpers() > task.min_helpers(),
            LedgerError::NoRetriesLeft,
        )?;
        task.set_removed_helpers(removed + 1);
    } else {
        require(
            voided - u32::from(removed) <= task.retries(),
            LedgerError::NoRetriesLeft,
        )?;
    }
    round.set_state(ROUND_VOID);
    task.set_voided(voided);
    // Settled or voided: either way the next round may open.
    task.set_settled(round.round());
    task.set_last_change(current_slot()?);
    Ok(())
}
