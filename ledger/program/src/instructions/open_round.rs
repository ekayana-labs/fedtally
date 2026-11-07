//! Open the next round. The aggregator signs and pays the rent of the round
//! account. The round's beacon is fixed here, from the model the round
//! starts from, so nobody can choose the cohort by choosing the beacon.
//!
//! The accounts are `[aggregator, task, round, system_program]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    sampling,
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [aggregator, task_account, round_account, system_program, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    check_payer(aggregator)?;
    check_writable(round_account)?;
    check_system_program(system_program)?;

    let (number, beacon) = {
        let mut task: Task<_> = load_mut(task_account)?;
        check_wallet(aggregator, task.aggregator())?;
        require(task.is_open(), LedgerError::TaskNotOpen)?;
        require(task.backed(), LedgerError::NotBacked)?;
        let steps = task
            .round()
            .checked_sub(task.voided())
            .ok_or(ProgramError::InvalidAccountData)?;
        require(steps < task.rounds(), LedgerError::NoRoundsLeft)?;
        require(
            task.settled() == task.round(),
            LedgerError::PreviousRoundNotFinal,
        )?;
        // A round that cannot afford the dearest cohort might never commit.
        let cost = task.dearest_round()?;
        let affordable = task
            .spent()
            .checked_add(cost)
            .is_some_and(|spent| spent <= task.budget());
        require(affordable, LedgerError::BudgetExhausted)?;

        let number = task.round() + 1;
        task.set_round(number);
        task.set_last_change(current_slot()?);
        (
            number,
            // A redo draws as the round it redoes would, so numbers it burned
            // don't change the draw.
            sampling::beacon(task.plan_id(), number - task.voided(), task.model()),
        )
    };

    let task_address = task_account.address().as_array();
    let bump = create(
        aggregator,
        round_account,
        &[ROUND_SEED, task_address, &number.to_le_bytes()],
        ROUND_SPACE,
        0,
    )?;

    let mut round = Round::init(round_account.try_borrow_mut()?)?;
    round.set_bump(bump);
    round.set_state(ROUND_COLLECTING);
    round.set_round(number);
    *round.task_mut() = *task_address;
    *round.beacon_mut() = beacon;
    Ok(())
}
