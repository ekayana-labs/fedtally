//! Close a dispute. Anyone may call it. A round can have any number of
//! disputes open, one per challenger, and each is decided on its own.
//!
//! - Every part was given and the inputs minus the masks differ from the
//!   aggregate leaf in some coordinate: the aggregate is proven wrong. The
//!   round becomes Fraud, the task is aborted, and the aggregator's stake
//!   and the bond go to the challenger.
//! - Every part was given and they agree: the bond goes to the aggregator.
//! - The deadline has passed with parts missing: the bond stays in the task,
//!   where it ends up with the publisher. It does not go to the aggregator,
//!   who could otherwise open empty disputes at no cost.
//! - Another dispute has already proven the round a fraud: the bond returns
//!   to the challenger.
//!
//! The rent of the dispute account returns to the challenger in every case.
//!
//! The accounts are `[task, round, dispute, challenger, aggregator]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    state::*,
};

enum Outcome {
    /// The challenger takes the aggregator's stake and keeps the bond.
    Fraud { stake: u64 },
    /// The aggregator takes the bond.
    Upheld { bond: u64 },
    /// The task keeps the bond.
    Abandoned { bond: u64 },
    /// The challenger keeps the bond.
    Moot,
}

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [task_account, round_account, dispute_account, challenger, aggregator, ..] = accounts
    else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    check_writable(challenger)?;
    check_writable(aggregator)?;

    let task_address = *task_account.address();
    let outcome = {
        let mut task: Task<_> = load_mut(task_account)?;
        let mut round: Round<_> = load_mut(round_account)?;
        let dispute: Dispute<_> = load_mut(dispute_account)?;
        require(
            dispute.task() == task_address.as_array()
                && round.task() == dispute.task()
                && round.round() == dispute.round()
                && challenger.address().as_array() == dispute.challenger()
                && aggregator.address().as_array() == task.aggregator(),
            LedgerError::AccountMismatch,
        )?;

        let slot = current_slot()?;
        let complete = dispute.is_complete(round.cohort_count(), task.helpers());
        let outcome = if round.state() == ROUND_FRAUD {
            Outcome::Moot
        } else if complete && !dispute.sums_match() {
            round.set_state(ROUND_FRAUD);
            task.set_state(TASK_ABORTED);
            task.set_fraud(true);
            let stake = task.stake_held();
            task.set_stake_held(0);
            Outcome::Fraud { stake }
        } else if complete {
            Outcome::Upheld {
                bond: task.dispute_bond(),
            }
        } else {
            require(slot > dispute.deadline(), LedgerError::DisputeNotOver)?;
            Outcome::Abandoned {
                bond: task.dispute_bond(),
            }
        };
        let disputes = round
            .disputes()
            .checked_sub(1)
            .ok_or(ProgramError::InvalidAccountData)?;
        round.set_disputes(disputes);
        task.set_last_change(slot);
        outcome
    };

    match outcome {
        Outcome::Fraud { stake } => pay(task_account, challenger, stake)?,
        Outcome::Upheld { bond } => pay(dispute_account, aggregator, bond)?,
        Outcome::Abandoned { bond } => pay(dispute_account, task_account, bond)?,
        Outcome::Moot => {}
    }
    close_to(dispute_account, challenger)
}
