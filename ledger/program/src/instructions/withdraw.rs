//! Take funds back once the task releases them.
//!
//! - A site takes its stake when the task is finished or aborted, or when
//!   its enrollment is revoked.
//! - The aggregator takes its stake when the task is finished, or aborted
//!   without fraud. If the task was aborted with a round posted, the stake
//!   stays until that round can no longer be disputed, so aborting is no
//!   way around a fraud proof.
//! - The publisher takes what the task holds above its rent, the rewards
//!   still owed to sites and the aggregator's stake, when the task is
//!   finished or aborted.
//!
//! The accounts are `[wallet, task]`, then the enrollment for a site, or the
//! latest round, if there is one, for the aggregator.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    reader::{Args, Reader},
    state::*,
};

pub const ROLE_SITE: u8 = 0;
pub const ROLE_AGGREGATOR: u8 = 1;
pub const ROLE_PUBLISHER: u8 = 2;

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [wallet, task_account, rest @ ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let mut r = Reader::<Args>::new(args);
    let role = r.u8()?;
    r.finish()?;
    check_payer(wallet)?;

    match role {
        ROLE_SITE => site(wallet, task_account, rest),
        ROLE_AGGREGATOR => aggregator(wallet, task_account, rest),
        ROLE_PUBLISHER => publisher(wallet, task_account),
        _ => Err(LedgerError::InvalidRole.into()),
    }
}

fn site(
    wallet: &mut AccountView,
    task_account: &AccountView,
    rest: &mut [AccountView],
) -> ProgramResult {
    let [enrollment, ..] = rest else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let stake = {
        let task: Task<_> = load(task_account)?;
        let mut site: Enrollment<_> = load_mut(enrollment)?;
        check_wallet(wallet, site.wallet())?;
        require(
            site.task() == task_account.address().as_array(),
            LedgerError::AccountMismatch,
        )?;
        require(
            site.status() != SITE_WITHDRAWN,
            LedgerError::AlreadyWithdrawn,
        )?;
        require(
            !task.is_open() || site.status() == SITE_REVOKED,
            LedgerError::FundsLocked,
        )?;
        site.set_status(SITE_WITHDRAWN);
        site.stake()
    };
    pay(enrollment, wallet, stake)
}

fn aggregator(
    wallet: &mut AccountView,
    task_account: &mut AccountView,
    rest: &[AccountView],
) -> ProgramResult {
    let task_address = *task_account.address();
    let stake = {
        let mut task: Task<_> = load_mut(task_account)?;
        check_wallet(wallet, task.aggregator())?;
        let released = match task.state() {
            TASK_FINISHED => true,
            TASK_ABORTED if !task.fraud() => match latest_round(&task, &task_address, rest)? {
                Some(round) => !is_disputable(&round, task.challenge_slots(), current_slot()?)?,
                None => true,
            },
            _ => false,
        };
        require(released, LedgerError::FundsLocked)?;
        let stake = task.stake_held();
        require(stake > 0, LedgerError::NothingToWithdraw)?;
        task.set_stake_held(0);
        stake
    };
    pay(task_account, wallet, stake)
}

fn publisher(wallet: &mut AccountView, task_account: &mut AccountView) -> ProgramResult {
    check_writable(task_account)?;
    let balance = task_account.lamports();
    let rent = rent_minimum_balance(TASK_SPACE)?;
    let surplus = {
        let task: Task<_> = load(task_account)?;
        check_wallet(wallet, task.publisher())?;
        require(!task.is_open(), LedgerError::FundsLocked)?;
        balance
            .checked_sub(rent)
            .and_then(|above_rent| above_rent.checked_sub(task.owed()))
            .and_then(|unowed| unowed.checked_sub(task.stake_held()))
            .ok_or(ProgramError::InsufficientFunds)?
    };
    require(surplus > 0, LedgerError::NothingToWithdraw)?;
    pay(task_account, wallet, surplus)
}
