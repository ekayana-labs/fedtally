//! Claim a site's share of a settled round. The share leaves the task for
//! the site's wallet. A site keeps this right after it is revoked or has
//! withdrawn its stake.
//!
//! When the task scores groups, the claim shows the site's group's share and
//! size against the round's shares root, which the whole cohort acknowledged.
//! A round never pays out more than its reward, whatever that root says.
//!
//! The accounts are `[wallet, enrollment, task, round]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    bitmap,
    error::{require, LedgerError},
    instructions::shared::*,
    merkle,
    reader::{Args, Reader},
    state::*,
};

/// What a scored claim pays, from `share u32 LE, members u16 LE, path`.
fn scored_pay(
    args: &[u8],
    task: &Task<impl core::ops::Deref<Target = [u8]>>,
    round: &Round<impl core::ops::Deref<Target = [u8]>>,
    index: u16,
) -> Result<u64, ProgramError> {
    let mut r = Reader::<Args>::new(args);
    let share = r.u32()?;
    let members = r.u16()?;
    let mut leaf = [0u8; 8];
    leaf[..2].copy_from_slice(&index.to_le_bytes());
    leaf[2..6].copy_from_slice(&share.to_le_bytes());
    leaf[6..].copy_from_slice(&members.to_le_bytes());
    let position = bitmap::rank(round.cohort(), index as usize);
    let root = merkle::root_from_path(
        merkle::leaf(&leaf),
        u64::from(position),
        u64::from(round.cohort_count()),
        &mut r,
    )?;
    r.finish()?;
    require(root == *round.shares_root(), LedgerError::InvalidShareProof)?;
    task.scored_pay(share, members)
}

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [wallet, enrollment, task_account, round_account, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    check_writable(wallet)?;

    let task_address = *task_account.address();
    let pay_out = {
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

        let pay_out = if task.is_scored() {
            let pay_out = scored_pay(args, &task, &round, site.index())?;
            let left = round
                .share()
                .checked_sub(pay_out)
                .ok_or(LedgerError::ShareTooLarge)?;
            round.set_share(left);
            pay_out
        } else {
            no_args(args)?;
            round.share()
        };
        bitmap::set(round.claimed_mut(), index);

        let owed = task
            .owed()
            .checked_sub(pay_out)
            .ok_or(ProgramError::InsufficientFunds)?;
        task.set_owed(owed);
        pay_out
    };
    pay(task_account, wallet, pay_out)
}
