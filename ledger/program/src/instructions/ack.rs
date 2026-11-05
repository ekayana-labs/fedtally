//! Acknowledge a posted round as one of its cohort. A site does this only
//! after it has checked the round off chain and seen that the roots posted
//! here are the ones in its own log, so the seal head it names has to be the
//! one posted.
//!
//! The accounts are `[wallet, enrollment, round]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    bitmap,
    error::{require, LedgerError},
    hash::Digest,
    instructions::shared::*,
    reader::{Args, Reader},
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [wallet, enrollment, round_account, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let mut r = Reader::<Args>::new(args);
    let number = r.u32()?;
    let seal_head: &Digest = r.array()?;
    r.finish()?;

    let site: Enrollment<_> = load(enrollment)?;
    check_wallet(wallet, site.wallet())?;
    let mut round: Round<_> = load_mut(round_account)?;
    require(
        round.task() == site.task() && round.round() == number,
        LedgerError::AccountMismatch,
    )?;
    require(
        round.state() == ROUND_POSTED,
        LedgerError::InvalidRoundState,
    )?;
    require(round.seal_head() == seal_head, LedgerError::SealMismatch)?;

    let index = site.index() as usize;
    require(bitmap::get(round.cohort(), index), LedgerError::NotInCohort)?;
    require(
        !bitmap::get(round.acked(), index),
        LedgerError::AlreadyAcked,
    )?;
    bitmap::set(round.acked_mut(), index);
    Ok(())
}
