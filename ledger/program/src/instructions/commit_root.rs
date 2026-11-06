//! Commit the round's cohort and the root over its masked inputs. This is
//! where the privacy budget is charged, for the cohort that actually showed
//! up, and where a sampled cohort is checked against the round's tickets.
//!
//! The accounts are `[aggregator, task, round]`.

use core::ops::Deref;

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    bitmap::{self, SiteBitmap},
    error::{require, LedgerError},
    hash::Digest,
    instructions::shared::*,
    reader::{Args, Reader},
    sampling::Tickets,
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let mut r = Reader::<Args>::new(args);
    let cohort: &SiteBitmap = r.array()?;
    let inputs_root: &Digest = r.array()?;
    let close_head: &Digest = r.array()?;
    let threshold: &Digest = r.array()?;
    r.finish()?;

    let (mut task, mut round) = aggregator_step(accounts)?;
    require(
        round.state() == ROUND_COLLECTING,
        LedgerError::InvalidRoundState,
    )?;
    let cohort_count = check_cohort(&task, cohort)?;
    if task.sample() > 0 {
        check_sample(&task, round.beacon(), cohort, threshold)?;
    }

    // `open_round` made sure the dearest cohort fits, so this holds for any
    // cohort the checks above let through. It is checked where it is charged.
    let spent = task
        .spent()
        .checked_add(task.round_cost(cohort_count)?)
        .filter(|spent| *spent <= task.budget())
        .ok_or(LedgerError::BudgetExhausted)?;
    task.set_spent(spent);
    task.set_last_change(current_slot()?);

    *round.cohort_mut() = *cohort;
    round.set_cohort_count(cohort_count);
    *round.inputs_root_mut() = *inputs_root;
    *round.close_head_mut() = *close_head;
    round.set_state(ROUND_COMMITTED);
    Ok(())
}

/// Checks the cohort's size and that every member is a site of the task and
/// enrolled. Returns the size.
///
/// A revoked site may still be in a cohort. The ledger runs behind the log, so
/// the log may have closed this round before the revocation was applied here,
/// and refusing would let anyone stall the task by applying it early. What
/// keeps a revoked site out is the log's own rule, and that no honest site
/// acknowledges a cohort other than the log's.
fn check_cohort<D: Deref<Target = [u8]>>(
    task: &Task<D>,
    cohort: &SiteBitmap,
) -> Result<u16, ProgramError> {
    let count = bitmap::count(cohort) as u16;
    require(
        (task.min_cohort()..=task.sites()).contains(&count),
        LedgerError::InvalidCohortSize,
    )?;
    require(
        !bitmap::any_from(cohort, task.sites() as usize)
            && bitmap::is_subset(cohort, task.enrolled()),
        LedgerError::InvalidCohortMember,
    )?;
    Ok(count)
}

/// Checks that `threshold` invites exactly the sampled number of the task's
/// sites, and that every cohort member is one of them.
fn check_sample<D: Deref<Target = [u8]>>(
    task: &Task<D>,
    beacon: &Digest,
    cohort: &SiteBitmap,
    threshold: &Digest,
) -> ProgramResult {
    let mut tickets = Tickets::new(beacon);
    let mut invited = 0;
    for site in 0..task.sites() {
        if tickets.of(site as u32) <= *threshold {
            invited += 1;
        } else {
            require(!bitmap::get(cohort, site as usize), LedgerError::NotInvited)?;
        }
    }
    require(invited == task.sample(), LedgerError::InvalidThreshold)
}
