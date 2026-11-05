//! Apply the status lists an enrollment names. Anyone may call it. If any
//! bit the enrollment names is set, an active enrollment becomes Revoked and
//! the task marks the site, which keeps it out of later cohorts. The stake
//! is not touched: a revoked site withdraws it itself.
//!
//! Every list the enrollment names has to be passed, in order. A named
//! address that holds no status list has no bit set, so a reference to a
//! list that was never created cannot shield an enrollment from the others.
//!
//! The accounts are `[task, enrollment]`, then the status lists.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    bitmap,
    error::{require, LedgerError},
    instructions::shared::*,
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [task_account, enrollment, lists @ ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    let task_address = *task_account.address();
    let mut task: Task<_> = load_mut(task_account)?;
    let mut site: Enrollment<_> = load_mut(enrollment)?;
    require(
        site.task() == task_address.as_array(),
        LedgerError::AccountMismatch,
    )?;
    require(
        lists.len() == site.status_ref_count() as usize,
        LedgerError::StatusListMismatch,
    )?;

    let mut revoked = false;
    for (i, list_account) in lists.iter().enumerate() {
        let (address, bit) = site.status_ref(i);
        require(
            list_account.address().as_array() == address,
            LedgerError::StatusListMismatch,
        )?;
        revoked |= is_set(list_account, bit);
    }
    if revoked && site.status() == SITE_ACTIVE {
        site.set_status(SITE_REVOKED);
        bitmap::set(task.revoked_mut(), site.index() as usize);
    }
    Ok(())
}

fn is_set(list_account: &AccountView, bit: u32) -> bool {
    load::<StatusList<_>>(list_account).is_ok_and(|status_list| status_list.is_set(bit))
}
