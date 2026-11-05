//! Enroll a site. The wallet proves its place in the task's sites root,
//! stakes, and records the digest of its Join record and the status list
//! bits that can revoke it.
//!
//! The accounts are `[wallet, task, enrollment, system_program]`.

use core::ops::Deref;

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    bitmap,
    error::{require, LedgerError},
    instructions::shared::*,
    merkle::{self, DIGEST_LEN},
    reader::{Args, Reader},
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [wallet, task_account, enrollment, system_program, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    check_payer(wallet)?;
    check_writable(enrollment)?;
    check_system_program(system_program)?;

    let mut r = Reader::<Args>::new(args);
    let index = r.u16()?;
    let path = r.items(DIGEST_LEN)?;
    let admission = r.array::<32>()?;
    let status_refs = r.items(STATUS_REF_LEN)?;
    r.finish()?;
    require(
        status_refs.len() <= MAX_STATUS_REFS * STATUS_REF_LEN,
        LedgerError::TooManyStatusRefs,
    )?;

    let stake = {
        let mut task: Task<_> = load_mut(task_account)?;
        require(task.is_open(), LedgerError::TaskNotOpen)?;
        check_site(&task, wallet.address().as_array(), index, path)?;
        bitmap::set(task.enrolled_mut(), index as usize);
        task.site_stake()
    };

    // A second enrollment at this index finds the account already there.
    let task_address = task_account.address().as_array();
    let bump = create(
        wallet,
        enrollment,
        &[SITE_SEED, task_address, &index.to_le_bytes()],
        ENROLLMENT_SPACE,
        stake,
    )?;

    let mut site = Enrollment::init(enrollment.try_borrow_mut()?)?;
    site.set_bump(bump);
    site.set_status(SITE_ACTIVE);
    site.set_index(index);
    site.set_status_ref_count((status_refs.len() / STATUS_REF_LEN) as u8);
    *site.task_mut() = *task_address;
    *site.wallet_mut() = *wallet.address().as_array();
    site.set_stake(stake);
    *site.admission_mut() = *admission;
    site.status_refs_mut()[..status_refs.len()].copy_from_slice(status_refs);
    Ok(())
}

/// Checks that `wallet` is the leaf at `index` of the task's sites root.
fn check_site<D: Deref<Target = [u8]>>(
    task: &Task<D>,
    wallet: &[u8; 32],
    index: u16,
    path: &[u8],
) -> ProgramResult {
    let (index, sites) = (index as u64, task.sites() as u64);
    let path_len = merkle::path_len(index, sites).ok_or(LedgerError::InvalidSiteProof)?;
    require(
        path.len() == path_len * DIGEST_LEN,
        LedgerError::InvalidSiteProof,
    )?;
    let root = merkle::root_from_path(
        merkle::leaf(wallet),
        index,
        sites,
        &mut Reader::<Args>::new(path),
    )?;
    require(&root == task.sites_root(), LedgerError::InvalidSiteProof)
}
