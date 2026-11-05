//! Create a status list. The issuer signs and pays its rent. Every bit
//! starts clear.
//!
//! The accounts are `[issuer, status_list, system_program]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    reader::{Args, Reader},
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [issuer, list_account, system_program, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let mut r = Reader::<Args>::new(args);
    let list = r.u32()?;
    let bits = r.u32()?;
    r.finish()?;
    check_payer(issuer)?;
    check_writable(list_account)?;
    check_system_program(system_program)?;
    require(
        (1..=MAX_STATUS_BITS).contains(&bits),
        LedgerError::InvalidStatusList,
    )?;

    let issuer_address = issuer.address().as_array();
    let bump = create(
        issuer,
        list_account,
        &[STATUS_SEED, issuer_address, &list.to_le_bytes()],
        status_list_space(bits),
        0,
    )?;

    let mut status_list = StatusList::init(list_account.try_borrow_mut()?)?;
    status_list.set_bump(bump);
    status_list.set_list(list);
    status_list.set_bits(bits);
    *status_list.issuer_mut() = *issuer_address;
    Ok(())
}
