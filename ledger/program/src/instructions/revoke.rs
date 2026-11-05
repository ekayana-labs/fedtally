//! Set one bit of a status list. Only the issuer can, bits are never
//! cleared, and every change moves the list's sequence on by one.
//!
//! The accounts are `[issuer, status_list]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    bitmap,
    error::{require, LedgerError},
    instructions::shared::*,
    reader::{Args, Reader},
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [issuer, list_account, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let mut r = Reader::<Args>::new(args);
    let bit = r.u32()?;
    r.finish()?;

    let mut status_list: StatusList<_> = load_mut(list_account)?;
    check_wallet(issuer, status_list.issuer())?;
    require(bit < status_list.bits(), LedgerError::InvalidBit)?;
    require(!status_list.is_set(bit), LedgerError::AlreadyRevoked)?;
    bitmap::set(status_list.bitmap_mut(), bit as usize);
    let sequence = status_list
        .sequence()
        .checked_add(1)
        .ok_or(ProgramError::ArithmeticOverflow)?;
    status_list.set_sequence(sequence);
    Ok(())
}
