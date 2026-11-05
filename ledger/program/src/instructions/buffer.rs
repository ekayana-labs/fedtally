//! A buffer is scratch space for a proof that does not fit one transaction.
//! Its owner creates it with a size, fills it in chunks and closes it to get
//! the rent back. A write at offset 0 starts the contents over and any other
//! write has to continue where the last one ended, so the contents never
//! have holes and one buffer serves any number of proofs.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    reader::{Args, Reader},
    state::*,
};

/// Creates the owner's buffer. The accounts are
/// `[owner, buffer, system_program]`.
pub fn init(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [owner, buffer_account, system_program, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let mut r = Reader::<Args>::new(args);
    let capacity = r.u32()?;
    r.finish()?;
    check_payer(owner)?;
    check_writable(buffer_account)?;
    check_system_program(system_program)?;
    require(
        (1..=MAX_BUFFER_CAPACITY).contains(&capacity),
        LedgerError::InvalidBufferSize,
    )?;

    let owner_address = owner.address().as_array();
    let bump = create(
        owner,
        buffer_account,
        &[BUFFER_SEED, owner_address],
        buffer_space(capacity),
        0,
    )?;

    let mut buffer = Buffer::init(buffer_account.try_borrow_mut()?)?;
    buffer.set_bump(bump);
    *buffer.owner_mut() = *owner_address;
    Ok(())
}

/// Writes a chunk. The accounts are `[owner, buffer]`.
pub fn write(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [owner, buffer_account, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let mut r = Reader::<Args>::new(args);
    let offset = r.u32()? as usize;
    let chunk = r.items(1)?;
    r.finish()?;

    let mut buffer: Buffer<_> = load_mut(buffer_account)?;
    check_wallet(owner, buffer.owner())?;
    let end = offset + chunk.len();
    require(
        !chunk.is_empty()
            && (offset == 0 || offset == buffer.used() as usize)
            && end <= buffer.capacity(),
        LedgerError::InvalidBufferWrite,
    )?;
    buffer.bytes_mut()[offset..end].copy_from_slice(chunk);
    buffer.set_used(end as u32);
    Ok(())
}

/// Closes the buffer and returns its rent. The accounts are
/// `[owner, buffer]`.
pub fn close(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [owner, buffer_account, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    no_args(args)?;
    check_writable(owner)?;
    {
        let buffer: Buffer<_> = load_mut(buffer_account)?;
        check_wallet(owner, buffer.owner())?;
    }
    close_to(buffer_account, owner)
}
