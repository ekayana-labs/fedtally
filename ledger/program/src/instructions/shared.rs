//! Account checks, loaders and lamport plumbing shared by the instructions.

use pinocchio::{
    account::{Ref, RefMut},
    cpi::{Seed, Signer},
    error::ProgramError,
    sysvars::{clock::Clock, get_sysvar, rent::RENT_ID, Sysvar},
    AccountView, Address, ProgramResult,
};
use pinocchio_system::instructions::{Allocate, Assign, CreateAccount, Transfer};

use crate::{
    error::{require, LedgerError},
    reader::{Args, Reader},
    state::{Derived, Round, Task, ROUND_POSTED},
};

#[inline]
pub fn check_signer(account: &AccountView) -> ProgramResult {
    if !account.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    Ok(())
}

#[inline]
pub fn check_writable(account: &AccountView) -> ProgramResult {
    if !account.is_writable() {
        return Err(ProgramError::Immutable);
    }
    Ok(())
}

/// A wallet that pays or is paid signs and is writable.
#[inline]
pub fn check_payer(account: &AccountView) -> ProgramResult {
    check_signer(account)?;
    check_writable(account)
}

#[inline]
pub fn check_system_program(account: &AccountView) -> ProgramResult {
    if account.address() != &pinocchio_system::ID {
        return Err(ProgramError::IncorrectProgramId);
    }
    Ok(())
}

/// Checks that `account` signs and is the wallet `expected`.
#[inline]
pub fn check_wallet(account: &AccountView, expected: &[u8; 32]) -> ProgramResult {
    check_signer(account)?;
    require(
        account.address().as_array() == expected,
        LedgerError::Unauthorized,
    )
}

/// For instructions without arguments, which refuse any.
#[inline]
pub fn no_args(args: &[u8]) -> ProgramResult {
    Reader::<Args>::new(args).finish()
}

#[inline]
pub fn current_slot() -> Result<u64, ProgramError> {
    Ok(Clock::get()?.slot)
}

/// Loads a program account read-only. It must be owned by this program,
/// carry the discriminator of `V` and sit at the address its own seeds and
/// bump derive.
pub fn load<'a, V: Derived<Ref<'a, [u8]>>>(account: &'a AccountView) -> Result<V, ProgramError> {
    check_owner(account)?;
    let view = V::parse(account.try_borrow()?)?;
    check_address(account.address(), &view.address())?;
    Ok(view)
}

/// Loads a program account for editing. It checks everything [`load`]
/// checks, and that the account is writable.
pub fn load_mut<'a, V: Derived<RefMut<'a, [u8]>>>(
    account: &'a mut AccountView,
) -> Result<V, ProgramError> {
    check_writable(account)?;
    check_owner(account)?;
    let address = *account.address();
    let view = V::parse(account.try_borrow_mut()?)?;
    check_address(&address, &view.address())?;
    Ok(view)
}

#[inline]
fn check_owner(account: &AccountView) -> ProgramResult {
    if !account.owned_by(&crate::ID) {
        return Err(ProgramError::InvalidAccountOwner);
    }
    Ok(())
}

#[inline]
fn check_address(actual: &Address, derived: &Address) -> ProgramResult {
    if actual != derived {
        return Err(ProgramError::InvalidSeeds);
    }
    Ok(())
}

/// A task and one of its rounds, both loaded for editing.
pub type AggregatorStep<'a> = (Task<RefMut<'a, [u8]>>, Round<RefMut<'a, [u8]>>);

/// The accounts `[aggregator, task, round]` of a step only the aggregator
/// takes: the task and its current round, loaded for editing once the
/// aggregator's signature, the open task and the round's task are checked.
pub fn aggregator_step(accounts: &mut [AccountView]) -> Result<AggregatorStep<'_>, ProgramError> {
    let [aggregator, task_account, round_account, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let task_address = *task_account.address();
    let task: Task<_> = load_mut(task_account)?;
    let round: Round<_> = load_mut(round_account)?;
    check_wallet(aggregator, task.aggregator())?;
    require(task.is_open(), LedgerError::TaskNotOpen)?;
    require(
        round.task() == task_address.as_array(),
        LedgerError::AccountMismatch,
    )?;
    Ok((task, round))
}

/// The latest round of `task`, which is the first of `rest`, or `None`
/// before any round was opened.
pub fn latest_round<'a, D: core::ops::Deref<Target = [u8]>>(
    task: &Task<D>,
    task_address: &Address,
    rest: &'a [AccountView],
) -> Result<Option<Round<Ref<'a, [u8]>>>, ProgramError> {
    if task.round() == 0 {
        return Ok(None);
    }
    let account = rest.first().ok_or(ProgramError::NotEnoughAccountKeys)?;
    let round: Round<_> = load(account)?;
    require(
        round.task() == task_address.as_array() && round.round() == task.round(),
        LedgerError::AccountMismatch,
    )?;
    Ok(Some(round))
}

/// Whether a posted round can still be disputed at `slot`, or is.
pub fn is_disputable<D: core::ops::Deref<Target = [u8]>>(
    round: &Round<D>,
    challenge_slots: u64,
    slot: u64,
) -> Result<bool, ProgramError> {
    if round.state() != ROUND_POSTED {
        return Ok(false);
    }
    Ok(round.disputes() > 0 || slot < window_end(round.posted_slot(), challenge_slots)?)
}

/// The first slot after the challenge window of a round posted at `posted`.
#[inline]
pub fn window_end(posted: u64, challenge_slots: u64) -> Result<u64, ProgramError> {
    posted
        .checked_add(challenge_slots)
        .ok_or(ProgramError::ArithmeticOverflow)
}

/// Creates the program account at the address `seeds` derive, which must be
/// `account` and still untouched. The payer funds its rent and `deposit`
/// lamports on top. Returns the bump.
pub fn create<const N: usize>(
    payer: &AccountView,
    account: &AccountView,
    seeds: &[&[u8]; N],
    space: usize,
    deposit: u64,
) -> Result<u8, ProgramError> {
    const { assert!(N < 5) };
    let (address, bump) =
        Address::derive_program_address(seeds, &crate::ID).ok_or(ProgramError::InvalidSeeds)?;
    check_address(account.address(), &address)?;
    if !account.owned_by(&pinocchio_system::ID) || account.data_len() != 0 {
        return Err(ProgramError::AccountAlreadyInitialized);
    }

    let bump_seed = [bump];
    let signer_seeds: [Seed; 5] = core::array::from_fn(|i| match seeds.get(i) {
        Some(seed) => Seed::from(*seed),
        None => Seed::from(&bump_seed),
    });
    let signer = [Signer::from(&signer_seeds[..N + 1])];

    let rent = rent_minimum_balance(space)?;
    if account.lamports() == 0 {
        let lamports = rent
            .checked_add(deposit)
            .ok_or(ProgramError::ArithmeticOverflow)?;
        return CreateAccount {
            from: payer,
            to: account,
            lamports,
            space: space as u64,
            owner: &crate::ID,
        }
        .invoke_signed(&signer)
        .map(|()| bump);
    }

    // The address was pre-funded. What it holds counts towards the rent only,
    // so the deposit is always the payer's own.
    let due = rent
        .saturating_sub(account.lamports())
        .checked_add(deposit)
        .ok_or(ProgramError::ArithmeticOverflow)?;
    if due > 0 {
        Transfer {
            from: payer,
            to: account,
            lamports: due,
        }
        .invoke()?;
    }
    Allocate {
        account,
        space: space as u64,
    }
    .invoke_signed(&signer)?;
    Assign {
        account,
        owner: &crate::ID,
    }
    .invoke_signed(&signer)?;
    Ok(bump)
}

/// Moves `lamports` from a wallet into a program account.
pub fn deposit(from: &AccountView, to: &AccountView, lamports: u64) -> ProgramResult {
    Transfer { from, to, lamports }.invoke()
}

/// Moves `lamports` out of an account this program owns.
pub fn pay(from: &mut AccountView, to: &mut AccountView, lamports: u64) -> ProgramResult {
    let left = from
        .lamports()
        .checked_sub(lamports)
        .ok_or(ProgramError::InsufficientFunds)?;
    from.set_lamports(left);
    let credited = to
        .lamports()
        .checked_add(lamports)
        .ok_or(ProgramError::ArithmeticOverflow)?;
    to.set_lamports(credited);
    Ok(())
}

/// Moves every lamport of `account` to `to` and closes the account.
pub fn close_to(account: &mut AccountView, to: &mut AccountView) -> ProgramResult {
    pay(account, to, account.lamports())?;
    account.close()
}

/// The rent-exempt minimum for `data_len` bytes, read from the rent sysvar.
///
/// The sysvar has had two layouts. The classic one is 17 bytes,
/// `{ lamports_per_byte_year: u64, exemption_threshold: f64, burn_percent: u8 }`,
/// and the condensed one is `{ lamports_per_byte: u64 }`. Reading only the
/// condensed form would under-fund accounts on a classic runtime whose
/// threshold is 2.0, so both are handled.
pub fn rent_minimum_balance(data_len: usize) -> Result<u64, ProgramError> {
    const ACCOUNT_STORAGE_OVERHEAD: u64 = 128;
    // 1.0 and 2.0 as little-endian IEEE-754 doubles. Comparing the bytes
    // keeps float arithmetic off the usual paths.
    const ONE: [u8; 8] = [0, 0, 0, 0, 0, 0, 0xf0, 0x3f];
    const TWO: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 0x40];

    let bytes = ACCOUNT_STORAGE_OVERHEAD
        .checked_add(data_len as u64)
        .ok_or(ProgramError::ArithmeticOverflow)?;
    let scaled = |rate: u64| {
        bytes
            .checked_mul(rate)
            .ok_or(ProgramError::ArithmeticOverflow)
    };

    let mut classic = [0u8; 17];
    match get_sysvar(&mut classic, &RENT_ID, 0) {
        Ok(()) => {
            let mut r = Reader::<crate::reader::Account>::new(&classic);
            let base = scaled(r.u64()?)?;
            let threshold = *r.array::<8>()?;
            if threshold == ONE {
                Ok(base)
            } else if threshold == TWO {
                base.checked_mul(2).ok_or(ProgramError::ArithmeticOverflow)
            } else {
                let threshold = f64::from_le_bytes(threshold);
                if !(threshold.is_finite() && threshold >= 0.0) {
                    return Err(ProgramError::InvalidArgument);
                }
                Ok((base as f64 * threshold) as u64)
            }
        }
        // A sysvar shorter than 17 bytes has the condensed layout.
        Err(ProgramError::InvalidArgument) => {
            let mut condensed = [0u8; 8];
            get_sysvar(&mut condensed, &RENT_ID, 0)?;
            scaled(u64::from_le_bytes(condensed))
        }
        Err(e) => Err(e),
    }
}
