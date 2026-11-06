//! Post the round's result: the root of each helper's mask vector, the root
//! of the aggregate, the digest of the model it leads to and the head of the
//! log at the seal. The challenge window starts here.
//!
//! The accounts are `[aggregator, task, round]`.

use pinocchio::{AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    hash::Digest,
    instructions::shared::*,
    merkle::DIGEST_LEN,
    reader::{Args, Reader},
    state::*,
};

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let mut r = Reader::<Args>::new(args);
    let mask_roots = r.items(DIGEST_LEN)?;
    let aggregate_root: &Digest = r.array()?;
    let next_model: &Digest = r.array()?;
    let seal_head: &Digest = r.array()?;
    r.finish()?;

    let (mut task, mut round) = aggregator_step(accounts)?;
    require(
        round.state() == ROUND_COMMITTED,
        LedgerError::InvalidRoundState,
    )?;
    require(
        mask_roots.len() == task.active_helpers() as usize * DIGEST_LEN,
        LedgerError::InvalidMaskRoots,
    )?;

    let slot = current_slot()?;
    round.mask_roots_mut()[..mask_roots.len()].copy_from_slice(mask_roots);
    *round.aggregate_root_mut() = *aggregate_root;
    *round.next_model_mut() = *next_model;
    *round.seal_head_mut() = *seal_head;
    round.set_posted_slot(slot);
    round.set_state(ROUND_POSTED);
    task.set_last_change(slot);
    Ok(())
}
