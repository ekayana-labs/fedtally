//! Give one part of a dispute's proof: the disputed leaf of one cohort
//! member's input, of one helper's mask, or of the aggregate. The proof is
//! read from the challenger's buffer, since a leaf and its paths do not fit
//! one transaction. It holds, back to back:
//!
//! - the leaf's coordinates, 8 bytes each,
//! - for an input, the member's vector root and then its path in the
//!   round's inputs root,
//! - the leaf's path in its vector root.
//!
//! A path has exactly the hashes its tree position needs, so the proof has
//! one valid length. An input is added to the dispute's running sum, a mask
//! is subtracted from it, and the aggregate leaf is kept for the comparison
//! at the close. Each part is taken once.
//!
//! The accounts are `[challenger, task, round, dispute, buffer]`.

use core::ops::Deref;

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    bitmap,
    error::{require, LedgerError},
    field::{self, COORD_LEN},
    hash::Digest,
    instructions::shared::*,
    merkle::{self, DIGEST_LEN},
    reader::{Account, Args, Reader},
    state::*,
};

pub const PART_INPUT: u8 = 0;
pub const PART_MASK: u8 = 1;
pub const PART_AGGREGATE: u8 = 2;

#[derive(Clone, Copy)]
pub enum Part {
    /// The input of the member at this cohort position.
    Input(u16),
    /// The mask of this helper.
    Mask(u8),
    Aggregate,
}

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [challenger, task_account, round_account, dispute_account, buffer_account, ..] = accounts
    else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let mut r = Reader::<Args>::new(args);
    let kind = r.u8()?;
    let position = r.u16()?;
    r.finish()?;

    let task: Task<_> = load(task_account)?;
    let round: Round<_> = load(round_account)?;
    let mut dispute: Dispute<_> = load_mut(dispute_account)?;
    let buffer: Buffer<_> = load(buffer_account)?;
    check_wallet(challenger, dispute.challenger())?;
    require(
        buffer.owner() == dispute.challenger()
            && dispute.task() == task_account.address().as_array()
            && round.task() == dispute.task()
            && round.round() == dispute.round(),
        LedgerError::AccountMismatch,
    )?;

    let part = part(kind, position, task.active_helpers(), round.cohort_count())?;
    require(!is_seen(&dispute, part), LedgerError::ProofPartRepeated)?;
    let leaf = proven_leaf(part, &task, &round, dispute.leaf_index(), buffer.contents())?;
    absorb(&mut dispute, part, leaf);
    Ok(())
}

pub fn part(kind: u8, position: u16, helpers: u8, cohort_count: u16) -> Result<Part, ProgramError> {
    match kind {
        PART_INPUT if position < cohort_count => Ok(Part::Input(position)),
        PART_MASK if position < helpers as u16 => Ok(Part::Mask(position as u8)),
        PART_AGGREGATE if position == 0 => Ok(Part::Aggregate),
        _ => Err(LedgerError::InvalidProofPart.into()),
    }
}

fn is_seen<D: Deref<Target = [u8]>>(dispute: &Dispute<D>, part: Part) -> bool {
    match part {
        Part::Input(position) => bitmap::get(dispute.seen_inputs(), position as usize),
        Part::Mask(helper) => dispute.seen_masks() >> helper & 1 == 1,
        Part::Aggregate => dispute.aggregate_given(),
    }
}

fn leaf_len<T: Deref<Target = [u8]>>(task: &Task<T>, leaf_index: u64) -> usize {
    task.leaf_coords(leaf_index) as usize * COORD_LEN
}

/// The one length a proof of this leaf can have.
pub fn proof_len<T, R>(
    part: Part,
    task: &Task<T>,
    round: &Round<R>,
    leaf_index: u64,
) -> Result<usize, ProgramError>
where
    T: Deref<Target = [u8]>,
    R: Deref<Target = [u8]>,
{
    let mut path_len = merkle::path_len(leaf_index, task.leaves());
    if let Part::Input(position) = part {
        let to_inputs_root = merkle::path_len(position as u64, round.cohort_count() as u64);
        path_len = path_len.zip(to_inputs_root).map(|(a, b)| a + b + 1);
    }
    let path_len = path_len.ok_or(ProgramError::InvalidAccountData)?;
    Ok(leaf_len(task, leaf_index) + path_len * DIGEST_LEN)
}

/// Checks `proof` against the round's roots and returns the leaf it proves.
pub fn proven_leaf<'a, T, R>(
    part: Part,
    task: &Task<T>,
    round: &'a Round<R>,
    leaf_index: u64,
    proof: &'a [u8],
) -> Result<&'a [u8], ProgramError>
where
    T: Deref<Target = [u8]>,
    R: Deref<Target = [u8]>,
{
    let leaves = task.leaves();
    let members = round.cohort_count() as u64;
    let leaf_len = leaf_len(task, leaf_index);
    require(
        proof.len() == proof_len(part, task, round, leaf_index)?,
        LedgerError::InvalidProofLength,
    )?;

    let mut r = Reader::<Account>::new(proof);
    let leaf = r.bytes(leaf_len)?;
    let vector_root: &Digest = match part {
        Part::Input(position) => {
            let vector_root = r.array()?;
            let inputs_root = merkle::root_from_path(
                merkle::leaf(vector_root),
                position as u64,
                members,
                &mut r,
            )?;
            require(
                &inputs_root == round.inputs_root(),
                LedgerError::InvalidProof,
            )?;
            vector_root
        }
        Part::Mask(helper) => round.mask_root(helper as usize),
        Part::Aggregate => round.aggregate_root(),
    };
    let root = merkle::root_from_path(merkle::leaf(leaf), leaf_index, leaves, &mut r)?;
    require(&root == vector_root, LedgerError::InvalidProof)?;
    r.finish()?;
    Ok(leaf)
}

fn absorb<D: core::ops::DerefMut<Target = [u8]>>(
    dispute: &mut Dispute<D>,
    part: Part,
    leaf: &[u8],
) {
    match part {
        Part::Input(position) => {
            combine(dispute.acc_mut(), leaf, field::add);
            bitmap::set(dispute.seen_inputs_mut(), position as usize);
            dispute.set_inputs_seen(dispute.inputs_seen() + 1);
        }
        Part::Mask(helper) => {
            combine(dispute.acc_mut(), leaf, field::sub);
            dispute.set_seen_masks(dispute.seen_masks() | 1 << helper);
        }
        Part::Aggregate => {
            combine(dispute.aggregate_mut(), leaf, |_, coordinate| coordinate);
            dispute.set_aggregate_given(true);
        }
    }
}

/// Replaces each coordinate of `into` by `op` of it and the reduced
/// coordinate of `leaf` at the same place. A short leaf leaves the rest of
/// `into` alone.
fn combine(into: &mut [u8; LEAF_LEN], leaf: &[u8], op: impl Fn(u64, u64) -> u64) {
    let (slots, _) = into.as_chunks_mut::<COORD_LEN>();
    let (coordinates, _) = leaf.as_chunks::<COORD_LEN>();
    for (slot, coordinate) in slots.iter_mut().zip(coordinates) {
        let value = field::reduce(u64::from_le_bytes(*coordinate));
        *slot = op(u64::from_le_bytes(*slot), value).to_le_bytes();
    }
}
