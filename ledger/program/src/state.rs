//! Account layouts.
//!
//! Every account is program derived. Its data is an 8 byte discriminator,
//! `sha256("account:<Name>")[..8]`, followed by the fields of its table below
//! in order, packed, integers little-endian. A flag is one byte, 0 or 1. A
//! `[u8; 128]` bitmap has one bit per site, numbered as in [`crate::bitmap`].
//!
//! Each table generates a layout struct made of byte arrays, so the struct
//! has no padding and its field offsets are the account's, and a view over
//! the account data with one getter and one setter per field. Handlers edit
//! accounts in place through these views. Nothing is copied out or
//! deserialized.

use core::{
    mem::{offset_of, size_of},
    ops::{Deref, DerefMut},
};

use pinocchio::{error::ProgramError, Address};

use crate::{hash::Digest, privacy};

/// The most sites a task can have.
pub const MAX_SITES: usize = 1024;
/// The most helpers a task can have.
pub const MAX_HELPERS: usize = 4;
/// How many coordinates a full leaf of a vector tree holds.
pub const LEAF_COORDS: u64 = 128;
/// The bytes of a full leaf.
pub const LEAF_LEN: usize = LEAF_COORDS as usize * crate::field::COORD_LEN;
/// The most status list bits an enrollment can name.
pub const MAX_STATUS_REFS: usize = 4;
/// One status reference: the list's address, then the bit index as a u32.
pub const STATUS_REF_LEN: usize = 32 + 4;
/// The most bits a status list can have.
pub const MAX_STATUS_BITS: u32 = 80_000;
/// The most bytes a buffer can hold.
pub const MAX_BUFFER_CAPACITY: u32 = 10_000;

pub const TASK_SEED: &[u8] = b"task";
pub const SITE_SEED: &[u8] = b"site";
pub const ROUND_SEED: &[u8] = b"round";
pub const DISPUTE_SEED: &[u8] = b"dispute";
pub const STATUS_SEED: &[u8] = b"status";
pub const BUFFER_SEED: &[u8] = b"buffer";

pub const TASK_DISCRIMINATOR: [u8; 8] = [79, 34, 229, 55, 88, 90, 55, 84];
pub const ENROLLMENT_DISCRIMINATOR: [u8; 8] = [249, 210, 64, 145, 197, 241, 57, 51];
pub const ROUND_DISCRIMINATOR: [u8; 8] = [87, 127, 165, 51, 73, 78, 116, 174];
pub const DISPUTE_DISCRIMINATOR: [u8; 8] = [36, 49, 241, 67, 40, 36, 241, 74];
pub const STATUS_LIST_DISCRIMINATOR: [u8; 8] = [128, 202, 59, 123, 106, 255, 246, 185];
pub const BUFFER_DISCRIMINATOR: [u8; 8] = [115, 5, 212, 192, 85, 30, 46, 41];

pub const TASK_SPACE: usize = size_of::<TaskLayout>();
pub const ENROLLMENT_SPACE: usize = size_of::<EnrollmentLayout>();
pub const ROUND_SPACE: usize = size_of::<RoundLayout>();
pub const DISPUTE_SPACE: usize = size_of::<DisputeLayout>();
/// The fixed head of a status list, which its bitmap follows.
pub const STATUS_LIST_HEADER: usize = size_of::<StatusListLayout>();
/// The fixed head of a buffer, which its bytes follow.
pub const BUFFER_HEADER: usize = size_of::<BufferLayout>();

pub const TASK_OPEN: u8 = 0;
pub const TASK_FINISHED: u8 = 1;
pub const TASK_ABORTED: u8 = 2;

pub const SITE_ACTIVE: u8 = 0;
pub const SITE_REVOKED: u8 = 1;
pub const SITE_WITHDRAWN: u8 = 2;

pub const ROUND_COLLECTING: u8 = 0;
pub const ROUND_COMMITTED: u8 = 1;
pub const ROUND_POSTED: u8 = 2;
pub const ROUND_FINAL: u8 = 3;
pub const ROUND_FRAUD: u8 = 4;

#[inline(always)]
fn array_at<const N: usize>(data: &[u8], offset: usize) -> &[u8; N] {
    data[offset..offset + N].try_into().unwrap()
}

#[inline(always)]
fn array_at_mut<const N: usize>(data: &mut [u8], offset: usize) -> &mut [u8; N] {
    (&mut data[offset..offset + N]).try_into().unwrap()
}

macro_rules! getter {
    ($layout:ident, $(#[$meta:meta])* $field:ident: u8) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $field(&self) -> u8 {
            self.0[offset_of!($layout, $field)]
        }
    };
    ($layout:ident, $(#[$meta:meta])* $field:ident: bool) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $field(&self) -> bool {
            self.0[offset_of!($layout, $field)] != 0
        }
    };
    ($layout:ident, $(#[$meta:meta])* $field:ident: [u8; $len:expr]) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $field(&self) -> &[u8; $len] {
            array_at(&self.0, offset_of!($layout, $field))
        }
    };
    ($layout:ident, $(#[$meta:meta])* $field:ident: $int:ident) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $field(&self) -> $int {
            $int::from_le_bytes(*array_at(&self.0, offset_of!($layout, $field)))
        }
    };
}

macro_rules! setter {
    ($layout:ident, $field:ident / $setter:ident: u8) => {
        #[inline(always)]
        pub fn $setter(&mut self, value: u8) {
            self.0[offset_of!($layout, $field)] = value;
        }
    };
    ($layout:ident, $field:ident / $setter:ident: bool) => {
        #[inline(always)]
        pub fn $setter(&mut self, value: bool) {
            self.0[offset_of!($layout, $field)] = value as u8;
        }
    };
    ($layout:ident, $field:ident / $setter:ident: [u8; $len:expr]) => {
        #[inline(always)]
        pub fn $setter(&mut self) -> &mut [u8; $len] {
            array_at_mut(&mut self.0, offset_of!($layout, $field))
        }
    };
    ($layout:ident, $field:ident / $setter:ident: $int:ident) => {
        #[inline(always)]
        pub fn $setter(&mut self, value: $int) {
            *array_at_mut(&mut self.0, offset_of!($layout, $field)) = value.to_le_bytes();
        }
    };
}

/// An account view that knows the address its account is derived at.
pub trait Derived<D>: Sized {
    fn parse(data: D) -> Result<Self, ProgramError>;
    fn address(&self) -> Address;
}

fn derive<const N: usize>(seeds: &[&[u8]; N], bump: u8) -> Address {
    Address::derive_address(seeds, Some(bump), &crate::ID)
}

/// Defines an account from its field table: the layout struct, and the view
/// with a getter and a setter per field. An array's setter hands out the
/// array to write into.
macro_rules! account {
    (
        $(#[$meta:meta])*
        $view:ident, $layout:ident, $discriminator:ident {
            $( $(#[$field_meta:meta])* $field:ident / $setter:ident: $ty:tt, )*
        }
    ) => {
        #[repr(C)]
        pub struct $layout {
            pub discriminator: [u8; 8],
            $( pub $field: [u8; size_of::<$ty>()], )*
        }

        $(#[$meta])*
        #[derive(Clone, Copy, Debug)]
        pub struct $view<D>(D);

        impl<D: Deref<Target = [u8]>> $view<D> {
            /// A view of account data that carries this account's
            /// discriminator and is long enough for its layout.
            pub fn parse(data: D) -> Result<Self, ProgramError> {
                if data.len() < size_of::<$layout>() || data[..8] != $discriminator {
                    return Err(ProgramError::InvalidAccountData);
                }
                Ok(Self(data))
            }

            $( getter!($layout, $(#[$field_meta])* $field: $ty); )*
        }

        impl<D: DerefMut<Target = [u8]>> $view<D> {
            /// Stamps the discriminator on freshly allocated, zeroed data.
            pub fn init(mut data: D) -> Result<Self, ProgramError> {
                if data.len() < size_of::<$layout>() {
                    return Err(ProgramError::AccountDataTooSmall);
                }
                data[..8].copy_from_slice(&$discriminator);
                Ok(Self(data))
            }

            $( setter!($layout, $field / $setter: $ty); )*
        }

        impl<D: Deref<Target = [u8]>> Derived<D> for $view<D> {
            fn parse(data: D) -> Result<Self, ProgramError> {
                $view::parse(data)
            }

            fn address(&self) -> Address {
                $view::address(self)
            }
        }
    };
}

account! {
    /// A task, at `["task", publisher, plan_id]`. It holds the reward escrow and the
    /// aggregator's stake above its rent.
    Task, TaskLayout, TASK_DISCRIMINATOR {
        bump / set_bump: u8,
        /// `TASK_OPEN`, `TASK_FINISHED` or `TASK_ABORTED`.
        state / set_state: u8,
        /// Whether the aggregator has staked.
        backed / set_backed: bool,
        /// Whether the task was aborted by a fraud proof.
        fraud / set_fraud: bool,
        helpers / set_helpers: u8,
        sites / set_sites: u16,
        min_cohort / set_min_cohort: u16,
        /// How many sites a round invites, or 0 when it invites them all.
        sample / set_sample: u16,
        colluding / set_colluding: u16,
        rounds / set_rounds: u32,
        /// The latest round opened, 0 before any.
        round / set_round: u32,
        /// The latest round settled, 0 before any.
        settled / set_settled: u32,
        publisher / publisher_mut: [u8; 32],
        aggregator / aggregator_mut: [u8; 32],
        plan_id / plan_id_mut: [u8; 32],
        sites_root / sites_root_mut: [u8; 32],
        /// The digest of the latest settled model, the initial one at first.
        model / model_mut: [u8; 32],
        /// How many coordinates a vector of the task has.
        coords / set_coords: u64,
        sensitivity / set_sensitivity: u64,
        noise / set_noise: u64,
        budget / set_budget: u64,
        spent / set_spent: u64,
        site_stake / set_site_stake: u64,
        aggregator_stake / set_aggregator_stake: u64,
        /// The aggregator's stake while the task holds it.
        stake_held / set_stake_held: u64,
        reward_per_round / set_reward_per_round: u64,
        dispute_bond / set_dispute_bond: u64,
        /// Rewards of settled rounds that sites have not claimed yet.
        owed / set_owed: u64,
        challenge_slots / set_challenge_slots: u64,
        stall_slots / set_stall_slots: u64,
        /// The slot of the last change to the task or its current round.
        last_change / set_last_change: u64,
        enrolled / enrolled_mut: [u8; 128],
        revoked / revoked_mut: [u8; 128],
    }
}

account! {
    /// A site's enrollment, at `["site", task, index u16 LE]`. It holds the
    /// site's stake above its rent.
    Enrollment, EnrollmentLayout, ENROLLMENT_DISCRIMINATOR {
        bump / set_bump: u8,
        /// `SITE_ACTIVE`, `SITE_REVOKED` or `SITE_WITHDRAWN`.
        status / set_status: u8,
        index / set_index: u16,
        status_ref_count / set_status_ref_count: u8,
        task / task_mut: [u8; 32],
        wallet / wallet_mut: [u8; 32],
        stake / set_stake: u64,
        /// The digest of the site's Join record.
        admission / admission_mut: [u8; 32],
        /// Up to four `(status list address, bit index u32)` pairs.
        status_refs / status_refs_mut: [u8; 144],
    }
}

account! {
    /// A round, at `["round", task, round u32 LE]`.
    Round, RoundLayout, ROUND_DISCRIMINATOR {
        bump / set_bump: u8,
        /// `ROUND_COLLECTING` through `ROUND_FRAUD`.
        state / set_state: u8,
        /// How many disputes are open.
        disputes / set_disputes: u32,
        round / set_round: u32,
        cohort_count / set_cohort_count: u16,
        task / task_mut: [u8; 32],
        beacon / beacon_mut: [u8; 32],
        inputs_root / inputs_root_mut: [u8; 32],
        close_head / close_head_mut: [u8; 32],
        /// One root per helper, in helper order. Unused slots are zero.
        mask_roots / mask_roots_mut: [u8; 128],
        aggregate_root / aggregate_root_mut: [u8; 32],
        next_model / next_model_mut: [u8; 32],
        seal_head / seal_head_mut: [u8; 32],
        posted_slot / set_posted_slot: u64,
        /// Lamports per cohort member, set when the round settles.
        share / set_share: u64,
        cohort / cohort_mut: [u8; 128],
        acked / acked_mut: [u8; 128],
        claimed / claimed_mut: [u8; 128],
    }
}

account! {
    /// One challenger's dispute over one leaf of a round, at
    /// `["dispute", task, round u32 LE, challenger]`.
    /// It holds the challenger's bond above its rent.
    Dispute, DisputeLayout, DISPUTE_DISCRIMINATOR {
        bump / set_bump: u8,
        /// Bit `h` is set once the mask leaf of helper `h` has been given.
        seen_masks / set_seen_masks: u8,
        aggregate_given / set_aggregate_given: bool,
        round / set_round: u32,
        /// How many coordinates the disputed leaf holds.
        leaf_coords / set_leaf_coords: u16,
        /// How many input leaves have been given.
        inputs_seen / set_inputs_seen: u16,
        task / task_mut: [u8; 32],
        challenger / challenger_mut: [u8; 32],
        leaf_index / set_leaf_index: u64,
        deadline / set_deadline: u64,
        /// One bit per cohort position whose input leaf has been given.
        seen_inputs / seen_inputs_mut: [u8; 128],
        /// The inputs given so far minus the masks given so far, 128 reduced
        /// coordinates.
        acc / acc_mut: [u8; 1024],
        /// The aggregate leaf, 128 reduced coordinates.
        aggregate / aggregate_mut: [u8; 1024],
    }
}

account! {
    /// The head of a status list, at `["status", issuer, list u32 LE]`. The
    /// bitmap follows it, `ceil(bits / 8)` bytes.
    StatusList, StatusListLayout, STATUS_LIST_DISCRIMINATOR {
        bump / set_bump: u8,
        list / set_list: u32,
        bits / set_bits: u32,
        issuer / issuer_mut: [u8; 32],
        /// Goes up by one with every bit set.
        sequence / set_sequence: u64,
    }
}

account! {
    /// The head of a buffer, at `["buffer", owner]`. The buffer's bytes
    /// follow it.
    Buffer, BufferLayout, BUFFER_DISCRIMINATOR {
        bump / set_bump: u8,
        /// How many of the buffer's bytes are in use, from its start.
        used / set_used: u32,
        owner / owner_mut: [u8; 32],
    }
}

// Clients read these layouts, so a field that moves fails the build.
const _: () = {
    assert!(TASK_SPACE == 561);
    assert!(offset_of!(TaskLayout, sites) == 13);
    assert!(offset_of!(TaskLayout, publisher) == 33);
    assert!(offset_of!(TaskLayout, coords) == 193);
    assert!(offset_of!(TaskLayout, last_change) == 297);
    assert!(offset_of!(TaskLayout, enrolled) == 305);
    assert!(ENROLLMENT_SPACE == 261);
    assert!(offset_of!(EnrollmentLayout, task) == 13);
    assert!(offset_of!(EnrollmentLayout, status_refs) == 117);
    assert!(ROUND_SPACE == 772);
    assert!(offset_of!(RoundLayout, task) == 20);
    assert!(offset_of!(RoundLayout, posted_slot) == 372);
    assert!(offset_of!(RoundLayout, cohort) == 388);
    assert!(DISPUTE_SPACE == 2275);
    assert!(offset_of!(DisputeLayout, task) == 19);
    assert!(offset_of!(DisputeLayout, acc) == 227);
    assert!(STATUS_LIST_HEADER == 57);
    assert!(BUFFER_HEADER == 45);
    assert!(MAX_STATUS_REFS * STATUS_REF_LEN == 144);
    assert!(MAX_HELPERS * 32 == 128);
    assert!(LEAF_LEN == 1024);
};

impl<D: Deref<Target = [u8]>> Task<D> {
    /// The address the stored seeds and bump derive.
    pub fn address(&self) -> Address {
        derive(&[TASK_SEED, self.publisher(), self.plan_id()], self.bump())
    }

    pub fn is_open(&self) -> bool {
        self.state() == TASK_OPEN
    }

    /// How many leaves a vector tree of the task has.
    pub fn leaves(&self) -> u64 {
        self.coords().div_ceil(LEAF_COORDS)
    }

    /// How many coordinates the leaf at `leaf_index` holds, where
    /// `leaf_index < leaves()`. Only the last leaf can be short.
    pub fn leaf_coords(&self, leaf_index: u64) -> u16 {
        let before = leaf_index * LEAF_COORDS;
        (self.coords() - before).min(LEAF_COORDS) as u16
    }

    /// The privacy cost of a round with a cohort of `cohort` sites.
    pub fn round_cost(&self, cohort: u16) -> Result<u64, ProgramError> {
        privacy::round_cost(self.sensitivity(), self.noise(), self.colluding(), cohort)
    }
}

impl<D: Deref<Target = [u8]>> Enrollment<D> {
    /// The address the stored seeds and bump derive.
    pub fn address(&self) -> Address {
        let index = self.index().to_le_bytes();
        derive(&[SITE_SEED, self.task(), &index], self.bump())
    }

    /// The `i`-th status reference: the list's address and the bit index.
    pub fn status_ref(&self, i: usize) -> (&[u8; 32], u32) {
        let entry = &self.status_refs()[i * STATUS_REF_LEN..];
        (array_at(entry, 0), u32::from_le_bytes(*array_at(entry, 32)))
    }
}

impl<D: Deref<Target = [u8]>> Round<D> {
    /// The address the stored seeds and bump derive.
    pub fn address(&self) -> Address {
        let round = self.round().to_le_bytes();
        derive(&[ROUND_SEED, self.task(), &round], self.bump())
    }

    /// The mask root of helper `helper`.
    pub fn mask_root(&self, helper: usize) -> &Digest {
        array_at(self.mask_roots(), helper * 32)
    }
}

impl<D: Deref<Target = [u8]>> Dispute<D> {
    /// The address the stored seeds and bump derive.
    pub fn address(&self) -> Address {
        let round = self.round().to_le_bytes();
        derive(
            &[DISPUTE_SEED, self.task(), &round, self.challenger()],
            self.bump(),
        )
    }

    /// Whether every input, every mask and the aggregate leaf are in.
    pub fn is_complete(&self, cohort_count: u16, helpers: u8) -> bool {
        self.inputs_seen() == cohort_count
            && self.seen_masks() == (1 << helpers) - 1
            && self.aggregate_given()
    }

    /// Whether the running sum equals the aggregate leaf.
    pub fn sums_match(&self) -> bool {
        self.acc() == self.aggregate()
    }
}

impl<D: Deref<Target = [u8]>> StatusList<D> {
    /// The address the stored seeds and bump derive.
    pub fn address(&self) -> Address {
        let list = self.list().to_le_bytes();
        derive(&[STATUS_SEED, self.issuer(), &list], self.bump())
    }

    pub fn bitmap(&self) -> &[u8] {
        &self.0[STATUS_LIST_HEADER..]
    }

    /// Whether bit `bit` is set. A bit the list does not have is not set.
    pub fn is_set(&self, bit: u32) -> bool {
        bit < self.bits() && crate::bitmap::get(self.bitmap(), bit as usize)
    }
}

impl<D: DerefMut<Target = [u8]>> StatusList<D> {
    pub fn bitmap_mut(&mut self) -> &mut [u8] {
        &mut self.0[STATUS_LIST_HEADER..]
    }
}

impl<D: Deref<Target = [u8]>> Buffer<D> {
    /// The address the stored seeds and bump derive.
    pub fn address(&self) -> Address {
        derive(&[BUFFER_SEED, self.owner()], self.bump())
    }

    pub fn capacity(&self) -> usize {
        self.0.len() - BUFFER_HEADER
    }

    /// The bytes in use.
    pub fn contents(&self) -> &[u8] {
        &self.0[BUFFER_HEADER..][..self.used() as usize]
    }
}

impl<D: DerefMut<Target = [u8]>> Buffer<D> {
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0[BUFFER_HEADER..]
    }
}

/// The bytes a status list of `bits` bits takes.
pub const fn status_list_space(bits: u32) -> usize {
    STATUS_LIST_HEADER + bits.div_ceil(8) as usize
}

/// The bytes a buffer of `capacity` bytes takes.
pub const fn buffer_space(capacity: u32) -> usize {
    BUFFER_HEADER + capacity as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_read_what_they_wrote() {
        let mut data = vec![0u8; TASK_SPACE];
        assert!(Task::parse(data.as_slice()).is_err());
        let mut task = Task::init(data.as_mut_slice()).unwrap();
        task.set_backed(true);
        task.set_sites(1024);
        task.set_rounds(7);
        task.set_budget(u64::MAX - 1);
        *task.model_mut() = [9; 32];
        crate::bitmap::set(task.enrolled_mut(), 1023);

        let task = Task::parse(data.as_slice()).unwrap();
        assert!(task.backed() && !task.fraud());
        assert_eq!(task.sites(), 1024);
        assert_eq!(task.rounds(), 7);
        assert_eq!(task.budget(), u64::MAX - 1);
        assert_eq!(task.model(), &[9; 32]);
        assert_eq!(task.enrolled()[127], 0x80);
        assert_eq!(data[..8], TASK_DISCRIMINATOR);
        assert_eq!(data[13..15], 1024u16.to_le_bytes());
        assert!(Task::parse(&data[..TASK_SPACE - 1]).is_err());
        assert!(Round::parse(data.as_slice()).is_err());
    }

    #[test]
    fn leaves_cut_a_vector_every_128_coordinates() {
        let mut data = vec![0u8; TASK_SPACE];
        let mut task = Task::init(data.as_mut_slice()).unwrap();
        for (coords, leaves, last) in [(1, 1, 1), (128, 1, 128), (129, 2, 1), (296, 3, 40)] {
            task.set_coords(coords);
            assert_eq!(task.leaves(), leaves);
            assert_eq!(task.leaf_coords(leaves - 1), last);
            assert_eq!(task.leaf_coords(0), coords.min(128) as u16);
        }
        task.set_coords(0);
        assert_eq!(task.leaves(), 0);
    }

    #[test]
    fn status_refs_and_bits() {
        let mut data = vec![0u8; ENROLLMENT_SPACE];
        let mut site = Enrollment::init(data.as_mut_slice()).unwrap();
        site.status_refs_mut()[STATUS_REF_LEN..][..32].fill(5);
        site.status_refs_mut()[STATUS_REF_LEN + 32..][..4].copy_from_slice(&77u32.to_le_bytes());
        assert_eq!(site.status_ref(1), (&[5; 32], 77));

        let mut data = vec![0u8; status_list_space(12)];
        assert_eq!(data.len(), STATUS_LIST_HEADER + 2);
        let mut list = StatusList::init(data.as_mut_slice()).unwrap();
        list.set_bits(12);
        crate::bitmap::set(list.bitmap_mut(), 11);
        assert!(list.is_set(11) && !list.is_set(10) && !list.is_set(12) && !list.is_set(9000));
    }
}
