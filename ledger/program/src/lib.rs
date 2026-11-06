//! The on-chain anchor and arbiter of a fedtally run.
//!
//! A run is driven by a signed log off chain. This program holds what has to
//! be neutral: the stake and reward escrow, the privacy budget, the cohort
//! sampling check, each round's commitments and acknowledgements, and the
//! fraud proof that slashes an aggregator whose posted aggregate is not the
//! sum of the committed vectors. It never sees a model, an update or patient
//! data. Everything it stores is a 32 byte digest, a bitmap or a number.
//!
//! The program is built on [Pinocchio](https://github.com/anza-xyz/pinocchio).
//! It is `no_std` and allocation free, every account has a fixed layout that
//! handlers edit in place, and all the hashing goes through the SHA-256
//! syscall.
//!
//! The modules [`merkle`], [`field`], [`privacy`] and [`sampling`] hold the
//! definitions this program shares with the off-chain code. They compile for
//! the host as well, so a client can call the very functions the program runs.

#![cfg_attr(target_os = "solana", no_std)]

pub mod bitmap;
#[cfg(not(target_os = "solana"))]
pub mod client;
pub mod error;
pub mod field;
pub mod hash;
pub mod instructions;
pub mod merkle;
pub mod privacy;
pub mod reader;
pub mod sampling;
pub mod state;

use pinocchio::{error::ProgramError, AccountView, Address, ProgramResult};

#[cfg(not(feature = "no-entrypoint"))]
pinocchio::program_entrypoint!(process_instruction);
#[cfg(not(feature = "no-entrypoint"))]
pinocchio::no_allocator!();
#[cfg(not(feature = "no-entrypoint"))]
pinocchio::nostd_panic_handler!();

/// The program ID, `BZ3EJcg4FUF2Yzp48FnADkw5FhobY1NAvgneoSV2v6gD`. It is a
/// placeholder whose keypair was never kept.
pub const ID: Address = Address::new_from_array(five8_const::decode_32_const(
    "BZ3EJcg4FUF2Yzp48FnADkw5FhobY1NAvgneoSV2v6gD",
));

/// Instruction discriminators, each `sha256("global:<name>")[..8]`.
pub mod ix {
    pub const PUBLISH: [u8; 8] = [129, 177, 182, 160, 184, 224, 219, 5];
    pub const BACK: [u8; 8] = [1, 144, 109, 177, 110, 131, 124, 0];
    pub const ENROLL: [u8; 8] = [58, 12, 36, 3, 142, 28, 1, 43];
    pub const OPEN_ROUND: [u8; 8] = [66, 235, 123, 240, 8, 35, 185, 159];
    pub const COMMIT_ROOT: [u8; 8] = [142, 242, 81, 248, 39, 238, 194, 125];
    pub const POST_AGGREGATE: [u8; 8] = [192, 227, 68, 121, 216, 212, 173, 210];
    pub const ACK: [u8; 8] = [227, 136, 23, 119, 174, 79, 201, 17];
    pub const SETTLE: [u8; 8] = [175, 42, 185, 87, 144, 131, 102, 212];
    pub const CLAIM: [u8; 8] = [62, 198, 214, 193, 213, 159, 108, 210];
    pub const DISPUTE_OPEN: [u8; 8] = [171, 66, 231, 238, 207, 254, 151, 219];
    pub const DISPUTE_LEAF: [u8; 8] = [180, 26, 101, 135, 251, 140, 111, 60];
    pub const DISPUTE_CLOSE: [u8; 8] = [238, 219, 107, 139, 243, 19, 45, 225];
    pub const DEMAND_OPEN: [u8; 8] = [62, 50, 59, 206, 85, 252, 38, 26];
    pub const DEMAND_ANSWER: [u8; 8] = [76, 254, 20, 65, 181, 107, 171, 157];
    pub const DEMAND_CLOSE: [u8; 8] = [107, 127, 189, 229, 211, 29, 172, 254];
    pub const ROUND_CLOSE: [u8; 8] = [5, 157, 205, 71, 76, 34, 229, 65];
    pub const VOID_ROUND: [u8; 8] = [102, 72, 126, 187, 12, 18, 16, 96];
    pub const STATUS_INIT: [u8; 8] = [20, 18, 138, 138, 31, 14, 2, 29];
    pub const REVOKE: [u8; 8] = [170, 23, 31, 34, 133, 173, 93, 242];
    pub const REVOKE_CHECK: [u8; 8] = [239, 53, 173, 18, 53, 253, 110, 115];
    pub const ABORT: [u8; 8] = [73, 205, 102, 177, 241, 200, 145, 80];
    pub const WITHDRAW: [u8; 8] = [183, 18, 70, 156, 148, 109, 161, 34];
    pub const BUFFER_INIT: [u8; 8] = [236, 53, 136, 9, 113, 231, 227, 24];
    pub const BUFFER_WRITE: [u8; 8] = [123, 29, 39, 50, 68, 109, 63, 94];
    pub const BUFFER_CLOSE: [u8; 8] = [193, 218, 99, 231, 212, 8, 29, 178];
}

#[inline]
pub fn process_instruction(
    program_id: &Address,
    accounts: &mut [AccountView],
    instruction_data: &[u8],
) -> ProgramResult {
    if program_id != &ID {
        return Err(ProgramError::IncorrectProgramId);
    }
    let (discriminator, args) = instruction_data
        .split_first_chunk::<8>()
        .ok_or(ProgramError::InvalidInstructionData)?;

    use instructions::*;
    match *discriminator {
        ix::PUBLISH => publish::process(accounts, args),
        ix::BACK => back::process(accounts, args),
        ix::ENROLL => enroll::process(accounts, args),
        ix::OPEN_ROUND => open_round::process(accounts, args),
        ix::COMMIT_ROOT => commit_root::process(accounts, args),
        ix::POST_AGGREGATE => post_aggregate::process(accounts, args),
        ix::ACK => ack::process(accounts, args),
        ix::SETTLE => settle::process(accounts, args),
        ix::CLAIM => claim::process(accounts, args),
        ix::DISPUTE_OPEN => dispute_open::process(accounts, args),
        ix::DISPUTE_LEAF => dispute_leaf::process(accounts, args),
        ix::DISPUTE_CLOSE => dispute_close::process(accounts, args),
        ix::DEMAND_OPEN => demand::open(accounts, args),
        ix::DEMAND_ANSWER => demand::answer(accounts, args),
        ix::DEMAND_CLOSE => demand::close(accounts, args),
        ix::ROUND_CLOSE => round_close::process(accounts, args),
        ix::VOID_ROUND => void_round::process(accounts, args),
        ix::STATUS_INIT => status_init::process(accounts, args),
        ix::REVOKE => revoke::process(accounts, args),
        ix::REVOKE_CHECK => revoke_check::process(accounts, args),
        ix::ABORT => abort::process(accounts, args),
        ix::WITHDRAW => withdraw::process(accounts, args),
        ix::BUFFER_INIT => buffer::init(accounts, args),
        ix::BUFFER_WRITE => buffer::write(accounts, args),
        ix::BUFFER_CLOSE => buffer::close(accounts, args),
        _ => Err(ProgramError::InvalidInstructionData),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix(name: &str) -> [u8; 8] {
        let digest = hash::sha256(&[name.as_bytes()]);
        let mut out = [0u8; 8];
        out.copy_from_slice(&digest[..8]);
        out
    }

    #[test]
    fn discriminators_are_hashes_of_their_names() {
        let instructions = [
            ("publish", ix::PUBLISH),
            ("back", ix::BACK),
            ("enroll", ix::ENROLL),
            ("open_round", ix::OPEN_ROUND),
            ("commit_root", ix::COMMIT_ROOT),
            ("post_aggregate", ix::POST_AGGREGATE),
            ("ack", ix::ACK),
            ("settle", ix::SETTLE),
            ("claim", ix::CLAIM),
            ("dispute_open", ix::DISPUTE_OPEN),
            ("dispute_leaf", ix::DISPUTE_LEAF),
            ("dispute_close", ix::DISPUTE_CLOSE),
            ("demand_open", ix::DEMAND_OPEN),
            ("demand_answer", ix::DEMAND_ANSWER),
            ("demand_close", ix::DEMAND_CLOSE),
            ("round_close", ix::ROUND_CLOSE),
            ("void_round", ix::VOID_ROUND),
            ("status_init", ix::STATUS_INIT),
            ("revoke", ix::REVOKE),
            ("revoke_check", ix::REVOKE_CHECK),
            ("abort", ix::ABORT),
            ("withdraw", ix::WITHDRAW),
            ("buffer_init", ix::BUFFER_INIT),
            ("buffer_write", ix::BUFFER_WRITE),
            ("buffer_close", ix::BUFFER_CLOSE),
        ];
        for (i, (name, discriminator)) in instructions.iter().enumerate() {
            assert_eq!(prefix(&format!("global:{name}")), *discriminator, "{name}");
            assert!(
                instructions[..i].iter().all(|(_, d)| d != discriminator),
                "{name} collides"
            );
        }
        let accounts = [
            ("Task", state::TASK_DISCRIMINATOR),
            ("Enrollment", state::ENROLLMENT_DISCRIMINATOR),
            ("Round", state::ROUND_DISCRIMINATOR),
            ("Dispute", state::DISPUTE_DISCRIMINATOR),
            ("Demand", state::DEMAND_DISCRIMINATOR),
            ("StatusList", state::STATUS_LIST_DISCRIMINATOR),
            ("Buffer", state::BUFFER_DISCRIMINATOR),
        ];
        for (name, discriminator) in accounts {
            assert_eq!(prefix(&format!("account:{name}")), discriminator, "{name}");
        }
    }
}
