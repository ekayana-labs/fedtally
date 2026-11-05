//! Domain errors, surfaced as custom program error codes starting at 6000.
//! Clients match on the numbers, so new codes are only ever appended.

use pinocchio::error::ProgramError;

macro_rules! ledger_errors {
    ($( $name:ident = $code:literal, $message:literal; )*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u32)]
        #[non_exhaustive]
        pub enum LedgerError {
            $( #[doc = $message] $name = $code, )*
        }

        impl LedgerError {
            /// Every error, in code order from 6000.
            pub const ALL: &'static [LedgerError] = &[ $( LedgerError::$name, )* ];

            /// What the error means.
            pub const fn message(self) -> &'static str {
                match self {
                    $( LedgerError::$name => $message, )*
                }
            }
        }
    };
}

ledger_errors! {
    InvalidSites = 6000, "A task has between 1 and 1024 sites";
    InvalidHelpers = 6001, "A task has at most 4 helpers";
    InvalidMinCohort = 6002, "The minimum cohort is between 1 and the number of sites";
    InvalidSample = 6003, "The sample is 0, or between the minimum cohort and the number of sites";
    InvalidRounds = 6004, "A task has at least one round";
    InvalidCollusion = 6005, "The cohort is not larger than the colluding sites the task assumes";
    BudgetTooSmall = 6006, "The privacy budget does not cover one round at the minimum cohort";
    CostOverflow = 6007, "The privacy cost of a round does not fit 64 bits";
    Unauthorized = 6008, "The signer is not the wallet this instruction is for";
    AccountMismatch = 6009, "An account does not belong with the others it was passed with";
    TaskNotOpen = 6010, "The task is finished or aborted";
    AlreadyBacked = 6011, "The aggregator has already staked on this task";
    NotBacked = 6012, "The aggregator has not staked on this task yet";
    InvalidSiteProof = 6013, "The wallet is not at this index under the task's sites root";
    TooManyStatusRefs = 6014, "An enrollment names at most 4 status list bits";
    NoRoundsLeft = 6015, "Every round of the task has been opened";
    PreviousRoundNotFinal = 6016, "The previous round has not been settled";
    BudgetExhausted = 6017, "The privacy budget does not cover this round";
    InvalidRoundState = 6018, "The round is not in the state this instruction needs";
    InvalidCohortSize = 6019, "The cohort is smaller than the minimum or larger than the task";
    InvalidCohortMember = 6020, "A cohort member is not enrolled, is revoked, or is not a site";
    InvalidThreshold = 6021, "The threshold does not invite exactly the sampled number of sites";
    NotInvited = 6022, "A cohort member's ticket is above the threshold";
    InvalidMaskRoots = 6023, "There must be one mask root per helper";
    SealMismatch = 6024, "The seal head differs from the one posted for the round";
    NotInCohort = 6025, "The site is not in the round's cohort";
    AlreadyAcked = 6026, "The site has already acknowledged the round";
    RoundDisputed = 6027, "The round has an open dispute";
    AcksMissing = 6028, "Not every cohort member has acknowledged the round";
    ChallengeWindowOpen = 6029, "The round's challenge window has not passed yet";
    ChallengeWindowClosed = 6030, "The round's challenge window has passed";
    AlreadyClaimed = 6031, "The site has already claimed its share of the round";
    InvalidLeafIndex = 6032, "The vectors have no leaf with this index";
    InvalidProofPart = 6033, "The dispute has no such input, mask or aggregate part";
    ProofPartRepeated = 6034, "This part of the proof has already been given";
    InvalidProofLength = 6035, "The proof is not as long as the disputed leaf and its paths";
    InvalidProof = 6036, "The proof does not lead to the committed root";
    DisputeNotOver = 6037, "The dispute is incomplete and its deadline has not passed";
    InvalidStatusList = 6038, "A status list has between 1 and 80000 bits";
    InvalidBit = 6039, "The status list has no bit with this index";
    AlreadyRevoked = 6040, "The bit is already set";
    StatusListMismatch = 6041, "The accounts are not the status lists the enrollment names";
    NotStalled = 6042, "Only the aggregator or the publisher may abort before the task stalls";
    FundsLocked = 6043, "The task or enrollment does not release these funds yet";
    NothingToWithdraw = 6044, "There is nothing to withdraw";
    AlreadyWithdrawn = 6045, "The stake has already been withdrawn";
    InvalidBufferSize = 6046, "A buffer holds between 1 and 10000 bytes";
    InvalidBufferWrite = 6047, "A write starts the buffer over or continues it, within its size";
    InvalidRole = 6048, "A withdrawal is for a site, the aggregator or the publisher";
    InvalidWindows = 6049, "A task may only be called stalled after a challenge window has had time to pass";
}

impl LedgerError {
    /// The custom program error code.
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// The error behind a custom program error code, if the code is one of
    /// this program's.
    pub fn from_code(code: u32) -> Option<Self> {
        Self::ALL.iter().copied().find(|e| e.code() == code)
    }
}

impl TryFrom<u32> for LedgerError {
    /// The code, when it is not one of this program's.
    type Error = u32;

    fn try_from(code: u32) -> Result<Self, u32> {
        Self::from_code(code).ok_or(code)
    }
}

impl core::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.message())
    }
}

impl core::error::Error for LedgerError {}

impl From<LedgerError> for ProgramError {
    #[inline(always)]
    fn from(e: LedgerError) -> Self {
        ProgramError::Custom(e as u32)
    }
}

/// Early return a domain error unless `cond` holds.
#[inline(always)]
pub fn require(cond: bool, err: LedgerError) -> Result<(), ProgramError> {
    if cond {
        Ok(())
    } else {
        Err(err.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_run_from_6000_without_gaps() {
        for (i, e) in LedgerError::ALL.iter().enumerate() {
            assert_eq!(e.code(), 6000 + i as u32);
            assert_eq!(LedgerError::from_code(e.code()), Some(*e));
            assert_eq!(LedgerError::try_from(e.code()), Ok(*e));
            assert_eq!(ProgramError::from(*e), ProgramError::Custom(e.code()));
            assert!(!e.message().is_empty());
        }
        let next = 6000 + LedgerError::ALL.len() as u32;
        assert_eq!(LedgerError::from_code(next), None);
        assert_eq!(LedgerError::try_from(5999), Err(5999));
        assert_eq!(
            LedgerError::SealMismatch.to_string(),
            "The seal head differs from the one posted for the round"
        );
    }
}
