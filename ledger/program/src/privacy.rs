//! The privacy cost of a round, which the program charges against a task's
//! budget.

use pinocchio::error::ProgramError;

use crate::error::LedgerError;

/// The fixed-point scale of a cost, 40 fractional bits.
const SCALE_BITS: u32 = 40;

/// A group's share of a round's reward is in millionths of it.
pub const WHOLE_SHARE: u32 = 1_000_000;

/// The size of the smallest group a cohort splits into, with groups of at least
/// `group` sites. The log splits a cohort the same way.
pub fn smallest_group(cohort: u16, group: u16) -> u16 {
    cohort / (cohort / group.max(1)).max(1)
}

/// What a round with a cohort of `cohort` sites costs:
/// `floor((s * s << 40) / (2 * noise * (cohort - colluding))) + 2`.
///
/// A task without noise has no privacy terms and every round costs 0.
/// Otherwise the cohort has to be larger than the colluding sites the task
/// assumes, and the cost has to fit 64 bits.
pub fn round_cost(
    sensitivity: u64,
    noise: u64,
    colluding: u16,
    cohort: u16,
) -> Result<u64, ProgramError> {
    if noise == 0 {
        return Ok(0);
    }
    let honest = cohort
        .checked_sub(colluding)
        .filter(|honest| *honest > 0)
        .ok_or(LedgerError::InvalidCollusion)?;
    let square = (sensitivity as u128) * (sensitivity as u128);
    if square > u128::MAX >> SCALE_BITS {
        return Err(LedgerError::CostOverflow.into());
    }
    // 2 * noise * honest stays below 2^81.
    let divisor = 2 * (noise as u128) * (honest as u128);
    let cost = (square << SCALE_BITS) / divisor + 2;
    u64::try_from(cost).map_err(|_| LedgerError::CostOverflow.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SENSITIVITY: u64 = 262146;
    const NOISE: u64 = 1 << 34;

    #[test]
    fn published_vectors() {
        let cost = |cohort, colluding| round_cost(SENSITIVITY, NOISE, colluding, cohort);
        assert_eq!(cost(4, 0), Ok(549764202530));
        assert_eq!(cost(3, 0), Ok(733018936706));
        assert_eq!(cost(8, 2), Ok(366509468354));
        assert_eq!(cost(2, 1), Ok(2199056810114));
    }

    #[test]
    fn a_cohort_within_the_colluding_sites_has_no_cost() {
        let err = Err(LedgerError::InvalidCollusion.into());
        assert_eq!(round_cost(SENSITIVITY, NOISE, 2, 2), err);
        assert_eq!(round_cost(SENSITIVITY, NOISE, 3, 2), err);
        assert_eq!(round_cost(SENSITIVITY, NOISE, 0, 0), err);
    }

    #[test]
    fn no_noise_means_no_cost() {
        assert_eq!(round_cost(SENSITIVITY, 0, 0, 4), Ok(0));
        assert_eq!(round_cost(u64::MAX, 0, 9, 4), Ok(0));
    }

    #[test]
    fn costs_that_do_not_fit_are_refused() {
        let err = Err(LedgerError::CostOverflow.into());
        assert_eq!(round_cost(1 << 44, NOISE, 0, 1), err);
        assert_eq!(round_cost(u64::MAX, u64::MAX, 0, 1024), err);
        assert_eq!(round_cost(1 << 32, 1, 0, 1), err);
        assert_eq!(
            round_cost((1 << 44) - 1, u64::MAX, 0, 1),
            Ok(9223372036853727234)
        );
    }
}
