//! The prime field the vectors live in, p = 2^64 - 2^32 + 1.
//!
//! A coordinate travels as a little-endian u64. A stored value at or above p
//! stands for that value minus p, so coordinates are reduced with [`reduce`]
//! before they are used, and [`add`] and [`sub`] take reduced operands.

pub const P: u64 = 0xffff_ffff_0000_0001;

/// 2^64 mod p, what a carry out of 64 bits is worth.
const CARRY: u64 = 0xffff_ffff;

pub const COORD_LEN: usize = 8;

/// The canonical form of a stored coordinate.
#[inline(always)]
pub const fn reduce(value: u64) -> u64 {
    if value >= P {
        value - P
    } else {
        value
    }
}

#[inline(always)]
pub const fn add(a: u64, b: u64) -> u64 {
    let (sum, carried) = a.overflowing_add(b);
    if carried {
        // The true sum is below 2p, so this cannot wrap again.
        sum.wrapping_add(CARRY)
    } else {
        reduce(sum)
    }
}

#[inline(always)]
pub const fn sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + (P - b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: [u64; 9] = [
        0,
        1,
        2,
        CARRY,
        1 << 32,
        1 << 63,
        P - 2,
        P - 1,
        0x1234_5678_9abc_def0,
    ];

    #[test]
    fn modulus_is_the_shared_one() {
        assert_eq!(P, 18446744069414584321);
        assert_eq!(P as u128, (1u128 << 64) - (1 << 32) + 1);
        assert_eq!(CARRY as u128, (1u128 << 64) % P as u128);
    }

    #[test]
    fn values_at_or_above_p_stand_for_the_remainder() {
        assert_eq!(reduce(P), 0);
        assert_eq!(reduce(P + 5), 5);
        assert_eq!(reduce(u64::MAX), CARRY - 1);
        assert_eq!(reduce(P - 1), P - 1);
    }

    #[test]
    fn add_and_sub_agree_with_wide_arithmetic() {
        let p = P as u128;
        for a in SAMPLES {
            for b in SAMPLES {
                assert_eq!(add(a, b) as u128, (a as u128 + b as u128) % p, "{a} + {b}");
                assert_eq!(
                    sub(a, b) as u128,
                    (a as u128 + p - b as u128) % p,
                    "{a} - {b}"
                );
                assert_eq!(sub(add(a, b), b), a);
            }
        }
    }
}
