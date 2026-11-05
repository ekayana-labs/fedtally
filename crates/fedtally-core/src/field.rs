//! The Goldilocks field and its quadratic extension.

use std::ops::{Add, Mul, Neg, Sub};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// 2^64 - 2^32 + 1.
pub const P: u64 = 0xFFFF_FFFF_0000_0001;
const EPSILON: u64 = 0xFFFF_FFFF;

/// An element of F_p, always in canonical form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp(u64);

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(1);

    /// Reduces any u64.
    #[inline]
    pub const fn new(v: u64) -> Fp {
        Fp(if v >= P { v - P } else { v })
    }

    #[inline]
    pub const fn value(self) -> u64 {
        self.0
    }

    #[inline]
    pub fn from_i64(v: i64) -> Fp {
        if v >= 0 {
            Fp::new(v as u64)
        } else {
            -Fp::new(v.unsigned_abs())
        }
    }

    /// The representative closest to zero.
    #[inline]
    pub const fn centered(self) -> i64 {
        if self.0 > P / 2 {
            -((P - self.0) as i64)
        } else {
            self.0 as i64
        }
    }

    pub fn pow(self, mut e: u64) -> Fp {
        let (mut base, mut acc) = (self, Fp::ONE);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            e >>= 1;
        }
        acc
    }

    #[inline]
    fn reduce(x: u128) -> Fp {
        let (lo, hi) = (x as u64, (x >> 64) as u64);
        let (hh, hl) = (hi >> 32, hi & EPSILON);
        let (mut t, borrow) = lo.overflowing_sub(hh);
        if borrow {
            t = t.wrapping_sub(EPSILON);
        }
        let (r, carry) = t.overflowing_add(hl * EPSILON);
        Fp::new(if carry { r.wrapping_add(EPSILON) } else { r })
    }
}

impl Add for Fp {
    type Output = Fp;
    #[inline]
    fn add(self, rhs: Fp) -> Fp {
        let (s, carry) = self.0.overflowing_add(rhs.0);
        Fp::new(if carry { s.wrapping_add(EPSILON) } else { s })
    }
}

impl Sub for Fp {
    type Output = Fp;
    #[inline]
    fn sub(self, rhs: Fp) -> Fp {
        let (d, borrow) = self.0.overflowing_sub(rhs.0);
        Fp(if borrow { d.wrapping_sub(EPSILON) } else { d })
    }
}

impl Neg for Fp {
    type Output = Fp;
    #[inline]
    fn neg(self) -> Fp {
        Fp::ZERO - self
    }
}

impl Mul for Fp {
    type Output = Fp;
    #[inline]
    fn mul(self, rhs: Fp) -> Fp {
        Fp::reduce(self.0 as u128 * rhs.0 as u128)
    }
}

/// a + b*u in F_p[u] / (u^2 - 7). 7 generates F_p*, so it is a non-residue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp2 {
    pub a: Fp,
    pub b: Fp,
}

impl Fp2 {
    pub const ZERO: Fp2 = Fp2 {
        a: Fp::ZERO,
        b: Fp::ZERO,
    };
    pub const ONE: Fp2 = Fp2 {
        a: Fp::ONE,
        b: Fp::ZERO,
    };

    pub fn pow(self, mut e: u64) -> Fp2 {
        let (mut base, mut acc) = (self, Fp2::ONE);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            e >>= 1;
        }
        acc
    }

    #[inline]
    pub fn add_base(self, v: Fp) -> Fp2 {
        Fp2 {
            a: self.a + v,
            b: self.b,
        }
    }
}

impl Add for Fp2 {
    type Output = Fp2;
    #[inline]
    fn add(self, rhs: Fp2) -> Fp2 {
        Fp2 {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl Sub for Fp2 {
    type Output = Fp2;
    #[inline]
    fn sub(self, rhs: Fp2) -> Fp2 {
        Fp2 {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl Mul for Fp2 {
    type Output = Fp2;
    #[inline]
    fn mul(self, rhs: Fp2) -> Fp2 {
        let seven = Fp::new(7);
        Fp2 {
            a: self.a * rhs.a + seven * (self.b * rhs.b),
            b: self.a * rhs.b + self.b * rhs.a,
        }
    }
}

impl std::iter::Sum for Fp2 {
    fn sum<I: Iterator<Item = Fp2>>(iter: I) -> Fp2 {
        iter.fold(Fp2::ZERO, |acc, x| acc + x)
    }
}

impl Serialize for Fp2 {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("{:016x}{:016x}", self.a.value(), self.b.value()))
    }
}

impl<'de> Deserialize<'de> for Fp2 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Fp2, D::Error> {
        use serde::de::Error;
        let text = String::deserialize(d)?;
        let limb = |range: std::ops::Range<usize>| {
            let v = text
                .get(range)
                .and_then(|h| u64::from_str_radix(h, 16).ok())
                .ok_or_else(|| D::Error::custom("bad field element"))?;
            if v < P {
                Ok(Fp::new(v))
            } else {
                Err(D::Error::custom("field element out of range"))
            }
        };
        if text.len() != 32 {
            return Err(D::Error::custom("bad field element length"));
        }
        Ok(Fp2 {
            a: limb(0..16)?,
            b: limb(16..32)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn slow_mul(a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % P as u128) as u64
    }

    proptest! {
        #[test]
        fn mul_matches_big_int(a in any::<u64>(), b in any::<u64>()) {
            let (x, y) = (Fp::new(a), Fp::new(b));
            prop_assert_eq!((x * y).value(), slow_mul(x.value(), y.value()));
        }

        #[test]
        fn add_sub_match_big_int(a in any::<u64>(), b in any::<u64>()) {
            let (x, y) = (Fp::new(a), Fp::new(b));
            let sum = ((x.value() as u128 + y.value() as u128) % P as u128) as u64;
            prop_assert_eq!((x + y).value(), sum);
            prop_assert_eq!(x + y - y, x);
        }

        #[test]
        fn centered_round_trips(v in -(1i64 << 62)..(1i64 << 62)) {
            prop_assert_eq!(Fp::from_i64(v).centered(), v);
        }

        #[test]
        fn extension_distributes(
            a in any::<[u64; 2]>(), b in any::<[u64; 2]>(), c in any::<[u64; 2]>()
        ) {
            let e = |v: [u64; 2]| Fp2 { a: Fp::new(v[0]), b: Fp::new(v[1]) };
            let (x, y, z) = (e(a), e(b), e(c));
            prop_assert_eq!(x * (y + z), x * y + x * z);
            prop_assert_eq!(x * y, y * x);
            prop_assert_eq!(x + y - y, x);
            prop_assert_eq!(x - x, Fp2::ZERO);
        }
    }

    #[test]
    fn edge_values_reduce() {
        assert_eq!(Fp::new(P), Fp::ZERO);
        assert_eq!(Fp::new(u64::MAX).value(), u64::MAX - P);
        let max = Fp::new(P - 1);
        assert_eq!((max * max).value(), 1);
        assert_eq!(max + Fp::ONE, Fp::ZERO);
    }

    #[test]
    fn centering_splits_the_field_in_half() {
        assert_eq!(Fp::new(P / 2).centered(), (P / 2) as i64);
        assert_eq!(Fp::new(P / 2 + 1).centered(), -((P / 2) as i64));
        assert_eq!(Fp::new(P - 1).centered(), -1);
    }

    #[test]
    fn powers_match_repeated_multiplication() {
        let x = Fp2 {
            a: Fp::new(3),
            b: Fp::new(5),
        };
        assert_eq!(x.pow(0), Fp2::ONE);
        assert_eq!(x.pow(3), x * x * x);
        assert_eq!(Fp::new(9).pow(0), Fp::ONE);
        assert_eq!(Fp::new(2).pow(10).value(), 1024);
    }

    #[test]
    fn seven_is_a_non_residue() {
        assert_eq!(Fp::new(7).pow((P - 1) / 2), -Fp::ONE);
    }

    #[test]
    fn extension_has_no_zero_divisors_at_u() {
        let u = Fp2 {
            a: Fp::ZERO,
            b: Fp::ONE,
        };
        assert_eq!(u * u, Fp2::ONE.add_base(Fp::new(6)));
    }

    #[test]
    fn serde_round_trips_and_rejects_out_of_range() {
        let v = Fp2 {
            a: Fp::new(P - 1),
            b: Fp::new(42),
        };
        let text = serde_json::to_string(&v).unwrap();
        assert_eq!(serde_json::from_str::<Fp2>(&text).unwrap(), v);
        let bad = format!("\"{:016x}{:016x}\"", P, 0);
        assert!(serde_json::from_str::<Fp2>(&bad).is_err());
    }
}
