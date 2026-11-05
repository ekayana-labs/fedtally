//! Bitmaps over byte slices. Bit `i` is bit `i % 8` of byte `i / 8`, counting
//! from the least significant bit.

/// The bytes of a bitmap with one bit per site of the largest task.
pub const SITE_BITMAP_LEN: usize = crate::state::MAX_SITES / 8;

pub type SiteBitmap = [u8; SITE_BITMAP_LEN];

#[inline(always)]
pub fn get(bits: &[u8], index: usize) -> bool {
    bits[index / 8] >> (index % 8) & 1 == 1
}

#[inline(always)]
pub fn set(bits: &mut [u8], index: usize) {
    bits[index / 8] |= 1 << (index % 8);
}

/// How many bits are set.
pub fn count(bits: &[u8]) -> u32 {
    words(bits).map(u64::count_ones).sum()
}

/// Whether every bit set in `bits` is also set in `of`.
pub fn is_subset(bits: &SiteBitmap, of: &SiteBitmap) -> bool {
    words(bits).zip(words(of)).all(|(a, b)| a & !b == 0)
}

/// Whether `a` and `b` have a set bit in common.
pub fn intersects(a: &SiteBitmap, b: &SiteBitmap) -> bool {
    words(a).zip(words(b)).any(|(a, b)| a & b != 0)
}

/// Whether any bit at index `from` or above is set.
pub fn any_from(bits: &SiteBitmap, from: usize) -> bool {
    let (byte, bit) = (from / 8, from % 8);
    match bits.get(byte) {
        Some(first) => first >> bit != 0 || bits[byte + 1..].iter().any(|b| *b != 0),
        None => false,
    }
}

fn words(bits: &[u8]) -> impl Iterator<Item = u64> + '_ {
    bits.chunks(8).map(|chunk| {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        u64::from_le_bytes(word)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bitmap(indexes: &[usize]) -> SiteBitmap {
        let mut bits = [0u8; SITE_BITMAP_LEN];
        for index in indexes {
            set(&mut bits, *index);
        }
        bits
    }

    #[test]
    fn bits_are_numbered_from_the_low_end_of_each_byte() {
        let bits = bitmap(&[0, 9, 1023]);
        assert_eq!(bits[0], 0b0000_0001);
        assert_eq!(bits[1], 0b0000_0010);
        assert_eq!(bits[127], 0b1000_0000);
        assert!(get(&bits, 9) && !get(&bits, 8));
        assert_eq!(count(&bits), 3);
        assert_eq!(count(&[0xff, 0x01, 0x80]), 10);
    }

    #[test]
    fn set_relations() {
        let all = bitmap(&[1, 2, 3, 700]);
        assert!(is_subset(&bitmap(&[2, 700]), &all));
        assert!(!is_subset(&bitmap(&[2, 701]), &all));
        assert!(intersects(&bitmap(&[5, 700]), &all));
        assert!(!intersects(&bitmap(&[5, 701]), &all));
    }

    #[test]
    fn bits_at_or_above_a_bound() {
        let bits = bitmap(&[3, 20]);
        assert!(any_from(&bits, 0));
        assert!(any_from(&bits, 20));
        assert!(!any_from(&bits, 21));
        assert!(!any_from(&bits, 1024));
        assert!(any_from(&bitmap(&[1023]), 1023));
        assert!(!any_from(&bitmap(&[15]), 16));
    }
}
