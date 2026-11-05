//! A bounds checked cursor over borrowed bytes, used for instruction
//! arguments and for the proofs a buffer holds.
//!
//! The cursor consumes its input from the front. A read either returns the
//! bytes it consumed or fails with the error of the input's [`Source`], and a
//! failed read leaves the cursor where it was.

use core::marker::PhantomData;

use pinocchio::error::ProgramError;

/// The kind of input a [`Reader`] walks, which decides the error of a short
/// read.
pub trait Source {
    const ERROR: ProgramError;
}

/// Instruction arguments, where a short read is `InvalidInstructionData`.
#[derive(Clone, Copy, Debug)]
pub enum Args {}

impl Source for Args {
    const ERROR: ProgramError = ProgramError::InvalidInstructionData;
}

/// Account data, where a short read is `InvalidAccountData`.
#[derive(Clone, Copy, Debug)]
pub enum Account {}

impl Source for Account {
    const ERROR: ProgramError = ProgramError::InvalidAccountData;
}

/// A forward-only cursor. Every slice it returns borrows the input, so
/// nothing is copied.
#[derive(Debug)]
pub struct Reader<'a, S> {
    rest: &'a [u8],
    source: PhantomData<fn() -> S>,
}

impl<'a, S: Source> Reader<'a, S> {
    #[inline(always)]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            rest: data,
            source: PhantomData,
        }
    }

    /// The next `n` bytes.
    #[inline(always)]
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8], ProgramError> {
        let (head, rest) = self.rest.split_at_checked(n).ok_or(S::ERROR)?;
        self.rest = rest;
        Ok(head)
    }

    /// The next `N` bytes as an array.
    #[inline(always)]
    pub fn array<const N: usize>(&mut self) -> Result<&'a [u8; N], ProgramError> {
        let (head, rest) = self.rest.split_first_chunk::<N>().ok_or(S::ERROR)?;
        self.rest = rest;
        Ok(head)
    }

    #[inline(always)]
    pub fn u8(&mut self) -> Result<u8, ProgramError> {
        Ok(self.array::<1>()?[0])
    }

    #[inline(always)]
    pub fn u16(&mut self) -> Result<u16, ProgramError> {
        Ok(u16::from_le_bytes(*self.array()?))
    }

    #[inline(always)]
    pub fn u32(&mut self) -> Result<u32, ProgramError> {
        Ok(u32::from_le_bytes(*self.array()?))
    }

    #[inline(always)]
    pub fn u64(&mut self) -> Result<u64, ProgramError> {
        Ok(u64::from_le_bytes(*self.array()?))
    }

    /// A u32 count followed by that many items of `item_len` bytes each, the
    /// way borsh lays out a vector of fixed-size items.
    #[inline(always)]
    pub fn items(&mut self, item_len: usize) -> Result<&'a [u8], ProgramError> {
        let count = self.u32()? as usize;
        self.bytes(count.checked_mul(item_len).ok_or(S::ERROR)?)
    }

    /// Checks that the input ends here.
    #[inline(always)]
    pub fn finish(self) -> Result<(), ProgramError> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(S::ERROR)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_consume_from_the_front() {
        let mut r = Reader::<Args>::new(&[1, 2, 0, 3, 0, 0, 0, 9, 9]);
        assert_eq!(r.u8(), Ok(1));
        assert_eq!(r.u16(), Ok(2));
        assert_eq!(r.u32(), Ok(3));
        assert_eq!(r.u64(), Err(ProgramError::InvalidInstructionData));
        assert_eq!(r.bytes(2), Ok(&[9, 9][..]));
        assert_eq!(r.finish(), Ok(()));
    }

    #[test]
    fn items_are_counted() {
        let mut r = Reader::<Account>::new(&[2, 0, 0, 0, 7, 7, 8, 8, 5]);
        assert_eq!(r.items(2), Ok(&[7, 7, 8, 8][..]));
        assert_eq!(r.finish(), Err(ProgramError::InvalidAccountData));

        let mut short = Reader::<Account>::new(&[3, 0, 0, 0, 7, 7]);
        assert_eq!(short.items(2), Err(ProgramError::InvalidAccountData));
    }
}
