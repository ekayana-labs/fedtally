//! SHA-256 over the concatenation of byte slices. On chain it is the
//! `sol_sha256` syscall, elsewhere the `sha2` crate.

pub type Digest = [u8; 32];

#[cfg(target_os = "solana")]
#[inline]
pub fn sha256(parts: &[&[u8]]) -> Digest {
    let mut digest = core::mem::MaybeUninit::<Digest>::uninit();
    // The syscall takes a pointer to an array of `&[u8]` fat pointers and
    // writes all 32 bytes of the result.
    unsafe {
        pinocchio::syscalls::sol_sha256(
            parts.as_ptr() as *const u8,
            parts.len() as u64,
            digest.as_mut_ptr() as *mut u8,
        );
        digest.assume_init()
    }
}

#[cfg(not(target_os = "solana"))]
pub fn sha256(parts: &[&[u8]]) -> Digest {
    use sha2::{Digest as _, Sha256};

    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&hasher.finalize());
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_hash_as_their_concatenation() {
        assert_eq!(sha256(&[b"ab", b"", b"c"]), sha256(&[b"abc"]));
        assert_eq!(
            sha256(&[b"abc"])[..4],
            [0xba, 0x78, 0x16, 0xbf],
            "the FIPS 180 vector"
        );
    }
}
