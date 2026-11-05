//! Merkle trees as RFC 6962 defines them: SHA-256, with a zero byte in front of
//! a leaf and a one byte in front of a node.
//!
//! A vector's root is such a tree over its bytes in 1 KiB leaves. One leaf can
//! then be shown to anyone holding the root, including a program on a ledger.

use sha2::{Digest as _, Sha256};

pub const LEAF_BYTES: usize = 1024;

pub type Hash = [u8; 32];

pub fn leaf(data: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0u8]);
    hasher.update(data);
    hasher.finalize().into()
}

pub fn node(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([1u8]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn fold(level: &mut Vec<Hash>) {
    let mut next = 0;
    for i in (0..level.len()).step_by(2) {
        level[next] = match level.get(i + 1) {
            Some(right) => node(&level[i], right),
            // An unpaired last node moves up unchanged. This is what makes the
            // result the RFC 6962 tree for any number of leaves.
            None => level[i],
        };
        next += 1;
    }
    level.truncate(next);
}

/// The root over hashed leaves, or over subtrees of equal power-of-two size.
pub fn root(mut level: Vec<Hash>) -> Hash {
    if level.is_empty() {
        return Sha256::digest([]).into();
    }
    while level.len() > 1 {
        fold(&mut level);
    }
    level[0]
}

/// The sibling hashes from `index` up to the root.
pub fn path(mut level: Vec<Hash>, mut index: usize) -> Vec<Hash> {
    let mut out = Vec::new();
    while level.len() > 1 {
        if let Some(sibling) = level.get(index ^ 1) {
            out.push(*sibling);
        }
        fold(&mut level);
        index /= 2;
    }
    out
}

/// The root a path leads to, for leaf `index` in a tree of `size` leaves.
/// This is the check in RFC 9162, section 2.1.3.2. It ties the leaf to its
/// index. The size has to come from somewhere trusted, as the plan does here.
pub fn climb(leaf: Hash, index: u64, size: u64, path: &[Hash]) -> Option<Hash> {
    if index >= size {
        return None;
    }
    let (mut at, mut last, mut hash) = (index, size - 1, leaf);
    for sibling in path {
        if last == 0 {
            return None;
        }
        if at & 1 == 1 || at == last {
            hash = node(sibling, &hash);
            while at & 1 == 0 && at != 0 {
                at >>= 1;
                last >>= 1;
            }
        } else {
            hash = node(&hash, sibling);
        }
        at >>= 1;
        last >>= 1;
    }
    (last == 0).then_some(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(n: usize) -> Vec<Hash> {
        (0..n).map(|i| leaf(&(i as u32).to_le_bytes())).collect()
    }

    /// The definition in RFC 6962, section 2.1, written out directly.
    fn by_definition(level: &[Hash]) -> Hash {
        match level.len() {
            1 => level[0],
            n => {
                let split = n.next_power_of_two() / 2;
                node(
                    &by_definition(&level[..split]),
                    &by_definition(&level[split..]),
                )
            }
        }
    }

    #[test]
    fn matches_the_published_test_vectors() {
        // From the Certificate Transparency reference tests.
        let hex = |h: Hash| hex::encode(h);
        assert_eq!(
            hex(root(Vec::new())),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(leaf(&[])),
            "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
        );
        let inputs: [&[u8]; 3] = [&[], &[0x00], &[0x10]];
        let hashed: Vec<Hash> = inputs.iter().map(|d| leaf(d)).collect();
        assert_eq!(
            hex(root(hashed[..2].to_vec())),
            "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125"
        );
        assert_eq!(
            hex(root(hashed)),
            "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77"
        );
    }

    #[test]
    fn folding_gives_the_tree_the_rfc_defines() {
        for n in 1..=70 {
            let level = leaves(n);
            assert_eq!(root(level.clone()), by_definition(&level), "{n} leaves");
        }
    }

    #[test]
    fn every_path_climbs_to_the_root_and_only_from_its_own_place() {
        for n in 1..=33usize {
            let level = leaves(n);
            let top = root(level.clone());
            for i in 0..n {
                let proof = path(level.clone(), i);
                assert_eq!(climb(level[i], i as u64, n as u64, &proof), Some(top));
                for other in (0..n).filter(|o| *o != i) {
                    assert_ne!(climb(level[other], i as u64, n as u64, &proof), Some(top));
                    assert_ne!(
                        climb(level[i], other as u64, n as u64, &proof),
                        Some(top),
                        "{n} leaves, {i} shown as {other}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_path_of_the_wrong_length_is_refused() {
        let level = leaves(6);
        let top = root(level.clone());
        let proof = path(level.clone(), 2);
        assert_eq!(climb(level[2], 2, 6, &proof), Some(top));
        assert_eq!(climb(level[2], 2, 6, &proof[..proof.len() - 1]), None);
        let mut long = proof.clone();
        long.push([0; 32]);
        assert_eq!(climb(level[2], 2, 6, &long), None);
        assert_eq!(climb(level[2], 6, 6, &proof), None);
    }

    #[test]
    fn subtrees_of_equal_power_of_two_size_fold_to_the_same_root() {
        let level = leaves(21);
        let blocks: Vec<Hash> = level.chunks(4).map(|c| root(c.to_vec())).collect();
        assert_eq!(root(blocks), root(level));
    }

    #[test]
    fn leaves_and_nodes_cannot_stand_in_for_each_other() {
        let (a, b) = (leaf(b"a"), leaf(b"b"));
        let mut joined = a.to_vec();
        joined.extend_from_slice(&b);
        assert_ne!(node(&a, &b), leaf(&joined));
        assert_ne!(node(&a, &b), node(&b, &a));
    }
}
