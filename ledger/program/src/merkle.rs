//! RFC 6962 Merkle trees over SHA-256, as far as the program needs them: the
//! leaf and node hashes, and the root an inclusion path leads to, computed
//! with the algorithm of RFC 9162 section 2.1.3.2.

use pinocchio::error::ProgramError;

use crate::{
    hash::{sha256, Digest},
    reader::{Reader, Source},
};

pub const DIGEST_LEN: usize = 32;

/// The hash of a leaf holding `data`.
#[inline]
pub fn leaf(data: &[u8]) -> Digest {
    sha256(&[&[0x00], data])
}

/// The hash of the node above `left` and `right`.
#[inline]
pub fn node(left: &Digest, right: &Digest) -> Digest {
    sha256(&[&[0x01], left, right])
}

/// Which side of the running hash a path entry goes on.
enum Sibling {
    Left,
    Right,
}

/// The walk from a leaf up to the root. It yields one step per path entry
/// and ends at the root, so its length is the length of a valid path.
struct Walk {
    position: u64,
    last: u64,
}

impl Walk {
    /// The walk from the leaf at `index` in a tree of `size` leaves, where
    /// `index < size`.
    fn new(index: u64, size: u64) -> Self {
        Self {
            position: index,
            last: size - 1,
        }
    }
}

impl Iterator for Walk {
    type Item = Sibling;

    fn next(&mut self) -> Option<Sibling> {
        if self.last == 0 {
            return None;
        }
        let sibling = if self.position & 1 == 1 || self.position == self.last {
            // A node that is the last of its level has no right sibling
            // until the level where it becomes a right child.
            while self.position & 1 == 0 && self.position != 0 {
                self.position >>= 1;
                self.last >>= 1;
            }
            Sibling::Left
        } else {
            Sibling::Right
        };
        self.position >>= 1;
        self.last >>= 1;
        Some(sibling)
    }
}

/// How many hashes the inclusion path of the leaf at `index` holds in a tree
/// of `size` leaves, or `None` when there is no such leaf.
pub fn path_len(index: u64, size: u64) -> Option<usize> {
    (index < size).then(|| Walk::new(index, size).count())
}

/// The root that the leaf hash `leaf` at `index` reaches in a tree of `size`
/// leaves, taking exactly the path entries it needs from `path`. A path that
/// ends early fails with the reader's error. The caller checks that nothing
/// is left over and compares the root.
pub fn root_from_path<S: Source>(
    leaf: Digest,
    index: u64,
    size: u64,
    path: &mut Reader<S>,
) -> Result<Digest, ProgramError> {
    if index >= size {
        return Err(S::ERROR);
    }
    let mut root = leaf;
    for sibling in Walk::new(index, size) {
        let entry = path.array::<DIGEST_LEN>()?;
        root = match sibling {
            Sibling::Left => node(entry, &root),
            Sibling::Right => node(&root, entry),
        };
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::Args;

    fn hex(s: &str) -> Digest {
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
        }
        out
    }

    /// The root of the tree over `leaves`, by the recursive definition of
    /// RFC 6962 section 2.1.
    fn tree_root(leaves: &[Digest]) -> Digest {
        match leaves {
            [] => sha256(&[]),
            [only] => *only,
            _ => {
                let split = leaves.len().next_power_of_two() / 2;
                node(&tree_root(&leaves[..split]), &tree_root(&leaves[split..]))
            }
        }
    }

    /// The inclusion path of leaf `index`, by RFC 6962 section 2.1.1.
    fn tree_path(leaves: &[Digest], index: usize) -> Vec<Digest> {
        if leaves.len() <= 1 {
            return Vec::new();
        }
        let split = leaves.len().next_power_of_two() / 2;
        let (left, right) = leaves.split_at(split);
        if index < split {
            let mut path = tree_path(left, index);
            path.push(tree_root(right));
            path
        } else {
            let mut path = tree_path(right, index - split);
            path.push(tree_root(left));
            path
        }
    }

    fn root_of(leaf: Digest, index: u64, size: u64, path: &[Digest]) -> Option<Digest> {
        let bytes = path.concat();
        let mut reader = Reader::<Args>::new(&bytes);
        let root = root_from_path(leaf, index, size, &mut reader).ok()?;
        reader.finish().ok()?;
        Some(root)
    }

    #[test]
    fn published_leaf_vector() {
        assert_eq!(
            leaf(b""),
            hex("6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d")
        );
    }

    #[test]
    fn published_root_vector_from_every_leaf() {
        let root = hex("aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77");
        let leaves = [leaf(&[]), leaf(&[0x00]), leaf(&[0x10])];
        assert_eq!(tree_root(&leaves), root);
        for (index, hash) in leaves.iter().enumerate() {
            let path = tree_path(&leaves, index);
            assert_eq!(root_of(*hash, index as u64, 3, &path), Some(root));
            assert_eq!(path_len(index as u64, 3), Some(path.len()));
        }
    }

    #[test]
    fn paths_verify_in_trees_of_every_size() {
        for size in 1..=70usize {
            let leaves: Vec<Digest> = (0..size).map(|i| leaf(&(i as u32).to_le_bytes())).collect();
            let root = tree_root(&leaves);
            for index in 0..size {
                let path = tree_path(&leaves, index);
                let (i, n) = (index as u64, size as u64);
                assert_eq!(path_len(i, n), Some(path.len()), "{index} of {size}");
                assert_eq!(root_of(leaves[index], i, n, &path), Some(root));
                assert_ne!(root_of(leaf(b"other"), i, n, &path), Some(root));
                if let Some((_, shorter)) = path.split_last() {
                    assert_eq!(root_of(leaves[index], i, n, shorter), None);
                }
                let mut longer = path.clone();
                longer.push([0; 32]);
                assert_eq!(root_of(leaves[index], i, n, &longer), None);
            }
            assert_eq!(path_len(size as u64, size as u64), None);
        }
    }

    #[test]
    fn a_path_proves_one_index_only() {
        let leaves: Vec<Digest> = (0..11u32).map(|i| leaf(&i.to_le_bytes())).collect();
        let root = tree_root(&leaves);
        let path = tree_path(&leaves, 4);
        for index in (0..11).filter(|i| *i != 4) {
            assert_ne!(root_of(leaves[4], index, 11, &path), Some(root));
        }
        assert_eq!(root_of(leaves[4], 11, 11, &path), None);
    }
}
