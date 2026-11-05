//! A reference of the shared definitions, written from the specification
//! with nothing taken from the program crate: RFC 6962 trees by their
//! recursive definition, the vector and inputs roots, the privacy cost, the
//! beacon and the tickets, and the field.

use sha2::{Digest, Sha256};

pub type Hash = [u8; 32];

pub const P: u64 = 18446744069414584321;
pub const LEAF_COORDS: usize = 128;

pub fn sha(parts: &[&[u8]]) -> Hash {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

pub fn leaf_hash(data: &[u8]) -> Hash {
    sha(&[&[0x00], data])
}

pub fn node_hash(left: &Hash, right: &Hash) -> Hash {
    sha(&[&[0x01], left, right])
}

/// The largest power of two below `n`, where the tree over `n` leaves
/// splits.
fn split(n: usize) -> usize {
    let mut k = 1;
    while k * 2 < n {
        k *= 2;
    }
    k
}

/// MTH of RFC 6962 section 2.1 over leaf hashes.
pub fn root_of_hashes(hashes: &[Hash]) -> Hash {
    match hashes.len() {
        0 => sha(&[]),
        1 => hashes[0],
        n => {
            let (left, right) = hashes.split_at(split(n));
            node_hash(&root_of_hashes(left), &root_of_hashes(right))
        }
    }
}

/// PATH of RFC 6962 section 2.1.1 over leaf hashes.
pub fn path_of_hashes(hashes: &[Hash], index: usize) -> Vec<Hash> {
    let n = hashes.len();
    if n <= 1 {
        return Vec::new();
    }
    let (left, right) = hashes.split_at(split(n));
    let (mut path, sibling) = if index < left.len() {
        (path_of_hashes(left, index), root_of_hashes(right))
    } else {
        (
            path_of_hashes(right, index - left.len()),
            root_of_hashes(left),
        )
    };
    path.push(sibling);
    path
}

/// The root that `path` leads to from a leaf hash, by the same recursion.
pub fn root_from_path(leaf: Hash, index: usize, size: usize, path: &[Hash]) -> Hash {
    let Some((top, below)) = path.split_last() else {
        assert_eq!(size, 1, "the path ends at the leaf");
        return leaf;
    };
    let k = split(size);
    if index < k {
        node_hash(&root_from_path(leaf, index, k, below), top)
    } else {
        node_hash(top, &root_from_path(leaf, index - k, size - k, below))
    }
}

fn hashes<T: AsRef<[u8]>>(leaves: &[T]) -> Vec<Hash> {
    leaves.iter().map(|leaf| leaf_hash(leaf.as_ref())).collect()
}

/// The root of the tree whose leaves hold `leaves`.
pub fn root<T: AsRef<[u8]>>(leaves: &[T]) -> Hash {
    root_of_hashes(&hashes(leaves))
}

/// The inclusion path of leaf `index` in the tree whose leaves hold `leaves`.
pub fn path<T: AsRef<[u8]>>(leaves: &[T], index: usize) -> Vec<Hash> {
    path_of_hashes(&hashes(leaves), index)
}

/// A vector cut into leaves of 128 little-endian coordinates.
pub fn vector_leaves(coords: &[u64]) -> Vec<Vec<u8>> {
    coords
        .chunks(LEAF_COORDS)
        .map(|chunk| chunk.iter().flat_map(|c| c.to_le_bytes()).collect())
        .collect()
}

pub fn vector_root(coords: &[u64]) -> Hash {
    root(&vector_leaves(coords))
}

/// `floor((s * s << 40) / (2 * noise * (k - colluding))) + 2`, or `None`
/// where it is undefined or does not fit.
pub fn round_cost(sensitivity: u64, noise: u64, colluding: u64, k: u64) -> Option<u64> {
    if noise == 0 {
        return Some(0);
    }
    if k <= colluding {
        return None;
    }
    let s = sensitivity as u128;
    let numerator = (s * s).checked_mul(1 << 40)?;
    let cost = numerator / (2 * noise as u128 * (k - colluding) as u128) + 2;
    u64::try_from(cost).ok()
}

pub fn beacon(plan_id: &[u8; 32], round: u32, model: &Hash) -> Hash {
    sha(&[b"fedtally v1 beacon", plan_id, &round.to_le_bytes(), model])
}

pub fn ticket(beacon: &Hash, site: u32) -> Hash {
    sha(&[b"fedtally v1 ticket", beacon, &site.to_le_bytes()])
}

/// The sites in ascending order of their tickets, as big-endian numbers.
pub fn by_ticket(beacon: &Hash, sites: u16) -> Vec<(Hash, u16)> {
    let mut tickets: Vec<(Hash, u16)> = (0..sites)
        .map(|site| (ticket(beacon, site as u32), site))
        .collect();
    tickets.sort();
    tickets
}

/// The largest ticket among the `sample` smallest, and the sites holding
/// them in ascending site index.
pub fn invitation(beacon: &Hash, sites: u16, sample: u16) -> (Hash, Vec<u16>) {
    let tickets = by_ticket(beacon, sites);
    let invited = &tickets[..sample as usize];
    let mut indexes: Vec<u16> = invited.iter().map(|(_, site)| *site).collect();
    indexes.sort();
    (invited.last().unwrap().0, indexes)
}

/// A 32 byte big-endian number moved by one.
pub fn step(value: &Hash, up: bool) -> Hash {
    let mut out = *value;
    for byte in out.iter_mut().rev() {
        let (next, wrapped) = if up {
            byte.overflowing_add(1)
        } else {
            byte.overflowing_sub(1)
        };
        *byte = next;
        if !wrapped {
            break;
        }
    }
    out
}

pub fn field_add(a: u64, b: u64) -> u64 {
    ((a as u128 % P as u128 + b as u128 % P as u128) % P as u128) as u64
}

pub fn field_sub(a: u64, b: u64) -> u64 {
    ((a as u128 % P as u128 + P as u128 - b as u128 % P as u128) % P as u128) as u64
}

/// The honest aggregate: the sum of the inputs minus the sum of the masks,
/// coordinate by coordinate.
pub fn aggregate(inputs: &[Vec<u64>], masks: &[Vec<u64>]) -> Vec<u64> {
    let coords = inputs[0].len();
    (0..coords)
        .map(|i| {
            let sum = inputs.iter().fold(0, |acc, v| field_add(acc, v[i]));
            masks.iter().fold(sum, |acc, v| field_sub(acc, v[i]))
        })
        .collect()
}
