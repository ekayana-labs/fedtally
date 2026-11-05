//! The vectors of one round, built in the test: every cohort member's
//! masked input, every helper's mask, and the aggregate, with real Merkle
//! trees over them.

use fedtally_ledger::client::{self, ProofPart};

use super::{reference, Hash, Posting};

#[derive(Clone, Debug)]
pub struct Vectors {
    /// By cohort position, which is ascending site index.
    pub inputs: Vec<Vec<u64>>,
    /// By helper.
    pub masks: Vec<Vec<u64>>,
    pub aggregate: Vec<u64>,
}

/// A small generator, so the vectors are the same on every run.
fn stream(mut state: u64) -> impl FnMut() -> u64 {
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

impl Vectors {
    /// Vectors whose aggregate is the sum of the inputs minus the masks.
    /// Some coordinates are stored at or above p, as the field allows.
    pub fn honest(members: usize, helpers: usize, coords: usize, seed: u64) -> Self {
        let mut next = stream(seed | 1);
        let mut vector = |_| -> Vec<u64> {
            let mut vector: Vec<u64> = (0..coords).map(|_| next()).collect();
            vector[0] = 0;
            vector[1] = reference::P;
            vector[coords / 2] = reference::P + (next() >> 40);
            vector[coords - 2] = 0;
            vector[coords - 1] = u64::MAX;
            vector
        };
        let inputs: Vec<Vec<u64>> = (0..members).map(&mut vector).collect();
        let masks: Vec<Vec<u64>> = (0..helpers).map(&mut vector).collect();
        let mut aggregate = reference::aggregate(&inputs, &masks);
        for value in aggregate.iter_mut() {
            if *value < u64::MAX - reference::P {
                *value += reference::P;
            }
        }
        Self {
            inputs,
            masks,
            aggregate,
        }
    }

    /// The same vectors with one coordinate of the aggregate off by one.
    pub fn with_wrong_aggregate(mut self, coordinate: usize) -> Self {
        self.aggregate[coordinate] = reference::field_add(self.aggregate[coordinate], 1);
        self
    }

    pub fn input_roots(&self) -> Vec<Hash> {
        self.inputs
            .iter()
            .map(|input| reference::vector_root(input))
            .collect()
    }

    /// What the aggregator posts for these vectors.
    pub fn posting(&self, number: u32) -> Posting {
        Posting {
            inputs_root: reference::root(&self.input_roots()),
            mask_roots: self
                .masks
                .iter()
                .map(|mask| reference::vector_root(mask))
                .collect(),
            aggregate_root: reference::vector_root(&self.aggregate),
            ..Posting::filler(number, 0)
        }
    }

    /// Every part a complete proof gives.
    pub fn parts(&self) -> Vec<ProofPart> {
        let inputs = (0..self.inputs.len() as u16).map(ProofPart::Input);
        let masks = (0..self.masks.len() as u8).map(ProofPart::Mask);
        inputs.chain(masks).chain([ProofPart::Aggregate]).collect()
    }

    /// The proof of leaf `leaf_index` of one part, as the buffer holds it.
    pub fn proof(&self, part: ProofPart, leaf_index: usize) -> Vec<u8> {
        let vector = match part {
            ProofPart::Input(position) => &self.inputs[position as usize],
            ProofPart::Mask(helper) => &self.masks[helper as usize],
            ProofPart::Aggregate => &self.aggregate,
        };
        let leaves = reference::vector_leaves(vector);
        let leaf_path = reference::path(&leaves, leaf_index);
        match part {
            ProofPart::Input(position) => {
                let roots = self.input_roots();
                client::input_proof(
                    &leaves[leaf_index],
                    &roots[position as usize],
                    &reference::path(&roots, position as usize),
                    &leaf_path,
                )
            }
            _ => client::leaf_proof(&leaves[leaf_index], &leaf_path),
        }
    }
}
