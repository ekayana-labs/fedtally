//! The watcher's copy of a round's vectors: fetched from the aggregator,
//! checked against a root before they count, added up, and cut into the
//! proofs a dispute needs.

use std::collections::{HashMap, HashSet};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use fedtally_core::anchor;
use fedtally_core::vector::{self, Digest, LEAF_COORDS, Manifest, Shape};
use fedtally_ledger::client::{self, ProofPart};
use fedtally_node::NodeError;
use fedtally_node::observer::{self, ObserverConfig};
use fedtally_node::wire::Object;

use crate::error::ChainError;
use crate::ledger::RoundView;
use crate::owed::Verdict;
use crate::run::RoundLog;

const FETCH_LIMIT: Duration = Duration::from_secs(600);

pub(crate) struct Vault {
    pub source: ObserverConfig,
    /// Where to fetch from. Without one, only what is on disk counts.
    pub runtime: Option<tokio::runtime::Handle>,
    pub shape: Shape,
    manifests: HashMap<PathBuf, Manifest>,
    /// Leaves that came from an answered demand, not from a download.
    patched: HashSet<(Object, u64)>,
}

/// A round's vectors on disk: inputs in cohort order, masks in helper order.
struct Held {
    inputs: Vec<PathBuf>,
    masks: Vec<PathBuf>,
    aggregate: PathBuf,
}

fn bytes(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The cohort's vector roots as the log has them, in cohort order.
fn vector_roots(cohort: &[u16], log: &RoundLog) -> Option<Vec<Digest>> {
    cohort.iter().map(|s| log.vectors.get(s).copied()).collect()
}

impl Vault {
    pub fn new(
        source: ObserverConfig,
        runtime: Option<tokio::runtime::Handle>,
        shape: Shape,
    ) -> Vault {
        Vault {
            source,
            runtime,
            shape,
            manifests: HashMap::new(),
            patched: HashSet::new(),
        }
    }

    fn path(&self, object: Object) -> PathBuf {
        let (round, name) = match object {
            Object::Upload { round, site } => (round, format!("site-{site}.bin")),
            Object::Mask { round, helper } => (round, format!("mask-{helper}.bin")),
            Object::Aggregate { round } => (round, "aggregate.bin".to_owned()),
            Object::GroupMask {
                round,
                helper,
                group,
            } => (round, format!("mask-{helper}-g{group}.bin")),
            Object::GroupAggregate { round, group } => (round, format!("aggregate-g{group}.bin")),
            Object::Model { round } => (round, "model".to_owned()),
            Object::InitialModel => (0, "model".to_owned()),
        };
        self.source.dir.join(format!("round-{round}")).join(name)
    }

    fn manifest(&mut self, path: &Path) -> std::io::Result<&Manifest> {
        if !self.manifests.contains_key(path) {
            let manifest = vector::manifest_of(path, self.shape)?;
            self.manifests.insert(path.to_owned(), manifest);
        }
        Ok(&self.manifests[path])
    }

    /// The file that holds the vector with `root`, downloaded if it is not
    /// here already.
    fn hold(&mut self, object: Object, root: Digest) -> Result<PathBuf, NodeError> {
        let path = self.path(object);
        if path.exists() && self.manifest(&path).is_ok_and(|m| m.root == root) {
            return Ok(path);
        }
        let Some(runtime) = self.runtime.clone() else {
            return Err(NodeError::Disconnected(format!(
                "{object:?} is not on disk"
            )));
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let partial = path.with_extension("partial");
        let _ = std::fs::remove_file(&partial);
        let fetch = observer::fetch_vector(&self.source, object, root, &partial);
        runtime
            .block_on(async { tokio::time::timeout(FETCH_LIMIT, fetch).await })
            .map_err(|_| NodeError::Disconnected(format!("{object:?} took too long")))??;
        std::fs::rename(&partial, &path)?;
        self.manifests.remove(&path);
        Ok(path)
    }

    /// Fetches a round's vectors under the roots in the log, while the
    /// aggregator still has them. Returns whether all of them are held.
    pub fn prefetch(&mut self, number: u32, log: &RoundLog) -> Result<bool, NodeError> {
        let (Some(committed), Some(aggregate)) = (&log.committed, log.aggregate) else {
            return Ok(false);
        };
        if log.masks.is_empty() {
            return Ok(false);
        }
        let cohort: Vec<u16> = committed.cohort.iter().map(|s| *s as u16).collect();
        let Some(roots) = vector_roots(&cohort, log) else {
            return Ok(false);
        };
        let round = number;
        for (site, root) in cohort.iter().zip(roots) {
            let site = u32::from(*site);
            self.hold(Object::Upload { round, site }, root)?;
        }
        for (helper, root) in log.masks.iter().enumerate() {
            let helper = helper as u32;
            self.hold(Object::Mask { round, helper }, *root)?;
        }
        self.hold(Object::Aggregate { round }, aggregate)?;
        Ok(true)
    }

    /// The files behind the roots the ledger holds for a round. `None` when
    /// the log's vector roots are not the ones under the ledger's inputs root.
    fn files(
        &mut self,
        number: u32,
        held: &RoundView,
        roots: &[Digest],
    ) -> Result<Option<Held>, NodeError> {
        if anchor::inputs_root(roots) != held.inputs_root {
            return Ok(None);
        }
        let round = number;
        let mut inputs = Vec::new();
        for (site, root) in held.cohort.members().into_iter().zip(roots) {
            let site = u32::from(site);
            inputs.push(self.hold(Object::Upload { round, site }, *root)?);
        }
        let mut masks = Vec::new();
        for (helper, root) in held.mask_roots.iter().enumerate() {
            let helper = helper as u32;
            masks.push(self.hold(Object::Mask { round, helper }, Digest(*root))?);
        }
        let aggregate = self.hold(Object::Aggregate { round }, Digest(held.aggregate_root))?;
        Ok(Some(Held {
            inputs,
            masks,
            aggregate,
        }))
    }

    /// Adds up the vectors the ledger's round commits to. `None` while the
    /// log does not name every cohort member's vector yet.
    pub fn examine(
        &mut self,
        number: u32,
        round: &RoundView,
        log: &RoundLog,
    ) -> Result<Option<Verdict>, NodeError> {
        let Some(roots) = vector_roots(&round.cohort.members(), log) else {
            return Ok(None);
        };
        let Some(held) = self.files(number, round, &roots)? else {
            return Ok(Some(Verdict::Unprovable));
        };
        let inputs: Vec<&Path> = held.inputs.iter().map(PathBuf::as_path).collect();
        let masks: Vec<&Path> = held.masks.iter().map(PathBuf::as_path).collect();
        let wrong = vector::first_wrong_leaf(self.shape, &inputs, &masks, &held.aggregate)?;
        Ok(Some(wrong.map_or(Verdict::Sound, Verdict::Wrong)))
    }

    /// A round's vectors, by the part of a proof each one is.
    fn vectors(
        &self,
        number: u32,
        round: &RoundView,
        roots: &[Digest],
    ) -> Vec<(ProofPart, Object, Digest)> {
        let cohort = round.cohort.members().into_iter().zip(roots);
        let inputs = cohort.enumerate().map(|(position, (site, root))| {
            let object = Object::Upload {
                round: number,
                site: u32::from(site),
            };
            (ProofPart::Input(position as u16), object, *root)
        });
        let masks = round.mask_roots.iter().enumerate().map(|(helper, root)| {
            let object = Object::Mask {
                round: number,
                helper: helper as u32,
            };
            (ProofPart::Mask(helper as u8), object, Digest(*root))
        });
        let aggregate = (
            ProofPart::Aggregate,
            Object::Aggregate { round: number },
            Digest(round.aggregate_root),
        );
        inputs.chain(masks).chain([aggregate]).collect()
    }

    /// Moves a salvaged vector into place once it has all its bytes.
    fn finish_if_whole(&mut self, object: Object, root: Digest) -> Result<bool, NodeError> {
        let path = self.path(object);
        let sparse = path.with_extension("sparse");
        if vector::manifest_of(&sparse, self.shape)?.root != root {
            return Ok(false);
        }
        std::fs::rename(&sparse, &path)?;
        self.manifests.remove(&path);
        Ok(true)
    }

    /// The leaves that can't be had off the ledger, up to `limit` of them.
    /// Whatever can still be fetched is kept.
    pub fn missing(
        &mut self,
        number: u32,
        round: &RoundView,
        log: &RoundLog,
        limit: usize,
    ) -> Result<Vec<(ProofPart, u64)>, NodeError> {
        let Some(roots) = vector_roots(&round.cohort.members(), log) else {
            return Ok(Vec::new());
        };
        let Some(runtime) = self.runtime.clone() else {
            return Ok(Vec::new());
        };
        let per_chunk = self.shape.chunk_coords / LEAF_COORDS;
        let mut found = Vec::new();
        for (part, object, root) in self.vectors(number, round, &roots) {
            let path = self.path(object);
            if path.exists() && self.manifest(&path).is_ok_and(|m| m.root == root) {
                continue;
            }
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let sparse = path.with_extension("sparse");
            let salvage = observer::salvage(&self.source, object, root, &sparse);
            let gone = runtime
                .block_on(async { tokio::time::timeout(FETCH_LIMIT, salvage).await })
                .map_err(|_| NodeError::Disconnected(format!("{object:?} took too long")))??;
            if self.finish_if_whole(object, root)? {
                continue;
            }
            let leaves = gone
                .iter()
                .flat_map(|chunk| (0..per_chunk).map(move |at| chunk * per_chunk + at))
                .filter(|leaf| *leaf < self.shape.leaves())
                .filter(|leaf| !self.patched.contains(&(object, *leaf)));
            found.extend(leaves.map(|leaf| (part, leaf)));
            if found.len() >= limit {
                found.truncate(limit);
                break;
            }
        }
        Ok(found)
    }

    /// Puts a leaf from an answered demand into the watcher's copy.
    pub fn patch(
        &mut self,
        number: u32,
        round: &RoundView,
        log: &RoundLog,
        part: ProofPart,
        leaf: u64,
        proof: &[u8],
    ) -> Result<(), ChainError> {
        let gone = || NodeError::Check(format!("round {number}: no such vector"));
        let roots = vector_roots(&round.cohort.members(), log).ok_or_else(gone)?;
        let vectors = self.vectors(number, round, &roots);
        let (_, object, root) = vectors
            .iter()
            .find(|(held, ..)| *held == part)
            .ok_or_else(gone)?;
        let path = self.path(*object);
        if path.exists() {
            return Ok(());
        }
        let sparse = path.with_extension("sparse");
        if let Some(dir) = sparse.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&sparse)?;
        file.set_len(self.shape.coords * 8)?;
        let start = leaf * LEAF_COORDS;
        let coords = (self.shape.coords - start).min(LEAF_COORDS);
        let values = proof
            .get(..coords as usize * 8)
            .ok_or_else(|| NodeError::Check(format!("round {number}: a short answer")))?;
        file.write_all_at(values, start * 8)
            .map_err(NodeError::from)?;
        self.patched.insert((*object, leaf));
        self.finish_if_whole(*object, *root)?;
        Ok(())
    }

    /// One part of the proof for `leaf`, as `dispute_leaf` reads it from the
    /// buffer.
    pub fn proof(
        &mut self,
        number: u32,
        round: &RoundView,
        log: &RoundLog,
        leaf: u64,
        part: ProofPart,
    ) -> Result<Vec<u8>, ChainError> {
        let gone = || NodeError::Check(format!("round {number}: the disputed vectors are gone"));
        let roots = vector_roots(&round.cohort.members(), log).ok_or_else(gone)?;
        let held = self.files(number, round, &roots)?.ok_or_else(gone)?;
        let (path, position) = match part {
            ProofPart::Input(position) => {
                (held.inputs.get(position as usize), Some(position as usize))
            }
            ProofPart::Mask(helper) => (held.masks.get(helper as usize), None),
            ProofPart::Aggregate => (Some(&held.aggregate), None),
        };
        let path = path.ok_or_else(gone)?;
        let shape = self.shape;
        let manifest = self.manifest(path)?;
        let proof = vector::prove_leaf(path, shape, manifest, leaf)?;
        let values = bytes(&proof.values);
        Ok(match position {
            Some(position) => {
                let within = anchor::inputs_path(&roots, position);
                client::input_proof(&values, &manifest.root.0, &within, &proof.path)
            }
            None => client::leaf_proof(&values, &proof.path),
        })
    }
}
