//! The watcher's copy of a round's vectors: fetched from the aggregator,
//! checked against a root before they count, added up, and cut into the
//! proofs a dispute needs.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use fedtally_core::anchor;
use fedtally_core::vector::{self, Digest, LEAF_COORDS, Manifest, Shape};
use fedtally_ledger::client::{self, ProofPart};
use fedtally_node::NodeError;
use fedtally_node::observer::{self, ObserverConfig, Salvaged};
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
    /// Vectors that could not be downloaded whole.
    partials: HashMap<Object, Partial>,
    /// What has been added up already and found right, by round.
    clean: HashMap<u32, Clean>,
}

/// What is held of a vector that isn't here whole.
#[derive(Default)]
struct Partial {
    salvaged: Salvaged,
    /// Proofs from answered demands, as the ledger checked them, by leaf.
    answers: HashMap<u64, Vec<u8>>,
}

impl Partial {
    fn has_leaf(&self, leaf: u64, per_chunk: u64) -> bool {
        self.answers.contains_key(&leaf) || self.salvaged.held.contains(&(leaf / per_chunk))
    }
}

#[derive(Default)]
struct Clean {
    chunks: BTreeSet<u64>,
    leaves: BTreeSet<u64>,
}

/// A round's vectors on disk: inputs in cohort order, masks in helper order.
struct Held {
    inputs: Vec<PathBuf>,
    masks: Vec<PathBuf>,
    aggregate: PathBuf,
}

fn round_of(object: Object) -> u32 {
    match object {
        Object::Upload { round, .. }
        | Object::Mask { round, .. }
        | Object::Aggregate { round }
        | Object::GroupMask { round, .. }
        | Object::GroupAggregate { round, .. }
        | Object::Model { round } => round,
        Object::InitialModel => 0,
    }
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
            partials: HashMap::new(),
            clean: HashMap::new(),
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

    fn is_whole(&mut self, object: Object, root: Digest) -> bool {
        let path = self.path(object);
        path.exists() && self.manifest(&path).is_ok_and(|m| m.root == root)
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
        if self.is_whole(object, root) {
            return Ok(path);
        }
        let refused = || NodeError::Disconnected(format!("{object:?} is not to be had whole"));
        if self.partials.contains_key(&object) {
            self.salvage(object, root)?;
            return if path.exists() {
                Ok(path)
            } else {
                Err(refused())
            };
        }
        let Some(runtime) = self.runtime.clone() else {
            return Err(refused());
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
        for (root, helper) in log.masks.iter().zip(&log.active) {
            let helper = *helper as u32;
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
        active: &[usize],
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
        for (root, helper) in held.mask_roots.iter().zip(active) {
            let helper = *helper as u32;
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
        let Some(held) = self.files(number, round, &roots, &log.active)? else {
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
        active: &[usize],
    ) -> Vec<(ProofPart, Object, Digest)> {
        let cohort = round.cohort.members().into_iter().zip(roots);
        let inputs = cohort.enumerate().map(|(position, (site, root))| {
            let object = Object::Upload {
                round: number,
                site: u32::from(site),
            };
            (ProofPart::Input(position as u16), object, *root)
        });
        // A proof numbers masks by position, and the spool by the plan's index.
        let held = round.mask_roots.iter().zip(active).enumerate();
        let masks = held.map(|(position, (root, helper))| {
            let object = Object::Mask {
                round: number,
                helper: *helper as u32,
            };
            (ProofPart::Mask(position as u8), object, Digest(*root))
        });
        let aggregate = (
            ProofPart::Aggregate,
            Object::Aggregate { round: number },
            Digest(round.aggregate_root),
        );
        inputs.chain(masks).chain([aggregate]).collect()
    }

    /// Keeps whatever more of a vector can be fetched. A failure leaves what
    /// was had before.
    fn salvage(&mut self, object: Object, root: Digest) -> Result<(), NodeError> {
        let path = self.path(object);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let sparse = path.with_extension("sparse");
        let partial = self.partials.entry(object).or_default();
        if let Some(runtime) = &self.runtime {
            let fetch =
                observer::salvage(&self.source, object, root, &sparse, &mut partial.salvaged);
            match runtime.block_on(async { tokio::time::timeout(FETCH_LIMIT, fetch).await }) {
                Ok(fetched) => fetched?,
                Err(_) => tracing::debug!("{object:?} took too long"),
            }
        }
        self.finish_if_whole(object, root)
    }

    /// Moves a salvaged vector into place once it has all its bytes.
    fn finish_if_whole(&mut self, object: Object, root: Digest) -> Result<(), NodeError> {
        let held = self
            .partials
            .get(&object)
            .map_or(0, |p| p.salvaged.held.len());
        if held as u64 != self.shape.chunks() {
            return Ok(());
        }
        let path = self.path(object);
        let sparse = path.with_extension("sparse");
        if vector::manifest_of(&sparse, self.shape)?.root == root {
            std::fs::rename(&sparse, &path)?;
            self.manifests.remove(&path);
        }
        Ok(())
    }

    /// Looks for a wrong leaf in what can be had when not every vector can:
    /// every chunk held of all of them, and every answered leaf held of all of
    /// them. The aggregate is one sum per leaf, so a leaf needs nothing else
    /// to be proven wrong, and withholding the rest of a vector can't hide it.
    pub fn recover(
        &mut self,
        number: u32,
        round: &RoundView,
        log: &RoundLog,
    ) -> Result<Option<u64>, NodeError> {
        let Some(roots) = vector_roots(&round.cohort.members(), log) else {
            return Ok(None);
        };
        if anchor::inputs_root(&roots) != round.inputs_root {
            return Ok(None);
        }
        let vectors = self.vectors(number, round, &roots, &log.active);
        let mut whole = HashMap::new();
        for (_, object, root) in &vectors {
            if !self.is_whole(*object, *root) {
                self.salvage(*object, *root)?;
            }
            whole.insert(*object, self.is_whole(*object, *root));
        }
        let shape = self.shape;
        let per_chunk = shape.chunk_coords / LEAF_COORDS;
        let located = |object: Object| match whole[&object] {
            true => self.path(object),
            false => self.path(object).with_extension("sparse"),
        };
        let (mut add, mut sub, mut aggregate) = (Vec::new(), Vec::new(), PathBuf::new());
        for (part, object, _) in &vectors {
            match part {
                ProofPart::Input(_) => add.push(located(*object)),
                ProofPart::Mask(_) => sub.push(located(*object)),
                ProofPart::Aggregate => aggregate = located(*object),
            }
        }
        let add: Vec<&Path> = add.iter().map(PathBuf::as_path).collect();
        let sub: Vec<&Path> = sub.iter().map(PathBuf::as_path).collect();

        // Salvaging left an entry for every vector that isn't whole.
        let parts: Vec<&Partial> = vectors
            .iter()
            .filter(|(_, object, _)| !whole[object])
            .filter_map(|(_, object, _)| self.partials.get(object))
            .collect();
        let clean = self.clean.entry(number).or_default();
        let chunks: Vec<u64> = (0..shape.chunks())
            .filter(|chunk| !clean.chunks.contains(chunk))
            .filter(|chunk| parts.iter().all(|part| part.salvaged.held.contains(chunk)))
            .collect();
        if !chunks.is_empty() {
            let wrong = vector::first_wrong_leaf_in(shape, &add, &sub, &aggregate, &chunks)?;
            if wrong.is_some() {
                return Ok(wrong);
            }
            clean.chunks.extend(chunks);
        }

        let answered: BTreeSet<u64> = parts
            .iter()
            .flat_map(|part| part.answers.keys().copied())
            .filter(|leaf| !clean.leaves.contains(leaf))
            .filter(|leaf| !clean.chunks.contains(&(leaf / per_chunk)))
            .filter(|leaf| parts.iter().all(|part| part.has_leaf(*leaf, per_chunk)))
            .collect();
        for leaf in answered {
            if vector::wrong_at_leaf(shape, &add, &sub, &aggregate, leaf)? {
                return Ok(Some(leaf));
            }
            clean.leaves.insert(leaf);
        }
        Ok(None)
    }

    /// The leaves that can't be had off the ledger, up to `limit` of them. A
    /// leaf can be checked once every vector has it, so the leaves missing from
    /// the fewest vectors come first, and each is finished before the next.
    pub fn missing(
        &mut self,
        number: u32,
        round: &RoundView,
        log: &RoundLog,
        limit: usize,
    ) -> Vec<(ProofPart, u64)> {
        let Some(roots) = vector_roots(&round.cohort.members(), log) else {
            return Vec::new();
        };
        let mut gone: BTreeMap<u64, Vec<(ProofPart, Object)>> = BTreeMap::new();
        for (part, object, root) in self.vectors(number, round, &roots, &log.active) {
            if self.is_whole(object, root) {
                continue;
            }
            let held = self.partials.get(&object).map(|p| &p.salvaged.held);
            for chunk in 0..self.shape.chunks() {
                if held.is_none_or(|held| !held.contains(&chunk)) {
                    gone.entry(chunk).or_default().push((part, object));
                }
            }
        }
        let mut chunks: Vec<(u64, Vec<(ProofPart, Object)>)> = gone.into_iter().collect();
        chunks.sort_by_key(|(chunk, vectors)| (vectors.len(), *chunk));

        let per_chunk = self.shape.chunk_coords / LEAF_COORDS;
        let mut found = Vec::new();
        for (chunk, vectors) in chunks {
            let end = ((chunk + 1) * per_chunk).min(self.shape.leaves());
            for leaf in chunk * per_chunk..end {
                for (part, object) in &vectors {
                    let answered = self
                        .partials
                        .get(object)
                        .is_some_and(|p| p.answers.contains_key(&leaf));
                    if !answered {
                        found.push((*part, leaf));
                    }
                    if found.len() == limit {
                        return found;
                    }
                }
            }
        }
        found
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
        let vectors = self.vectors(number, round, &roots, &log.active);
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

        let per_chunk = self.shape.chunk_coords / LEAF_COORDS;
        let chunk = leaf / per_chunk;
        let end = ((chunk + 1) * per_chunk).min(self.shape.leaves());
        let partial = self.partials.entry(*object).or_default();
        partial.answers.insert(leaf, proof.to_vec());
        if (chunk * per_chunk..end).all(|leaf| partial.answers.contains_key(&leaf)) {
            partial.salvaged.held.insert(chunk);
        }
        self.finish_if_whole(*object, *root)?;
        Ok(())
    }

    /// One part of the proof for `leaf`, as `dispute_leaf` reads it from the
    /// buffer. Only that leaf of that vector has to be here.
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
        if anchor::inputs_root(&roots) != round.inputs_root {
            return Err(gone().into());
        }
        let vectors = self.vectors(number, round, &roots, &log.active);
        let (_, object, root) = vectors
            .iter()
            .find(|(held, ..)| *held == part)
            .ok_or_else(gone)?;
        let (object, root) = (*object, *root);

        let partial = self.partials.get(&object);
        if let Some(answer) = partial.and_then(|p| p.answers.get(&leaf)) {
            return Ok(answer.clone());
        }
        let shape = self.shape;
        let per_chunk = shape.chunk_coords / LEAF_COORDS;
        let salvaged = partial
            .map(|p| &p.salvaged)
            .filter(|salvaged| salvaged.held.contains(&(leaf / per_chunk)))
            .and_then(|salvaged| salvaged.manifest.clone());
        let (path, manifest) = match salvaged {
            Some(manifest) if !self.path(object).exists() => {
                (self.path(object).with_extension("sparse"), manifest)
            }
            _ => {
                let path = self.hold(object, root)?;
                let manifest = self.manifest(&path)?.clone();
                (path, manifest)
            }
        };
        let proof = vector::prove_leaf(&path, shape, &manifest, leaf)?;
        let values = bytes(&proof.values);
        Ok(match part {
            ProofPart::Input(position) => {
                let within = anchor::inputs_path(&roots, position as usize);
                client::input_proof(&values, &root.0, &within, &proof.path)
            }
            _ => client::leaf_proof(&values, &proof.path),
        })
    }

    /// Lets go of what is held of a round.
    pub fn forget(&mut self, number: u32) {
        self.partials
            .retain(|object, _| round_of(*object) != number);
        self.clean.remove(&number);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fedtally_core::keys::{Identity, SigAlg};

    use super::*;

    const SHAPE: Shape = Shape {
        coords: 300,
        chunk_coords: 256,
    };
    const ROUND: u32 = 1;

    fn write(path: &Path, value: impl Fn(u64) -> u64 + Sync) -> Manifest {
        vector::build(path, SHAPE, |chunk, buf, _| {
            let (start, _) = SHAPE.range(chunk);
            for (at, slot) in buf.iter_mut().enumerate() {
                *slot = value(start + at as u64);
            }
            Ok(())
        })
        .unwrap()
    }

    /// A round of two sites and a helper on disk, with no way to fetch more.
    struct Fixture {
        _dir: tempfile::TempDir,
        vault: Vault,
        round: RoundView,
        log: RoundLog,
    }

    fn fixture(wrong_at: Option<u64>) -> Fixture {
        masked_by(wrong_at, 0)
    }

    /// `helper` is the plan's index of the round's one active helper.
    fn masked_by(wrong_at: Option<u64>, helper: usize) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let source = ObserverConfig {
            identity: Arc::new(Identity::generate(SigAlg::Ed25519).unwrap()),
            plan_line: Vec::new(),
            addr: String::new(),
            dir: dir.path().to_owned(),
            give_up_after: Duration::ZERO,
        };
        let spool = dir.path().join(format!("round-{ROUND}"));
        std::fs::create_dir_all(&spool).unwrap();
        let a = write(&spool.join("site-0.bin"), |i| i + 5);
        let b = write(&spool.join("site-1.bin"), |i| i * 3);
        let mask = write(&spool.join(format!("mask-{helper}.bin")), |i| i);
        let aggregate = write(&spool.join("aggregate.bin"), move |i| {
            let sum = i + 5 + i * 3 - i;
            sum + u64::from(wrong_at == Some(i))
        });
        let roots = [a.root, b.root];
        let round = RoundView {
            cohort: [0u16, 1].into_iter().collect(),
            inputs_root: anchor::inputs_root(&roots),
            mask_roots: vec![mask.root.0],
            aggregate_root: aggregate.root.0,
            ..RoundView::default()
        };
        let log = RoundLog {
            vectors: [(0, a.root), (1, b.root)].into_iter().collect(),
            active: vec![helper],
            ..RoundLog::default()
        };
        Fixture {
            _dir: dir,
            vault: Vault::new(source, None, SHAPE),
            round,
            log,
        }
    }

    impl Fixture {
        fn proof(&mut self, part: ProofPart, leaf: u64) -> Vec<u8> {
            self.vault
                .proof(ROUND, &self.round, &self.log, leaf, part)
                .unwrap()
        }

        fn lose(&self, object: Object) {
            std::fs::remove_file(self.vault.path(object)).unwrap();
        }

        fn patch(&mut self, part: ProofPart, leaf: u64, proof: &[u8]) {
            self.vault
                .patch(ROUND, &self.round, &self.log, part, leaf, proof)
                .unwrap();
        }

        fn recover(&mut self) -> Option<u64> {
            self.vault.recover(ROUND, &self.round, &self.log).unwrap()
        }

        fn missing(&mut self, limit: usize) -> Vec<(ProofPart, u64)> {
            self.vault.missing(ROUND, &self.round, &self.log, limit)
        }
    }

    const FIRST: Object = Object::Upload {
        round: ROUND,
        site: 0,
    };
    const SECOND: Object = Object::Upload {
        round: ROUND,
        site: 1,
    };

    #[test]
    fn one_answered_leaf_proves_a_wrong_sum_with_the_rest_of_the_vector_gone() {
        // Leaf 1 of three is wrong, and the first site's vector is lost.
        let mut held = fixture(Some(130));
        let answer = held.proof(ProofPart::Input(0), 1);
        held.lose(FIRST);
        assert_eq!(held.recover(), None, "nothing is known of that leaf yet");

        held.patch(ProofPart::Input(0), 1, &answer);
        assert_eq!(held.recover(), Some(1));

        // Every part of the proof can be had, the lost vector's from the answer.
        assert_eq!(held.proof(ProofPart::Input(0), 1), answer);
        held.proof(ProofPart::Input(1), 1);
        held.proof(ProofPart::Mask(0), 1);
        held.proof(ProofPart::Aggregate, 1);
    }

    #[test]
    fn a_mask_is_found_by_its_helper_after_a_removal_and_proven_by_its_position() {
        let mut held = masked_by(Some(130), 2);
        let verdict = held.vault.examine(ROUND, &held.round, &held.log).unwrap();
        assert!(matches!(verdict, Some(Verdict::Wrong(1))), "{verdict:?}");
        held.proof(ProofPart::Mask(0), 1);
    }

    #[test]
    fn right_leaves_prove_nothing_and_are_not_added_up_twice() {
        let mut held = fixture(Some(260));
        let answer = held.proof(ProofPart::Input(0), 0);
        held.lose(FIRST);
        held.patch(ProofPart::Input(0), 0, &answer);
        assert_eq!(held.recover(), None);
        assert!(held.vault.clean[&ROUND].leaves.contains(&0));
    }

    #[test]
    fn a_vector_answered_in_full_becomes_whole_again() {
        let mut held = fixture(None);
        let answers: Vec<Vec<u8>> = (0..3)
            .map(|leaf| held.proof(ProofPart::Input(0), leaf))
            .collect();
        held.lose(FIRST);
        for (leaf, answer) in answers.iter().enumerate() {
            held.patch(ProofPart::Input(0), leaf as u64, answer);
        }
        assert!(held.vault.path(FIRST).exists());
    }

    #[test]
    fn demands_finish_one_leaf_before_starting_the_next() {
        let mut held = fixture(None);
        held.lose(FIRST);
        held.lose(SECOND);
        // Both vectors miss both chunks, so a leaf takes one demand from each.
        assert_eq!(
            held.missing(3),
            vec![
                (ProofPart::Input(0), 0),
                (ProofPart::Input(1), 0),
                (ProofPart::Input(0), 1),
            ]
        );
        // A chunk missing from one vector only comes first: one demand settles
        // each of its leaves.
        let partial = held.vault.partials.entry(SECOND).or_default();
        partial.salvaged.held.insert(1);
        assert_eq!(held.missing(1), vec![(ProofPart::Input(0), 2)]);
    }

    #[test]
    fn what_was_answered_is_not_asked_for_again() {
        let mut held = fixture(None);
        let answer = held.proof(ProofPart::Input(0), 0);
        held.lose(FIRST);
        held.patch(ProofPart::Input(0), 0, &answer);
        assert_eq!(held.missing(1), vec![(ProofPart::Input(0), 1)]);
    }
}
