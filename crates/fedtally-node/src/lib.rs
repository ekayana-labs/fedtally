//! The three long-running roles. All of them react to the log and nothing
//! else, so a restarted process picks up from what the log says.

pub mod aggregator;
mod follower;
pub mod helper;
pub mod local;
pub mod net;
pub mod observer;
pub mod registry;
pub mod site;
pub mod wire;

use std::fs::File;
use std::path::{Path, PathBuf};

use fedtally_core::keys::KeyError;
use fedtally_core::model::ModelError;
use fedtally_core::plan::Plan;
use fedtally_core::record::RecordError;
use fedtally_core::replay::{Fault, Replay};
use fedtally_core::status::StatusList;
use thiserror::Error;

const FRAME_SLACK: usize = 1 << 23;
const MANIFEST_BYTES_PER_CHUNK: usize = 70;

/// Largest frame a run can need. Refused frames are never allocated.
pub fn frame_limit(plan: &Plan) -> usize {
    let manifest = plan.shape().chunks() as usize * MANIFEST_BYTES_PER_CHUNK;
    let proofs = plan.norm().map_or(0, |params| {
        params
            .proof_bytes()
            .max(params.share_bytes() * plan.sites.len())
    });
    (plan.chunk_coords as usize * 8).max(manifest).max(proofs) + FRAME_SLACK
}

#[derive(Debug, Error)]
pub enum NodeError {
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("tls: {0}")]
    Tls(String),
    #[error("protocol: {0}")]
    Protocol(String),
    #[error("refused: {0}")]
    Refused(String),
    #[error("disconnected: {0}")]
    Disconnected(String),
    #[error("the log breaks the rules: {0}")]
    Fault(#[from] Fault),
    #[error(transparent)]
    Record(#[from] RecordError),
    #[error(transparent)]
    Key(#[from] KeyError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("check failed, not acknowledging: {0}")]
    Check(String),
    #[error("the trainer failed: {0}")]
    Trainer(String),
    #[error("the run was aborted: {0}")]
    Aborted(String),
    #[error("task: {0}")]
    Join(String),
    #[error("site {site}: {why}")]
    BadSeed { site: usize, why: String },
}

impl NodeError {
    /// True when reconnecting might help.
    pub(crate) fn is_transient(&self) -> bool {
        matches!(
            self,
            NodeError::Io(_)
                | NodeError::Disconnected(_)
                | NodeError::Tls(_)
                | NodeError::Protocol(_)
        )
    }
}

pub(crate) async fn blocking<T, F>(f: F) -> Result<T, NodeError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, NodeError> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| NodeError::Join(e.to_string()))?
}

/// The revocation lists in `dir` that a run relies on and its log lacks or
/// holds an older copy of, oldest first. A file that isn't a list is skipped.
pub(crate) fn newer_lists(dir: &Path, replay: &Replay) -> Vec<String> {
    let Some(policy) = replay.plan().and_then(|p| p.admission.as_ref()) else {
        return Vec::new();
    };
    let mut found: Vec<StatusList> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|text| StatusList::parse(text.trim()).ok())
        .filter(|list| policy.relies_on(&list.issuer))
        .filter(|list| {
            replay
                .list(&list.issuer, &list.id)
                .is_none_or(|held| held.issued < list.issued || !held.keeps(list))
        })
        .collect();
    found.sort_by_key(|list| list.issued);
    found.into_iter().map(|list| list.text).collect()
}

pub(crate) struct Dirs(PathBuf);

/// Writes a file a power cut must not lose or leave half written: the data
/// reaches the disk, and then the name does. For what the log commits to and
/// nothing can rebuild, such as a site's seeds.
pub(crate) fn write_durable(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let partial = path.with_extension("partial");
    std::fs::write(&partial, data)?;
    File::open(&partial)?.sync_data()?;
    std::fs::rename(&partial, path)?;
    sync_parent(path)
}

/// A rename or a new file is only safe once its directory has reached the disk.
pub fn sync_parent(path: &Path) -> std::io::Result<()> {
    match path.parent() {
        Some(dir) => File::open(dir)?.sync_all(),
        None => Ok(()),
    }
}

impl Dirs {
    pub fn new(root: &Path) -> std::io::Result<Dirs> {
        std::fs::create_dir_all(root.join("models"))?;
        Ok(Dirs(root.to_owned()))
    }

    pub fn log(&self) -> PathBuf {
        self.0.join("log.jsonl")
    }

    pub fn round_path(&self, round: u32) -> PathBuf {
        self.0.join(format!("round-{round}"))
    }

    pub fn round(&self, round: u32) -> std::io::Result<PathBuf> {
        let dir = self.round_path(round);
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    pub fn model(&self, round: u32) -> PathBuf {
        self.0
            .join("models")
            .join(format!("round-{round}.safetensors"))
    }

    pub fn moments(&self, round: u32) -> PathBuf {
        self.0.join("models").join(format!("round-{round}.moments"))
    }

    pub fn drop_round(&self, round: u32) {
        let _ = std::fs::remove_dir_all(self.round_path(round));
    }
}
