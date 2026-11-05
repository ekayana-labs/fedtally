//! A party outside the plan that the aggregator lets read: it keeps a replica
//! of the log, replayed like any other, and downloads a round's vectors.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use fedtally_core::keys::Identity;
use fedtally_core::plan::Plan;
use fedtally_core::record::{Body, Record};
use fedtally_core::replay::Phase;
use fedtally_core::vector::Digest;

use crate::follower::{Follower, Patience, Start};
use crate::net::Client;
use crate::wire::Object;
use crate::{NodeError, frame_limit};

#[derive(Clone)]
pub struct ObserverConfig {
    pub identity: Arc<Identity>,
    pub plan_line: Vec<u8>,
    pub addr: String,
    /// Where the replica goes, as `log.jsonl`.
    pub dir: PathBuf,
    pub give_up_after: Duration,
}

/// Where an observer with this directory keeps its replica of the log.
pub fn log_path(dir: &Path) -> PathBuf {
    dir.join("log.jsonl")
}

/// Follows the log until the run ends, reconnecting while the aggregator is
/// away. Every entry is replayed before it is stored.
pub async fn mirror_log(config: ObserverConfig) -> Result<(), NodeError> {
    let mut follower = Follower::open(Start {
        identity: config.identity,
        plan_line: config.plan_line,
        addr: config.addr,
        dir: config.dir,
        give_up_after: config.give_up_after,
    })?;
    let mut patience = Patience::new(config.give_up_after);
    loop {
        match session(&mut follower).await {
            Ok(()) => return Ok(()),
            Err(e) if e.is_transient() && patience.allows(follower.replay.entries) => {
                tracing::info!("observer reconnecting: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

async fn session(f: &mut Follower) -> Result<(), NodeError> {
    let mut client = f.connect().await?;
    loop {
        if matches!(f.replay.phase, Phase::Done | Phase::Aborted) {
            return Ok(());
        }
        let batch = client.entries().await?;
        f.absorb(batch)?;
    }
}

/// Downloads one vector over a connection of its own and checks it against
/// `root`. An object the aggregator does not hold is `Disconnected`.
pub async fn fetch_vector(
    config: &ObserverConfig,
    object: Object,
    root: Digest,
    out: &Path,
) -> Result<(), NodeError> {
    let record = Record::parse(&config.plan_line)?;
    let Body::Plan(plan) = &record.body else {
        return Err(NodeError::Protocol("not a plan".into()));
    };
    let plan: &Plan = plan;
    let mut client = Client::connect(
        &config.addr,
        plan.aggregator.tls_pin,
        record.body_hash,
        &config.identity,
        frame_limit(plan),
    )
    .await?;
    Follower::fetch(&mut client, plan.shape(), object, root, out).await
}
