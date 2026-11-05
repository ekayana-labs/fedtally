//! What sites and helpers share: a local replica of the log, kept honest by
//! replaying every entry, and object transfer with the aggregator.

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fedtally_core::keys::Identity;
use fedtally_core::log::LogFile;
use fedtally_core::plan::Plan;
use fedtally_core::record::{Body, Record};
use fedtally_core::replay::Replay;
use fedtally_core::vector::{self, Digest, Manifest, Shape};

use crate::net::Client;
use crate::wire::{Msg, Object};
use crate::{Dirs, NodeError, blocking, frame_limit};

const FETCH_WINDOW: usize = 8;
const RETRY_FIRST: Duration = Duration::from_millis(100);
const RETRY_MAX: Duration = Duration::from_secs(5);

pub(crate) struct Follower {
    pub identity: Arc<Identity>,
    pub plan: Arc<Plan>,
    pub plan_line: Vec<u8>,
    pub plan_id: Digest,
    pub addr: String,
    pub dirs: Dirs,
    pub replay: Replay,
    pub give_up_after: Duration,
    log: LogFile,
}

/// Which chunks of a download have arrived. Counting replies isn't enough: an
/// aggregator repeating valid chunks could leave others unset, and so choose
/// which part of a vector the reader ends up checking.
struct Arrivals {
    seen: Vec<bool>,
    left: u64,
}

impl Arrivals {
    fn new(total: u64) -> Arrivals {
        Arrivals {
            seen: vec![false; total as usize],
            left: total,
        }
    }

    fn record(&mut self, chunk: u64) -> Result<(), NodeError> {
        match self.seen.get_mut(chunk as usize) {
            Some(seen) if !*seen => {
                *seen = true;
                self.left -= 1;
                Ok(())
            }
            _ => Err(NodeError::Check(format!(
                "chunk {chunk} was repeated or out of range"
            ))),
        }
    }

    fn is_complete(&self) -> bool {
        self.left == 0
    }
}

/// Decides when to stop reconnecting: after failing for `limit` with the log
/// not moving. Long quiet stretches between failures don't count.
pub(crate) struct Patience {
    entries: u64,
    since: Instant,
    last_failure: Option<Instant>,
    limit: Duration,
}

impl Patience {
    pub fn new(limit: Duration) -> Patience {
        Patience {
            entries: 0,
            since: Instant::now(),
            last_failure: None,
            limit,
        }
    }

    /// Records a failure and says whether to try again.
    pub fn allows(&mut self, entries: u64) -> bool {
        let now = Instant::now();
        let new_burst =
            entries != self.entries || self.last_failure.is_none_or(|last| now - last > self.limit);
        if new_burst {
            self.entries = entries;
            self.since = now;
        }
        self.last_failure = Some(now);
        now - self.since < self.limit
    }
}

pub(crate) struct Start {
    pub identity: Arc<Identity>,
    pub plan_line: Vec<u8>,
    pub addr: String,
    pub dir: std::path::PathBuf,
    pub give_up_after: Duration,
}

impl Follower {
    pub fn open(start: Start) -> Result<Follower, NodeError> {
        let record = Record::parse(&start.plan_line)?;
        let Body::Plan(plan) = &record.body else {
            return Err(NodeError::Protocol("not a plan".into()));
        };
        plan.validate()
            .map_err(|e| NodeError::Check(e.to_string()))?;
        let plan = Arc::new((**plan).clone());
        let dirs = Dirs::new(&start.dir)?;
        let mut log = LogFile::open(&dirs.log())?;
        let lines = log.read_all()?;
        let mut replay = Replay::new();
        for line in &lines {
            replay.apply_line(line)?;
        }
        if !lines.is_empty() && replay.plan_id != record.body_hash {
            return Err(NodeError::Protocol("local log is for another run".into()));
        }
        Ok(Follower {
            identity: start.identity,
            plan_id: record.body_hash,
            plan,
            plan_line: start.plan_line,
            addr: start.addr,
            dirs,
            replay,
            give_up_after: start.give_up_after,
            log,
        })
    }

    /// Connects, retrying while the aggregator is down.
    pub async fn connect(&self) -> Result<Client, NodeError> {
        let started = Instant::now();
        let mut wait = RETRY_FIRST;
        loop {
            let attempt = Client::connect(
                &self.addr,
                self.plan.aggregator.tls_pin,
                self.plan_id,
                &self.identity,
                frame_limit(&self.plan),
            )
            .await;
            match attempt {
                Ok(mut client) => {
                    client
                        .send(&Msg::Subscribe {
                            from: self.replay.entries,
                            head: self.replay.head,
                        })
                        .await?;
                    return Ok(client);
                }
                Err(e) if !e.is_transient() => return Err(e),
                Err(e) if started.elapsed() > self.give_up_after => {
                    return Err(NodeError::Disconnected(format!("gave up: {e}")));
                }
                Err(_) => {
                    tokio::time::sleep(wait).await;
                    wait = (wait * 2).min(RETRY_MAX);
                }
            }
        }
    }

    /// Replays a batch and only then stores it. A line that breaks the rules
    /// stops this party here: it never reacts to a log it can't accept.
    pub fn absorb(&mut self, lines: Vec<Vec<u8>>) -> Result<(), NodeError> {
        for line in lines {
            if self.replay.entries == 0 && line != self.plan_line {
                return Err(NodeError::Protocol(
                    "the aggregator sent a different plan".into(),
                ));
            }
            self.replay.apply_line(&line)?;
            self.log.append(&line)?;
        }
        Ok(())
    }

    /// Signs and sends a record, and waits to see it in the log. The record
    /// is only ever applied from the stream, so all copies stay in one order.
    pub async fn append(&mut self, client: &mut Client, body: &Body) -> Result<(), NodeError> {
        let record = Record::sign(&self.identity, body)?;
        client
            .send(&Msg::Append {
                line: record.line.clone(),
            })
            .await?;
        let moment = (self.replay.phase, self.replay.round_number());
        loop {
            let batch = client.entries().await?;
            let mine = batch.contains(&record.line);
            self.absorb(batch)?;
            if mine {
                return Ok(());
            }
            if (self.replay.phase, self.replay.round_number()) != moment {
                return Err(NodeError::Refused(
                    "the log moved on without this record".into(),
                ));
            }
        }
    }

    /// As `append`, for records that may lose a race with a deadline. Being
    /// refused is then something to report, not a reason to stop.
    pub async fn offer(&mut self, client: &mut Client, body: &Body) -> Result<bool, NodeError> {
        match self.append(client, body).await {
            Ok(()) => Ok(true),
            Err(NodeError::Refused(reason)) => {
                tracing::warn!("record refused: {reason}");
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    pub fn shape(&self) -> Shape {
        self.plan.shape()
    }

    /// Uploads a vector until the aggregator holds all of it.
    pub async fn upload(
        &self,
        client: &mut Client,
        object: Object,
        path: &Path,
        manifest: &Manifest,
    ) -> Result<(), NodeError> {
        client
            .send(&Msg::PutManifest {
                object,
                manifest: manifest.clone(),
            })
            .await?;
        let Msg::Missing { chunks, .. } = client.reply().await? else {
            return Err(NodeError::Protocol("expected the missing list".into()));
        };
        let shape = self.shape();
        let file = Arc::new(File::open(path)?);
        for chunk in chunks {
            let file = file.clone();
            let data = blocking(move || {
                let mut buf = Vec::new();
                vector::read_chunk(&file, shape, chunk, &mut buf)?;
                Ok(buf)
            })
            .await?;
            client
                .send(&Msg::PutChunk {
                    object,
                    chunk,
                    data: vector::chunk_bytes(&data).to_vec(),
                })
                .await?;
        }
        client.send(&Msg::PutDone { object }).await?;
        match client.reply().await? {
            Msg::Stored { .. } => Ok(()),
            other => Err(NodeError::Disconnected(format!(
                "upload incomplete: {other:?}"
            ))),
        }
    }

    /// Downloads a vector and checks it against the digest from the log.
    pub async fn fetch_vector(
        &self,
        client: &mut Client,
        object: Object,
        root: Digest,
        out: &Path,
    ) -> Result<(), NodeError> {
        client.send(&Msg::GetManifest { object }).await?;
        let manifest = match client.reply().await? {
            Msg::Manifest { manifest, .. } => manifest,
            Msg::Unavailable { .. } => {
                return Err(NodeError::Disconnected(format!("{object:?} is not ready")));
            }
            other => {
                return Err(NodeError::Protocol(format!(
                    "expected a manifest, got {other:?}"
                )));
            }
        };
        let shape = self.shape();
        if manifest.root != root
            || Manifest::root_of(&manifest.chunks) != root
            || manifest.chunks.len() as u64 != shape.chunks()
        {
            return Err(NodeError::Check(format!(
                "{object:?} has the wrong manifest"
            )));
        }
        let file = Arc::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(out)?,
        );
        file.set_len(shape.coords * 8)?;
        let total = shape.chunks();
        let mut arrived = Arrivals::new(total);
        let (mut next, mut in_flight) = (0u64, 0usize);
        while !arrived.is_complete() {
            while in_flight < FETCH_WINDOW && next < total {
                client
                    .send(&Msg::GetChunk {
                        object,
                        chunk: next,
                    })
                    .await?;
                next += 1;
                in_flight += 1;
            }
            let (chunk, data) = match client.reply().await? {
                Msg::Chunk { chunk, data, .. } => (chunk, data),
                Msg::Unavailable { .. } => {
                    return Err(NodeError::Disconnected(format!("{object:?} went away")));
                }
                other => {
                    return Err(NodeError::Protocol(format!(
                        "expected a chunk, got {other:?}"
                    )));
                }
            };
            arrived.record(chunk)?;
            let want = manifest.chunks[chunk as usize];
            let target = file.clone();
            blocking(move || {
                if data.len() != shape.chunk_bytes(chunk)
                    || vector::digest_chunk_bytes(&data) != want
                {
                    return Err(NodeError::Check(format!("chunk {chunk} does not match")));
                }
                vector::write_chunk_bytes(&target, shape, chunk, &data)?;
                Ok(())
            })
            .await?;
            in_flight -= 1;
        }
        file.sync_data()?;
        Ok(())
    }

    /// Downloads a whole file and checks its digest.
    pub async fn fetch_file(
        &self,
        client: &mut Client,
        object: Object,
        digest: Digest,
        out: &Path,
    ) -> Result<(), NodeError> {
        use std::io::Write;
        client.send(&Msg::GetFile { object }).await?;
        let partial = out.with_extension("partial");
        let mut file = File::create(&partial)?;
        loop {
            match client.reply().await? {
                Msg::FileChunk { data, last, .. } => {
                    file.write_all(&data)?;
                    if last {
                        break;
                    }
                }
                Msg::Unavailable { .. } => {
                    return Err(NodeError::Disconnected(format!("{object:?} is not ready")));
                }
                other => {
                    return Err(NodeError::Protocol(format!(
                        "expected file data, got {other:?}"
                    )));
                }
            }
        }
        file.sync_data()?;
        drop(file);
        if Digest::file(&partial)? != digest {
            return Err(NodeError::Check(format!("{object:?} has the wrong digest")));
        }
        std::fs::rename(&partial, out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_download_needs_every_chunk_once() {
        let mut arrived = Arrivals::new(3);
        arrived.record(1).unwrap();
        assert!(arrived.record(1).is_err(), "a repeat was counted");
        assert!(
            arrived.record(3).is_err(),
            "a chunk past the end was counted"
        );
        assert!(!arrived.is_complete());
        arrived.record(0).unwrap();
        assert!(!arrived.is_complete());
        arrived.record(2).unwrap();
        assert!(arrived.is_complete());
    }

    #[test]
    fn patience_runs_out_only_while_failing_without_progress() {
        let limit = Duration::from_millis(40);
        let mut patience = Patience::new(limit);
        std::thread::sleep(limit * 2);
        assert!(patience.allows(0), "a quiet stretch used up the patience");
        std::thread::sleep(limit / 2);
        assert!(patience.allows(0));
        std::thread::sleep(limit * 2 / 3);
        assert!(!patience.allows(0));
        assert!(patience.allows(1), "progress did not restore the patience");
    }
}
