//! The aggregator adds up vectors it can't read and writes down what it did.
//!
//! All round state is in the log and the spool directory. Killing the process
//! at any point loses nothing but the work in progress: on restart it replays
//! the log, rescans the spool, and the peers reconnect and resume.

use std::collections::{BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use fedtally_core::field::Fp2;
use fedtally_core::keys::Identity;
use fedtally_core::log::LogFile;
use fedtally_core::model;
use fedtally_core::norm::{self, Challenge, Params};
use fedtally_core::plan::Plan;
use fedtally_core::record::{Body, Included, Record};
use fedtally_core::replay::{Phase, Replay, RoundState};
use fedtally_core::scoring;
use fedtally_core::vector::{self, Digest, Manifest, Shape, challenge};
use tokio::sync::{Notify, mpsc};
use tokio_rustls::server::TlsStream;

use crate::net::{Listener, accept_peer, split_stream};
use crate::wire::{Blob, Msg, Object, PING_EVERY, SILENCE_LIMIT, read_msg, write_msg};
use crate::{
    Dirs, NodeError, blocking, frame_limit, newer_lists, replay_own, sync_parent, write_durable,
};

const TICK: Duration = Duration::from_millis(100);
const FILE_CHUNK: usize = 1 << 20;
const ENTRY_BATCH_BYTES: usize = 1 << 19;
const DATA_QUEUE: usize = 16;
const LINGER: Duration = Duration::from_secs(30);

/// Points where a test can act as a dishonest operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    BeforeVerdict { round: u32 },
    BeforeSum { round: u32 },
    BeforeOutcome { round: u32 },
    BeforeSeal { round: u32 },
}

pub type Hook = Arc<dyn Fn(Stage, &Path) + Send + Sync>;

pub struct AggregatorConfig {
    pub identity: Arc<Identity>,
    pub plan_line: Vec<u8>,
    pub listen: String,
    pub dir: PathBuf,
    pub initial_model: PathBuf,
    pub hook: Option<Hook>,
    pub on_ready: Option<Box<dyn FnOnce(String) + Send>>,
    /// Keep each round's vectors instead of deleting them. For inspection.
    pub keep_spool: bool,
    /// Where the newest revocation lists are kept, if the plan has a policy.
    /// Any the log lacks are added before each round opens.
    pub status_dir: Option<PathBuf>,
    /// Parties outside the plan that may follow the log and read the vectors
    /// of a round once its outcome is in the log. Only a spool that is kept
    /// stays readable after its round.
    pub watchers: Vec<String>,
}

struct Upload {
    manifest: Manifest,
    missing: BTreeSet<u64>,
    file: Arc<File>,
    /// On disk for certain. Nobody resends an upload after it's been counted.
    durable: bool,
}

impl Upload {
    fn is_ready(&self, root: &Digest) -> bool {
        self.durable && self.missing.is_empty() && self.manifest.root == *root
    }
}

struct Peer {
    control: mpsc::UnboundedSender<Msg>,
    subscribed: bool,
}

struct PendingAttest {
    record: Record,
    peer: u64,
}

struct State {
    replay: Replay,
    lines: Vec<Vec<u8>>,
    log: LogFile,
    peers: HashMap<u64, Peer>,
    next_peer: u64,
    /// The latest round whose aggregate each named watcher has come for.
    watched: HashMap<String, u32>,
    uploads: HashMap<Object, Upload>,
    manifests: HashMap<Object, Manifest>,
    attests: Vec<PendingAttest>,
    per_site: Option<(u32, Vec<Fp2>)>,
    /// The scored groups' shares for the round being sealed.
    shares: Vec<u32>,
    /// The disk refused a log write. Memory is now ahead of the log, so
    /// nothing more may be signed or served until a restart replays it.
    log_failed: bool,
}

struct Shared {
    identity: Arc<Identity>,
    plan: Arc<Plan>,
    plan_id: Digest,
    dirs: Dirs,
    shape: Shape,
    state: Mutex<State>,
    wake: Notify,
    hook: Option<Hook>,
    keep_spool: bool,
    status_dir: Option<PathBuf>,
    watchers: Vec<String>,
}

impl Shared {
    /// A run can end before a watcher has its last round, and that round's
    /// challenge window is still open. The aggregate is what a watcher fetches
    /// last, so asking for it means it's nearly done.
    fn awaited(&self) -> bool {
        let state = self.lock();
        let posted = state
            .replay
            .rounds
            .last()
            .is_some_and(|r| r.outcome.is_some());
        let last = state.replay.rounds.len() as u32;
        !state.peers.is_empty()
            || (posted
                && self
                    .watchers
                    .iter()
                    .any(|w| state.watched.get(w).is_none_or(|round| *round < last)))
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn apply(&self, state: &mut State, record: &Record) -> Result<(), NodeError> {
        if state.log_failed {
            return Err(NodeError::Check("the log can no longer be written".into()));
        }
        state.replay.apply(record)?;
        if let Err(e) = state.log.append(&record.line) {
            state.log_failed = true;
            return Err(e.into());
        }
        state.lines.push(record.line.clone());
        let batch = Msg::Entries {
            lines: vec![serde_bytes::ByteBuf::from(record.line.clone())],
        };
        for peer in state.peers.values().filter(|p| p.subscribed) {
            let _ = peer.control.send(batch.clone());
        }
        self.wake.notify_waiters();
        Ok(())
    }

    /// Signs a record against the current head and applies it in one step, so
    /// no peer's record can slip in between.
    fn append_own(&self, build: impl FnOnce(&Replay) -> Body) -> Result<(), NodeError> {
        let mut state = self.lock();
        let body = build(&state.replay);
        let record = Record::sign(&self.identity, &body)?;
        self.apply(&mut state, &record)
    }

    fn upload_path(&self, object: Object) -> PathBuf {
        match object {
            Object::Upload { round, site } => {
                self.dirs.round_path(round).join(format!("site-{site}.bin"))
            }
            Object::Mask { round, helper } => self
                .dirs
                .round_path(round)
                .join(format!("mask-{helper}.bin")),
            Object::Aggregate { round } => self.dirs.round_path(round).join("aggregate.bin"),
            Object::GroupMask {
                round,
                helper,
                group,
            } => self
                .dirs
                .round_path(round)
                .join(format!("mask-{helper}-g{group}.bin")),
            Object::GroupAggregate { round, group } => self
                .dirs
                .round_path(round)
                .join(format!("aggregate-g{group}.bin")),
            Object::Model { round } => self.dirs.model(round),
            Object::Moments { round } => self.dirs.moments(round),
            Object::InitialModel => self.dirs.model(0),
        }
    }

    fn sealed_path(&self, round: u32, site: u32, helper: u32) -> PathBuf {
        self.dirs
            .round_path(round)
            .join(format!("sealed-{site}-{helper}.bin"))
    }

    fn blob_path(&self, blob: Blob) -> PathBuf {
        match blob {
            Blob::Proof { round, site } => self
                .dirs
                .round_path(round)
                .join(format!("proof-{site}.bin")),
            Blob::Check { round, helper } => self
                .dirs
                .round_path(round)
                .join(format!("check-{helper}.bin")),
            Blob::Screened { round } => self.dirs.round_path(round).join("screened.bin"),
        }
    }

    fn holds(&self, blob: Blob, want: &Digest) -> bool {
        std::fs::read(self.blob_path(blob)).is_ok_and(|data| Digest::of(&data) == *want)
    }
}

/// Runs the aggregator until the run is over.
pub async fn run_aggregator(config: AggregatorConfig) -> Result<Replay, NodeError> {
    let record = Record::parse(&config.plan_line)?;
    let Body::Plan(plan) = &record.body else {
        return Err(NodeError::Protocol("not a plan".into()));
    };
    let plan = Arc::new((**plan).clone());
    if plan.aggregator.did != config.identity.did_string() {
        return Err(NodeError::Refused(
            "this key is not the plan's aggregator".into(),
        ));
    }

    let dirs = Dirs::new(&config.dir)?;
    let mut log = LogFile::open(&dirs.log())?;
    let (mut replay, mut lines) = replay_own(&mut log)?;
    if lines.is_empty() {
        replay.apply(&record)?;
        log.append(&record.line)?;
        lines.push(record.line.clone());
    } else if replay.plan_id != record.body_hash {
        return Err(NodeError::Protocol(
            "the log on disk is for another run".into(),
        ));
    }

    let first_model = dirs.model(0);
    if !first_model.exists() {
        if Digest::file(&config.initial_model)? != plan.init_model {
            return Err(NodeError::Check(
                "the initial model does not match the plan".into(),
            ));
        }
        let partial = first_model.with_extension("partial");
        std::fs::copy(&config.initial_model, &partial)?;
        File::open(&partial)?.sync_data()?;
        std::fs::rename(&partial, &first_model)?;
        sync_parent(&first_model)?;
    }

    let shared = Arc::new(Shared {
        identity: config.identity.clone(),
        shape: plan.shape(),
        plan_id: record.body_hash,
        plan,
        dirs,
        state: Mutex::new(State {
            replay,
            lines,
            log,
            peers: HashMap::new(),
            watched: HashMap::new(),
            next_peer: 0,
            uploads: HashMap::new(),
            manifests: HashMap::new(),
            attests: Vec::new(),
            per_site: None,
            shares: Vec::new(),
            log_failed: false,
        }),
        wake: Notify::new(),
        hook: config.hook,
        keep_spool: config.keep_spool,
        status_dir: config.status_dir,
        watchers: config.watchers,
    });
    recover_uploads(&shared).await?;

    let listener = Listener::bind(&config.listen, &config.identity).await?;
    let addr = listener.local_addr()?;
    tracing::info!("aggregator listening on {addr}");
    if let Some(ready) = config.on_ready {
        ready(addr);
    }

    let serving = tokio::spawn(serve(shared.clone(), listener));
    let result = drive(&shared).await;
    if result.is_ok() {
        linger(&shared).await;
    }
    serving.abort();
    result?;
    let replay = shared.lock().replay.clone();
    Ok(replay)
}

async fn linger(shared: &Shared) {
    let started = Instant::now();
    while started.elapsed() < LINGER && shared.awaited() {
        tokio::time::sleep(TICK).await;
    }
}

fn manifest_path(data: &Path) -> PathBuf {
    data.with_extension("manifest")
}

/// Rebuilds what the spool holds for the current round after a restart.
async fn recover_uploads(shared: &Arc<Shared>) -> Result<(), NodeError> {
    let round = shared.lock().replay.round_number();
    if round == 0 {
        return Ok(());
    }
    let mut objects = Vec::new();
    for site in 0..shared.plan.sites.len() as u32 {
        objects.push(Object::Upload { round, site });
    }
    for helper in 0..shared.plan.helpers.len() as u32 {
        objects.push(Object::Mask { round, helper });
    }
    for object in objects {
        let data = shared.upload_path(object);
        let Ok(text) = std::fs::read(manifest_path(&data)) else {
            continue;
        };
        let manifest: Manifest =
            serde_json::from_slice(&text).map_err(|e| NodeError::Protocol(e.to_string()))?;
        let upload = open_upload(&data, shared.shape, manifest).await?;
        shared.lock().uploads.insert(object, upload);
    }
    Ok(())
}

async fn open_upload(path: &Path, shape: Shape, manifest: Manifest) -> Result<Upload, NodeError> {
    let path = path.to_owned();
    blocking(move || {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        file.set_len(shape.coords * 8)?;
        let missing = vector::missing_chunks(&file, shape, &manifest)?;
        let durable = missing.is_empty();
        if durable {
            file.sync_data()?;
            sync_parent(&path)?;
        }
        Ok(Upload {
            manifest,
            missing,
            file: Arc::new(file),
            durable,
        })
    })
    .await
}

async fn serve(shared: Arc<Shared>, listener: Listener) {
    let limit = frame_limit(&shared.plan);
    loop {
        let Ok(stream) = listener.accept().await else {
            continue;
        };
        let (acceptor, shared) = (listener.acceptor(), shared.clone());
        tokio::spawn(async move {
            match accept_peer(acceptor, stream, shared.plan_id, limit).await {
                Ok((did, tls)) => {
                    if let Err(e) = serve_peer(shared, did, tls, limit).await {
                        tracing::debug!("peer closed: {e}");
                    }
                }
                Err(e) => tracing::debug!("handshake failed: {e}"),
            }
        });
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Site(u32),
    Helper(u32),
    Watcher,
}

async fn serve_peer<S>(
    shared: Arc<Shared>,
    did: String,
    tls: TlsStream<S>,
    limit: usize,
) -> Result<(), NodeError>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let role = if let Some(i) = shared.plan.site_index(&did) {
        Role::Site(i as u32)
    } else if let Some(i) = shared.plan.helper_index(&did) {
        Role::Helper(i as u32)
    } else if shared.watchers.contains(&did) {
        Role::Watcher
    } else {
        return Err(NodeError::Refused(format!("{did} is not in the plan")));
    };
    let (mut reader, mut writer) = split_stream(tls);
    let (control, mut control_rx) = mpsc::unbounded_channel();
    let (data, mut data_rx) = mpsc::channel::<Msg>(DATA_QUEUE);

    let id = {
        let mut state = shared.lock();
        let id = state.next_peer;
        state.next_peer += 1;
        state.peers.insert(
            id,
            Peer {
                control: control.clone(),
                subscribed: false,
            },
        );
        id
    };

    let writing = tokio::spawn(async move {
        let mut ping = tokio::time::interval(PING_EVERY);
        loop {
            let next = tokio::select! {
                biased;
                msg = control_rx.recv() => msg,
                msg = data_rx.recv() => msg,
                _ = ping.tick() => Some(Msg::Ping),
            };
            let Some(msg) = next else { return };
            let sent = tokio::time::timeout(SILENCE_LIMIT * 4, write_msg(&mut writer, &msg)).await;
            if !matches!(sent, Ok(Ok(()))) {
                return;
            }
        }
    });

    let result = async {
        loop {
            let msg = read_msg(&mut reader, limit).await?;
            handle(&shared, id, &did, role, msg, &data).await?;
        }
    }
    .await;
    shared.lock().peers.remove(&id);
    writing.abort();
    result
}

async fn handle(
    shared: &Arc<Shared>,
    peer: u64,
    did: &str,
    role: Role,
    msg: Msg,
    data: &mpsc::Sender<Msg>,
) -> Result<(), NodeError> {
    match msg {
        Msg::Subscribe { from, head } => subscribe(shared, peer, from, head),
        Msg::Append { line } => {
            if let Err(e) = try_append(shared, peer, did, role, &line) {
                tracing::warn!("refused a record from {did}: {e}");
                reply(
                    data,
                    Msg::Refused {
                        reason: e.to_string(),
                    },
                )
                .await?;
            }
            Ok(())
        }
        Msg::PutManifest { object, manifest } => {
            on_manifest(shared, data, role, object, manifest).await
        }
        Msg::PutChunk {
            object,
            chunk,
            data,
        } => on_chunk(shared, role, object, chunk, data).await,
        Msg::PutDone { object } => {
            let (missing, file) = {
                let state = shared.lock();
                match state.uploads.get(&object) {
                    Some(upload) => (
                        upload.missing.iter().copied().collect::<Vec<_>>(),
                        Some(upload.file.clone()),
                    ),
                    None => (Vec::new(), None),
                }
            };
            if missing.is_empty()
                && let Some(file) = file
            {
                let path = shared.upload_path(object);
                blocking(move || {
                    file.sync_data()?;
                    Ok(sync_parent(&path)?)
                })
                .await?;
                if let Some(upload) = shared.lock().uploads.get_mut(&object) {
                    upload.durable = true;
                }
            }
            let answer = if missing.is_empty() {
                Msg::Stored { object }
            } else {
                Msg::Missing {
                    object,
                    chunks: missing,
                }
            };
            shared.wake.notify_waiters();
            reply(data, answer).await
        }
        Msg::PutSealed {
            round,
            helper,
            data,
        } => {
            let Role::Site(site) = role else {
                return Err(NodeError::Protocol("only a site sends sealed seeds".into()));
            };
            let signed = {
                let state = shared.lock();
                if round != state.replay.round_number() || state.replay.phase != Phase::Collecting {
                    return Ok(());
                }
                state
                    .replay
                    .current()
                    .and_then(|r| r.contributions.get(&(site as usize)))
                    .and_then(|c| c.sealed.get(helper as usize).copied())
            };
            if signed != Some(Digest::of(&data)) {
                return Err(NodeError::Protocol(
                    "that seed is not in the signed contribution".into(),
                ));
            }
            shared.dirs.round(round)?;
            let path = shared.sealed_path(round, site, helper);
            blocking(move || write_atomic(&path, &data)).await?;
            shared.wake.notify_waiters();
            Ok(())
        }
        Msg::GetSealed { round, site } => {
            let Role::Helper(helper) = role else {
                return Err(NodeError::Protocol(
                    "only a helper reads sealed seeds".into(),
                ));
            };
            let path = shared.sealed_path(round, site, helper);
            let contents = std::fs::read(path).ok().map(serde_bytes::ByteBuf::from);
            reply(
                data,
                Msg::Sealed {
                    round,
                    site,
                    data: contents,
                },
            )
            .await
        }
        Msg::PutBlob { blob, data: bytes } => {
            let expected = {
                let state = shared.lock();
                let current = state.replay.current();
                let now = |round: u32, phase| {
                    round == state.replay.round_number() && state.replay.phase == phase
                };
                match (blob, role) {
                    (Blob::Proof { round, site }, Role::Site(me))
                        if me == site && now(round, Phase::Collecting) =>
                    {
                        current
                            .and_then(|r| r.contributions.get(&(site as usize)))
                            .and_then(|c| c.proof)
                    }
                    (Blob::Check { round, helper }, Role::Helper(me))
                        if me == helper && now(round, Phase::Screening) =>
                    {
                        Some(Digest::of(&bytes))
                    }
                    _ => return Ok(()),
                }
            };
            if expected != Some(Digest::of(&bytes)) {
                return Err(NodeError::Protocol(
                    "that proof is not in the signed contribution".into(),
                ));
            }
            let path = shared.blob_path(blob);
            blocking(move || write_atomic(&path, &bytes)).await?;
            shared.wake.notify_waiters();
            Ok(())
        }
        Msg::GetBlob { blob } => {
            let contents = std::fs::read(shared.blob_path(blob))
                .ok()
                .map(serde_bytes::ByteBuf::from);
            reply(
                data,
                Msg::Blob {
                    blob,
                    data: contents,
                },
            )
            .await
        }
        Msg::GetManifest { object } => {
            let answer = match manifest_for(shared, role, object).await {
                Ok(Some(manifest)) => {
                    if let (Role::Watcher, Object::Aggregate { round }) = (role, object) {
                        let mut state = shared.lock();
                        let seen = state.watched.entry(did.to_owned()).or_default();
                        *seen = round.max(*seen);
                    }
                    Msg::Manifest { object, manifest }
                }
                _ => Msg::Unavailable { object },
            };
            reply(data, answer).await
        }
        Msg::GetChunk { object, chunk } => {
            if !readable(shared, role, object) || chunk >= shared.shape.chunks() {
                return Err(NodeError::Protocol("that chunk is not readable".into()));
            }
            let path = shared.upload_path(object);
            let shape = shared.shape;
            let read = blocking(move || {
                let file = File::open(path)?;
                Ok(vector::read_chunk_bytes(&file, shape, chunk)?)
            })
            .await;
            let answer = match read {
                Ok(bytes) => Msg::Chunk {
                    object,
                    chunk,
                    data: bytes,
                },
                Err(_) => Msg::Unavailable { object },
            };
            reply(data, answer).await
        }
        Msg::GetFile { object } => {
            if !readable(shared, role, object) {
                return reply(data, Msg::Unavailable { object }).await;
            }
            send_file(shared, object, data).await
        }
        Msg::Ping => Ok(()),
        other => Err(NodeError::Protocol(format!("unexpected message {other:?}"))),
    }
}

/// Answers go through the bounded queue, so a peer that sends requests and
/// doesn't read the answers only stalls its own connection.
async fn reply(data: &mpsc::Sender<Msg>, msg: Msg) -> Result<(), NodeError> {
    data.send(msg)
        .await
        .map_err(|e| NodeError::Protocol(e.to_string()))
}

fn readable(shared: &Shared, role: Role, object: Object) -> bool {
    if role == Role::Watcher {
        return watchable(shared, object);
    }
    match object {
        Object::Aggregate { round }
        | Object::GroupAggregate { round, .. }
        | Object::Model { round }
        | Object::Moments { round } => !sum_held_back(&shared.lock(), round),
        Object::InitialModel => true,
        // A masked vector is noise to a site, so it's safe to audit, but only the
        // assigned auditor gets it. A site working with every helper could unmask
        // what it reads, so this keeps that to the vectors it was given.
        Object::Upload { round, site } => {
            let Role::Site(me) = role else {
                return false;
            };
            let state = shared.lock();
            let Some(current) = state.replay.current().filter(|r| r.round == round) else {
                return false;
            };
            let Some(challenge) = current.outcome.as_ref().map(|o| o.challenge) else {
                return false;
            };
            let position = |s: u32| current.included.iter().position(|i| *i == s as usize);
            let (Some(reader), Some(target)) = (position(me), position(site)) else {
                return false;
            };
            matches!(state.replay.phase, Phase::Attesting | Phase::Acking)
                && shared
                    .plan
                    .audit_targets(&challenge, reader, current.included.len())
                    .contains(&target)
        }
        Object::Mask { .. } | Object::GroupMask { .. } => false,
    }
}

/// A watcher holds no seeds, so the masked vectors and the mask totals of a
/// round tell it the aggregate and nothing more.
fn watchable(shared: &Shared, object: Object) -> bool {
    let round = match object {
        Object::Upload { round, .. }
        | Object::Mask { round, .. }
        | Object::Aggregate { round }
        | Object::GroupMask { round, .. }
        | Object::GroupAggregate { round, .. }
        | Object::Model { round }
        | Object::Moments { round } => round,
        Object::InitialModel => return true,
    };
    let state = shared.lock();
    let known = round.checked_sub(1).map(|i| i as usize);
    !sum_held_back(&state, round)
        && known
            .and_then(|i| state.replay.rounds.get(i))
            .is_some_and(|r| r.outcome.is_some())
}

/// A round's sum stays in until every helper has stood behind it, and a round
/// given up keeps it for good. A site could otherwise read a sum that may yet
/// be voided, and choose what it sends in the redo knowing it.
fn sum_held_back(state: &State, round: u32) -> bool {
    let at = round.checked_sub(1).map(|i| i as usize);
    at.and_then(|i| state.replay.rounds.get(i))
        .is_some_and(|r| r.voided || r.helper_attests.len() < r.active.len())
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<(), NodeError> {
    Ok(write_durable(path, data)?)
}

fn subscribe(shared: &Shared, peer: u64, from: u64, head: Digest) -> Result<(), NodeError> {
    let mut state = shared.lock();
    let state = &mut *state;
    let Some(p) = state.peers.get_mut(&peer) else {
        return Ok(());
    };
    if p.subscribed {
        return Err(NodeError::Protocol("already subscribed".into()));
    }
    let known = state.lines.get(..from as usize);
    let same_past = known.is_some_and(|lines| {
        let mut h = fedtally_core::log::GENESIS;
        for line in lines {
            h = fedtally_core::log::chain(&h, line);
        }
        h == head
    });
    if !same_past {
        let _ = p.control.send(Msg::Refused {
            reason: "this peer has seen a log this aggregator does not hold".into(),
        });
        return Ok(());
    }
    p.subscribed = true;
    let mut batch = Vec::new();
    let mut bytes = 0;
    for line in &state.lines[from as usize..] {
        bytes += line.len();
        batch.push(serde_bytes::ByteBuf::from(line.clone()));
        if bytes > ENTRY_BATCH_BYTES {
            let _ = p.control.send(Msg::Entries {
                lines: std::mem::take(&mut batch),
            });
            bytes = 0;
        }
    }
    let _ = p.control.send(Msg::Entries { lines: batch });
    Ok(())
}

fn try_append(
    shared: &Shared,
    peer: u64,
    did: &str,
    role: Role,
    line: &[u8],
) -> Result<(), NodeError> {
    let record = Record::parse(line)?;
    if record.signer != did {
        return Err(NodeError::Protocol("record signed by another party".into()));
    }
    if role == Role::Watcher {
        return Err(NodeError::Refused("a watcher only reads".into()));
    }
    let mut state = shared.lock();
    match &record.body {
        Body::Attest { .. } => {
            if state.replay.phase != Phase::Attesting {
                return Err(NodeError::Check("this round is not attesting".into()));
            }
            state.attests.retain(|p| p.record.signer != record.signer);
            state.attests.push(PendingAttest { record, peer });
            flush_attests(shared, &mut state);
            Ok(())
        }
        Body::MaskSum { round, mask, .. } => {
            let Role::Helper(helper) = role else {
                return Err(NodeError::Protocol("not a helper".into()));
            };
            let object = Object::Mask {
                round: *round,
                helper,
            };
            let complete = state.uploads.get(&object).is_some_and(|u| u.is_ready(mask));
            if !complete {
                return Err(NodeError::Check("mask total is not fully uploaded".into()));
            }
            shared.apply(&mut state, &record)
        }
        Body::Join { at, .. } if clock_gap(*at) > JOIN_CLOCK_SLACK => Err(NodeError::Check(
            "the time in this join is far from the aggregator's clock".into(),
        )),
        _ => {
            shared.apply(&mut state, &record)?;
            flush_attests(shared, &mut state);
            Ok(())
        }
    }
}

/// How far a site's stated join time may be from this machine's clock. The
/// time decides whether its credentials were in date.
const JOIN_CLOCK_SLACK: u64 = 600;

fn clock_gap(at: i64) -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    now.abs_diff(at)
}

/// Logs each waiting attestation that agrees with the fingerprints. One that
/// doesn't is refused here: logging it would make the whole seal invalid, so a
/// site could otherwise spoil a round just by lying.
fn flush_attests(shared: &Shared, state: &mut State) {
    let Some((round, per_site)) = state.per_site.clone() else {
        return;
    };
    let Some(current) = state.replay.current().cloned() else {
        return;
    };
    if current.round != round || state.replay.phase != Phase::Attesting {
        return;
    }
    if current.helper_attests.len() < current.active.len() {
        return;
    }
    for pending in std::mem::take(&mut state.attests) {
        let verdict = check_attest(shared, &current, &per_site, &pending.record);
        let outcome = verdict.and_then(|()| shared.apply(state, &pending.record));
        if let Err(e) = outcome {
            tracing::warn!("refused an attestation: {e}");
            if let Some(p) = state.peers.get(&pending.peer) {
                let _ = p.control.send(Msg::Refused {
                    reason: e.to_string(),
                });
            }
        }
    }
}

fn check_attest(
    shared: &Shared,
    round: &fedtally_core::replay::RoundState,
    per_site: &[Fp2],
    record: &Record,
) -> Result<(), NodeError> {
    let Body::Attest { masked, .. } = &record.body else {
        return Ok(());
    };
    let site = shared
        .plan
        .site_index(&record.signer)
        .ok_or_else(|| NodeError::Check("not a site".into()))?;
    let position = round
        .included
        .iter()
        .position(|s| *s == site)
        .ok_or_else(|| NodeError::Check("site is not in this round".into()))?;
    if per_site[position] != *masked {
        return Err(NodeError::Check(format!(
            "site {site} attests a different masked fingerprint"
        )));
    }
    Ok(())
}

async fn on_manifest(
    shared: &Arc<Shared>,
    data: &mpsc::Sender<Msg>,
    role: Role,
    object: Object,
    manifest: Manifest,
) -> Result<(), NodeError> {
    let allowed = match (object, role) {
        (Object::Upload { site, .. }, Role::Site(me)) => site == me,
        (Object::Mask { helper, .. } | Object::GroupMask { helper, .. }, Role::Helper(me)) => {
            helper == me
        }
        _ => false,
    };
    if !allowed || manifest.chunks.len() as u64 != shared.shape.chunks() {
        return Err(NodeError::Protocol("that upload is not allowed".into()));
    }
    if Manifest::root_of(&manifest.chunks) != manifest.root {
        return Err(NodeError::Protocol("manifest root is wrong".into()));
    }

    let late = {
        let state = shared.lock();
        let round = match object {
            Object::Upload { round, .. }
            | Object::Mask { round, .. }
            | Object::GroupMask { round, .. } => round,
            _ => 0,
        };
        let wanted = match object {
            Object::Upload { .. } => state.replay.phase == Phase::Collecting,
            _ => state.replay.phase == Phase::Masking,
        };
        round != state.replay.round_number() || !wanted
    };
    if late {
        return reply(
            data,
            Msg::Missing {
                object,
                chunks: Vec::new(),
            },
        )
        .await;
    }

    if let Object::Upload { round: _, site } = object {
        let signed = shared
            .lock()
            .replay
            .current()
            .and_then(|r| r.contributions.get(&(site as usize)).map(|c| c.masked));
        if signed != Some(manifest.root) {
            return Err(NodeError::Protocol(
                "that upload is not the signed contribution".into(),
            ));
        }
    }

    // A helper can't swap the file under a mask it has already signed for.
    let signed = shared.lock().replay.current().and_then(|r| match object {
        Object::Mask { helper, .. } => r.masks.get(&(helper as usize)).copied(),
        Object::GroupMask { helper, group, .. } => r
            .group_masks
            .get(&(helper as usize))
            .and_then(|groups| groups.get(group as usize).copied()),
        _ => None,
    });
    if signed.is_some_and(|root| root != manifest.root) {
        return Err(NodeError::Protocol(
            "that upload is not the mask its helper signed".into(),
        ));
    }

    let existing = {
        let state = shared.lock();
        state.uploads.get(&object).map(|u| u.manifest.root)
    };
    if existing != Some(manifest.root) {
        if let Object::Upload { round, .. }
        | Object::Mask { round, .. }
        | Object::GroupMask { round, .. } = object
        {
            shared.dirs.round(round)?;
        }
        let path = shared.upload_path(object);
        let text = serde_json::to_vec(&manifest).map_err(|e| NodeError::Protocol(e.to_string()))?;
        let meta = manifest_path(&path);
        blocking(move || write_atomic(&meta, &text)).await?;
        let upload = open_upload(&path, shared.shape, manifest).await?;
        shared.lock().uploads.insert(object, upload);
    }
    let chunks = {
        let state = shared.lock();
        state
            .uploads
            .get(&object)
            .map(|u| u.missing.iter().copied().collect())
            .unwrap_or_default()
    };
    shared.wake.notify_waiters();
    reply(data, Msg::Missing { object, chunks }).await
}

async fn on_chunk(
    shared: &Arc<Shared>,
    role: Role,
    object: Object,
    chunk: u64,
    data: Vec<u8>,
) -> Result<(), NodeError> {
    let allowed = match (object, role) {
        (Object::Upload { site, .. }, Role::Site(me)) => site == me,
        (Object::Mask { helper, .. } | Object::GroupMask { helper, .. }, Role::Helper(me)) => {
            helper == me
        }
        _ => false,
    };
    if !allowed {
        return Err(NodeError::Protocol("that upload is not allowed".into()));
    }
    let target = {
        let state = shared.lock();
        state.uploads.get(&object).and_then(|u| {
            u.manifest
                .chunks
                .get(chunk as usize)
                .map(|want| (*want, u.file.clone()))
        })
    };
    let Some((want, file)) = target else {
        return Ok(());
    };
    let shape = shared.shape;
    let stored = blocking(move || {
        if data.len() != shape.chunk_bytes(chunk) || vector::digest_chunk_bytes(&data) != want {
            return Ok(false);
        }
        vector::write_chunk_bytes(&file, shape, chunk, &data)?;
        Ok(true)
    })
    .await?;
    if stored && let Some(upload) = shared.lock().uploads.get_mut(&object) {
        upload.missing.remove(&chunk);
    }
    Ok(())
}

async fn manifest_for(
    shared: &Arc<Shared>,
    role: Role,
    object: Object,
) -> Result<Option<Manifest>, NodeError> {
    if !readable(shared, role, object) {
        return Ok(None);
    }
    {
        let state = shared.lock();
        if let Some(upload) = state.uploads.get(&object) {
            return Ok(Some(upload.manifest.clone()));
        }
        if let Some(known) = state.manifests.get(&object) {
            return Ok(Some(known.clone()));
        }
    }
    let path = shared.upload_path(object);
    if !path.exists() {
        return Ok(None);
    }
    // An upload's manifest is what was committed to, so a reader can still
    // check the chunks that are good if the file is damaged.
    let kept = std::fs::read(manifest_path(&path)).ok();
    if let Some(manifest) = kept.and_then(|text| serde_json::from_slice(&text).ok()) {
        return Ok(Some(manifest));
    }
    let shape = shared.shape;
    let manifest = blocking(move || Ok(vector::manifest_of(&path, shape)?)).await?;
    // A watcher is told what the disk holds now, so nothing is kept for it.
    if role != Role::Watcher {
        shared.lock().manifests.insert(object, manifest.clone());
    }
    Ok(Some(manifest))
}

async fn send_file(
    shared: &Arc<Shared>,
    object: Object,
    data: &mpsc::Sender<Msg>,
) -> Result<(), NodeError> {
    use std::os::unix::fs::FileExt;
    if !matches!(
        object,
        Object::Model { .. } | Object::Moments { .. } | Object::InitialModel
    ) {
        return Err(NodeError::Protocol("that object is not a file".into()));
    }
    let closed = |e: mpsc::error::SendError<Msg>| NodeError::Protocol(e.to_string());
    let Ok(file) = File::open(shared.upload_path(object)) else {
        return data.send(Msg::Unavailable { object }).await.map_err(closed);
    };
    let file = Arc::new(file);
    let length = file.metadata()?.len();
    let mut offset = 0;
    loop {
        let reader = file.clone();
        let bytes = blocking(move || {
            let n = (length - offset).min(FILE_CHUNK as u64) as usize;
            let mut buf = vec![0u8; n];
            reader.read_exact_at(&mut buf, offset)?;
            Ok(buf)
        })
        .await?;
        let n = bytes.len() as u64;
        let last = offset + n >= length;
        data.send(Msg::FileChunk {
            object,
            offset,
            last,
            data: bytes,
        })
        .await
        .map_err(closed)?;
        if last {
            return Ok(());
        }
        offset += n;
    }
}

enum Decision {
    Wait,
    Open,
    Screen,
    Close,
    Outcome,
    Fingerprints,
    Seal,
    Retry,
    RemoveHelper(usize),
    Next,
    Abort(String),
    Stop,
}

fn ready_sites(shared: &Shared, state: &State) -> Vec<usize> {
    let Some(current) = state.replay.current() else {
        return Vec::new();
    };
    let round = current.round;
    let sealed_ok = |site: usize, h: usize, want: &Digest| {
        std::fs::read(shared.sealed_path(round, site as u32, h as u32))
            .is_ok_and(|data| Digest::of(&data) == *want)
    };
    current
        .contributions
        .iter()
        .filter(|(site, _)| !state.replay.revoked.contains_key(site))
        .filter(|(site, contribution)| {
            let object = Object::Upload {
                round,
                site: **site as u32,
            };
            let uploaded = state
                .uploads
                .get(&object)
                .is_some_and(|u| u.is_ready(&contribution.masked));
            let proof = Blob::Proof {
                round,
                site: **site as u32,
            };
            uploaded
                && contribution
                    .sealed
                    .iter()
                    .enumerate()
                    .all(|(h, want)| *want == Digest::default() || sealed_ok(**site, h, want))
                && contribution
                    .proof
                    .is_none_or(|want| shared.holds(proof, &want))
        })
        .map(|(site, _)| *site)
        .collect()
}

/// `sites_given` is how long sites have had to attest, since they can't before
/// every helper has.
fn decide(shared: &Shared, state: &State, elapsed: Duration, sites_given: Duration) -> Decision {
    let plan = &shared.plan;
    let past = |secs: u64| elapsed > Duration::from_secs(secs);
    let min = plan.min_cohort as usize;
    match state.replay.phase {
        Phase::Empty | Phase::Done | Phase::Aborted => Decision::Stop,
        Phase::Joining => {
            let joined = state.replay.joined.len();
            if joined == plan.sites.len() {
                Decision::Open
            } else if past(plan.timeouts.join_secs) {
                if joined >= min {
                    Decision::Open
                } else {
                    Decision::Abort(format!("{joined} sites joined, the plan needs {min}"))
                }
            } else {
                Decision::Wait
            }
        }
        Phase::Collecting => {
            let ready = ready_sites(shared, state);
            let everyone = state
                .replay
                .expected(plan)
                .iter()
                .filter(|s| state.replay.joined.contains_key(s))
                .all(|s| ready.contains(s));
            let close = || {
                if plan.bounded {
                    Decision::Screen
                } else {
                    Decision::Close
                }
            };
            if let Some(cohort) = state.replay.current().and_then(|r| r.redo.as_ref()) {
                return if ready == *cohort {
                    close()
                } else if past(plan.timeouts.collect_secs) {
                    Decision::Abort("a site did not come back for the redo".into())
                } else {
                    Decision::Wait
                };
            }
            if everyone && ready.len() >= min {
                close()
            } else if past(plan.timeouts.collect_secs) {
                if ready.len() >= min {
                    close()
                } else {
                    Decision::Abort(format!(
                        "{} sites finished, the plan needs {min}",
                        ready.len()
                    ))
                }
            } else {
                Decision::Wait
            }
        }
        Phase::Screening => {
            let checked = state.replay.current().is_some_and(|r| {
                r.checks.len() == plan.helpers.len()
                    && r.checks.iter().all(|(helper, want)| {
                        let blob = Blob::Check {
                            round: r.round,
                            helper: *helper as u32,
                        };
                        shared.holds(blob, want)
                    })
            });
            if checked {
                Decision::Close
            } else if past(plan.timeouts.collect_secs) {
                Decision::Abort("a helper did not check the norm proofs".into())
            } else {
                Decision::Wait
            }
        }
        Phase::Masking => {
            let done = state
                .replay
                .current()
                .is_some_and(|r| r.masks.len() == r.active.len());
            if done {
                Decision::Outcome
            } else if past(plan.timeouts.collect_secs) {
                let long_past = past(plan.timeouts.collect_secs * 2);
                helper_late(state, long_past, "a helper did not deliver its mask total")
            } else {
                Decision::Wait
            }
        }
        Phase::Attesting => {
            let Some(current) = state.replay.current() else {
                return Decision::Stop;
            };
            let have_prints = state
                .per_site
                .as_ref()
                .is_some_and(|(round, _)| *round == current.round);
            if !have_prints {
                return Decision::Fingerprints;
            }
            let helpers_in = current.helper_attests.len() == current.active.len();
            let silent: Vec<&str> = current
                .included
                .iter()
                .filter(|s| !current.site_attests.contains_key(s))
                .map(|s| plan.sites[*s].as_str())
                .collect();
            let attest_for = Duration::from_secs(plan.timeouts.attest_secs);
            if helpers_in && silent.is_empty() {
                Decision::Seal
            } else if helpers_in && sites_given > attest_for {
                match state.replay.retry_cohort(plan) {
                    Ok(_) => Decision::Retry,
                    Err(_) => {
                        Decision::Abort(format!("no valid attestation from {}", silent.join(", ")))
                    }
                }
            } else if !helpers_in && past(plan.timeouts.attest_secs) {
                let long_past = past(plan.timeouts.attest_secs * 2);
                helper_late(state, long_past, "a helper did not attest")
            } else {
                Decision::Wait
            }
        }
        Phase::Acking => {
            let silent: Vec<&str> = state
                .replay
                .current()
                .iter()
                .flat_map(|r| r.included.iter().filter(|s| !r.acks.contains(s)))
                .map(|s| plan.sites[*s].as_str())
                .collect();
            if silent.is_empty() {
                Decision::Next
            } else if past(plan.timeouts.ack_secs) {
                Decision::Abort(format!("no acknowledgement from {}", silent.join(", ")))
            } else {
                Decision::Wait
            }
        }
    }
}

/// A helper is past its deadline. If the run can lose it, it gets longer than
/// that, since a removal is for good.
fn helper_late(state: &State, long_past: bool, reason: &str) -> Decision {
    let plan = state.replay.plan();
    let owing = plan.and_then(|plan| state.replay.removable(plan).ok());
    match owing.as_ref().and_then(|owing| owing.first()) {
        Some(helper) if long_past => Decision::RemoveHelper(*helper),
        Some(_) => Decision::Wait,
        None => Decision::Abort(reason.to_owned()),
    }
}

async fn drive(shared: &Arc<Shared>) -> Result<(), NodeError> {
    let mut clock = (Phase::Empty, 0u32, Instant::now());
    let mut helpers_in: Option<(u32, Instant)> = None;
    loop {
        let decision = {
            let state = shared.lock();
            if state.log_failed {
                return Err(NodeError::Check("the log can no longer be written".into()));
            }
            let key = (state.replay.phase, state.replay.round_number());
            if (clock.0, clock.1) != key {
                clock = (key.0, key.1, Instant::now());
            }
            let current = state.replay.current();
            let all_in = current.filter(|r| {
                state.replay.phase == Phase::Attesting && r.helper_attests.len() == r.active.len()
            });
            helpers_in = all_in.map(|r| match helpers_in {
                Some((round, since)) if round == r.round => (round, since),
                _ => (r.round, Instant::now()),
            });
            let sites_given = helpers_in.map_or(Duration::ZERO, |(_, since)| since.elapsed());
            decide(shared, &state, clock.2.elapsed(), sites_given)
        };
        match decision {
            Decision::Wait => {
                let _ = tokio::time::timeout(TICK, shared.wake.notified()).await;
            }
            Decision::Stop => return Ok(()),
            Decision::Abort(reason) => abort(shared, reason)?,
            Decision::Open => open_round(shared)?,
            Decision::Screen => screen_round(shared)?,
            Decision::Close => close_round(shared).await?,
            // What its signers committed to is gone, and they won't send it again.
            Decision::Outcome => match outcome_round(shared).await {
                Err(NodeError::Check(reason)) => abort(shared, reason)?,
                done => done?,
            },
            Decision::Fingerprints => fingerprints(shared).await?,
            Decision::Seal => seal_round(shared)?,
            Decision::Retry => retry_round(shared)?,
            Decision::RemoveHelper(helper) => remove_helper(shared, helper)?,
            Decision::Next => next_round(shared)?,
        }
    }
}

/// Gives up this round number and starts the next one without whoever went
/// silent. The old round's files stay until a seal, since a slow site may still
/// be reading them.
fn retry_round(shared: &Shared) -> Result<(), NodeError> {
    let mut state = shared.lock();
    // A late attestation may have got in since the decision.
    if state.replay.retry_cohort(&shared.plan).is_err() {
        return Ok(());
    }
    let round = state.replay.round_number();
    tracing::warn!("round {round}: a site went silent, starting over without it");
    let body = Body::Retry {
        plan: shared.plan_id,
        round,
        prev: state.replay.head,
    };
    let record = Record::sign(&shared.identity, &body)?;
    shared.apply(&mut state, &record)?;
    state.uploads.clear();
    state.manifests.clear();
    refuse_attests(&mut state, "the round was given up");
    Ok(())
}

/// Every attestation gets an answer, since its site is waiting for one.
fn refuse_attests(state: &mut State, reason: &str) {
    for pending in std::mem::take(&mut state.attests) {
        if let Some(p) = state.peers.get(&pending.peer) {
            let reason = reason.to_owned();
            let _ = p.control.send(Msg::Refused { reason });
        }
    }
}

/// Takes a silent helper out of the run. The round starts over, and sites mask
/// the same updates again without it.
fn remove_helper(shared: &Shared, helper: usize) -> Result<(), NodeError> {
    let mut state = shared.lock();
    let owing = state.replay.removable(&shared.plan);
    if !owing.is_ok_and(|owing| owing.contains(&helper)) {
        return Ok(());
    }
    let round = state.replay.round_number();
    tracing::warn!(
        "round {round}: removing silent helper {}",
        shared.plan.helpers[helper].did
    );
    let body = Body::RemoveHelper {
        plan: shared.plan_id,
        round,
        helper: helper as u32,
        prev: state.replay.head,
    };
    let record = Record::sign(&shared.identity, &body)?;
    shared.apply(&mut state, &record)?;
    state.uploads.clear();
    state.manifests.clear();
    refuse_attests(&mut state, "the round was given up");
    Ok(())
}

fn abort(shared: &Shared, reason: String) -> Result<(), NodeError> {
    tracing::warn!("aborting: {reason}");
    let plan = shared.plan_id;
    shared.append_own(|replay| Body::Abort {
        plan,
        round: replay.round_number(),
        reason,
    })
}

/// Adds any revocation list the log lacks. One that the rules refuse, such as
/// a copy a site got in first, is only logged here.
fn relay_lists(shared: &Shared) {
    let Some(dir) = &shared.status_dir else {
        return;
    };
    let lists = newer_lists(dir, &shared.lock().replay);
    for list in lists {
        let plan = shared.plan_id;
        if let Err(e) = shared.append_own(|_| Body::Status { plan, list }) {
            tracing::warn!("revocation list not added: {e}");
        }
    }
}

fn open_round(shared: &Shared) -> Result<(), NodeError> {
    relay_lists(shared);
    if shared.lock().replay.phase == Phase::Aborted {
        return Ok(());
    }
    let (spent, round) = {
        let state = shared.lock();
        (state.replay.spent, state.replay.round_number() + 1)
    };
    if let Some(privacy) = &shared.plan.privacy {
        let left = privacy.budget.saturating_sub(spent);
        let fits = shared.plan.dearest_round().is_some_and(|cost| cost <= left);
        if !fits {
            return abort(
                shared,
                format!("the privacy budget cannot cover round {round}"),
            );
        }
    }
    let plan = shared.plan_id;
    shared.append_own(|replay| Body::Open {
        plan,
        round: replay.round_number() + 1,
        prev: replay.head,
        base: replay.model,
    })
}

fn next_round(shared: &Arc<Shared>) -> Result<(), NodeError> {
    let (round, from, step) = {
        let state = shared.lock();
        let current = state.replay.current();
        let (from, step) = current.map_or((0, 0), |r| (r.from, r.step));
        (state.replay.round_number(), from, step)
    };
    if !shared.keep_spool {
        for spent in from + 1..=round {
            shared.dirs.drop_round(spent);
        }
    }
    if from > 0 {
        let _ = std::fs::remove_file(shared.dirs.model(from));
        let _ = std::fs::remove_file(shared.dirs.moments(from));
    }
    for given_up in from + 1..round {
        let _ = std::fs::remove_file(shared.dirs.model(given_up));
        let _ = std::fs::remove_file(shared.dirs.moments(given_up));
    }
    if step >= shared.plan.rounds {
        let plan = shared.plan_id;
        return shared.append_own(|replay| Body::Finish {
            plan,
            prev: replay.head,
        });
    }
    {
        let mut state = shared.lock();
        state.uploads.clear();
        state.manifests.clear();
        state.attests.clear();
    }
    open_round(shared)
}

/// Puts the finished contributions up for their norm proofs to be checked.
fn screen_round(shared: &Arc<Shared>) -> Result<(), NodeError> {
    let (round, candidates) = {
        let state = shared.lock();
        let current = state
            .replay
            .current()
            .ok_or_else(|| NodeError::Protocol("no round".into()))?;
        let candidates: Vec<Included> = ready_sites(shared, &state)
            .iter()
            .map(|s| Included {
                site: shared.plan.sites[*s].clone(),
                contribution: current.contributions[s].hash,
            })
            .collect();
        (current.round, candidates)
    };
    let plan = shared.plan_id;
    shared.append_own(|replay| Body::Screen {
        plan,
        round,
        prev: replay.head,
        candidates,
    })
}

/// Checks every screened proof. Returns the sites whose proofs hold and the
/// digest of the aggregator's own messages, which it keeps for auditors.
async fn check_proofs(
    shared: &Arc<Shared>,
    current: &RoundState,
    params: Params,
) -> Result<(Vec<usize>, Digest), NodeError> {
    let round = current.round;
    let helpers = (0..shared.plan.helpers.len())
        .map(|helper| {
            std::fs::read(shared.blob_path(Blob::Check {
                round,
                helper: helper as u32,
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let jobs: Vec<_> = current
        .screened
        .iter()
        .map(|site| {
            let contribution = &current.contributions[site];
            let index = *site as u32;
            (
                *site,
                norm::joint(
                    &shared.plan_id,
                    round,
                    index,
                    &contribution.masked,
                    &contribution.sealed,
                ),
                Challenge::derive(&current.screen_head, index),
                shared.upload_path(Object::Upload { round, site: index }),
                shared.blob_path(Blob::Proof { round, site: index }),
            )
        })
        .collect();
    let out = shared.blob_path(Blob::Screened { round });
    let hook = shared.hook.clone();
    blocking(move || {
        let size = params.share_bytes();
        let mut own = Vec::with_capacity(jobs.len() * size);
        for (_, joint, challenge, masked, proof) in &jobs {
            let proof = std::fs::read(proof)?;
            let share = norm::leader_share(&params, joint, challenge, masked, &proof)?;
            own.extend(share.unwrap_or_else(|_| vec![0; size]));
        }
        write_atomic(&out, &own)?;
        if let (Some(hook), Some(dir)) = (&hook, out.parent()) {
            hook(Stage::BeforeVerdict { round }, dir);
            own = std::fs::read(&out)?;
        }
        let mut proven = Vec::new();
        for (position, (site, ..)) in jobs.iter().enumerate() {
            if norm::accepted(&params, position, &own, &helpers) {
                proven.push(*site);
            } else {
                tracing::warn!("site {site} did not prove its update short, leaving it out");
            }
        }
        Ok((proven, Digest::of(&own)))
    })
    .await
}

async fn close_round(shared: &Arc<Shared>) -> Result<(), NodeError> {
    let (current, mut ready) = {
        let state = shared.lock();
        let current = state
            .replay
            .current()
            .cloned()
            .ok_or_else(|| NodeError::Protocol("no round".into()))?;
        (current, ready_sites(shared, &state))
    };
    let round = current.round;
    let mut shares = None;
    if let Some(params) = shared.plan.norm() {
        let (proven, digest) = check_proofs(shared, &current, params).await?;
        ready = proven;
        shares = Some(digest);
    }
    let weights: Vec<u64> = ready
        .iter()
        .map(|s| current.contributions[s].weight)
        .collect();
    let dir = shared.dirs.round(round)?;
    if let Some(hook) = &shared.hook {
        hook(Stage::BeforeSum { round }, &dir);
    }

    let paths: Vec<PathBuf> = ready
        .iter()
        .map(|site| {
            shared.upload_path(Object::Upload {
                round,
                site: *site as u32,
            })
        })
        .collect();
    let (plan, shape) = (shared.plan.clone(), shared.shape);
    let sum_path = dir.join("sum.bin");
    let task_ready = ready.clone();
    let (kept, manifest) = blocking(move || {
        let kept: Vec<usize> = if plan.is_masked() {
            task_ready.clone()
        } else {
            let plain = plan.layout.coords();
            let mut kept = Vec::new();
            for ((site, path), weight) in task_ready.iter().zip(&paths).zip(&weights) {
                if model::in_range(path, shape, plain, plan.encoding.bound(*weight))? {
                    kept.push(*site);
                } else {
                    tracing::warn!("site {site} sent coordinates out of range, leaving it out");
                }
            }
            kept
        };
        let chosen: Vec<&Path> = task_ready
            .iter()
            .zip(&paths)
            .filter(|(site, _)| kept.contains(site))
            .map(|(_, path)| path.as_path())
            .collect();
        let manifest = vector::combine_unsynced(&sum_path, shape, &chosen)?;
        Ok((kept, manifest))
    })
    .await?;

    if kept.len() < shared.plan.min_cohort as usize {
        return abort(
            shared,
            format!(
                "{} sites usable, the plan needs {}",
                kept.len(),
                shared.plan.min_cohort
            ),
        );
    }
    let plan_id = shared.plan_id;
    let sites = shared.plan.sites.clone();
    let contributions = shared
        .lock()
        .replay
        .current()
        .map(|r| r.contributions.clone())
        .unwrap_or_default();
    let included = kept
        .iter()
        .map(|s| Included {
            site: sites[*s].clone(),
            contribution: contributions[s].hash,
        })
        .collect();
    shared.append_own(|replay| Body::Close {
        plan: plan_id,
        round,
        prev: replay.head,
        included,
        sum: manifest.root,
        shares,
    })
}

/// Rebuilds a derived vector if a crash left it missing or partial.
fn ensure_vector(
    path: &Path,
    shape: Shape,
    expected: Digest,
    build: impl FnOnce() -> Result<Manifest, NodeError>,
) -> Result<(), NodeError> {
    let intact =
        path.exists() && vector::manifest_of(path, shape).is_ok_and(|m| m.root == expected);
    if intact {
        return Ok(());
    }
    let manifest = build()?;
    if manifest.root != expected {
        return Err(NodeError::Check(format!(
            "{} no longer matches the digest in the log",
            path.display()
        )));
    }
    Ok(())
}

/// A stored vector is what its signer committed to, or nothing may come of it.
fn check_stored(path: &Path, shape: Shape, expected: Digest) -> Result<(), NodeError> {
    match vector::manifest_of(path, shape).map(|manifest| manifest.root) {
        Ok(root) if root == expected => Ok(()),
        _ => Err(NodeError::Check(format!(
            "{} no longer matches the digest in the log",
            path.display()
        ))),
    }
}

async fn outcome_round(shared: &Arc<Shared>) -> Result<(), NodeError> {
    let (round, state_copy) = {
        let state = shared.lock();
        let current = state
            .replay
            .current()
            .cloned()
            .ok_or_else(|| NodeError::Protocol("no round".into()))?;
        (current.round, current)
    };
    let dir = shared.dirs.round(round)?;
    let shape = shared.shape;
    let sum_path = dir.join("sum.bin");
    let inputs: Vec<PathBuf> = state_copy
        .included
        .iter()
        .map(|s| {
            shared.upload_path(Object::Upload {
                round,
                site: *s as u32,
            })
        })
        .collect();
    let expected = state_copy.sum;
    let sum_for_build = sum_path.clone();
    blocking(move || {
        ensure_vector(&sum_for_build, shape, expected, || {
            let refs: Vec<&Path> = inputs.iter().map(PathBuf::as_path).collect();
            Ok(vector::combine_unsynced(&sum_for_build, shape, &refs)?)
        })
    })
    .await?;

    let masks: Vec<PathBuf> = state_copy
        .active
        .iter()
        .map(|h| {
            shared.upload_path(Object::Mask {
                round,
                helper: *h as u32,
            })
        })
        .collect();
    let held: Vec<Digest> = state_copy
        .active
        .iter()
        .map(|h| state_copy.masks[h])
        .collect();
    let stored = masks.clone();
    blocking(move || {
        for (path, expected) in stored.iter().zip(held) {
            check_stored(path, shape, expected)?;
        }
        Ok(())
    })
    .await?;
    let aggregate = dir.join("aggregate.bin");
    let next_model = shared.dirs.model(round);
    let base = shared.dirs.model(state_copy.from);
    let plan = shared.plan.clone();
    let total_weight: u64 = state_copy
        .included
        .iter()
        .map(|s| state_copy.contributions[s].weight)
        .sum();
    let hook = shared.hook.clone();
    let hook_dir = dir.clone();
    let from = state_copy.from;
    let before_moments = (from > 0).then(|| shared.dirs.moments(from));
    let next_moments = shared.dirs.moments(round);
    let (root, model_digest, manifest, moments) = blocking(move || {
        let refs: Vec<&Path> = masks.iter().map(PathBuf::as_path).collect();
        let manifest = vector::combine(&aggregate, shape, &[&sum_path], &refs)?;
        let manifest = match hook {
            Some(hook) => {
                hook(Stage::BeforeOutcome { round }, &hook_dir);
                vector::manifest_of(&aggregate, shape)?
            }
            None => manifest,
        };
        let mut moments = None;
        match plan.server_adam {
            Some(params) => {
                let before = before_moments.as_deref();
                let adam = model::AdamStep {
                    params,
                    moments: before,
                    moments_out: &next_moments,
                };
                model::apply_aggregate_adam(
                    &plan.layout,
                    plan.encoding,
                    plan.chunk_coords,
                    &base,
                    &aggregate,
                    total_weight,
                    &adam,
                    &next_model,
                )?;
                File::open(&next_moments)?.sync_data()?;
                moments = Some(Digest::file(&next_moments)?);
            }
            None => model::apply_aggregate(
                &plan.layout,
                plan.encoding,
                plan.chunk_coords,
                &base,
                &aggregate,
                total_weight,
                &next_model,
            )?,
        }
        // Sites fetch these once the Outcome is logged, and they have only one source.
        File::open(&next_model)?.sync_data()?;
        sync_parent(&next_model)?;
        sync_parent(&aggregate)?;
        Ok((manifest.root, Digest::file(&next_model)?, manifest, moments))
    })
    .await?;
    shared
        .lock()
        .manifests
        .insert(Object::Aggregate { round }, manifest);

    let mut groups = Vec::new();
    for (group, members) in shared
        .plan
        .groups(&state_copy.close_head, state_copy.included.len())
        .into_iter()
        .enumerate()
    {
        let group = group as u32;
        let add: Vec<PathBuf> = members
            .iter()
            .map(|p| {
                shared.upload_path(Object::Upload {
                    round,
                    site: state_copy.included[*p] as u32,
                })
            })
            .collect();
        let sub: Vec<PathBuf> = (0..shared.plan.helpers.len())
            .map(|helper| {
                shared.upload_path(Object::GroupMask {
                    round,
                    helper: helper as u32,
                    group,
                })
            })
            .collect();
        let held: Vec<Digest> = (0..sub.len())
            .map(|helper| state_copy.group_masks[&helper][group as usize])
            .collect();
        let object = Object::GroupAggregate { round, group };
        let out = shared.upload_path(object);
        let manifest = blocking(move || {
            for (path, expected) in sub.iter().zip(held) {
                check_stored(path, shape, expected)?;
            }
            let add: Vec<&Path> = add.iter().map(PathBuf::as_path).collect();
            let sub: Vec<&Path> = sub.iter().map(PathBuf::as_path).collect();
            let manifest = vector::combine(&out, shape, &add, &sub)?;
            sync_parent(&out)?;
            Ok(manifest)
        })
        .await?;
        groups.push(manifest.root);
        shared.lock().manifests.insert(object, manifest);
    }
    let plan_id = shared.plan_id;
    shared.append_own(|replay| Body::Outcome {
        plan: plan_id,
        round,
        prev: replay.head,
        total_weight,
        aggregate: root,
        model: model_digest,
        groups,
        moments,
    })
}

/// Each scored group's share of the round's reward, from the group sums.
async fn score_groups(
    shared: &Arc<Shared>,
    round: u32,
    close_head: &Digest,
    cohort: usize,
) -> Result<Vec<u32>, NodeError> {
    let groups = shared.plan.groups(close_head, cohort).len() as u32;
    if groups == 0 {
        return Ok(Vec::new());
    }
    let total = shared.upload_path(Object::Aggregate { round });
    let sums: Vec<PathBuf> = (0..groups)
        .map(|group| shared.upload_path(Object::GroupAggregate { round, group }))
        .collect();
    let (plan, shape) = (shared.plan.clone(), shared.shape);
    blocking(move || {
        let plain = plan.layout.coords();
        let alignments = sums
            .iter()
            .map(|sum| scoring::alignment(sum, &total, shape, plain))
            .collect::<Result<Vec<i128>, _>>()?;
        Ok(scoring::shares(&alignments))
    })
    .await
}

async fn fingerprints(shared: &Arc<Shared>) -> Result<(), NodeError> {
    let close_head;
    let (round, included, head) = {
        let state = shared.lock();
        let current = state
            .replay
            .current()
            .ok_or_else(|| NodeError::Protocol("no round".into()))?;
        let head = current
            .outcome
            .as_ref()
            .map(|o| o.challenge)
            .ok_or_else(|| NodeError::Protocol("no outcome".into()))?;
        close_head = current.close_head;
        (current.round, current.included.clone(), head)
    };
    let paths: Vec<PathBuf> = included
        .iter()
        .map(|s| {
            shared.upload_path(Object::Upload {
                round,
                site: *s as u32,
            })
        })
        .collect();
    let shape = shared.shape;
    let rho = challenge(&head);
    let per_site = blocking(move || {
        paths
            .iter()
            .map(|p| Ok(vector::fingerprint_file(p, shape, rho)?))
            .collect::<Result<Vec<Fp2>, NodeError>>()
    })
    .await?;
    let shares = score_groups(shared, round, &close_head, included.len()).await?;
    let mut state = shared.lock();
    state.shares = shares;
    state.per_site = Some((round, per_site));
    flush_attests(shared, &mut state);
    shared.wake.notify_waiters();
    Ok(())
}

fn seal_round(shared: &Arc<Shared>) -> Result<(), NodeError> {
    if let Some(hook) = &shared.hook {
        let round = shared.lock().replay.round_number();
        hook(Stage::BeforeSeal { round }, &shared.dirs.round_path(round));
    }
    let mut state = shared.lock();
    let state = &mut *state;
    let (round, per_site) = state
        .per_site
        .clone()
        .ok_or_else(|| NodeError::Protocol("no fingerprints".into()))?;
    let current = state
        .replay
        .current()
        .ok_or_else(|| NodeError::Protocol("no round".into()))?;
    let sum: Fp2 = per_site.iter().copied().sum();
    let masks: Fp2 = current.helper_attests.values().map(|a| a.total).sum();
    let body = Body::Seal {
        plan: shared.plan_id,
        round,
        prev: state.replay.head,
        per_site,
        sum,
        aggregate: sum - masks,
        shares: state.shares.clone(),
    };
    refuse_attests(state, "the round was sealed before this attestation agreed");
    let record = Record::sign(&shared.identity, &body)?;
    shared.apply(state, &record)
}
