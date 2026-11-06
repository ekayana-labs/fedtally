//! A site trains on its own data, masks the result, and refuses to acknowledge
//! a round unless the numbers add up over the bytes it holds.
//!
//! A site never receives code from the aggregator. It runs the trainer it was
//! started with.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fedtally_core::admission;
use fedtally_core::credential::Credential;
use fedtally_core::field::{Fp, Fp2, P};
use fedtally_core::keys::{Identity, random_bytes, seal};
use fedtally_core::model::{
    self, EncodeRequest, Encoded, ModelError, PAD, PrivateRequest, Remask, TensorView,
};
use fedtally_core::norm::{self, Challenge, Params};
use fedtally_core::pq_ucan::Cid;
use fedtally_core::pq_ucan::did::Did;
use fedtally_core::pq_ucan::time::Timestamp;
use fedtally_core::pq_ucan::validate::RevocationStore;
use fedtally_core::privacy::Privacy;
use fedtally_core::record::{Body, encode_token};
use fedtally_core::replay::{Phase, RoundState};
use fedtally_core::scoring;
use fedtally_core::vector::{self, Digest, Manifest, challenge};
use serde::{Deserialize, Serialize};

use crate::follower::{Follower, Patience, Start};
use crate::helper::seal_context;
use crate::net::Client;
use crate::registry::{self, Registry, Unsure};
use crate::wire::{Blob, Msg, Object};
use crate::{NodeError, blocking, newer_lists, sync_parent, write_durable};

pub struct TrainContext {
    pub round: u32,
    pub base_model: PathBuf,
}

/// Encodes trained tensors and their weight straight to disk.
pub type Submit<'a> = dyn FnMut(&[TensorView<'_>], u64) -> Result<Encoded, ModelError> + 'a;

/// Local training. `submit` can only be called once.
pub trait Trainer: Send + 'static {
    fn train(&mut self, ctx: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String>;
}

/// Delegation CIDs a site has taken back, one per line. Read fresh each time,
/// and an unreadable file counts as everything being revoked.
pub struct RevocationFile(pub PathBuf);

impl RevocationStore for RevocationFile {
    fn is_revoked(&self, cid: &Cid, _issuers: &[&Did]) -> bool {
        match std::fs::read_to_string(&self.0) {
            Ok(text) => text.lines().any(|line| line.trim() == cid.to_string()),
            Err(e) => e.kind() != std::io::ErrorKind::NotFound,
        }
    }
}

pub struct SiteConfig {
    pub identity: Arc<Identity>,
    pub plan_line: Vec<u8>,
    pub addr: String,
    pub dir: PathBuf,
    pub invocation: Vec<u8>,
    pub proofs: Vec<Vec<u8>>,
    /// Delegations this site has taken back. Checked before every round.
    pub revocations: Option<Arc<dyn RevocationStore + Send + Sync>>,
    pub give_up_after: Duration,
    /// What this site holds for a plan that admits on credentials.
    pub standing: Standing,
    /// Where to ask whether `did:bio` operators and issuers still stand.
    pub registry: Option<Arc<dyn Registry>>,
    /// Terms to encode with in place of the plan's, without the checks an
    /// honest site makes on itself. Only for playing a dishonest site.
    pub cheat: Option<Privacy>,
}

/// A site's dataset and credentials, and where it keeps the issuers' newest
/// revocation lists. It brings any list the log lacks before each round.
#[derive(Clone, Debug, Default)]
pub struct Standing {
    pub dataset: Option<String>,
    pub credentials: Vec<String>,
    pub status_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct SiteSummary {
    pub final_model: PathBuf,
    pub final_digest: Digest,
    pub rounds_contributed: Vec<u32>,
    pub rounds_refused: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct RoundFile {
    seeds: Vec<String>,
    weight: u64,
    manifest: Manifest,
    sealed: Vec<Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    proof: Option<Digest>,
}

struct Site {
    f: Follower,
    me: usize,
    invocation: Vec<u8>,
    proofs: Vec<Vec<u8>>,
    revocations: Option<Arc<dyn RevocationStore + Send + Sync>>,
    standing: Standing,
    held: Vec<Credential>,
    registry: Option<Arc<dyn Registry>>,
    cheat: Option<Privacy>,
    trainer: Option<Box<dyn Trainer>>,
    refused: Vec<u32>,
    checks: Option<Checks>,
    uploaded: Option<u32>,
}

#[derive(Clone)]
struct Checks {
    round: u32,
    masked: Fp2,
    aggregate: Fp2,
    audits: Vec<(usize, Fp2)>,
}

pub async fn run_site(
    config: SiteConfig,
    trainer: Box<dyn Trainer>,
) -> Result<SiteSummary, NodeError> {
    let f = Follower::open(Start {
        identity: config.identity,
        plan_line: config.plan_line,
        addr: config.addr,
        dir: config.dir,
        give_up_after: config.give_up_after,
    })?;
    let did = f.identity.did_string();
    let me = f
        .plan
        .site_index(&did)
        .ok_or_else(|| NodeError::Refused("this key is not a site in the plan".into()))?;
    let held = config
        .standing
        .credentials
        .iter()
        .map(|text| Credential::parse(text))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| NodeError::Refused(format!("a credential of this site: {e}")))?;
    let mut site = Site {
        f,
        me,
        invocation: config.invocation,
        proofs: config.proofs,
        revocations: config.revocations,
        standing: config.standing,
        held,
        registry: config.registry,
        cheat: config.cheat,
        trainer: Some(trainer),
        refused: Vec::new(),
        checks: None,
        uploaded: None,
    };
    site.admit()?;
    let mut patience = Patience::new(config.give_up_after);
    loop {
        match site.session().await {
            Ok(()) => return site.summary(),
            Err(e) if e.is_transient() && patience.allows(site.f.replay.entries) => {
                tracing::info!("site reconnecting: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

fn now() -> Result<Timestamp, NodeError> {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| NodeError::Protocol(e.to_string()))?
        .as_secs() as i64;
    Timestamp::from_unix(secs).map_err(|e| NodeError::Protocol(e.to_string()))
}

fn random_field() -> Fp {
    loop {
        let v = u64::from_le_bytes(random_bytes::<8>());
        if v < P {
            return Fp::new(v);
        }
    }
}

impl Site {
    /// Checks this site's own grant. Run before joining and before each round,
    /// so an expiry or a revocation takes effect at the next round.
    fn admit(&self) -> Result<(), NodeError> {
        let did = Did::parse(&self.f.identity.did_string())
            .map_err(|e| NodeError::Protocol(e.to_string()))?;
        let store = self
            .revocations
            .as_ref()
            .map(|r| r.as_ref() as &dyn RevocationStore);
        admission::check(
            &did,
            &self.f.plan_id,
            &self.f.plan,
            &self.invocation,
            &self.proofs,
            now()?,
            store,
        )
        .map_err(|e| NodeError::Refused(e.to_string()))?;
        if let Some(state) = self
            .f
            .replay
            .current()
            .filter(|_| !self.f.replay.removed.is_empty())
        {
            admission::grants_cover_helpers(
                &self.f.plan_id,
                &self.f.plan,
                &self.proofs,
                &state.active,
            )
            .map_err(|e| NodeError::Refused(e.to_string()))?;
        }
        Ok(())
    }

    /// Whether this site's credentials still meet the plan's policy, by the
    /// lists in the log and this site's own clock.
    fn in_standing(&self) -> Result<(), NodeError> {
        if let Some(why) = self.f.replay.revoked.get(&self.me) {
            return Err(NodeError::Refused(why.clone()));
        }
        self.f
            .replay
            .site_meets(
                &self.f.plan,
                &self.f.identity.did_string(),
                self.standing.dataset.as_deref(),
                &self.held,
                now()?.as_unix(),
            )
            .map_err(NodeError::Refused)
    }

    /// Asks the registry about every `did:bio` party this site relies on.
    /// A site that can't reach the registry sits a round out, except a redo,
    /// which can't close without it. That one it keeps trying.
    async fn registry_agrees(&mut self, redo: bool) -> Result<(), NodeError> {
        let Some(registry) = self.registry.clone() else {
            return Ok(());
        };
        let plan = self.f.plan.clone();
        let removed = self.f.replay.removed.clone();
        blocking(
            move || match registry::check(registry.as_ref(), &plan, &removed) {
                Ok(()) => Ok(()),
                Err(Unsure::Unreachable(why)) if redo => Err(NodeError::Disconnected(why)),
                Err(e) => Err(NodeError::Refused(e.to_string())),
            },
        )
        .await
    }

    /// Brings the log any revocation list this site holds a newer copy of.
    async fn relay_lists(&mut self, client: &mut Client) -> Result<(), NodeError> {
        let Some(dir) = self.standing.status_dir.clone() else {
            return Ok(());
        };
        let offered = newer_lists(&dir, &self.f.replay);
        for list in &offered {
            let body = Body::Status {
                plan: self.f.plan_id,
                list: list.clone(),
            };
            self.f.offer(client, &body).await?;
        }
        // An aggregator that won't log a revocation doesn't get this site's data.
        // A list that turned up after the offers waits for the next round.
        let missing = newer_lists(&dir, &self.f.replay);
        if offered.iter().all(|list| !missing.contains(list)) {
            Ok(())
        } else {
            Err(NodeError::Refused(
                "the log lacks a revocation list this site holds".into(),
            ))
        }
    }

    async fn session(&mut self) -> Result<(), NodeError> {
        // Back after the end with nothing missing: the aggregator may be long gone.
        if self.f.replay.phase == Phase::Done {
            let last = self.f.dirs.model(self.f.replay.round_number());
            if last.exists() && Digest::file(&last)? == self.f.replay.model {
                self.tidy_given_up();
                return Ok(());
            }
        }
        let mut client = self.f.connect().await?;
        self.uploaded = None;
        let mut wait = true;
        loop {
            if wait || client.has_backlog() {
                let batch = client.entries().await?;
                self.f.absorb(batch)?;
            }
            let before = self.f.replay.entries;
            match self.f.replay.phase {
                Phase::Done => {
                    self.settle(&mut client).await?;
                    return Ok(());
                }
                Phase::Aborted => {
                    return Err(NodeError::Aborted(
                        self.f.replay.abort.clone().unwrap_or_default(),
                    ));
                }
                Phase::Empty => {}
                _ => {
                    self.join(&mut client).await?;
                    self.step(&mut client).await?;
                }
            }
            wait = self.f.replay.entries == before;
        }
    }

    async fn join(&mut self, client: &mut Client) -> Result<(), NodeError> {
        if self.f.replay.joined.contains_key(&self.me)
            || !matches!(
                self.f.replay.phase,
                Phase::Joining | Phase::Collecting | Phase::Acking
            )
        {
            return Ok(());
        }
        self.relay_lists(client).await?;
        let body = Body::Join {
            plan: self.f.plan_id,
            at: now()?.as_unix(),
            invocation: encode_token(&self.invocation),
            proofs: self.proofs.iter().map(|p| encode_token(p)).collect(),
            dataset: self.standing.dataset.clone(),
            credentials: self.standing.credentials.clone(),
        };
        self.f.append(client, &body).await
    }

    async fn step(&mut self, client: &mut Client) -> Result<(), NodeError> {
        let Some(state) = self.f.replay.current().cloned() else {
            return Ok(());
        };
        // Every phase, as a batch of the log can carry a site past Collecting.
        self.tidy(&state);
        match self.f.replay.phase {
            Phase::Collecting => self.collect(client, &state).await,
            Phase::Attesting => self.attest(client, &state).await,
            Phase::Acking => self.acknowledge(client, &state).await,
            _ => Ok(()),
        }
    }

    /// Makes sure the model the round starts from is on disk. Model 0 and any
    /// model this site missed come from the aggregator, checked against the log.
    async fn ensure_model(
        &mut self,
        client: &mut Client,
        round: u32,
        digest: Digest,
    ) -> Result<(), NodeError> {
        let path = self.f.dirs.model(round);
        if path.exists() && Digest::file(&path)? == digest {
            return Ok(());
        }
        let object = if round == 0 {
            Object::InitialModel
        } else {
            Object::Model { round }
        };
        self.f.fetch_file(client, object, digest, &path).await
    }

    /// Drops what no round will read again: the model before the one this
    /// round starts from, and whatever given up rounds left, seeds included.
    fn tidy(&self, state: &RoundState) {
        let rounds = &self.f.replay.rounds;
        let before = state
            .from
            .checked_sub(1)
            .and_then(|i| rounds.get(i as usize))
            .map_or(0, |sealed| sealed.from);
        if before > 0 {
            let _ = std::fs::remove_file(self.f.dirs.model(before));
            let _ = std::fs::remove_file(self.f.dirs.moments(before));
        }
        // A redo is built from the round before it, so that one stays until
        // the redo's own file is written.
        let in_redo = state.redo.as_ref().is_some_and(|c| c.contains(&self.me));
        let building = in_redo
            && !self
                .f
                .dirs
                .round_path(state.round)
                .join("round.json")
                .exists();
        let needed = |r: &&RoundState| building && r.round + 1 == state.round;
        for given_up in rounds.iter().filter(|r| r.voided && !needed(r)) {
            self.f.dirs.drop_round(given_up.round);
            let _ = std::fs::remove_file(self.f.dirs.model(given_up.round));
            let _ = std::fs::remove_file(self.f.dirs.moments(given_up.round));
        }
    }

    /// A site that sat out the round before fetches Adam's moments, checked
    /// against the log. The first round starts from nothing.
    async fn ensure_moments(
        &mut self,
        client: &mut Client,
        state: &RoundState,
    ) -> Result<(), NodeError> {
        if self.f.plan.server_adam.is_none() || state.from == 0 {
            return Ok(());
        }
        let sealed = self.f.replay.rounds.get(state.from as usize - 1);
        let digest = sealed
            .and_then(|r| r.outcome.as_ref())
            .and_then(|o| o.moments)
            .ok_or_else(|| NodeError::Protocol("no moments in the log".into()))?;
        let path = self.f.dirs.moments(state.from);
        if path.exists() && Digest::file(&path)? == digest {
            return Ok(());
        }
        let object = Object::Moments { round: state.from };
        self.f.fetch_file(client, object, digest, &path).await
    }

    async fn collect(&mut self, client: &mut Client, state: &RoundState) -> Result<(), NodeError> {
        let round = state.round;
        let round_dir = self.f.dirs.round(round)?;
        let meta = round_dir.join("round.json");
        let spool = round_dir.join("masked.bin");

        if !state.contributions.contains_key(&self.me) {
            let drawn = state.invited.as_ref().is_none_or(|i| i.contains(&self.me));
            if self.refused.contains(&round) || !drawn {
                return Ok(());
            }
            let mut admitted = self.relay_lists(client).await;
            if admitted.is_ok() {
                admitted = self.admit().and_then(|()| self.in_standing());
            }
            if admitted.is_ok() {
                admitted = self.registry_agrees(state.redo.is_some()).await;
            }
            if let Err(e) = admitted {
                if e.is_transient() {
                    return Err(e);
                }
                tracing::warn!("sitting out round {round}: {e}");
                self.refused.push(round);
                return Ok(());
            }
            self.ensure_model(client, state.from, state.base).await?;
            if !meta.exists() {
                if state.redo.is_some() {
                    if !self
                        .f
                        .dirs
                        .round_path(round - 1)
                        .join("round.json")
                        .exists()
                    {
                        tracing::warn!("round {round}: nothing left to redo it from");
                        self.refused.push(round);
                        return Ok(());
                    }
                    self.remask_round(state, &meta, &spool).await?;
                } else {
                    self.train_round(state, &meta, &spool).await?;
                }
            }
            let file: RoundFile = read_round_file(&meta)?;
            let body = Body::Contribute {
                plan: self.f.plan_id,
                round,
                base: state.base,
                masked: file.manifest.root,
                weight: file.weight,
                sealed: file.sealed.clone(),
                proof: file.proof,
            };
            if !self.f.offer(client, &body).await? {
                self.refused.push(round);
                return Ok(());
            }
        }
        if !meta.exists() || self.uploaded == Some(round) {
            return Ok(());
        }
        let file = read_round_file(&meta)?;
        for h in state.active.iter().copied() {
            let data = std::fs::read(round_dir.join(format!("sealed-{h}.bin")))?;
            client
                .send(&Msg::PutSealed {
                    round,
                    helper: h as u32,
                    data,
                })
                .await?;
        }
        if file.proof.is_some() {
            let blob = Blob::Proof {
                round,
                site: self.me as u32,
            };
            let data = std::fs::read(round_dir.join("proof.bin"))?;
            client.send(&Msg::PutBlob { blob, data }).await?;
        }
        let object = Object::Upload {
            round,
            site: self.me as u32,
        };
        self.f
            .upload(client, object, &spool, &file.manifest)
            .await?;
        self.uploaded = Some(round);
        Ok(())
    }

    /// Draws a seed for each active helper and seals it. The digests are in
    /// plan order, zero for a helper that has been removed.
    fn seal_seeds(&self, state: &RoundState) -> Result<(Vec<[u8; 32]>, Vec<Digest>), NodeError> {
        let plan = &self.f.plan;
        let did = self.f.identity.did_string();
        let round_dir = self.f.dirs.round(state.round)?;
        let mut seeds = Vec::with_capacity(state.active.len());
        let mut sealed = vec![Digest::default(); plan.helpers.len()];
        for h in state.active.iter().copied() {
            let helper = &plan.helpers[h];
            let seed = random_bytes::<32>();
            let context = seal_context(&self.f.plan_id, state.round, &did, &helper.did);
            let boxed = seal(&helper.seal, &context, &seed)?;
            sealed[h] = Digest::of(&boxed);
            write_durable(&round_dir.join(format!("sealed-{h}.bin")), &boxed)?;
            seeds.push(seed);
        }
        Ok((seeds, sealed))
    }

    /// Prepares a redo: the update from the round before, under new masks and a
    /// new pad. Nothing is trained, so its sum says nothing new.
    async fn remask_round(
        &mut self,
        state: &RoundState,
        meta: &Path,
        spool: &Path,
    ) -> Result<(), NodeError> {
        let old_dir = self.f.dirs.round_path(state.round - 1);
        let old = read_round_file(&old_dir.join("round.json"))?;
        let old_seeds = decode_seeds(&old)?;
        let (seeds, sealed) = self.seal_seeds(state)?;
        let plan = self.f.plan.clone();
        let (meta, spool) = (meta.to_owned(), spool.to_owned());
        blocking(move || {
            let from = old_dir.join("masked.bin");
            if vector::manifest_of(&from, plan.shape())?.root != old.manifest.root {
                return Err(NodeError::Check(
                    "the update to redo is not the one this site committed to".into(),
                ));
            }
            let manifest = model::remask(&Remask {
                layout: &plan.layout,
                chunk_coords: plan.chunk_coords,
                from: &from,
                old_seeds: &old_seeds,
                pad: [random_field(), random_field()],
                seeds: &seeds,
                out: &spool,
            })?;
            let file = RoundFile {
                seeds: seeds.iter().map(hex::encode).collect(),
                weight: old.weight,
                manifest,
                sealed,
                proof: None,
            };
            let text = serde_json::to_vec(&file).map_err(|e| NodeError::Protocol(e.to_string()))?;
            write_durable(&meta, &text)?;
            Ok(())
        })
        .await
    }

    async fn train_round(
        &mut self,
        state: &RoundState,
        meta: &Path,
        spool: &Path,
    ) -> Result<(), NodeError> {
        let (round, step) = (state.round, state.step);
        let plan = self.f.plan.clone();
        let base = self.f.dirs.model(state.from);
        let (seeds, sealed) = self.seal_seeds(state)?;
        let round_dir = self.f.dirs.round(round)?;

        let mut trainer = self
            .trainer
            .take()
            .ok_or_else(|| NodeError::Trainer("trainer lost after a failed round".into()))?;
        let (meta, spool) = (meta.to_owned(), spool.to_owned());
        let (plan_id, me, cheat) = (self.f.plan_id, self.me as u32, self.cheat);
        let proof_path = round_dir.join("proof.bin");
        let task = blocking(move || {
            let mut outcome: Option<(Encoded, u64)> = None;
            let mut proof = None;
            let terms = cheat.as_ref().or(plan.privacy.as_ref());
            let bound = plan.norm();
            let ctx = TrainContext {
                round: step,
                base_model: base.clone(),
            };
            let pad: [Fp; PAD as usize] = [random_field(), random_field()];
            let trained = trainer.train(&ctx, &mut |tensors, weight| {
                if outcome.is_some() {
                    return Err(ModelError::Mismatch("submit called twice".into()));
                }
                // A private run counts every site the same, whatever it trained on.
                let weight = if plan.privacy.is_some() { 1 } else { weight };
                if weight == 0 || weight > plan.max_weight {
                    return Err(ModelError::Mismatch(format!(
                        "weight {weight} is outside the plan's 1 to {}",
                        plan.max_weight
                    )));
                }
                let mut tries = 0;
                let encoded = loop {
                    let encoded = match terms {
                        Some(privacy) => model::encode_private(&PrivateRequest {
                            layout: &plan.layout,
                            frac_bits: plan.encoding.frac_bits,
                            chunk_coords: plan.chunk_coords,
                            base: &base,
                            local: tensors,
                            privacy,
                            pad,
                            seeds: &seeds,
                            out: &spool,
                        })?,
                        None => model::encode_update(&EncodeRequest {
                            layout: &plan.layout,
                            encoding: plan.encoding,
                            chunk_coords: plan.chunk_coords,
                            base: &base,
                            local: tensors,
                            weight,
                            pad,
                            seeds: &seeds,
                            out: &spool,
                        })?,
                    };
                    let Some(params) = &bound else {
                        break encoded;
                    };
                    let joint = norm::joint(&plan_id, round, me, &encoded.manifest.root, &sealed);
                    let made = match cheat {
                        Some(_) => Some(norm::prove_unchecked(params, &joint, &spool, &seeds)?),
                        None => norm::prove(params, &joint, &spool, &seeds)?,
                    };
                    if let Some(bytes) = made {
                        proof = Some(Digest::of(&bytes));
                        write_durable(&proof_path, &bytes)?;
                        break encoded;
                    }
                    // Fresh noise and seeds each time, and each try fails
                    // with negligible probability.
                    tries += 1;
                    if tries == 3 {
                        return Err(ModelError::Mismatch(
                            "the update does not fit the plan's norm bound".into(),
                        ));
                    }
                };
                outcome = Some((
                    Encoded {
                        manifest: encoded.manifest.clone(),
                        clipped: encoded.clipped,
                    },
                    weight,
                ));
                Ok(encoded)
            });
            trained.map_err(NodeError::Trainer)?;
            let (encoded, weight) = outcome
                .ok_or_else(|| NodeError::Trainer("the trainer submitted nothing".into()))?;
            if encoded.clipped > 0 && plan.privacy.is_none() {
                tracing::warn!("{} coordinates were clipped this round", encoded.clipped);
            }
            let file = RoundFile {
                seeds: seeds.iter().map(hex::encode).collect(),
                weight,
                manifest: encoded.manifest,
                sealed,
                proof,
            };
            let text = serde_json::to_vec(&file).map_err(|e| NodeError::Protocol(e.to_string()))?;
            write_durable(&meta, &text)?;
            Ok(trainer)
        });
        self.trainer = Some(task.await?);
        Ok(())
    }

    /// Works out everything this site will vouch for, over bytes it holds.
    /// Reused between attesting and acknowledging, and rebuilt after a restart.
    async fn checks(
        &mut self,
        client: &mut Client,
        state: &RoundState,
        position: usize,
    ) -> Result<Checks, NodeError> {
        let round = state.round;
        if let Some(checks) = self.checks.as_ref().filter(|c| c.round == round) {
            return Ok(checks.clone());
        }
        let outcome = state
            .outcome
            .clone()
            .ok_or_else(|| NodeError::Protocol("no outcome".into()))?;
        let round_dir = self.f.dirs.round(round)?;
        let file = read_round_file(&round_dir.join("round.json"))?;
        if state.contributions[&self.me].masked != file.manifest.root {
            return Err(NodeError::Check(
                "the log holds a different contribution from this site".into(),
            ));
        }

        let shape = self.f.shape();
        let rho = challenge(&outcome.challenge);
        let seeds = decode_seeds(&file)?;
        let spool = round_dir.join("masked.bin");
        let (masked, shares) = blocking(move || {
            let masked = vector::fingerprint_file(&spool, shape, rho)?;
            let shares = seeds
                .iter()
                .map(|seed| Ok(vector::fingerprint_mask(seed, shape, rho)?))
                .collect::<Result<Vec<_>, NodeError>>()?;
            Ok((masked, shares))
        })
        .await?;
        for (h, share) in state.active.iter().copied().zip(&shares) {
            if state.helper_attests[&h].per_site[position] != *share {
                return Err(NodeError::Check(format!(
                    "round {round}: helper {} claims a mask this site's seed did not produce",
                    self.f.plan.helpers[h].did
                )));
            }
        }

        let aggregate_path = round_dir.join("aggregate.bin");
        self.f
            .fetch_vector(
                client,
                Object::Aggregate { round },
                outcome.aggregate,
                &aggregate_path,
            )
            .await?;
        // Models are written without a sync, and a restart comes back here
        // without passing through collect.
        self.ensure_model(client, state.from, state.base).await?;
        self.ensure_moments(client, state).await?;
        let plan = self.f.plan.clone();
        let base = self.f.dirs.model(state.from);
        let next = self.f.dirs.model(round);
        let total_weight = outcome.total_weight;
        let from = state.from;
        let before_moments = self.f.dirs.moments(from);
        let next_moments = self.f.dirs.moments(round);
        let (aggregate, model_digest, moments) = blocking(move || {
            let aggregate = vector::fingerprint_file(&aggregate_path, shape, rho)?;
            let mut moments = None;
            match plan.server_adam {
                Some(params) => {
                    let adam = model::AdamStep {
                        params,
                        moments: (from > 0).then_some(before_moments.as_path()),
                        moments_out: &next_moments,
                    };
                    model::apply_aggregate_adam(
                        &plan.layout,
                        plan.encoding,
                        plan.chunk_coords,
                        &base,
                        &aggregate_path,
                        total_weight,
                        &adam,
                        &next,
                    )?;
                    moments = Some(Digest::file(&next_moments)?);
                }
                None => model::apply_aggregate(
                    &plan.layout,
                    plan.encoding,
                    plan.chunk_coords,
                    &base,
                    &aggregate_path,
                    total_weight,
                    &next,
                )?,
            }
            Ok((aggregate, Digest::file(&next)?, moments))
        })
        .await?;
        if model_digest != outcome.model || moments != outcome.moments {
            return Err(NodeError::Check(
                "the model rebuilt from the aggregate has a different digest".into(),
            ));
        }
        // Once the ack is out the aggregator can leave, so this model is for keeps.
        if state.step >= self.f.plan.rounds {
            let last = self.f.dirs.model(round);
            File::open(&last)?.sync_data()?;
            sync_parent(&last)?;
        }

        let bound = match self.f.plan.norm() {
            Some(params) => Some((params, self.proof_messages(client, state, &params).await?)),
            None => None,
        };
        let mut audits = Vec::new();
        let targets = self
            .f
            .plan
            .audit_targets(&outcome.challenge, position, state.included.len());
        for target in targets {
            let site = state.included[target];
            let object = Object::Upload {
                round,
                site: site as u32,
            };
            let root = state.contributions[&site].masked;
            // A norm proof is rechecked against the vector itself, so a
            // bounded plan keeps it on disk until that is done.
            let print = match &bound {
                Some((params, (own, _))) => {
                    let path = round_dir.join(format!("audit-{site}.bin"));
                    self.f.fetch_vector(client, object, root, &path).await?;
                    let reading = path.clone();
                    let print =
                        blocking(move || Ok(vector::fingerprint_file(&reading, shape, rho)?))
                            .await?;
                    self.audit_proof(client, state, *params, site, &path, own)
                        .await?;
                    std::fs::remove_file(&path)?;
                    print
                }
                None => self.f.fetch_fingerprint(client, object, root, rho).await?,
            };
            audits.push((target, print));
        }

        let checks = Checks {
            round,
            masked,
            aggregate,
            audits,
        };
        self.checks = Some(checks.clone());
        Ok(checks)
    }

    /// Fetches every party's messages about the round's norm proofs and
    /// checks that they accept each site the aggregator let in.
    async fn proof_messages(
        &mut self,
        client: &mut Client,
        state: &RoundState,
        params: &Params,
    ) -> Result<(Vec<u8>, Vec<Vec<u8>>), NodeError> {
        let round = state.round;
        let missing = || NodeError::Check(format!("round {round}: no record of the proof checks"));
        let own = fetch_blob(
            client,
            Blob::Screened { round },
            state.shares.ok_or_else(missing)?,
        )
        .await?;
        let mut helpers = Vec::with_capacity(self.f.plan.helpers.len());
        for helper in 0..self.f.plan.helpers.len() {
            let blob = Blob::Check {
                round,
                helper: helper as u32,
            };
            let want = *state.checks.get(&helper).ok_or_else(missing)?;
            helpers.push(fetch_blob(client, blob, want).await?);
        }
        for site in &state.included {
            let position = state.screened.iter().position(|s| s == site);
            if !position.is_some_and(|p| norm::accepted(params, p, &own, &helpers)) {
                return Err(NodeError::Check(format!(
                    "round {round}: {} is in the round without a valid norm proof",
                    self.f.plan.sites[*site]
                )));
            }
        }
        Ok((own, helpers))
    }

    /// Recomputes the aggregator's message about `site` from the vector and
    /// the proof share that site committed to.
    async fn audit_proof(
        &mut self,
        client: &mut Client,
        state: &RoundState,
        params: Params,
        site: usize,
        masked: &Path,
        own: &[u8],
    ) -> Result<(), NodeError> {
        let round = state.round;
        let index = site as u32;
        let contribution = &state.contributions[&site];
        let wrong = || {
            NodeError::Check(format!(
                "round {round}: the aggregator misreported the norm proof of {}",
                self.f.plan.sites[site]
            ))
        };
        let proof = fetch_blob(
            client,
            Blob::Proof { round, site: index },
            contribution.proof.ok_or_else(wrong)?,
        )
        .await?;
        let joint = norm::joint(
            &self.f.plan_id,
            round,
            index,
            &contribution.masked,
            &contribution.sealed,
        );
        let challenge = Challenge::derive(&state.screen_head, index);
        let masked = masked.to_owned();
        let share = blocking(move || {
            Ok(norm::leader_share(
                &params, &joint, &challenge, &masked, &proof,
            )?)
        })
        .await?
        .map_err(|_| wrong())?;
        let size = params.share_bytes();
        let position = state
            .screened
            .iter()
            .position(|s| *s == site)
            .ok_or_else(wrong)?;
        if own.get(position * size..(position + 1) * size) != Some(share.as_slice()) {
            return Err(wrong());
        }
        Ok(())
    }

    /// Fetches every scored group's sum, checks it against the fingerprint
    /// the signed values give it, and recomputes the shares.
    async fn check_scores(
        &mut self,
        client: &mut Client,
        state: &RoundState,
    ) -> Result<(), NodeError> {
        let plan = self.f.plan.clone();
        let groups = plan.groups(&state.close_head, state.included.len());
        if groups.is_empty() {
            return Ok(());
        }
        let round = state.round;
        let wrong = |what: &str| NodeError::Check(format!("round {round}: {what}"));
        let (Some(outcome), Some(seal), Some(prints)) =
            (&state.outcome, &state.seal, state.group_prints(&plan))
        else {
            return Err(wrong("the group sums cannot be checked yet"));
        };
        let round_dir = self.f.dirs.round(round)?;
        let total = round_dir.join("aggregate.bin");
        let shape = self.f.shape();
        let rho = challenge(&outcome.challenge);
        let mut alignments = Vec::with_capacity(groups.len());
        for (group, print) in prints.iter().enumerate() {
            let path = round_dir.join(format!("group-{group}.bin"));
            let object = Object::GroupAggregate {
                round,
                group: group as u32,
            };
            let root = *outcome
                .groups
                .get(group)
                .ok_or_else(|| wrong("a group sum is missing"))?;
            self.f.fetch_vector(client, object, root, &path).await?;
            let (total, plain) = (total.clone(), plan.layout.coords());
            let (found, alignment) = blocking(move || {
                let found = vector::fingerprint_file(&path, shape, rho)?;
                let alignment = scoring::alignment(&path, &total, shape, plain)?;
                std::fs::remove_file(&path)?;
                Ok((found, alignment))
            })
            .await?;
            if found != *print {
                return Err(wrong("a group sum is not the sum of its members"));
            }
            alignments.push(alignment);
        }
        if scoring::shares(&alignments) != seal.shares {
            return Err(wrong("the sealed shares are not what the group sums give"));
        }
        Ok(())
    }

    async fn attest(&mut self, client: &mut Client, state: &RoundState) -> Result<(), NodeError> {
        let Some(position) = state.included.iter().position(|s| *s == self.me) else {
            return Ok(());
        };
        if state.site_attests.contains_key(&self.me)
            || state.helper_attests.len() < state.active.len()
        {
            return Ok(());
        }
        let outcome = state
            .outcome
            .clone()
            .ok_or_else(|| NodeError::Protocol("no outcome".into()))?;
        let checks = self.checks(client, state, position).await?;
        let body = Body::Attest {
            plan: self.f.plan_id,
            round: state.round,
            challenge: outcome.challenge,
            masked: checks.masked,
        };
        self.f.offer(client, &body).await?;
        Ok(())
    }

    async fn acknowledge(
        &mut self,
        client: &mut Client,
        state: &RoundState,
    ) -> Result<(), NodeError> {
        let round = state.round;
        let Some(position) = state.included.iter().position(|s| *s == self.me) else {
            self.f.dirs.drop_round(round);
            return Ok(());
        };
        if state.acks.contains(&self.me) {
            return Ok(());
        }
        let seal = state
            .seal
            .clone()
            .ok_or_else(|| NodeError::Protocol("no seal".into()))?;
        let checks = self.checks(client, state, position).await?;
        if checks.aggregate != seal.aggregate {
            return Err(NodeError::Check(format!(
                "round {round}: the aggregate received does not match the sealed fingerprint"
            )));
        }
        if checks.masked != seal.per_site[position] {
            return Err(NodeError::Check(format!(
                "round {round}: the aggregator sealed a different fingerprint for this site"
            )));
        }
        for (target, print) in &checks.audits {
            if seal.per_site[*target] != *print {
                return Err(NodeError::Check(format!(
                    "round {round}: site {} attested to a vector it did not commit to",
                    self.f.plan.sites[state.included[*target]]
                )));
            }
        }
        self.check_scores(client, state).await?;
        let body = Body::Ack {
            plan: self.f.plan_id,
            round,
            head: seal.head,
        };
        self.f.offer(client, &body).await?;
        self.checks = None;
        self.f.dirs.drop_round(round);
        Ok(())
    }

    /// The final state may have arrived while this site was offline.
    async fn settle(&mut self, client: &mut Client) -> Result<(), NodeError> {
        self.tidy_given_up();
        let (round, digest) = (self.f.replay.round_number(), self.f.replay.model);
        self.ensure_model(client, round, digest).await?;
        // The aggregator is leaving, so nothing could fetch this one again.
        let path = self.f.dirs.model(round);
        File::open(&path)?.sync_data()?;
        Ok(sync_parent(&path)?)
    }

    fn tidy_given_up(&self) {
        for given_up in self.f.replay.rounds.iter().filter(|r| r.voided) {
            self.f.dirs.drop_round(given_up.round);
            let _ = std::fs::remove_file(self.f.dirs.model(given_up.round));
            let _ = std::fs::remove_file(self.f.dirs.moments(given_up.round));
        }
    }

    fn summary(&self) -> Result<SiteSummary, NodeError> {
        let round = self.f.replay.round_number();
        Ok(SiteSummary {
            final_model: self.f.dirs.model(round),
            final_digest: self.f.replay.model,
            rounds_contributed: self
                .f
                .replay
                .rounds
                .iter()
                .filter(|r| r.included.contains(&self.me))
                .map(|r| r.round)
                .collect(),
            rounds_refused: self.refused.clone(),
        })
    }
}

fn read_round_file(path: &Path) -> Result<RoundFile, NodeError> {
    serde_json::from_slice(&std::fs::read(path)?).map_err(|e| NodeError::Protocol(e.to_string()))
}

fn decode_seeds(file: &RoundFile) -> Result<Vec<[u8; 32]>, NodeError> {
    file.seeds
        .iter()
        .map(|text| {
            let mut seed = [0u8; 32];
            hex::decode_to_slice(text, &mut seed)
                .map_err(|e| NodeError::Protocol(e.to_string()))?;
            Ok(seed)
        })
        .collect()
}

async fn fetch_blob(client: &mut Client, blob: Blob, want: Digest) -> Result<Vec<u8>, NodeError> {
    client.send(&Msg::GetBlob { blob }).await?;
    let Msg::Blob { data, .. } = client.reply().await? else {
        return Err(NodeError::Protocol("expected a stored file".into()));
    };
    data.map(serde_bytes::ByteBuf::into_vec)
        .filter(|data| Digest::of(data) == want)
        .ok_or_else(|| NodeError::Check(format!("{blob:?} is not what the log names")))
}
