//! A helper turns the seeds it was sent into a mask total, once per round, for
//! the cohort the log fixed. It never sees an update.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use fedtally_core::field::Fp2;
use fedtally_core::keys::Identity;
use fedtally_core::norm::{self, Challenge};
use fedtally_core::record::Body;
use fedtally_core::replay::Phase;
use fedtally_core::vector::{self, Digest, challenge};

use crate::follower::{Follower, Patience, Start};
use crate::net::Client;
use crate::wire::{Blob, Msg, Object};
use crate::{NodeError, blocking};

/// The seeds opened so far in one round, by site.
type Seeds = Option<(u32, BTreeMap<usize, [u8; 32]>)>;

/// A point where a test can act as a dishonest helper.
pub type ShareHook = Arc<dyn Fn(&mut Vec<Fp2>) + Send + Sync>;

pub struct HelperConfig {
    pub identity: Arc<Identity>,
    pub plan_line: Vec<u8>,
    pub addr: String,
    pub dir: PathBuf,
    pub give_up_after: Duration,
    pub share_hook: Option<ShareHook>,
}

/// The context a sealed seed is bound to.
pub fn seal_context(plan_id: &Digest, round: u32, site: &str, helper: &str) -> Vec<u8> {
    format!("{plan_id}/{round}/{site}/{helper}").into_bytes()
}

pub async fn run_helper(config: HelperConfig) -> Result<(), NodeError> {
    let mut follower = Follower::open(Start {
        identity: config.identity,
        plan_line: config.plan_line,
        addr: config.addr,
        dir: config.dir,
        give_up_after: config.give_up_after,
    })?;
    let me = follower
        .plan
        .helper_index(&follower.identity.did_string())
        .ok_or_else(|| NodeError::Refused("this key is not a helper in the plan".into()))?;
    let hook = config.share_hook;
    let mut seeds: Seeds = None;
    let mut patience = Patience::new(config.give_up_after);
    loop {
        match session(&mut follower, me, &mut seeds, hook.as_ref()).await {
            Ok(()) => return Ok(()),
            Err(e) if e.is_transient() && patience.allows(follower.replay.entries) => {
                tracing::info!("helper reconnecting: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            // It may have been removed while it worked, which the log it has
            // yet to read would say.
            Err(NodeError::Refused(why))
                if follower.plan.min_helpers > 0 && patience.allows(follower.replay.entries) =>
            {
                tracing::warn!("record refused: {why}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

async fn session(
    f: &mut Follower,
    me: usize,
    seeds: &mut Seeds,
    hook: Option<&ShareHook>,
) -> Result<(), NodeError> {
    let mut client = f.connect().await?;
    let mut wait = true;
    loop {
        if wait || client.has_backlog() {
            let batch = client.entries().await?;
            f.absorb(batch)?;
        }
        let before = f.replay.entries;
        // Nothing more is asked about a given up round, and its mask is large.
        for given_up in f.replay.rounds.iter().filter(|r| r.voided) {
            f.dirs.drop_round(given_up.round);
        }
        match f.replay.phase {
            Phase::Done => return Ok(()),
            Phase::Aborted => {
                return Err(NodeError::Aborted(
                    f.replay.abort.clone().unwrap_or_default(),
                ));
            }
            // Out of the run: it only follows the log to the end.
            _ if f.replay.removed.contains(&me) => {}
            Phase::Screening => check_step(f, &mut client, me, seeds).await?,
            Phase::Masking => mask_step(f, &mut client, me, seeds).await?,
            Phase::Attesting => attest_step(f, &mut client, me, seeds, hook).await?,
            _ => {}
        }
        wait = f.replay.entries == before;
    }
}

async fn load_seed(
    f: &Follower,
    client: &mut Client,
    me: usize,
    round: u32,
    site: usize,
) -> Result<[u8; 32], NodeError> {
    let state = f
        .replay
        .current()
        .ok_or_else(|| NodeError::Protocol("no round".into()))?;
    let helper_did = f.identity.did_string();
    let bad = |site: usize, why: &str| NodeError::BadSeed {
        site,
        why: why.to_owned(),
    };
    client
        .send(&Msg::GetSealed {
            round,
            site: site as u32,
        })
        .await?;
    let Msg::Sealed { data, .. } = client.reply().await? else {
        return Err(NodeError::Protocol("expected a sealed seed".into()));
    };
    let data = data.ok_or_else(|| bad(site, "the aggregator has no sealed seed"))?;
    if Digest::of(&data) != state.contributions[&site].sealed[me] {
        return Err(bad(
            site,
            "sealed seed does not match the signed contribution",
        ));
    }
    let context = seal_context(&f.plan_id, round, &f.plan.sites[site], &helper_did);
    f.identity
        .open(&context, &data)
        .map_err(|_| bad(site, "sealed seed does not open"))
}

/// Seeds for the round, kept so the fingerprints don't need them fetched again.
async fn ensure_seeds(
    f: &Follower,
    client: &mut Client,
    me: usize,
    round: u32,
    included: &[usize],
    seeds: &mut Seeds,
) -> Result<Vec<[u8; 32]>, NodeError> {
    if seeds.as_ref().is_none_or(|(r, _)| *r != round) {
        *seeds = Some((round, BTreeMap::new()));
    }
    let mut out = Vec::with_capacity(included.len());
    for site in included {
        let known = seeds
            .as_ref()
            .and_then(|(_, known)| known.get(site).copied());
        let seed = match known {
            Some(seed) => seed,
            None => load_seed(f, client, me, round, *site).await?,
        };
        if let Some((_, known)) = seeds {
            known.insert(*site, seed);
        }
        out.push(seed);
    }
    Ok(out)
}

/// Sends this helper's messages about the screened norm proofs. A site whose
/// seed doesn't open gets a message that cannot pass, and the round goes on
/// without it.
async fn check_step(
    f: &mut Follower,
    client: &mut Client,
    me: usize,
    seeds: &mut Seeds,
) -> Result<(), NodeError> {
    let Some(params) = f.plan.norm() else {
        return Ok(());
    };
    let state = f
        .replay
        .current()
        .ok_or_else(|| NodeError::Protocol("no round".into()))?
        .clone();
    if state.checks.contains_key(&me) {
        return Ok(());
    }
    let round = state.round;
    let mut jobs = Vec::with_capacity(state.screened.len());
    for site in &state.screened {
        let seed = match ensure_seeds(f, client, me, round, &[*site], seeds).await {
            Ok(seed) => Some(seed[0]),
            Err(NodeError::BadSeed { why, .. }) => {
                tracing::warn!("no proof check for {}: {why}", f.plan.sites[*site]);
                None
            }
            Err(e) => return Err(e),
        };
        let contribution = &state.contributions[site];
        let index = *site as u32;
        jobs.push((
            seed,
            norm::joint(
                &f.plan_id,
                round,
                index,
                &contribution.masked,
                &contribution.sealed,
            ),
            Challenge::derive(&state.screen_head, index),
        ));
    }
    let shares = blocking(move || {
        let size = params.share_bytes();
        let mut out = Vec::with_capacity(jobs.len() * size);
        for (seed, joint, challenge) in jobs {
            match seed {
                Some(seed) => out.extend(norm::helper_share(&params, &joint, &challenge, &seed)?),
                None => out.resize(out.len() + size, 0),
            }
        }
        Ok(out)
    })
    .await?;
    let body = Body::Check {
        plan: f.plan_id,
        round,
        screen: state.screen_head,
        shares: Digest::of(&shares),
    };
    let blob = Blob::Check {
        round,
        helper: me as u32,
    };
    client.send(&Msg::PutBlob { blob, data: shares }).await?;
    f.append(client, &body).await
}

async fn mask_step(
    f: &mut Follower,
    client: &mut Client,
    me: usize,
    seeds: &mut Seeds,
) -> Result<(), NodeError> {
    let state = f
        .replay
        .current()
        .ok_or_else(|| NodeError::Protocol("no round".into()))?
        .clone();
    if state.masks.contains_key(&me) {
        return Ok(());
    }
    let loaded = match ensure_seeds(f, client, me, state.round, &state.included, seeds).await {
        Ok(loaded) => loaded,
        Err(NodeError::BadSeed { site, why }) => {
            let body = Body::Abort {
                plan: f.plan_id,
                round: state.round,
                reason: format!("helper cannot mask site {}: {why}", f.plan.sites[site]),
            };
            return f.append(client, &body).await;
        }
        Err(e) => return Err(e),
    };
    let shape = f.shape();
    let all = loaded.clone();
    let path = f.dirs.round(state.round)?.join("mask.bin");
    let out = path.clone();
    let manifest = blocking(move || {
        Ok(vector::build(&out, shape, |chunk, buf, _| {
            for seed in &loaded {
                vector::apply_mask(seed, chunk, buf);
            }
            Ok(())
        })?)
    })
    .await?;
    let object = Object::Mask {
        round: state.round,
        helper: me as u32,
    };
    f.upload(client, object, &path, &manifest).await?;
    let mut groups = Vec::new();
    for (group, members) in f
        .plan
        .groups(&state.close_head, state.included.len())
        .into_iter()
        .enumerate()
    {
        let seeds: Vec<[u8; 32]> = members.iter().map(|p| all[*p]).collect();
        let path = f
            .dirs
            .round(state.round)?
            .join(format!("mask-g{group}.bin"));
        let out = path.clone();
        let manifest = blocking(move || {
            Ok(vector::build(&out, shape, |chunk, buf, _| {
                for seed in &seeds {
                    vector::apply_mask(seed, chunk, buf);
                }
                Ok(())
            })?)
        })
        .await?;
        let object = Object::GroupMask {
            round: state.round,
            helper: me as u32,
            group: group as u32,
        };
        f.upload(client, object, &path, &manifest).await?;
        groups.push(manifest.root);
    }
    let body = Body::MaskSum {
        plan: f.plan_id,
        round: state.round,
        close: state.close_head,
        mask: manifest.root,
        groups,
    };
    f.append(client, &body).await
}

async fn attest_step(
    f: &mut Follower,
    client: &mut Client,
    me: usize,
    seeds: &mut Seeds,
    hook: Option<&ShareHook>,
) -> Result<(), NodeError> {
    let state = f
        .replay
        .current()
        .ok_or_else(|| NodeError::Protocol("no round".into()))?
        .clone();
    if state.helper_attests.contains_key(&me) {
        return Ok(());
    }
    let outcome = state
        .outcome
        .clone()
        .ok_or_else(|| NodeError::Protocol("no outcome".into()))?;
    let loaded = ensure_seeds(f, client, me, state.round, &state.included, seeds).await?;
    let shape = f.shape();
    let rho = challenge(&outcome.challenge);
    let mut per_site = blocking(move || {
        loaded
            .iter()
            .map(|seed| Ok(vector::fingerprint_mask(seed, shape, rho)?))
            .collect::<Result<Vec<Fp2>, NodeError>>()
    })
    .await?;
    if let Some(hook) = hook {
        hook(&mut per_site);
    }
    let body = Body::HelperAttest {
        plan: f.plan_id,
        round: state.round,
        challenge: outcome.challenge,
        total: per_site.iter().copied().sum(),
        per_site,
    };
    f.append(client, &body).await?;
    f.dirs.drop_round(state.round);
    Ok(())
}
