//! Why a site took part. Authority starts at the site's own key: it delegates
//! to a sponsor with conditions, and refuses any plan outside them.

use std::collections::BTreeMap;

use pq_ucan::Ipld;
use pq_ucan::command::Command;
use pq_ucan::crypto::Signer;
use pq_ucan::delegation::{Delegation, Subject};
use pq_ucan::did::Did;
use pq_ucan::invocation::Invocation;
use pq_ucan::nonce::Nonce;
use pq_ucan::policy::Policy;
use pq_ucan::time::Timestamp;
use pq_ucan::validate::{MemoryStore, RevocationStore, Validator};
use serde::Serialize;
use thiserror::Error;

use crate::keys::Principals;
use crate::plan::Plan;
use crate::vector::Digest;

pub const COMMAND: &str = "/fedtally/contribute";
const MAX_CHAIN: usize = 8;

#[derive(Debug, Error)]
pub enum AdmissionError {
    #[error("bad token: {0}")]
    Token(String),
    #[error("invocation is for another site, command or plan")]
    Mismatch,
    #[error("authority check failed: {0}")]
    Refused(String),
}

fn text(value: &str) -> Ipld {
    Ipld::String(value.to_owned())
}

/// The arguments an invocation must carry for this plan. A site's grant
/// constrains them with an ordinary UCAN policy.
pub fn arguments(plan_id: &Digest, plan: &Plan) -> BTreeMap<String, Ipld> {
    BTreeMap::from([
        ("plan".to_owned(), text(&plan_id.hex())),
        ("purpose".to_owned(), text(&plan.purpose)),
        ("rounds".to_owned(), Ipld::Integer(plan.rounds.into())),
        (
            "min_cohort".to_owned(),
            Ipld::Integer(plan.min_cohort.into()),
        ),
        ("aggregator".to_owned(), text(&plan.aggregator.did)),
        (
            "helpers".to_owned(),
            Ipld::List(plan.helpers.iter().map(|h| text(&h.did)).collect()),
        ),
        (
            "sites".to_owned(),
            Ipld::List(plan.sites.iter().map(|s| text(s)).collect()),
        ),
        ("audits".to_owned(), Ipld::Integer(plan.audits.into())),
        ("privacy".to_owned(), privacy_argument(plan)),
    ])
}

/// The plan's privacy terms, or null for a plan that makes no such promise.
fn privacy_argument(plan: &Plan) -> Ipld {
    let Some(privacy) = &plan.privacy else {
        return Ipld::Null;
    };
    let entry = |name: &str, value: u64| (name.to_owned(), Ipld::Integer(value.into()));
    Ipld::Map(BTreeMap::from([
        entry("budget", privacy.budget),
        entry("clip", privacy.clip),
        entry("colluding", privacy.colluding.into()),
        entry("noise", privacy.noise),
    ]))
}

/// The conditions a site puts on its participation.
#[derive(Clone, Debug, Default)]
pub struct Terms {
    pub purpose: Option<String>,
    /// The site only joins plans using at least one of these helpers.
    pub trusted_helpers: Vec<String>,
    pub min_cohort: Option<u32>,
    pub max_rounds: Option<u32>,
    /// The only sites this site will be averaged with, if set. Stops a sponsor
    /// meeting a minimum cohort with parties of its own.
    pub allowed_sites: Vec<String>,
    /// Accept a plan with no helpers, where the aggregator reads every update.
    pub allow_plain: bool,
    /// The fewest audits per site to accept. Without one, a masked plan must
    /// have at least one: otherwise the aggregator and one site can change the
    /// sum unnoticed.
    pub min_audits: Option<u32>,
    /// Only join a private plan, whose whole budget is at most this much rho
    /// in units of 2^-40.
    pub max_budget: Option<u64>,
    /// Only join a private plan that assumes at least this many sites in a
    /// round give their noise away.
    pub min_colluding: Option<u32>,
}

impl Terms {
    pub fn policy(&self) -> Result<Policy, AdmissionError> {
        let list = |items: Vec<Ipld>| Ipld::List(items);
        let mut statements = Vec::new();
        if let Some(purpose) = &self.purpose {
            statements.push(list(vec![text("=="), text(".purpose"), text(purpose)]));
        }
        if !self.trusted_helpers.is_empty() {
            let any_of = self
                .trusted_helpers
                .iter()
                .map(|h| list(vec![text("=="), text("."), text(h)]))
                .collect();
            statements.push(list(vec![
                text("any"),
                text(".helpers"),
                list(vec![text("or"), list(any_of)]),
            ]));
        }
        if let Some(n) = self.min_cohort {
            statements.push(list(vec![
                text(">="),
                text(".min_cohort"),
                Ipld::Integer(n.into()),
            ]));
        }
        if !self.allowed_sites.is_empty() {
            let any_of = self
                .allowed_sites
                .iter()
                .map(|s| list(vec![text("=="), text("."), text(s)]))
                .collect();
            statements.push(list(vec![
                text("all"),
                text(".sites"),
                list(vec![text("or"), list(any_of)]),
            ]));
        }
        if !self.allow_plain {
            statements.push(list(vec![text("!="), text(".helpers"), list(Vec::new())]));
            let audits = self.min_audits.unwrap_or(1);
            statements.push(list(vec![
                text(">="),
                text(".audits"),
                Ipld::Integer(audits.into()),
            ]));
        }
        if let Some(n) = self.max_rounds {
            statements.push(list(vec![
                text("<="),
                text(".rounds"),
                Ipld::Integer(n.into()),
            ]));
        }
        if let Some(n) = self.max_budget {
            statements.push(list(vec![
                text("<="),
                text(".privacy.budget"),
                Ipld::Integer(n.into()),
            ]));
        }
        if let Some(n) = self.min_colluding {
            statements.push(list(vec![
                text(">="),
                text(".privacy.colluding"),
                Ipld::Integer(n.into()),
            ]));
        }
        Policy::from_ipld(&Ipld::List(statements)).map_err(|e| AdmissionError::Token(e.to_string()))
    }
}

fn command() -> Result<Command, AdmissionError> {
    Command::parse(COMMAND).map_err(|e| AdmissionError::Token(e.to_string()))
}

/// From a site to an audience: may ask this site to contribute, under `terms`.
pub fn grant(
    site: &impl Signer,
    audience: &Did,
    terms: &Terms,
    expires: Timestamp,
) -> Result<Delegation, AdmissionError> {
    Delegation::builder(audience.clone(), Subject::Did(site.did()), command()?)
        .policy(terms.policy()?)
        .nonce(Nonce::random(&mut pq_ucan::rng::system()))
        .expires_at(expires)
        .sign(site)
        .map_err(|e| AdmissionError::Token(e.to_string()))
}

/// Sponsor: exercises its chain to ask `site` to run `plan`.
pub fn invoke(
    sponsor: &impl Signer,
    plan_id: &Digest,
    plan: &Plan,
    site: &Did,
    chain: &[&Delegation],
    expires: Timestamp,
) -> Result<Invocation, AdmissionError> {
    Invocation::builder(site.clone(), command()?)
        .args(arguments(plan_id, plan))
        .proofs(chain.iter().map(|d| *d.cid()))
        .nonce(Nonce::random(&mut pq_ucan::rng::system()))
        .expires_at(expires)
        .sign(sponsor)
        .map_err(|e| AdmissionError::Token(e.to_string()))
}

/// What an accepted chain says, for the report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Admission {
    pub invocation: String,
    pub invoked_by: String,
    pub chain_len: usize,
}

/// Checks the invocation and chain as `site`, at time `now`.
pub fn check(
    site: &Did,
    plan_id: &Digest,
    plan: &Plan,
    invocation: &[u8],
    proofs: &[Vec<u8>],
    now: Timestamp,
    revocations: Option<&dyn RevocationStore>,
) -> Result<Admission, AdmissionError> {
    let invocation =
        Invocation::decode(invocation).map_err(|e| AdmissionError::Token(e.to_string()))?;
    let payload = invocation.payload();
    if !payload.subject().same_principal(site)
        || payload.command().as_str() != COMMAND
        || payload.args() != &arguments(plan_id, plan)
    {
        return Err(AdmissionError::Mismatch);
    }
    let mut store = MemoryStore::new();
    for proof in proofs {
        store.insert(Delegation::decode(proof).map_err(|e| AdmissionError::Token(e.to_string()))?);
    }
    let mut validator = Validator::new(&store, now)
        .resolver(&Principals)
        .executor(site)
        .max_proofs(MAX_CHAIN);
    if let Some(revocations) = revocations {
        validator = validator.revocations(revocations);
    }
    let proof = validator
        .validate(&invocation)
        .map_err(|e| AdmissionError::Refused(e.to_string()))?;
    Ok(Admission {
        invocation: invocation.cid().to_string(),
        invoked_by: payload.issuer().as_str().to_owned(),
        chain_len: proof.chain().len(),
    })
}
