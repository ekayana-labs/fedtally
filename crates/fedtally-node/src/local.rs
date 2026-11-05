//! Sets up a whole federation in one directory: keys, plan and grants. Used by
//! tests, notebooks and benchmarks. A real deployment makes these separately,
//! each party holding only its own key.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use fedtally_core::admission::{self, Terms};
use fedtally_core::keys::{Identity, SigAlg, random_bytes};
use fedtally_core::model::{Encoding, Layout};
use fedtally_core::plan::{AggregatorSpec, HelperSpec, Plan, Timeouts};
use fedtally_core::pq_ucan::crypto::Signer;
use fedtally_core::pq_ucan::time::Timestamp;
use fedtally_core::record::{Body, Record, decode_token, encode_token};
use fedtally_core::vector::Digest;
use serde::{Deserialize, Serialize};

use crate::NodeError;
use crate::aggregator::AggregatorConfig;
use crate::helper::HelperConfig;
use crate::net::{generate_tls, tls_pin};
use crate::site::SiteConfig;

const LONG_GRANT_SECS: i64 = 60 * 60 * 24 * 365;

#[derive(Clone, Debug)]
pub struct Options {
    pub sites: usize,
    pub helpers: usize,
    pub rounds: u32,
    pub min_cohort: u32,
    pub chunk_coords: u64,
    pub encoding: Encoding,
    pub max_weight: u64,
    pub audits: u32,
    pub timeouts: Timeouts,
    pub purpose: String,
    /// The purpose sites grant, if it differs from the one the plan states.
    pub granted_purpose: Option<String>,
    pub sig_alg: SigAlg,
}

/// Three quarters of the sites. The aggregator chooses who is in a round, so
/// with `k` sites on its side it can isolate an honest site's update unless the
/// cohort has to be larger than `k + 1`.
pub fn default_min_cohort(sites: usize) -> u32 {
    (sites * 3).div_ceil(4).clamp(sites.min(2), sites) as u32
}

impl Options {
    pub fn new(sites: usize, helpers: usize, rounds: u32) -> Options {
        Options {
            sites,
            helpers,
            rounds,
            min_cohort: default_min_cohort(sites),
            chunk_coords: 1 << 19,
            encoding: Encoding {
                frac_bits: 24,
                clip: 128.0,
            },
            max_weight: 1 << 20,
            audits: 1,
            timeouts: Timeouts {
                join_secs: 120,
                collect_secs: 600,
                attest_secs: 300,
                ack_secs: 120,
            },
            purpose: "local run".into(),
            granted_purpose: None,
            sig_alg: SigAlg::Ed25519,
        }
    }
}

pub struct Grant {
    pub invocation: Vec<u8>,
    pub proofs: Vec<Vec<u8>>,
}

/// A site's authority for one plan, as a file.
#[derive(Serialize, Deserialize)]
pub struct GrantFile {
    pub invocation: String,
    pub proofs: Vec<String>,
}

impl GrantFile {
    pub fn load(path: &Path) -> Result<Grant, NodeError> {
        let file: GrantFile = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| NodeError::Protocol(e.to_string()))?;
        Ok(Grant {
            invocation: decode_token(&file.invocation)?,
            proofs: file
                .proofs
                .iter()
                .map(|p| decode_token(p))
                .collect::<Result<_, _>>()?,
        })
    }
}

pub struct LocalFederation {
    pub root: PathBuf,
    pub plan: Arc<Plan>,
    pub plan_line: Vec<u8>,
    pub plan_id: Digest,
    pub initial_model: PathBuf,
    pub sponsor: Arc<Identity>,
    pub aggregator: Arc<Identity>,
    pub helpers: Vec<Arc<Identity>>,
    pub sites: Vec<Arc<Identity>>,
    pub grants: Vec<Grant>,
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or_default()
}

impl LocalFederation {
    pub fn create(
        root: &Path,
        layout: Layout,
        initial_model: &Path,
        options: &Options,
    ) -> Result<LocalFederation, NodeError> {
        std::fs::create_dir_all(root)?;
        let new = || Identity::generate(options.sig_alg).map(Arc::new);
        let sponsor = new()?;
        let mut aggregator = Identity::generate(options.sig_alg)?;
        let credentials = generate_tls()?;
        let pin = tls_pin(&credentials)?;
        aggregator.tls = Some(credentials);
        let aggregator = Arc::new(aggregator);
        let helpers = (0..options.helpers)
            .map(|_| new())
            .collect::<Result<Vec<_>, _>>()?;
        let sites = (0..options.sites)
            .map(|_| new())
            .collect::<Result<Vec<_>, _>>()?;

        let plan = Plan {
            version: 1,
            nonce: hex::encode(random_bytes::<16>()),
            purpose: options.purpose.clone(),
            sponsor: sponsor.did_string(),
            aggregator: AggregatorSpec {
                did: aggregator.did_string(),
                tls_pin: pin,
            },
            helpers: helpers
                .iter()
                .map(|h| {
                    Ok(HelperSpec {
                        did: h.did_string(),
                        seal: h.seal_public().clone(),
                        seal_proof: h.seal_proof()?,
                    })
                })
                .collect::<Result<_, NodeError>>()?,
            sites: sites.iter().map(|s| s.did_string()).collect(),
            layout,
            init_model: Digest::file(initial_model)?,
            rounds: options.rounds,
            min_cohort: options.min_cohort,
            chunk_coords: options.chunk_coords,
            encoding: options.encoding,
            max_weight: options.max_weight,
            audits: options.audits,
            timeouts: options.timeouts,
        };
        plan.validate()
            .map_err(|e| NodeError::Check(e.to_string()))?;
        let record = Record::sign(&sponsor, &Body::Plan(Box::new(plan.clone())))?;

        let expires = Timestamp::from_unix(unix_now() + LONG_GRANT_SECS)
            .map_err(|e| NodeError::Protocol(e.to_string()))?;
        let terms = Terms {
            purpose: Some(
                options
                    .granted_purpose
                    .clone()
                    .unwrap_or_else(|| options.purpose.clone()),
            ),
            trusted_helpers: plan.helpers.iter().map(|h| h.did.clone()).collect(),
            allow_plain: plan.helpers.is_empty(),
            min_audits: Some(options.audits),
            ..Terms::default()
        };
        let mut grants = Vec::with_capacity(sites.len());
        for site in &sites {
            let grant = admission::grant(site.as_ref(), &sponsor.did(), &terms, expires)
                .map_err(|e| NodeError::Check(e.to_string()))?;
            let invocation = admission::invoke(
                sponsor.as_ref(),
                &record.body_hash,
                &plan,
                &site.did(),
                &[&grant],
                expires,
            )
            .map_err(|e| NodeError::Check(e.to_string()))?;
            grants.push(Grant {
                invocation: invocation.bytes().to_vec(),
                proofs: vec![grant.bytes().to_vec()],
            });
        }

        Ok(LocalFederation {
            root: root.to_owned(),
            plan: Arc::new(plan),
            plan_line: record.line.clone(),
            plan_id: record.body_hash,
            initial_model: initial_model.to_owned(),
            sponsor,
            aggregator,
            helpers,
            sites,
            grants,
        })
    }

    /// Writes each party's files, ready for separate processes.
    pub fn save(&self) -> Result<(), NodeError> {
        std::fs::write(self.root.join("plan.jsonl"), &self.plan_line)?;
        self.aggregator.save(&self.root.join("aggregator.key"))?;
        self.sponsor.save(&self.root.join("sponsor.key"))?;
        for (h, helper) in self.helpers.iter().enumerate() {
            helper.save(&self.root.join(format!("helper-{h}.key")))?;
        }
        for (i, site) in self.sites.iter().enumerate() {
            site.save(&self.root.join(format!("site-{i}.key")))?;
            let grant = GrantFile {
                invocation: encode_token(&self.grants[i].invocation),
                proofs: self.grants[i]
                    .proofs
                    .iter()
                    .map(|p| encode_token(p))
                    .collect(),
            };
            let text =
                serde_json::to_vec(&grant).map_err(|e| NodeError::Protocol(e.to_string()))?;
            std::fs::write(self.root.join(format!("site-{i}.grant.json")), text)?;
        }
        Ok(())
    }

    pub fn aggregator_dir(&self) -> PathBuf {
        self.root.join("aggregator")
    }

    pub fn site_dir(&self, site: usize) -> PathBuf {
        self.root.join(format!("site-{site}"))
    }

    pub fn aggregator_config(&self, listen: &str) -> AggregatorConfig {
        AggregatorConfig {
            identity: self.aggregator.clone(),
            plan_line: self.plan_line.clone(),
            listen: listen.to_owned(),
            dir: self.aggregator_dir(),
            initial_model: self.initial_model.clone(),
            hook: None,
            on_ready: None,
            keep_spool: false,
        }
    }

    pub fn helper_config(&self, helper: usize, addr: &str) -> HelperConfig {
        HelperConfig {
            identity: self.helpers[helper].clone(),
            plan_line: self.plan_line.clone(),
            addr: addr.to_owned(),
            dir: self.root.join(format!("helper-{helper}")),
            give_up_after: Duration::from_secs(60),
            share_hook: None,
        }
    }

    pub fn site_config(&self, site: usize, addr: &str) -> SiteConfig {
        SiteConfig {
            identity: self.sites[site].clone(),
            plan_line: self.plan_line.clone(),
            addr: addr.to_owned(),
            dir: self.site_dir(site),
            invocation: self.grants[site].invocation.clone(),
            proofs: self.grants[site].proofs.clone(),
            revocations: None,
            give_up_after: Duration::from_secs(60),
        }
    }
}
