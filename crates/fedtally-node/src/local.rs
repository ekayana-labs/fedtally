//! Sets up a whole federation in one directory: keys, plan and grants. Used by
//! tests, notebooks and benchmarks. A real deployment makes these separately,
//! each party holding only its own key.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use fedtally_core::admission::{self, Terms};
use fedtally_core::credential::{Credential, Draft, StatusEntry};
use fedtally_core::duo::{self, Purpose, Research};
use fedtally_core::keys::{Identity, SigAlg, random_bytes};
use fedtally_core::model::{Encoding, Layout};
use fedtally_core::plan::{AggregatorSpec, HelperSpec, Plan, Timeouts};
use fedtally_core::policy::{About, Atom, Policy, Rule};
use fedtally_core::pq_ucan::crypto::Signer;
use fedtally_core::pq_ucan::time::Timestamp;
use fedtally_core::privacy;
use fedtally_core::record::{Body, Record, decode_token, encode_token};
use fedtally_core::scoring::Scoring;
use fedtally_core::status::StatusList;
use fedtally_core::vector::Digest;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::NodeError;
use crate::aggregator::AggregatorConfig;
use crate::helper::HelperConfig;
use crate::net::{generate_tls, tls_pin};
use crate::site::{SiteConfig, Standing};

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
    /// A privacy target. It replaces `encoding` and `max_weight`.
    pub privacy: Option<privacy::Request>,
    /// How many sites each round invites. Everyone, if absent.
    pub sample: Option<u32>,
    /// Gate the run on credentials: an affiliation, an ethics approval and an
    /// access grant for each site, and provenance for each dataset.
    pub credentials: bool,
    /// Name every party by `did:bio` on this network, not by `did:key`.
    pub did_bio: Option<String>,
    /// Have every site prove its update is within the privacy terms.
    pub bounded: bool,
    /// Score groups of at least this many sites for rewards.
    pub scoring: Option<u32>,
    /// How many times a round may be redone without a site that went silent.
    pub retries: u32,
    /// The fewest helpers a run may carry on with, if it may lose any.
    pub min_helpers: u32,
    /// Sites that play dishonest, with how many times past the plan's clip
    /// each one reaches. For tests and attack benchmarks.
    pub cheats: BTreeMap<usize, u64>,
}

/// The options a caller may set from outside, such as from Python. Anything
/// left out keeps the value `Options::new` gives it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Overrides {
    sites: usize,
    helpers: usize,
    rounds: u32,
    min_cohort: Option<u32>,
    chunk_coords: Option<u64>,
    frac_bits: Option<u32>,
    clip: Option<f64>,
    max_weight: Option<u64>,
    audits: Option<u32>,
    purpose: Option<String>,
    granted_purpose: Option<String>,
    join_secs: Option<u64>,
    collect_secs: Option<u64>,
    attest_secs: Option<u64>,
    ack_secs: Option<u64>,
    sig_alg: Option<SigAlg>,
    privacy: Option<privacy::Request>,
    sample: Option<u32>,
    credentials: Option<bool>,
    did_bio: Option<String>,
    bounded: Option<bool>,
    scoring: Option<u32>,
    retries: Option<u32>,
    min_helpers: Option<u32>,
    cheats: Option<BTreeMap<usize, u64>>,
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
            privacy: None,
            sample: None,
            credentials: false,
            did_bio: None,
            bounded: false,
            scoring: None,
            retries: 0,
            min_helpers: 0,
            cheats: BTreeMap::new(),
        }
    }

    pub fn from_json(text: &str) -> Result<Options, NodeError> {
        let given: Overrides =
            serde_json::from_str(text).map_err(|e| NodeError::Protocol(e.to_string()))?;
        let mut options = Options::new(given.sites, given.helpers, given.rounds);
        fn set<T>(slot: &mut T, value: Option<T>) {
            if let Some(value) = value {
                *slot = value;
            }
        }
        set(&mut options.min_cohort, given.min_cohort);
        set(&mut options.chunk_coords, given.chunk_coords);
        set(&mut options.encoding.frac_bits, given.frac_bits);
        set(&mut options.encoding.clip, given.clip);
        set(&mut options.max_weight, given.max_weight);
        set(&mut options.audits, given.audits);
        set(&mut options.purpose, given.purpose);
        set(&mut options.timeouts.join_secs, given.join_secs);
        set(&mut options.timeouts.collect_secs, given.collect_secs);
        set(&mut options.timeouts.attest_secs, given.attest_secs);
        set(&mut options.timeouts.ack_secs, given.ack_secs);
        set(&mut options.sig_alg, given.sig_alg);
        set(&mut options.credentials, given.credentials);
        options.granted_purpose = given.granted_purpose;
        options.privacy = given.privacy;
        options.sample = given.sample;
        options.did_bio = given.did_bio;
        set(&mut options.bounded, given.bounded);
        options.scoring = given.scoring;
        set(&mut options.retries, given.retries);
        set(&mut options.min_helpers, given.min_helpers);
        set(&mut options.cheats, given.cheats);
        Ok(options)
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

/// The bodies that vouch for sites in a federation gated on credentials: a
/// university for affiliation, a review board for ethics approval, and a data
/// access committee for access grants and dataset provenance.
pub struct Issuers {
    pub university: Identity,
    pub board: Identity,
    pub committee: Identity,
    issued: AtomicI64,
}

/// Dataset credentials sit above every site's slot in the committee's list.
const DATASET_SLOTS_FROM: u64 = 1 << 16;

impl Issuers {
    fn new(now: i64, alg: SigAlg) -> Result<Issuers, NodeError> {
        let new = || Identity::generate(alg);
        Ok(Issuers {
            university: new()?,
            board: new()?,
            committee: new()?,
            // Dated a little back, so a list never postdates a site's Join.
            issued: AtomicI64::new(now - 3600),
        })
    }

    fn list_id(issuer: &Identity) -> String {
        format!("urn:fedtally:status:{}", issuer.did_string())
    }

    fn credential(
        issuer: &Identity,
        kind: &str,
        subject: &str,
        index: u64,
        claims: serde_json::Value,
        now: i64,
    ) -> Result<String, NodeError> {
        let draft = Draft {
            kind: kind.to_owned(),
            subject: subject.to_owned(),
            claims: claims.as_object().cloned().unwrap_or_default(),
            valid_from: now - 60,
            valid_until: Some(now + LONG_GRANT_SECS),
            status: Some(StatusEntry {
                list: Issuers::list_id(issuer),
                index,
            }),
        };
        Credential::issue(issuer, &draft, now)
            .map(|c| c.text)
            .map_err(|e| NodeError::Check(e.to_string()))
    }

    /// What site `index` holds: an affiliation, an ethics approval, a grant
    /// for its dataset under health research terms, and the dataset's provenance.
    fn wallet(
        &self,
        index: usize,
        site: &str,
        dataset: &str,
        now: i64,
    ) -> Result<Vec<String>, NodeError> {
        let visa = |kind: &str, value: &str, by: &str| {
            json!({"ga4gh_visa_v1": {
                "type": kind, "value": value, "by": by,
                "source": "https://issuer.example", "asserted": now - 60,
            }})
        };
        let mut grant = visa("ControlledAccessGrants", dataset, "dac");
        grant["data_use"] = json!({
            "permission": duo::HEALTH,
            "modifiers": [{"term": duo::ETHICS_APPROVAL_REQUIRED}, {"term": duo::NON_COMMERCIAL}],
        });
        let slot = index as u64;
        let affiliation = visa("AffiliationAndRole", "researcher@university.example", "so");
        Ok(vec![
            Issuers::credential(
                &self.university,
                "AffiliationAndRole",
                site,
                slot,
                affiliation,
                now,
            )?,
            Issuers::credential(&self.board, "EthicsApproval", site, slot, json!({}), now)?,
            Issuers::credential(
                &self.committee,
                "ControlledAccessGrants",
                site,
                slot,
                grant,
                now,
            )?,
            Issuers::credential(
                &self.committee,
                "Provenance",
                dataset,
                DATASET_SLOTS_FROM + slot,
                json!({}),
                now,
            )?,
        ])
    }

    fn policy(&self) -> Policy {
        let holds = |kind: &str, issuer: &Identity| Atom {
            kind: kind.to_owned(),
            issuers: vec![issuer.did_string()],
            about: About::Holder,
            claims: Default::default(),
            names_dataset: None,
            data_use: None,
        };
        let mut grant = holds("ControlledAccessGrants", &self.committee);
        grant.claims.insert("ga4gh_visa_v1.by".into(), json!("dac"));
        grant.names_dataset = Some("ga4gh_visa_v1.value".into());
        grant.data_use = Some("data_use".into());
        let mut provenance = holds("Provenance", &self.committee);
        provenance.about = About::Dataset;
        Policy {
            purpose: Purpose {
                research: Research::Health,
                not_for_profit: true,
                ethics_approval: true,
                ..Purpose::default()
            },
            sites: Rule::All(vec![
                Rule::Holds(holds("AffiliationAndRole", &self.university)),
                Rule::Holds(holds("EthicsApproval", &self.board)),
                Rule::Holds(grant),
                Rule::Holds(provenance),
            ]),
            operators: None,
        }
    }

    /// Writes an issuer's list to `dir`, newer than any written before.
    fn publish(&self, issuer: &Identity, revoked: &[u64], dir: &Path) -> Result<(), NodeError> {
        let issued = self.issued.fetch_add(1, Ordering::SeqCst);
        let list = StatusList::issue(issuer, &Issuers::list_id(issuer), revoked, issued)
            .map_err(|e| NodeError::Check(e.to_string()))?;
        let name = format!("{}.json", Digest::of(list.id.as_bytes()).hex());
        let partial = dir.join(format!("{name}.partial"));
        std::fs::write(&partial, &list.text)?;
        std::fs::rename(&partial, dir.join(name))?;
        Ok(())
    }
}

/// What a site holds for a plan with a policy, as a file.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StandingFile {
    pub dataset: Option<String>,
    pub credentials: Vec<String>,
}

impl StandingFile {
    pub fn load(path: &Path, status_dir: Option<PathBuf>) -> Result<Standing, NodeError> {
        let file: StandingFile = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| NodeError::Protocol(e.to_string()))?;
        Ok(Standing {
            dataset: file.dataset,
            credentials: file.credentials,
            status_dir,
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
    /// Set when the federation is gated on credentials.
    pub issuers: Option<Issuers>,
    /// What each site holds for the plan's policy.
    pub standing: Vec<Standing>,
    cheats: BTreeMap<usize, u64>,
}

fn to_json(value: &impl Serialize) -> Result<Vec<u8>, NodeError> {
    serde_json::to_vec(value).map_err(|e| NodeError::Protocol(e.to_string()))
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
        LocalFederation::create_with(root, layout, initial_model, options, |_| {})
    }

    /// As `create`, with a last change to the plan before the sponsor signs it.
    pub fn create_with(
        root: &Path,
        layout: Layout,
        initial_model: &Path,
        options: &Options,
        edit: impl FnOnce(&mut Plan),
    ) -> Result<LocalFederation, NodeError> {
        std::fs::create_dir_all(root)?;
        let check = |e: &dyn std::fmt::Display| NodeError::Check(e.to_string());
        let removals = match options.min_helpers {
            0 => 0,
            floor => (options.helpers as u32).saturating_sub(floor),
        };
        let (encoding, max_weight, privacy) = match &options.privacy {
            Some(request) => {
                let (encoding, privacy) = request
                    .terms(
                        layout.coords(),
                        options.rounds + options.retries + removals,
                        options.scoring.unwrap_or(options.min_cohort),
                    )
                    .map_err(|e| check(&e))?;
                (encoding, 1, Some(privacy))
            }
            None => (options.encoding, options.max_weight, None),
        };
        let named = || -> Result<Identity, NodeError> {
            let identity = Identity::generate(options.sig_alg)?;
            Ok(match &options.did_bio {
                Some(network) => identity.on_registry(network)?,
                None => identity,
            })
        };
        let new = || named().map(Arc::new);
        let sponsor = new()?;
        let mut aggregator = named()?;
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

        let status_dir = root.join("status");
        let (issuers, standing) = if options.credentials {
            std::fs::create_dir_all(&status_dir)?;
            let now = unix_now();
            let issuers = Issuers::new(now, options.sig_alg)?;
            let mut standing = Vec::with_capacity(sites.len());
            for (i, site) in sites.iter().enumerate() {
                let dataset = Identity::generate(SigAlg::Ed25519)?.did_string();
                standing.push(Standing {
                    credentials: issuers.wallet(i, &site.did_string(), &dataset, now)?,
                    dataset: Some(dataset),
                    status_dir: Some(status_dir.clone()),
                });
            }
            for issuer in [&issuers.university, &issuers.board, &issuers.committee] {
                issuers.publish(issuer, &[], &status_dir)?;
            }
            (Some(issuers), standing)
        } else {
            (None, vec![Standing::default(); sites.len()])
        };

        let mut plan = Plan {
            version: 1,
            nonce: hex::encode(random_bytes::<16>()),
            purpose: options.purpose.clone(),
            sponsor: sponsor.did_string(),
            aggregator: AggregatorSpec {
                did: aggregator.did_string(),
                tls_pin: pin,
                credentials: Vec::new(),
            },
            helpers: helpers
                .iter()
                .map(|h| {
                    Ok(HelperSpec {
                        did: h.did_string(),
                        seal: h.seal_public().clone(),
                        seal_proof: h.seal_proof()?,
                        credentials: Vec::new(),
                    })
                })
                .collect::<Result<_, NodeError>>()?,
            sites: sites.iter().map(|s| s.did_string()).collect(),
            layout,
            init_model: Digest::file(initial_model)?,
            rounds: options.rounds,
            min_cohort: options.min_cohort,
            chunk_coords: options.chunk_coords,
            encoding,
            max_weight,
            audits: options.audits,
            timeouts: options.timeouts,
            privacy,
            admission: issuers.as_ref().map(Issuers::policy),
            sample: options.sample,
            bounded: options.bounded,
            scoring: options.scoring.map(|group| Scoring { group }),
            retries: options.retries,
            min_helpers: options.min_helpers,
            server_adam: None,
        };
        edit(&mut plan);
        plan.validate().map_err(|e| check(&e))?;
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
            max_budget: privacy.map(|p| p.budget),
            max_retries: (plan.retries > 0).then_some(plan.retries),
            min_helpers: (plan.min_helpers > 0).then_some(plan.min_helpers),
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
            issuers,
            standing,
            cheats: options.cheats.clone(),
        })
    }

    pub fn status_dir(&self) -> PathBuf {
        self.root.join("status")
    }

    /// Has the data access committee take back a site's access grant. The
    /// newer list reaches the log before the next round opens.
    pub fn revoke_access(&self, site: usize) -> Result<(), NodeError> {
        let issuers = self
            .issuers
            .as_ref()
            .ok_or_else(|| NodeError::Check("this federation has no credentials".into()))?;
        issuers.publish(&issuers.committee, &[site as u64], &self.status_dir())
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
            std::fs::write(
                self.root.join(format!("site-{i}.grant.json")),
                to_json(&grant)?,
            )?;
            if self.issuers.is_some() {
                let standing = StandingFile {
                    dataset: self.standing[i].dataset.clone(),
                    credentials: self.standing[i].credentials.clone(),
                };
                let path = self.root.join(format!("site-{i}.standing.json"));
                std::fs::write(path, to_json(&standing)?)?;
            }
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
            status_dir: self.issuers.as_ref().map(|_| self.status_dir()),
            watchers: Vec::new(),
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
            standing: self.standing[site].clone(),
            registry: None,
            cheat: self
                .plan
                .privacy
                .zip(self.cheats.get(&site))
                .map(|(mut terms, reach)| {
                    terms.clip = terms.clip.saturating_mul(*reach);
                    terms
                }),
        }
    }
}
