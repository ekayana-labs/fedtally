use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};
use fedtally_chain::{Anchor, Role, Stakes, Watching};
use fedtally_core::admission::{self, Terms};
use fedtally_core::credential::{Credential, Draft, StatusEntry};
use fedtally_core::keys::{Identity, SigAlg, random_bytes};
use fedtally_core::model::{Encoding, Layout, ModelFile, TensorView};
use fedtally_core::plan::{AggregatorSpec, HelperSpec, Plan, Timeouts};
use fedtally_core::policy::Policy;
use fedtally_core::pq_ucan::delegation::Delegation;
use fedtally_core::pq_ucan::did::Did;
use fedtally_core::pq_ucan::time::Timestamp;
use fedtally_core::privacy::{self, Privacy};
use fedtally_core::record::{Body, Record, encode_token};
use fedtally_core::report::verify_log;
use fedtally_core::scoring::Scoring;
use fedtally_core::status::StatusList;
use fedtally_core::vector::Digest;
use fedtally_node::aggregator::{AggregatorConfig, run_aggregator};
use fedtally_node::helper::{HelperConfig, run_helper};
use fedtally_node::local::{GrantFile, StandingFile};
use fedtally_node::net::{generate_tls, tls_pin};
use fedtally_node::registry::RpcRegistry;
use fedtally_node::site::{RevocationFile, SiteConfig, Submit, TrainContext, Trainer, run_site};
use serde_json::json;

type Failure = Box<dyn std::error::Error>;

#[derive(Parser)]
#[command(
    name = "fedtally",
    version,
    about = "Federated averaging that every party can check"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Alg {
    Ed25519,
    #[value(name = "ml-dsa-65")]
    MlDsa65,
}

#[derive(Clone, Copy, ValueEnum)]
enum Party {
    Publisher,
    Aggregator,
    Site,
    Watcher,
    Issuer,
}

#[derive(Subcommand)]
enum Command {
    /// Create a key file and print the public details for a plan
    Keygen {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "ed25519")]
        alg: Alg,
        /// Also create the TLS certificate an aggregator needs
        #[arg(long)]
        tls: bool,
    },
    /// Print the public details of a key file
    Whoami {
        #[arg(long)]
        key: PathBuf,
    },
    /// As a site, let someone ask this site to contribute, on your conditions
    Grant {
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        to: String,
        #[arg(long)]
        purpose: Option<String>,
        /// Only join plans using one of these helpers. Repeatable.
        #[arg(long = "trust-helper")]
        trust_helpers: Vec<String>,
        /// Accept any helper the sponsor names, including one it runs itself
        #[arg(long)]
        any_helper: bool,
        /// The smallest round to be part of. Required unless --allow-plain
        #[arg(long)]
        min_cohort: Option<u32>,
        #[arg(long)]
        max_rounds: Option<u32>,
        /// Only be averaged with these sites. Repeatable.
        #[arg(long = "allow-site")]
        allowed_sites: Vec<String>,
        /// Accept a plan with no helpers, where the aggregator reads every update
        #[arg(long)]
        allow_plain: bool,
        /// The fewest audits per site to accept
        #[arg(long)]
        min_audits: Option<u32>,
        /// Only join a private plan that spends at most this epsilon, at --delta
        #[arg(long, requires = "delta")]
        max_epsilon: Option<f64>,
        #[arg(long, requires = "max_epsilon")]
        delta: Option<f64>,
        /// Only join a private plan that assumes this many sites give their noise away
        #[arg(long)]
        assume_colluding: Option<u32>,
        /// Join a plan that may redo a round this many times. Needs --max-epsilon
        #[arg(long)]
        max_retries: Option<u32>,
        #[arg(long, default_value_t = 90)]
        days: u32,
        #[arg(long)]
        out: PathBuf,
    },
    /// As the sponsor, sign a plan from a JSON description
    Plan {
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// As the sponsor, exercise a site's grant for a plan
    Invoke {
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        site: String,
        /// Delegations from the site down to the sponsor, in order. Repeatable.
        #[arg(long = "grant", required = true)]
        grants: Vec<PathBuf>,
        #[arg(long, default_value_t = 30)]
        days: u32,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run the aggregator until the run ends
    /// Run a site, training with a command of your own
    ///
    /// Each round the command runs with FEDTALLY_ROUND, FEDTALLY_BASE (the
    /// model to start from) and FEDTALLY_OUT (where to write the trained one,
    /// as safetensors). The last line it prints is its weight, 1 if it prints
    /// no number.
    Site {
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        grant: PathBuf,
        #[arg(long)]
        connect: String,
        #[arg(long)]
        dir: PathBuf,
        /// This site's dataset and credentials, for a plan that admits on them
        #[arg(long)]
        standing: Option<PathBuf>,
        /// A folder of issuers' revocation lists this site brings to the log
        #[arg(long)]
        status_dir: Option<PathBuf>,
        /// Delegations this site has taken back, one per line
        #[arg(long)]
        revocations: Option<PathBuf>,
        /// Where to ask whether did:bio parties still stand
        #[arg(long)]
        registry_rpc: Option<String>,
        #[arg(long, default_value_t = 600)]
        give_up_secs: u64,
        /// The training command and its arguments
        #[arg(last = true, required = true)]
        train: Vec<String>,
    },
    Aggregator {
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        dir: PathBuf,
        #[arg(long, default_value = "127.0.0.1:7443")]
        listen: String,
        #[arg(long)]
        keep_spool: bool,
        /// A folder of issuers' revocation lists, added to the log as they change
        #[arg(long)]
        status_dir: Option<PathBuf>,
        /// A party outside the plan that may read the log and each round's vectors. Repeatable.
        #[arg(long = "watcher")]
        watchers: Vec<String>,
    },
    /// Post to the ledger what a role owes, as the log calls for it
    Anchor {
        #[arg(long, value_enum)]
        role: Party,
        #[arg(long)]
        rpc: String,
        #[arg(long)]
        program: String,
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        /// The folder of this party's node, whose log is followed. A watcher keeps its own here.
        #[arg(long, required_unless_present = "stakes")]
        dir: Option<PathBuf>,
        /// As a watcher, the aggregator to follow
        #[arg(long)]
        connect: Option<String>,
        /// Micro-lamports per compute unit to offer on every transaction
        #[arg(long)]
        priority_fee: Option<u64>,
        /// As the publisher, six amounts: the site stake, the aggregator stake, the reward
        /// per round and the dispute bond in lamports, then the challenge window and the
        /// stall limit in slots
        #[arg(long, num_args = 6, value_names = ["SITE", "AGGREGATOR", "REWARD", "BOND", "CHALLENGE", "STALL"])]
        stakes: Option<Vec<u64>>,
    },
    /// Run a helper until the run ends
    Helper {
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        dir: PathBuf,
        #[arg(long)]
        connect: String,
    },
    /// Check a log. Exit 0 if complete, 1 if it fails, 2 if valid with gaps
    Verify {
        log: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
        /// List the rounds this site's data went into
        #[arg(long)]
        site: Option<String>,
    },
    /// As an issuer, sign a credential about a site, a dataset or an operator
    Issue {
        #[arg(long)]
        key: PathBuf,
        /// The credential type, such as ControlledAccessGrants
        #[arg(long = "type")]
        kind: String,
        /// The DID the credential is about
        #[arg(long)]
        subject: String,
        /// A JSON object of claims about the subject
        #[arg(long)]
        claims: Option<PathBuf>,
        #[arg(long, default_value_t = 365)]
        days: u32,
        /// The issuer's revocation list this credential answers to
        #[arg(long, requires = "index")]
        list: Option<String>,
        #[arg(long, requires = "list")]
        index: Option<u64>,
        #[arg(long)]
        out: PathBuf,
    },
    /// As an issuer, sign a revocation list with these entries revoked
    Revoke {
        #[arg(long)]
        key: PathBuf,
        /// The list's id, as named in the credentials that answer to it
        #[arg(long)]
        list: String,
        /// An index to revoke. Repeatable. Give every index revoked so far.
        #[arg(long = "index")]
        indexes: Vec<u64>,
        #[arg(long)]
        out: PathBuf,
    },
    /// As a site, gather what it holds for a plan that admits on credentials
    Standing {
        /// The dataset this site trains on
        #[arg(long)]
        dataset: Option<String>,
        /// A credential file. Repeatable.
        #[arg(long = "credential")]
        credentials: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Check a credential or a revocation list and print what it says
    Show { file: PathBuf },
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanSpec {
    purpose: String,
    aggregator: AggregatorSpec,
    #[serde(default)]
    helpers: Vec<HelperSpec>,
    sites: Vec<String>,
    layout: Layout,
    rounds: u32,
    min_cohort: u32,
    chunk_coords: u64,
    /// Left out of a private plan, which derives both from its clip.
    #[serde(default)]
    encoding: Option<Encoding>,
    #[serde(default)]
    max_weight: Option<u64>,
    #[serde(default = "one_audit")]
    audits: u32,
    timeouts: Timeouts,
    #[serde(default)]
    privacy: Option<privacy::Request>,
    #[serde(default)]
    admission: Option<Policy>,
    #[serde(default)]
    sample: Option<u32>,
    #[serde(default)]
    bounded: bool,
    #[serde(default)]
    scoring: Option<Scoring>,
    #[serde(default)]
    retries: u32,
}

impl PlanSpec {
    fn terms(&self) -> Result<(Encoding, u64, Option<Privacy>), Failure> {
        let Some(spec) = &self.privacy else {
            let encoding = self.encoding.ok_or("the spec needs an encoding")?;
            return Ok((
                encoding,
                self.max_weight.ok_or("the spec needs max_weight")?,
                None,
            ));
        };
        if self.encoding.is_some() || self.max_weight.is_some() {
            return Err("a private plan sets its own encoding and weights: leave them out".into());
        }
        let (encoding, terms) = spec.terms(
            self.layout.coords(),
            self.rounds + self.retries,
            self.scoring.map_or(self.min_cohort, |s| s.group),
        )?;
        Ok((encoding, 1, Some(terms)))
    }
}

/// Trains by running a command, which needs no Python in this process.
struct CommandTrainer {
    command: Vec<String>,
    layout: Layout,
}

impl Trainer for CommandTrainer {
    fn train(&mut self, ctx: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        let fail = |e: &dyn std::fmt::Display| e.to_string();
        let out = ctx
            .base_model
            .with_extension(format!("trained-{}", ctx.round));
        let ran = std::process::Command::new(&self.command[0])
            .args(&self.command[1..])
            .env("FEDTALLY_ROUND", ctx.round.to_string())
            .env("FEDTALLY_BASE", &ctx.base_model)
            .env("FEDTALLY_OUT", &out)
            .stderr(std::process::Stdio::inherit())
            .output()
            .map_err(|e| fail(&e))?;
        if !ran.status.success() {
            return Err(format!("the training command ended with {}", ran.status));
        }
        let text = String::from_utf8_lossy(&ran.stdout);
        let weight = text
            .lines()
            .last()
            .and_then(|line| line.trim().parse().ok())
            .unwrap_or(1);
        let tensors = ModelFile::open(&out, &self.layout)
            .map_err(|e| fail(&e))?
            .read_all()
            .map_err(|e| fail(&e))?;
        let views: Vec<TensorView<'_>> = tensors.iter().map(|t| TensorView::F64(t)).collect();
        submit(&views, weight).map_err(|e| fail(&e))?;
        let _ = std::fs::remove_file(&out);
        Ok(())
    }
}

fn one_audit() -> u32 {
    1
}

fn unix_now() -> Result<i64, Failure> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
    Ok(now.as_secs() as i64)
}

fn expiry(days: u32) -> Result<Timestamp, Failure> {
    Ok(Timestamp::from_unix(
        unix_now()? + i64::from(days) * 86_400,
    )?)
}

fn public_details(identity: &Identity) -> Result<serde_json::Value, Failure> {
    let pin = match &identity.tls {
        Some(credentials) => Some(tls_pin(credentials)?),
        None => None,
    };
    Ok(json!({
        "did": identity.did_string(),
        "seal": identity.seal_public(),
        "seal_proof": identity.seal_proof()?,
        "tls_pin": pin,
    }))
}

fn read_plan(path: &PathBuf) -> Result<(Vec<u8>, Record, Plan), Failure> {
    let line = std::fs::read(path)?;
    let line = line.strip_suffix(b"\n").map(<[u8]>::to_vec).unwrap_or(line);
    let record = Record::parse(&line)?;
    let Body::Plan(plan) = &record.body else {
        return Err("that file does not hold a plan".into());
    };
    let plan = (**plan).clone();
    Ok((line, record, plan))
}

async fn run(command: Command) -> Result<ExitCode, Failure> {
    match command {
        Command::Keygen { out, alg, tls } => {
            let alg = match alg {
                Alg::Ed25519 => SigAlg::Ed25519,
                Alg::MlDsa65 => SigAlg::MlDsa65,
            };
            let mut identity = Identity::generate(alg)?;
            if tls {
                identity.tls = Some(generate_tls()?);
            }
            identity.save(&out)?;
            println!("{}", public_details(&identity)?);
        }
        Command::Whoami { key } => println!("{}", public_details(&Identity::load(&key)?)?),
        Command::Grant {
            key,
            to,
            purpose,
            trust_helpers,
            any_helper,
            min_cohort,
            max_rounds,
            allowed_sites,
            allow_plain,
            min_audits,
            max_epsilon,
            delta,
            assume_colluding,
            max_retries,
            days,
            out,
        } => {
            if !allow_plain {
                if trust_helpers.is_empty() && !any_helper {
                    return Err(
                        "say which helpers this site trusts: --trust-helper or --any-helper".into(),
                    );
                }
                if min_cohort.is_none() {
                    return Err("say how small a round this site will join: --min-cohort".into());
                }
            }
            let site = Identity::load(&key)?;
            let terms = Terms {
                purpose,
                trusted_helpers: trust_helpers,
                min_cohort,
                max_rounds,
                allowed_sites,
                allow_plain,
                min_audits,
                max_budget: max_epsilon
                    .zip(delta)
                    .map(|(epsilon, delta)| privacy::budget_for(epsilon, delta)),
                min_colluding: assume_colluding,
                max_retries,
            };
            let grant = admission::grant(&site, &Did::parse(&to)?, &terms, expiry(days)?)?;
            std::fs::write(&out, grant.bytes())?;
            println!("{}", grant.cid());
        }
        Command::Plan {
            key,
            spec,
            model,
            out,
        } => {
            let sponsor = Identity::load(&key)?;
            let spec: PlanSpec = serde_json::from_slice(&std::fs::read(&spec)?)?;
            let (encoding, max_weight, privacy) = spec.terms()?;
            let plan = Plan {
                version: 1,
                nonce: hex::encode(random_bytes::<16>()),
                purpose: spec.purpose,
                sponsor: sponsor.did_string(),
                aggregator: spec.aggregator,
                helpers: spec.helpers,
                sites: spec.sites,
                layout: spec.layout,
                init_model: Digest::file(&model)?,
                rounds: spec.rounds,
                min_cohort: spec.min_cohort,
                chunk_coords: spec.chunk_coords,
                encoding,
                max_weight,
                audits: spec.audits,
                timeouts: spec.timeouts,
                privacy,
                admission: spec.admission,
                sample: spec.sample,
                bounded: spec.bounded,
                scoring: spec.scoring,
                retries: spec.retries,
            };
            plan.validate()?;
            let record = Record::sign(&sponsor, &Body::Plan(Box::new(plan)))?;
            std::fs::write(&out, &record.line)?;
            println!("{}", record.body_hash);
        }
        Command::Invoke {
            key,
            plan,
            site,
            grants,
            days,
            out,
        } => {
            let sponsor = Identity::load(&key)?;
            let (_, record, plan) = read_plan(&plan)?;
            let chain = grants
                .iter()
                .map(|path| Ok(Delegation::decode(&std::fs::read(path)?)?))
                .collect::<Result<Vec<_>, Failure>>()?;
            let refs: Vec<&Delegation> = chain.iter().collect();
            let invocation = admission::invoke(
                &sponsor,
                &record.body_hash,
                &plan,
                &Did::parse(&site)?,
                &refs,
                expiry(days)?,
            )?;
            let file = GrantFile {
                invocation: encode_token(invocation.bytes()),
                proofs: chain.iter().map(|d| encode_token(d.bytes())).collect(),
            };
            std::fs::write(&out, serde_json::to_vec(&file)?)?;
        }
        Command::Site {
            key,
            plan,
            grant,
            connect,
            dir,
            standing,
            status_dir,
            revocations,
            registry_rpc,
            give_up_secs,
            train,
        } => {
            let (plan_line, _, body) = read_plan(&plan)?;
            let grant = GrantFile::load(&grant)?;
            let standing = match standing {
                Some(path) => StandingFile::load(&path, status_dir)?,
                None => Default::default(),
            };
            let config = SiteConfig {
                identity: Arc::new(Identity::load(&key)?),
                plan_line,
                addr: connect,
                dir,
                invocation: grant.invocation,
                proofs: grant.proofs,
                revocations: revocations.map(|path| Arc::new(RevocationFile(path)) as _),
                give_up_after: Duration::from_secs(give_up_secs),
                standing,
                registry: registry_rpc.map(|url| Arc::new(RpcRegistry { url }) as _),
                cheat: None,
            };
            let trainer = CommandTrainer {
                command: train,
                layout: body.layout,
            };
            let summary = run_site(config, Box::new(trainer)).await?;
            println!("final model {}", summary.final_digest);
        }
        Command::Aggregator {
            key,
            plan,
            model,
            dir,
            listen,
            keep_spool,
            status_dir,
            watchers,
        } => {
            let (plan_line, _, _) = read_plan(&plan)?;
            let replay = run_aggregator(AggregatorConfig {
                identity: Arc::new(Identity::load(&key)?),
                plan_line,
                listen,
                dir,
                initial_model: model,
                hook: None,
                on_ready: Some(Box::new(|addr| println!("listening on {addr}"))),
                keep_spool,
                status_dir,
                watchers,
            })
            .await?;
            println!("final model {}", replay.model);
        }
        Command::Anchor {
            role,
            rpc,
            program,
            key,
            plan,
            dir,
            connect,
            priority_fee,
            stakes,
        } => {
            let (plan_line, _, _) = read_plan(&plan)?;
            let program: [u8; 32] = bs58::decode(&program)
                .into_vec()?
                .try_into()
                .map_err(|_| "a program id is 32 bytes")?;
            let dir = dir.unwrap_or_default();
            let role = match (role, stakes.as_deref(), connect) {
                (
                    Party::Publisher,
                    Some(&[site, aggregator, reward, bond, challenge, stall]),
                    _,
                ) => Role::Publisher(Stakes {
                    site_stake: site,
                    aggregator_stake: aggregator,
                    reward_per_round: reward,
                    dispute_bond: bond,
                    challenge_slots: challenge,
                    stall_slots: stall,
                }),
                (Party::Publisher, _, _) => return Err("a publisher sets the --stakes".into()),
                (Party::Aggregator, _, _) => Role::Aggregator,
                (Party::Site, _, _) => Role::Site,
                (Party::Issuer, _, _) => Role::Issuer,
                (Party::Watcher, _, Some(addr)) => Role::Watcher(Watching {
                    addr,
                    dir: dir.clone(),
                    runtime: tokio::runtime::Handle::current(),
                }),
                (Party::Watcher, _, None) => {
                    return Err("a watcher needs the aggregator to follow: --connect".into());
                }
            };
            let mut anchor = Anchor::new(fedtally_chain::Config {
                role,
                identity: Arc::new(Identity::load(&key)?),
                plan_line,
                log: dir.join("log.jsonl"),
                rpc,
                program,
                price: priority_fee,
                poll: Duration::from_secs(1),
                give_up_after: Duration::from_secs(600),
                sink: None,
            })?;
            println!("task {}", bs58::encode(anchor.task()).into_string());
            let never = std::sync::atomic::AtomicBool::new(false);
            tokio::task::spawn_blocking(move || anchor.run(&never)).await??;
        }
        Command::Helper {
            key,
            plan,
            dir,
            connect,
        } => {
            let (plan_line, _, _) = read_plan(&plan)?;
            run_helper(HelperConfig {
                identity: Arc::new(Identity::load(&key)?),
                plan_line,
                addr: connect,
                dir,
                give_up_after: Duration::from_secs(600),
                share_hook: None,
            })
            .await?;
        }
        Command::Verify { log, model, site } => {
            let report = match verify_log(&log, model.as_deref()) {
                Ok(report) => report,
                Err(e) => {
                    eprintln!("FAILED: {e}");
                    return Ok(ExitCode::from(1));
                }
            };
            if let Some(site) = site {
                println!("{}", serde_json::to_string(&report.rounds_with(&site))?);
                return Ok(ExitCode::SUCCESS);
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.complete() {
                eprintln!("valid, but with gaps: the run did not finish or a site was left out");
                return Ok(ExitCode::from(2));
            }
        }
        Command::Issue {
            key,
            kind,
            subject,
            claims,
            days,
            list,
            index,
            out,
        } => {
            let issuer = Identity::load(&key)?;
            let claims = match claims {
                Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
                None => serde_json::Map::new(),
            };
            Did::parse(&subject)?;
            let now = unix_now()?;
            let draft = Draft {
                kind,
                subject,
                claims,
                valid_from: now,
                valid_until: Some(now + i64::from(days) * 86_400),
                status: list
                    .zip(index)
                    .map(|(list, index)| StatusEntry { list, index }),
            };
            let credential = Credential::issue(&issuer, &draft, now)?;
            std::fs::write(&out, &credential.text)?;
            println!("{}", credential.digest);
        }
        Command::Revoke {
            key,
            list,
            indexes,
            out,
        } => {
            let issuer = Identity::load(&key)?;
            let list = StatusList::issue(&issuer, &list, &indexes, unix_now()?)?;
            std::fs::write(&out, &list.text)?;
            println!("{} revoked", indexes.len());
        }
        Command::Standing {
            dataset,
            credentials,
            out,
        } => {
            let credentials = credentials
                .iter()
                .map(|path| {
                    let text = std::fs::read_to_string(path)?;
                    Ok(Credential::parse(text.trim())?.text)
                })
                .collect::<Result<Vec<_>, Failure>>()?;
            let standing = StandingFile {
                dataset,
                credentials,
            };
            std::fs::write(&out, serde_json::to_vec(&standing)?)?;
        }
        Command::Show { file } => {
            let text = std::fs::read_to_string(&file)?;
            let credential = Credential::parse(text.trim())?;
            let mut shown = json!({
                "issuer": credential.issuer,
                "subject": credential.subject,
                "types": credential.types,
                "claims": credential.claims,
                "valid_from": credential.valid_from,
                "valid_until": credential.valid_until,
                "valid_now": credential.valid_at(unix_now()?),
                "digest": credential.digest,
            });
            if let Some(status) = &credential.status {
                shown["status"] = json!({"list": status.list, "index": status.index});
            }
            if let Ok(list) = StatusList::parse(text.trim()) {
                shown["list"] = json!({"id": list.id, "issued": list.issued});
            }
            println!("{}", serde_json::to_string_pretty(&shown)?);
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    match run(Cli::parse().command).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
