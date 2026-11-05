use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};
use fedtally_core::admission::{self, Terms};
use fedtally_core::keys::{Identity, SigAlg, random_bytes};
use fedtally_core::model::{Encoding, Layout};
use fedtally_core::plan::{AggregatorSpec, HelperSpec, Plan, Timeouts};
use fedtally_core::pq_ucan::delegation::Delegation;
use fedtally_core::pq_ucan::did::Did;
use fedtally_core::pq_ucan::time::Timestamp;
use fedtally_core::record::{Body, Record, encode_token};
use fedtally_core::report::verify_log;
use fedtally_core::vector::Digest;
use fedtally_node::aggregator::{AggregatorConfig, run_aggregator};
use fedtally_node::helper::{HelperConfig, run_helper};
use fedtally_node::local::GrantFile;
use fedtally_node::net::{generate_tls, tls_pin};
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
    encoding: Encoding,
    max_weight: u64,
    #[serde(default = "one_audit")]
    audits: u32,
    timeouts: Timeouts,
}

fn one_audit() -> u32 {
    1
}

fn expiry(days: u32) -> Result<Timestamp, Failure> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
    Ok(Timestamp::from_unix(
        now.as_secs() as i64 + i64::from(days) * 86_400,
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
                encoding: spec.encoding,
                max_weight: spec.max_weight,
                audits: spec.audits,
                timeouts: spec.timeouts,
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
        Command::Aggregator {
            key,
            plan,
            model,
            dir,
            listen,
            keep_spool,
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
            })
            .await?;
            println!("final model {}", replay.model);
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
