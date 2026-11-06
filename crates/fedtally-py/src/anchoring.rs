//! Anchoring from Python: the same roles as `fedtally anchor`, on threads of
//! their own beside the run, with the GIL released.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use fedtally_chain::ledger::Sink;
use fedtally_chain::{Anchor, ChainError, Config, Role, Stakes, Watching};
use fedtally_core::anchor;
use fedtally_core::keys::Identity;
use fedtally_node::local::LocalFederation;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const GIVE_UP_AFTER: Duration = Duration::from_secs(600);

/// The ledger a run anchors to, as `fedtally.Ledger` hands it over.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub struct Spec {
    pub rpc: String,
    pub program: Option<String>,
    pub priority_fee: Option<u64>,
    pub poll_ms: u64,
    /// The six amounts of `fedtally anchor --stakes`, for the publisher.
    pub stakes: Option<[u64; 6]>,
    pub watch: bool,
}

fn stakes_of([site, aggregator, reward, bond, challenge, stall]: [u64; 6]) -> Stakes {
    Stakes {
        site_stake: site,
        aggregator_stake: aggregator,
        reward_per_round: reward,
        dispute_bond: bond,
        challenge_slots: challenge,
        stall_slots: stall,
    }
}

impl Spec {
    fn config(
        &self,
        role: Role,
        identity: Arc<Identity>,
        plan_line: Vec<u8>,
        log: PathBuf,
        sink: Option<Sink>,
    ) -> PyResult<Config> {
        Ok(Config {
            role,
            identity,
            plan_line,
            log,
            rpc: self.rpc.clone(),
            program: program_of(self.program.as_deref())?,
            price: self.priority_fee,
            poll: Duration::from_millis(self.poll_ms),
            give_up_after: GIVE_UP_AFTER,
            sink,
        })
    }
}

fn program_of(given: Option<&str>) -> PyResult<[u8; 32]> {
    let Some(given) = given else {
        return Ok(*fedtally_chain::program::ID.as_array());
    };
    bs58::decode(given)
        .into_vec()
        .ok()
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| PyValueError::new_err("a program id is 32 bytes of base58"))
}

/// Every party's wallet, to fund before a run. A plan whose parties aren't
/// named by `did:bio` or a key has none, and can't be anchored.
pub fn wallets<'py>(
    py: Python<'py>,
    fed: &LocalFederation,
    watcher: &Identity,
) -> PyResult<Bound<'py, PyDict>> {
    let wallet = |did: &str| -> PyResult<String> {
        let address = anchor::wallet(did).map_err(PyValueError::new_err)?;
        Ok(bs58::encode(address).into_string())
    };
    let out = PyDict::new(py);
    out.set_item("publisher", wallet(&fed.plan.sponsor)?)?;
    out.set_item("aggregator", wallet(&fed.plan.aggregator.did)?)?;
    out.set_item("watcher", wallet(&watcher.did_string())?)?;
    let sites = fed.plan.sites.iter().map(|site| wallet(site));
    out.set_item("sites", sites.collect::<PyResult<Vec<_>>>()?)?;
    Ok(out)
}

type Ended = Result<(), ChainError>;

/// The anchors of one run.
pub struct Anchors {
    task: [u8; 32],
    threads: Vec<(String, JoinHandle<Ended>)>,
    stop: Arc<AtomicBool>,
    sent: Arc<Mutex<BTreeMap<&'static str, u64>>>,
}

fn spawn(
    config: Config,
    stop: Arc<AtomicBool>,
) -> Result<([u8; 32], JoinHandle<Ended>), ChainError> {
    let mut anchor = Anchor::new(config)?;
    let task = *anchor.task();
    Ok((task, std::thread::spawn(move || anchor.run(&stop))))
}

impl Anchors {
    /// Starts every party's anchor. The aggregator is up at `addr`.
    pub fn start(
        spec: &Spec,
        fed: &LocalFederation,
        watcher: &Arc<Identity>,
        addr: &str,
        runtime: tokio::runtime::Handle,
    ) -> PyResult<Anchors> {
        let stakes = spec
            .stakes
            .ok_or_else(|| PyValueError::new_err("a ledger for a whole run needs its stakes"))?;
        let sent: Arc<Mutex<BTreeMap<&'static str, u64>>> = Arc::default();
        let counted = sent.clone();
        let sink: Sink = Arc::new(move |one| {
            *counted.lock().unwrap().entry(one.what).or_default() += 1;
        });
        let log = |dir: PathBuf| dir.join("log.jsonl");

        let mut parties = vec![
            (
                "publisher".to_owned(),
                Role::Publisher(stakes_of(stakes)),
                fed.sponsor.clone(),
                PathBuf::new(),
            ),
            (
                "aggregator".to_owned(),
                Role::Aggregator,
                fed.aggregator.clone(),
                log(fed.aggregator_dir()),
            ),
        ];
        for (site, identity) in fed.sites.iter().enumerate() {
            parties.push((
                format!("site {site}"),
                Role::Site,
                identity.clone(),
                log(fed.site_dir(site)),
            ));
        }
        if spec.watch {
            let watching = Watching {
                addr: addr.to_owned(),
                dir: fed.root.join("watcher"),
                runtime,
            };
            parties.push((
                "watcher".to_owned(),
                Role::Watcher(watching),
                watcher.clone(),
                PathBuf::new(),
            ));
        }

        let stop = Arc::new(AtomicBool::new(false));
        let mut anchors = Anchors {
            task: [0; 32],
            threads: Vec::new(),
            stop: stop.clone(),
            sent,
        };
        for (name, role, identity, log) in parties {
            let config = spec.config(
                role,
                identity,
                fed.plan_line.clone(),
                log,
                Some(sink.clone()),
            )?;
            let (task, thread) =
                spawn(config, stop.clone()).map_err(|e| PyValueError::new_err(e.to_string()))?;
            anchors.task = task;
            anchors.threads.push((name, thread));
        }
        Ok(anchors)
    }

    /// Waits for every anchor. With `give_up`, the run has failed and what it
    /// still owed will never come, so they are told to stop.
    pub fn finish(self, give_up: bool) -> Report {
        self.stop.store(give_up, Ordering::Relaxed);
        let errors = self
            .threads
            .into_iter()
            .filter_map(|(name, thread)| match thread.join() {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some((name, e.to_string())),
                Err(_) => Some((name, "it panicked".to_owned())),
            })
            .collect();
        Report {
            task: bs58::encode(self.task).into_string(),
            errors,
            sent: self.sent.lock().unwrap().clone(),
        }
    }
}

pub struct Report {
    task: String,
    errors: BTreeMap<String, String>,
    sent: BTreeMap<&'static str, u64>,
}

impl Report {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("task", &self.task)?;
        out.set_item("errors", &self.errors)?;
        out.set_item("sent", &self.sent)?;
        Ok(out)
    }
}

/// Runs one party's anchor until nothing more is owed. For deployments, where
/// every party is its own process.
#[pyfunction]
#[pyo3(signature = (role, key, plan, ledger, dir = None, connect = None))]
pub fn run_anchor(
    py: Python<'_>,
    role: &str,
    key: PathBuf,
    plan: PathBuf,
    ledger: Spec,
    dir: Option<PathBuf>,
    connect: Option<String>,
) -> PyResult<String> {
    let fail = |e: &dyn std::fmt::Display| crate::fail(e);
    let line = std::fs::read(&plan).map_err(|e| fail(&e))?;
    let plan_line = line.strip_suffix(b"\n").map(<[u8]>::to_vec).unwrap_or(line);
    let identity = Arc::new(Identity::load(&key).map_err(|e| fail(&e))?);
    let dir = dir.unwrap_or_default();
    let runtime = crate::runtime()?;
    let role = match role {
        "publisher" => {
            let stakes = ledger
                .stakes
                .ok_or_else(|| PyValueError::new_err("a publisher sets the stakes"))?;
            Role::Publisher(stakes_of(stakes))
        }
        "aggregator" => Role::Aggregator,
        "site" => Role::Site,
        "issuer" => Role::Issuer,
        "watcher" => Role::Watcher(Watching {
            addr: connect.ok_or_else(|| {
                PyValueError::new_err("a watcher needs the aggregator to follow: connect")
            })?,
            dir: dir.clone(),
            runtime: runtime.handle().clone(),
        }),
        other => return Err(PyValueError::new_err(format!("no such role: {other}"))),
    };
    let config = ledger.config(role, identity, plan_line, dir.join("log.jsonl"), None)?;
    py.detach(move || {
        let _entered = runtime.enter();
        let mut anchor = Anchor::new(config).map_err(|e| fail(&e))?;
        let never = AtomicBool::new(false);
        anchor.run(&never).map_err(|e| fail(&e))?;
        Ok(bs58::encode(anchor.task()).into_string())
    })
}
