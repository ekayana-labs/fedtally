//! The four roles against a local validator. A real federation trains over
//! TLS while its parties anchor it with their own keys, first honestly and
//! then with an aggregator that posts an aggregate it did not compute.
//!
//! Ignored by default, since it needs `solana-test-validator` and the built
//! program:
//!
//! ```console
//! cargo build-sbf --manifest-path ledger/program/Cargo.toml
//! cargo test -p fedtally-chain --test validator -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use fedtally_chain::ledger::{self, Ledger, RoundView, Sent, Sink, TaskView};
use fedtally_chain::program::client;
use fedtally_chain::program::state::{
    ENROLLMENT_SPACE, ROUND_FRAUD, SITE_WITHDRAWN, TASK_ABORTED, TASK_FINISHED,
};
use fedtally_chain::rpc::Rpc;
use fedtally_chain::run::Run;
use fedtally_chain::tx::{BUDGET_IX_UNITS, Wallet};
use fedtally_chain::{Anchor, ChainError, Config, Role, Stakes, Watching};
use fedtally_core::anchor;
use fedtally_core::keys::{Identity, SigAlg};
use fedtally_core::model::{Dtype, Layout, TensorSpec, TensorView, write_model};
use fedtally_core::replay::Replay;
use fedtally_core::vector::{self, LEAF_COORDS};
use fedtally_node::NodeError;
use fedtally_node::aggregator::run_aggregator;
use fedtally_node::helper::run_helper;
use fedtally_node::local::{LocalFederation, Options};
use fedtally_node::site::{Submit, TrainContext, Trainer, run_site};

const SITES: usize = 3;
const ROUNDS: u32 = 2;
const SOL: u64 = 1_000_000_000;
const WALLET: u64 = 20 * SOL;
const AGGREGATOR_STAKE: u64 = 5 * SOL;
const REWARD: u64 = 300_000_000;
const POLL: Duration = Duration::from_millis(200);
const PATIENCE: Duration = Duration::from_secs(300);

/// The README's LiteSVM figure for each instruction, for the nearest case it
/// measures.
const LITESVM: &[(&str, u64)] = &[
    ("round_close", 3935),
    ("publish", 2398),
    ("back", 1722),
    ("enroll", 3371),
    ("open_round", 3035),
    ("commit_root", 3004),
    ("post_aggregate", 957),
    ("ack", 750),
    ("settle", 911),
    ("claim", 1074),
    ("withdraw (site)", 749),
    ("withdraw (aggregator)", 470),
    ("withdraw (publisher)", 651),
    ("buffer_init", 2322),
    ("buffer_write", 427),
    ("dispute_open", 2851),
    ("dispute_leaf (input)", 6164),
    ("dispute_leaf (mask)", 4910),
    ("dispute_leaf (aggregate)", 4006),
    ("dispute_close", 1312),
    ("buffer_close", 380),
];

struct Validator {
    child: Child,
    url: String,
    _ledger: tempfile::TempDir,
}

impl Drop for Validator {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn binary() -> PathBuf {
    let name = "solana-test-validator";
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    let installed = home.join(".local/share/solana/install/active_release/bin");
    std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .chain([installed])
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| panic!("{name} is not on the PATH"))
}

/// A port with `span` free ports from it on.
fn free_ports(span: u16) -> u16 {
    loop {
        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .unwrap()
            .port();
        let free = |p: u16| TcpListener::bind(("127.0.0.1", p)).is_ok();
        if port.checked_add(span).is_some() && (port..port + span).all(free) {
            return port;
        }
    }
}

impl Validator {
    fn start() -> Validator {
        let program = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../ledger/target/deploy/fedtally_ledger.so");
        assert!(
            program.is_file(),
            "{} not found, run `cargo build-sbf --manifest-path ledger/program/Cargo.toml` first",
            program.display()
        );
        let ledger = tempfile::tempdir().unwrap();
        let id = *fedtally_chain::program::ID.as_array();
        let (rpc, faucet, gossip, dynamic) =
            (free_ports(2), free_ports(1), free_ports(1), free_ports(40));
        let child = Command::new(binary())
            .arg("--ledger")
            .arg(ledger.path())
            .args(["--reset", "--quiet", "--bind-address", "127.0.0.1"])
            .args(["--rpc-port", &rpc.to_string()])
            .args(["--faucet-port", &faucet.to_string()])
            .args(["--gossip-port", &gossip.to_string()])
            .args([
                "--dynamic-port-range",
                &format!("{dynamic}-{}", dynamic + 39),
            ])
            // Any address will do. Without one the validator wants a keypair file.
            .args(["--mint", &bs58::encode([7u8; 32]).into_string()])
            .args(["--bpf-program", &bs58::encode(id).into_string()])
            .arg(&program)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let validator = Validator {
            child,
            url: format!("http://127.0.0.1:{rpc}"),
            _ledger: ledger,
        };
        let rpc = validator.rpc();
        let started = Instant::now();
        loop {
            let ready = rpc.slot().is_ok_and(|slot| slot > 0)
                && rpc.account(&id).is_ok_and(|held| held.is_some());
            if ready {
                return validator;
            }
            let waited = started.elapsed();
            assert!(waited < PATIENCE, "the validator did not start");
            std::thread::sleep(POLL);
        }
    }

    fn rpc(&self) -> Rpc {
        Rpc::new(&self.url)
    }

    fn fund(&self, wallets: &[[u8; 32]]) {
        let rpc = self.rpc();
        let pending: Vec<String> = wallets
            .iter()
            .map(|wallet| rpc.request_airdrop(wallet, WALLET).unwrap())
            .collect();
        for signature in pending {
            eventually("an airdrop", || {
                rpc.confirmed(&signature)
                    .unwrap()
                    .map(|ended| ended.unwrap())
            });
        }
    }
}

fn eventually<T>(what: &str, mut check: impl FnMut() -> Option<T>) -> T {
    let started = Instant::now();
    loop {
        if let Some(found) = check() {
            return found;
        }
        assert!(started.elapsed() < PATIENCE, "{what} never came");
        std::thread::sleep(POLL);
    }
}

fn layout() -> Layout {
    Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![300],
        }],
    }
}

/// Lets training past the first round go ahead only once it is opened.
#[derive(Clone, Default)]
struct Gate(Arc<(Mutex<bool>, Condvar)>);

impl Gate {
    fn open(&self) {
        *self.0.0.lock().unwrap() = true;
        self.0.1.notify_all();
    }

    fn pass(&self) {
        let mut open = self.0.0.lock().unwrap();
        while !*open {
            open = self.0.1.wait(open).unwrap();
        }
    }
}

struct Towards(f32, Gate);

impl Trainer for Towards {
    fn train(&mut self, ctx: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        if ctx.round > 1 {
            self.1.pass();
        }
        submit(&[TensorView::F32(&[self.0; 300])], 1)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

/// A federation that is training, and everything sent to the ledger for it.
struct Running {
    _dir: tempfile::TempDir,
    fed: LocalFederation,
    watcher: Arc<Identity>,
    addr: String,
    aggregator: Option<tokio::task::JoinHandle<Result<Replay, NodeError>>>,
    sites: Vec<tokio::task::JoinHandle<Result<(), NodeError>>>,
    gate: Gate,
    sent: Arc<Mutex<Vec<Sent>>>,
    price: Option<u64>,
    url: String,
    window: u64,
}

fn start(
    runtime: &tokio::runtime::Runtime,
    validator: &Validator,
    price: Option<u64>,
    window: u64,
    gated: bool,
) -> Running {
    start_with(runtime, validator, price, window, gated, |_| {})
}

fn start_with(
    runtime: &tokio::runtime::Runtime,
    validator: &Validator,
    price: Option<u64>,
    window: u64,
    gated: bool,
    edit: impl FnOnce(&mut Options),
) -> Running {
    let dir = tempfile::tempdir().unwrap();
    let init = dir.path().join("init.safetensors");
    write_model(&init, &layout(), &[TensorView::F32(&[0.5; 300])]).unwrap();
    let mut options = Options::new(SITES, 1, ROUNDS);
    options.chunk_coords = LEAF_COORDS;
    options.did_bio = Some("localnet".into());
    edit(&mut options);
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options).unwrap();
    let watcher = Arc::new(Identity::generate(SigAlg::Ed25519).unwrap());

    let wallet = |did: &str| anchor::wallet(did).unwrap();
    let mut wallets: Vec<[u8; 32]> = fed.plan.sites.iter().map(|site| wallet(site)).collect();
    wallets.push(wallet(&fed.plan.sponsor));
    wallets.push(wallet(&fed.plan.aggregator.did));
    wallets.push(wallet(&watcher.did_string()));
    validator.fund(&wallets);

    let gate = Gate::default();
    if !gated {
        gate.open();
    }
    let (tx, rx) = mpsc::channel();
    let mut config = fed.aggregator_config("127.0.0.1:0");
    config.keep_spool = true;
    config.watchers = vec![watcher.did_string()];
    config.on_ready = Some(Box::new(move |addr| {
        let _ = tx.send(addr);
    }));
    let aggregator = Some(runtime.spawn(run_aggregator(config)));
    let addr = rx.recv_timeout(PATIENCE).unwrap();
    runtime.spawn(run_helper(fed.helper_config(0, &addr)));
    let sites = (0..SITES)
        .map(|i| {
            let trainer = Box::new(Towards(i as f32, gate.clone()));
            let site = run_site(fed.site_config(i, &addr), trainer);
            runtime.spawn(async move { site.await.map(|_| ()) })
        })
        .collect();
    Running {
        _dir: dir,
        fed,
        watcher,
        addr,
        aggregator,
        sites,
        gate,
        sent: Arc::default(),
        price,
        url: validator.url.clone(),
        window,
    }
}

type Ended = (&'static str, Result<(), ChainError>);

/// The anchors of a run, each on a thread of its own.
struct Anchors {
    done: mpsc::Sender<Ended>,
    ended: mpsc::Receiver<Ended>,
    running: usize,
}

impl Anchors {
    fn new() -> Anchors {
        let (done, ended) = mpsc::channel();
        Anchors {
            done,
            ended,
            running: 0,
        }
    }

    fn start(&mut self, name: &'static str, config: Config) -> Arc<AtomicBool> {
        let stop = Arc::new(AtomicBool::new(false));
        let (flag, done) = (stop.clone(), self.done.clone());
        std::thread::spawn(move || {
            let ended = Anchor::new(config).and_then(|mut anchor| anchor.run(&flag));
            let _ = done.send((name, ended));
        });
        self.running += 1;
        stop
    }

    fn next(&mut self) -> &'static str {
        let (name, ended) = self.ended.recv_timeout(PATIENCE).expect("an anchor hangs");
        ended.unwrap_or_else(|e| panic!("the {name} anchor failed: {e}"));
        self.running -= 1;
        name
    }

    /// Stops an anchor between two steps, as a crash would, while every
    /// other one is still at work.
    fn kill(&mut self, name: &'static str, stop: &AtomicBool) {
        stop.store(true, Ordering::Relaxed);
        assert_eq!(self.next(), name);
    }

    /// Waits until every anchor has found nothing more to do.
    fn finish(mut self) {
        while self.running > 0 {
            self.next();
        }
    }
}

impl Running {
    fn stakes(&self) -> Stakes {
        Stakes {
            site_stake: SOL,
            aggregator_stake: AGGREGATOR_STAKE,
            reward_per_round: REWARD,
            dispute_bond: SOL,
            challenge_slots: self.window,
            stall_slots: 10 * self.window,
        }
    }

    fn sink(&self) -> Sink {
        let sent = self.sent.clone();
        Arc::new(move |one| sent.lock().unwrap().push(one))
    }

    fn config(&self, role: Role, identity: &Arc<Identity>, log: PathBuf) -> Config {
        Config {
            role,
            identity: identity.clone(),
            plan_line: self.fed.plan_line.clone(),
            log,
            rpc: self.url.clone(),
            program: *fedtally_chain::program::ID.as_array(),
            price: self.price,
            poll: POLL,
            give_up_after: Duration::from_secs(60),
            sink: Some(self.sink()),
        }
    }

    fn publisher(&self) -> Config {
        let role = Role::Publisher(self.stakes());
        self.config(role, &self.fed.sponsor, PathBuf::new())
    }

    fn aggregator(&self) -> Config {
        let log = self.fed.aggregator_dir().join("log.jsonl");
        self.config(Role::Aggregator, &self.fed.aggregator, log)
    }

    fn site(&self, site: usize) -> Config {
        let log = self.fed.site_dir(site).join("log.jsonl");
        self.config(Role::Site, &self.fed.sites[site], log)
    }

    fn watching(&self, runtime: &tokio::runtime::Runtime) -> Config {
        let watching = Watching {
            addr: self.addr.clone(),
            dir: self.fed.root.join("watcher"),
            runtime: runtime.handle().clone(),
        };
        self.config(Role::Watcher(watching), &self.watcher, PathBuf::new())
    }

    /// Starts every anchor but the aggregator's.
    fn anchors(&self, runtime: &tokio::runtime::Runtime) -> (Anchors, Arc<AtomicBool>) {
        let mut anchors = Anchors::new();
        anchors.start("publisher", self.publisher());
        for site in 0..SITES {
            anchors.start("site", self.site(site));
        }
        let watcher = anchors.start("watcher", self.watching(runtime));
        (anchors, watcher)
    }

    fn was_sent(&self, what: &str) -> Option<()> {
        let sent = self.sent.lock().unwrap();
        sent.iter().any(|one| one.what == what).then_some(())
    }

    fn rpc(&self) -> Rpc {
        Rpc::new(&self.url)
    }

    fn task_address(&self) -> [u8; 32] {
        let publisher = anchor::wallet(&self.fed.plan.sponsor).unwrap();
        client::task(&publisher, &self.fed.plan_id.0)
    }

    fn task(&self) -> Option<TaskView> {
        ledger::task(self.rpc().account(&self.task_address()).unwrap().as_ref())
    }

    fn round(&self, number: u32) -> Option<RoundView> {
        let address = client::round(&self.task_address(), number);
        ledger::round(self.rpc().account(&address).unwrap().as_ref(), 1)
    }

    fn balance(&self, identity: &Identity) -> u64 {
        self.rpc().balance(&wallet_of(identity)).unwrap()
    }
}

fn wallet_of(identity: &Identity) -> [u8; 32] {
    anchor::wallet(&identity.did_string()).unwrap()
}

/// What one transaction cost, as the validator reports it.
struct Cost {
    sent: Sent,
    slot: u64,
    fee: u64,
    units: u64,
}

fn costs(running: &Running) -> Vec<Cost> {
    let rpc = running.rpc();
    let sent = running.sent.lock().unwrap().clone();
    sent.into_iter()
        .map(|sent| {
            let landed = eventually("a transaction's record", || {
                rpc.landed(&sent.signature).unwrap()
            });
            assert!(!landed.failed, "{} failed on the ledger", sent.what);
            Cost {
                slot: landed.slot,
                fee: landed.fee,
                units: landed.units.expect("the validator reports the units"),
                sent,
            }
        })
        .collect()
}

fn by<'a>(costs: &'a [Cost], identity: &Identity) -> impl Iterator<Item = &'a Cost> {
    let wallet = wallet_of(identity);
    costs.iter().filter(move |cost| cost.sent.by == wallet)
}

fn fees_of(costs: &[Cost], identity: &Identity) -> u64 {
    by(costs, identity).map(|cost| cost.fee).sum()
}

fn label(price: Option<u64>) -> String {
    match price {
        None => "no priority fee".to_owned(),
        Some(price) => format!("priority fee of {price} micro-lamports a unit"),
    }
}

fn print_rounds(running: &Running, costs: &[Cost]) {
    println!(
        "\nlatency per round, {}, challenge window of {} slots",
        label(running.price),
        running.window
    );
    println!(
        "{:>5} {:>12} {:>13} {:>13} {:>9} {:>6}",
        "round", "open>posted", "posted>acked", "acked>settled", "total", "slots"
    );
    for round in 1..=ROUNDS {
        let of = |what: &'static str| {
            costs
                .iter()
                .filter(move |c| c.sent.round == Some(round) && c.sent.what == what)
        };
        let open = of("open_round").next().unwrap();
        let posted = of("post_aggregate").next().unwrap();
        let settled = of("settle").next().unwrap();
        let acked = of("ack").map(|c| c.sent.confirmed).max().unwrap();
        let secs = |from: Instant, to: Instant| to.saturating_duration_since(from).as_secs_f64();
        println!(
            "{round:>5} {:>11.2}s {:>12.2}s {:>12.2}s {:>8.2}s {:>6}",
            secs(open.sent.sent, posted.sent.confirmed),
            secs(posted.sent.confirmed, acked),
            secs(acked, settled.sent.confirmed),
            secs(open.sent.sent, settled.sent.confirmed),
            settled.slot - open.slot,
        );
    }
}

fn print_fees(running: &Running, costs: &[Cost]) {
    let fed = &running.fed;
    println!("\nfees paid in lamports, {}", label(running.price));
    let mut parties = vec![("publisher", &fed.sponsor), ("aggregator", &fed.aggregator)];
    parties.extend(fed.sites.iter().map(|site| ("site", site)));
    parties.push(("watcher", &running.watcher));
    for (name, identity) in parties {
        let count = by(costs, identity).count();
        let fees = fees_of(costs, identity);
        println!("{name:>12} {fees:>8} in {count} transactions");
    }
    let total: u64 = costs.iter().map(|c| c.fee).sum();
    println!("{:>12} {total:>8} in {} transactions", "all", costs.len());
    for round in 1..=ROUNDS {
        let of: Vec<&Cost> = costs
            .iter()
            .filter(|c| c.sent.round == Some(round))
            .collect();
        let fee: u64 = of.iter().map(|c| c.fee).sum();
        let name = format!("round {round}");
        println!("{name:>12} {fee:>8} in {} transactions", of.len());
    }
}

/// Units per instruction, net of what the compute budget instructions take.
type Units = BTreeMap<&'static str, Vec<u64>>;

fn units(costs: &[Cost]) -> Units {
    let mut units = Units::new();
    for cost in costs {
        let own = cost.units - BUDGET_IX_UNITS * cost.sent.budget.instructions();
        units.entry(cost.sent.what).or_default().push(own);
    }
    units
}

fn add(all: &mut Units, more: Units) {
    for (what, units) in more {
        all.entry(what).or_default().extend(units);
    }
}

fn print_units(plain: &Units, priced: &Units, price: u64) {
    println!("\ncompute units per instruction on the validator, least to most over n transactions");
    println!(
        "with a priority fee a transaction spends {} more, on its two compute budget instructions",
        2 * BUDGET_IX_UNITS
    );
    println!(
        "{:<26} {:>8} {:>12} {:>4} {:>12} {:>4}",
        "instruction",
        "LiteSVM",
        "no fee",
        "n",
        format!("fee {price}"),
        "n"
    );
    let range = |units: Option<&Vec<u64>>| {
        let units = units.map(Vec::as_slice).unwrap_or_default();
        let text = match (units.iter().min(), units.iter().max()) {
            (Some(least), Some(most)) if least == most => least.to_string(),
            (Some(least), Some(most)) => format!("{least}-{most}"),
            _ => "-".to_owned(),
        };
        (text, units.len())
    };
    for (what, reference) in LITESVM {
        let (a, n) = range(plain.get(what));
        let (b, m) = range(priced.get(what));
        println!("{what:<26} {reference:>8} {a:>12} {n:>4} {b:>12} {m:>4}");
    }
    for what in plain.keys().chain(priced.keys()) {
        let known = LITESVM.iter().any(|(known, _)| known == what);
        assert!(known, "{what} has no reference figure");
    }
}

/// Every role runs to the end of an honest run: the rounds settle, each site
/// is paid its share and gets its stake back.
fn honest(runtime: &tokio::runtime::Runtime, validator: &Validator, price: Option<u64>) -> Units {
    let mut running = start(runtime, validator, price, 25, false);
    let sites = running.fed.sites.clone();
    let before: Vec<u64> = sites.iter().map(|site| running.balance(site)).collect();
    let (mut anchors, _) = running.anchors(runtime);
    // The aggregator's anchor dies with a round posted, and a new one takes
    // over from what the ledger holds.
    let first = anchors.start("aggregator", running.aggregator());
    eventually("the first round on the ledger", || {
        running.was_sent("post_aggregate")
    });
    anchors.kill("aggregator", &first);
    anchors.start("aggregator", running.aggregator());
    anchors.finish();
    let aggregator = running.aggregator.take().unwrap();
    let replay = runtime.block_on(aggregator).unwrap().unwrap();

    let task = running.task().unwrap();
    assert_eq!(task.state, TASK_FINISHED);
    assert_eq!((task.round, task.settled), (ROUNDS, ROUNDS));
    assert_eq!((task.owed, task.stake_held), (0, 0));
    let outcome = replay.rounds[ROUNDS as usize - 1].outcome.as_ref().unwrap();
    assert_eq!(task.model, outcome.model.0);
    // Every round is paid out and closed, and its rent is back.
    for number in 1..=ROUNDS {
        assert!(
            running.round(number).is_none(),
            "round {number} is still open"
        );
    }
    assert!(running.was_sent("settle").is_some());

    let costs = costs(&running);
    let rent = running.rpc().rent_exempt_minimum(ENROLLMENT_SPACE).unwrap();
    let share = REWARD / SITES as u64;
    for (site, before) in sites.iter().zip(before) {
        let paid = fees_of(&costs, site);
        assert_eq!(
            running.balance(site),
            before + u64::from(ROUNDS) * share - rent - paid,
            "a site ends with its stake and its shares, less only the rent and the fees"
        );
    }
    let disputes = by(&costs, &running.watcher).count();
    assert_eq!(disputes, 0, "the watcher disputed an honest round");
    for number in 1..=ROUNDS {
        let held = format!("watcher/round-{number}/aggregate.bin");
        let held = running.fed.root.join(held);
        assert!(held.is_file(), "the watcher never fetched round {number}");
    }

    print_rounds(&running, &costs);
    print_fees(&running, &costs);
    units(&costs)
}

fn flip(path: &Path, shape: vector::Shape, chunk_at: u64) -> vector::Digest {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut chunk = Vec::new();
    vector::read_chunk(&file, shape, chunk_at, &mut chunk).unwrap();
    chunk[17] ^= 1;
    vector::write_chunk_bytes(&file, shape, chunk_at, vector::chunk_bytes(&chunk)).unwrap();
    file.sync_data().unwrap();
    vector::manifest_of(path, shape).unwrap().root
}

/// The log is honest, but the aggregator swaps the aggregate it serves and
/// posts the root of that one. No site acknowledges it, and the watcher takes
/// the aggregator's stake.
/// A finished run, anchored afterwards, with a watcher that can't reach the
/// aggregator. Everything it needs must come through the ledger.
fn blind_watcher(running: &Running, runtime: &tokio::runtime::Runtime) -> Config {
    let dir = running.fed.root.join("watcher");
    std::fs::create_dir_all(&dir).unwrap();
    let log = running.fed.aggregator_dir().join("log.jsonl");
    std::fs::copy(log, dir.join("log.jsonl")).unwrap();
    let watching = Watching {
        addr: "127.0.0.1:1".to_owned(),
        dir,
        runtime: runtime.handle().clone(),
    };
    running.config(Role::Watcher(watching), &running.watcher, PathBuf::new())
}

fn finish_offline(runtime: &tokio::runtime::Runtime, running: &mut Running) {
    let aggregator = running.aggregator.take().unwrap();
    let sites = std::mem::take(&mut running.sites);
    runtime.block_on(async {
        aggregator.await.unwrap().unwrap();
        for site in sites {
            site.await.unwrap().unwrap();
        }
    });
}

fn anchor_parties(running: &Running, anchors: &mut Anchors) {
    anchors.start("publisher", running.publisher());
    for site in 0..SITES {
        anchors.start("site", running.site(site));
    }
    anchors.start("aggregator", running.aggregator());
}

/// The aggregator has lost its vectors, so nobody can answer: it pays.
fn withheld(runtime: &tokio::runtime::Runtime, validator: &Validator) {
    let mut running = start(runtime, validator, None, 80, false);
    finish_offline(runtime, &mut running);
    let spool = running.fed.aggregator_dir();
    for number in 1..=ROUNDS {
        std::fs::remove_dir_all(spool.join(format!("round-{number}"))).unwrap();
    }
    let before = running.balance(&running.watcher);
    let mut anchors = Anchors::new();
    anchor_parties(&running, &mut anchors);
    anchors.start("watcher", blind_watcher(&running, runtime));
    let started = Instant::now();
    anchors.finish();

    let task = running.task().unwrap();
    assert_eq!(
        (task.state, task.fraud, task.stake_held),
        (TASK_ABORTED, true, 0)
    );
    assert!(running.round(1).is_none(), "its rent was not returned");
    let costs = costs(&running);
    let paid = fees_of(&costs, &running.watcher);
    assert_eq!(
        running.balance(&running.watcher),
        before + AGGREGATOR_STAKE - paid,
        "the watcher ends with the stake and its bonds back, less the fees"
    );
    println!(
        "\nwithheld vectors: the aggregator was slashed in {:.1}s, {} transactions by the watcher, {paid} lamports in fees",
        started.elapsed().as_secs_f64(),
        by(&costs, &running.watcher).count(),
    );
}

/// The aggregator holds its vectors and answers from its spool, so the watcher
/// gets them through the ledger, finds the sum right, and everyone is paid.
fn answered(runtime: &tokio::runtime::Runtime, validator: &Validator) {
    let mut running = start(runtime, validator, None, 80, false);
    finish_offline(runtime, &mut running);
    let mut anchors = Anchors::new();
    anchor_parties(&running, &mut anchors);
    anchors.start("watcher", blind_watcher(&running, runtime));
    let started = Instant::now();
    anchors.finish();

    let task = running.task().unwrap();
    assert_eq!((task.state, task.fraud), (TASK_FINISHED, false));
    assert_eq!((task.settled, task.stake_held), (ROUNDS, 0));
    let costs = costs(&running);
    let answers = costs
        .iter()
        .filter(|c| c.sent.what == "demand_answer")
        .count();
    assert!(answers >= 15, "only {answers} answers");
    let spool = running.fed.aggregator_dir().join("round-1");
    assert!(!spool.exists(), "a settled round's vectors were kept");
    println!(
        "\nanswered demands: {answers} leaves through the ledger, the run settled in {:.1}s, {} lamports in fees for the watcher",
        started.elapsed().as_secs_f64(),
        fees_of(&costs, &running.watcher),
    );
}

/// A data access committee revokes a grant in mid run. Its issuer anchor posts
/// the list, and the watcher applies it to the site's enrollment.
fn revoked(runtime: &tokio::runtime::Runtime, validator: &Validator) {
    let mut running = start_with(runtime, validator, None, 60, true, |options| {
        options.credentials = true;
        options.did_bio = None;
        options.min_cohort = 2;
        options.timeouts.join_secs = 30;
        options.rounds = 3;
    });
    let issuers = running.fed.issuers.as_ref().unwrap();
    let key = running.fed.root.join("committee.key");
    issuers.committee.save(&key).unwrap();
    let committee = Arc::new(Identity::load(&key).unwrap());
    validator.fund(&[wallet_of(&committee)]);

    // The grant goes while round 2 trains, so round 3 is the first without it.
    let log = running.fed.aggregator_dir().join("log.jsonl");
    let mut run = Run::open(&running.fed.plan_line, &log).unwrap();
    eventually("round 2 in the log", || {
        run.refresh().unwrap();
        (run.view().rounds.len() == 2).then_some(())
    });
    running.fed.revoke_access(2).unwrap();
    running.gate.open();

    let (mut anchors, _) = running.anchors(runtime);
    anchors.start("aggregator", running.aggregator());
    anchors.start("issuer", running.config(Role::Issuer, &committee, log));
    anchors.finish();

    let replay = runtime.block_on(running.aggregator.take().unwrap());
    assert_eq!(replay.unwrap().unwrap().abort, None);

    let task = running.task().unwrap();
    assert_eq!(task.state, TASK_FINISHED);
    assert!(
        task.revoked.has(2),
        "the revocation never reached the ledger"
    );
    let address = client::enrollment(&running.task_address(), 2);
    let site = ledger::enrollment(running.rpc().account(&address).unwrap().as_ref()).unwrap();
    // Revoked, it took its stake back without waiting for the task to end.
    assert_eq!(site.status, SITE_WITHDRAWN);
    let sent = running.sent.lock().unwrap();
    for what in ["status_init", "revoke", "revoke_check"] {
        assert!(sent.iter().any(|one| one.what == what), "no {what}");
    }
}

/// The aggregator's side, by hand: an honest round up to the aggregate, and
/// then the root of one whose leaf 1 is not the sum.
fn post_wrong_aggregate(running: &Running, price: Option<u64>) {
    let aggregator = Ledger {
        rpc: running.rpc(),
        wallet: Wallet::new(running.fed.aggregator.clone()).unwrap(),
        price,
        confirm_every: POLL,
        sink: Some(running.sink()),
    };
    let me = *aggregator.wallet.address();
    let task = running.task_address();
    let mut run = Run::open(
        &running.fed.plan_line,
        &running.fed.aggregator_dir().join("log.jsonl"),
    )
    .unwrap();
    let (committed, posted) = eventually("the first round in the log", || {
        run.refresh().unwrap();
        let log = run.view().round(1)?;
        let sealed = log.acks.len() == SITES;
        log.committed
            .clone()
            .zip(log.posted.clone())
            .filter(|_| sealed)
    });
    eventually("the task", || running.task());
    let send = |what, ix, ceiling| aggregator.send(what, Some(1), ix, ceiling).unwrap();
    send("back", client::back(&me, &task), 10_000);
    eventually("every enrollment", || {
        running.task().filter(|task| task.enrolled.len() == SITES)
    });
    send("open_round", client::open_round(&me, &task, 1), 40_000);
    let cohort = client::site_bitmap(committed.cohort.iter().map(|site| *site as u16));
    let commit = client::commit_root(
        &me,
        &task,
        1,
        &cohort,
        &committed.inputs_root,
        &committed.close_head.0,
        &committed.threshold,
    );
    send("commit_root", commit, 10_000);

    let spool = running.fed.aggregator_dir().join("round-1/aggregate.bin");
    let wrong = flip(&spool, running.fed.plan.shape(), 1);
    assert_ne!(wrong, posted.aggregate_root);
    let masks: Vec<[u8; 32]> = posted.mask_roots.iter().map(|m| m.0).collect();
    let post = client::post_aggregate(
        &me,
        &task,
        1,
        &masks,
        &wrong.0,
        &posted.next_model.0,
        &posted.seal_head.0,
    );
    send("post_aggregate", post, 10_000);
}

/// The aggregator cheats at leaf 1 and spoils another chunk of a site's vector
/// so that no vector can be had whole. The watcher proves the wrong leaf from
/// the chunks it did get, without asking the ledger for anything.
fn partly_withheld(runtime: &tokio::runtime::Runtime, validator: &Validator) {
    let running = start(runtime, validator, None, 150, true);
    let mut anchors = Anchors::new();
    anchors.start("publisher", running.publisher());
    for site in 0..SITES {
        anchors.start("site", running.site(site));
    }
    post_wrong_aggregate(&running, None);
    let vector = running.fed.aggregator_dir().join("round-1/site-0.bin");
    flip(&vector, running.fed.plan.shape(), 0);
    anchors.start("watcher", running.watching(runtime));

    let ended = eventually("the dispute", || {
        running.task().filter(|task| !task.is_open())
    });
    assert_eq!(
        (ended.state, ended.fraud, ended.stake_held),
        (TASK_ABORTED, true, 0)
    );
    assert_eq!(running.round(1).unwrap().state, ROUND_FRAUD);
    assert!(
        running.was_sent("demand_open").is_none(),
        "it had no need to ask"
    );
    running.gate.open();
    anchors.finish();
}

fn fraud(runtime: &tokio::runtime::Runtime, validator: &Validator, price: Option<u64>) -> Units {
    let running = start(runtime, validator, price, 150, true);
    let fed = &running.fed;
    let before = running.balance(&running.watcher);
    let (mut anchors, watcher) = running.anchors(runtime);

    post_wrong_aggregate(&running, price);

    // The watcher dies in the middle of its proof, and a new one finishes it.
    eventually("the first leaf of the proof", || {
        running.was_sent("dispute_leaf (input)")
    });
    anchors.kill("watcher", &watcher);
    anchors.start("watcher", running.watching(runtime));

    let ended = eventually("the dispute", || {
        running.task().filter(|task| !task.is_open())
    });
    let slashed = (TASK_ABORTED, true, 0);
    assert_eq!((ended.state, ended.fraud, ended.stake_held), slashed);
    let round = running.round(1).unwrap();
    assert_eq!(round.state, ROUND_FRAUD);
    let acked = round.acked.is_empty();
    assert!(acked, "a site acknowledged a round that is not in its log");

    // The run itself goes on. Every party then takes back what is its own.
    running.gate.open();
    anchors.finish();
    let costs = costs(&running);
    let paid = fees_of(&costs, &running.watcher);
    assert_eq!(
        running.balance(&running.watcher),
        before + AGGREGATOR_STAKE - paid,
        "the watcher ends with the stake, its bond and its rent back, less the fees"
    );
    let rent = running.rpc().rent_exempt_minimum(ENROLLMENT_SPACE).unwrap();
    for site in &fed.sites {
        let left = WALLET - rent - fees_of(&costs, site);
        assert_eq!(running.balance(site), left);
    }
    let disputed: Vec<&Cost> = costs
        .iter()
        .filter(|c| c.sent.what.starts_with("dispute"))
        .collect();
    let (first, last) = (disputed.first().unwrap(), disputed.last().unwrap());
    let took = last
        .sent
        .confirmed
        .saturating_duration_since(first.sent.sent);
    println!(
        "\nfraud proof, {}: {} transactions by the watcher, {:.2}s and {} slots from dispute_open to dispute_close, {paid} lamports in fees",
        label(price),
        by(&costs, &running.watcher).count(),
        took.as_secs_f64(),
        last.slot - first.slot,
    );
    let units = units(&costs);
    runtime.block_on(async {
        for site in running.sites {
            site.await.unwrap().unwrap();
        }
        running.aggregator.unwrap().await.unwrap().unwrap();
    });
    units
}

#[test]
#[ignore = "needs solana-test-validator and the built program"]
fn the_roles_settle_an_honest_run_and_slash_a_dishonest_aggregator() {
    const PRICE: u64 = 1_000;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let validator = Validator::start();

    let (mut plain, mut priced) = (Units::new(), Units::new());
    add(&mut plain, honest(&runtime, &validator, None));
    add(&mut priced, honest(&runtime, &validator, Some(PRICE)));
    add(&mut plain, fraud(&runtime, &validator, None));
    add(&mut priced, fraud(&runtime, &validator, Some(PRICE)));
    print_units(&plain, &priced, PRICE);
}

fn with_validator(scenario: fn(&tokio::runtime::Runtime, &Validator)) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    scenario(&runtime, &Validator::start());
}

#[test]
#[ignore = "needs solana-test-validator and the built program"]
fn an_aggregator_that_cannot_answer_a_demand_loses_its_stake() {
    with_validator(withheld);
}

#[test]
#[ignore = "needs solana-test-validator and the built program"]
fn a_watcher_cut_off_from_the_aggregator_gets_what_it_needs_through_the_ledger() {
    with_validator(answered);
}

#[test]
#[ignore = "needs solana-test-validator and the built program"]
fn a_wrong_leaf_is_proven_though_no_vector_can_be_had_whole() {
    with_validator(partly_withheld);
}

#[test]
#[ignore = "needs solana-test-validator and the built program"]
fn a_revocation_in_the_log_reaches_the_ledger() {
    with_validator(revoked);
}
