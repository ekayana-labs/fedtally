//! Whole runs over real TLS on loopback.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fedtally_core::field::P;
use fedtally_core::keys::{Identity, SigAlg};
use fedtally_core::model::{Dtype, Layout, ModelFile, TensorSpec, TensorView, write_model};
use fedtally_core::plan::Plan;
use fedtally_core::pq_ucan::Cid;
use fedtally_core::pq_ucan::did::Did;
use fedtally_core::pq_ucan::validate::RevocationStore;
use fedtally_core::privacy::Request;
use fedtally_core::replay::{Phase, Replay};
use fedtally_core::report::verify_log;
use fedtally_core::vector::{self, Digest};
use fedtally_node::NodeError;
use fedtally_node::aggregator::{Hook, Stage, run_aggregator};
use fedtally_node::helper::{ShareHook, run_helper};
use fedtally_node::local::{LocalFederation, Options};
use fedtally_node::observer::{ObserverConfig, fetch_vector};
use fedtally_node::site::{SiteSummary, Submit, TrainContext, Trainer, run_site};
use fedtally_node::wire::Object;
use tokio::sync::oneshot;
use tokio::task::{AbortHandle, JoinHandle};

fn layout() -> Layout {
    Layout {
        tensors: vec![
            TensorSpec {
                name: "w".into(),
                dtype: Dtype::F32,
                shape: vec![40, 25],
            },
            TensorSpec {
                name: "b".into(),
                dtype: Dtype::F64,
                shape: vec![24],
            },
        ],
    }
}

fn initial_model(dir: &Path) -> PathBuf {
    let path = dir.join("init.safetensors");
    write_model(
        &path,
        &layout(),
        &[TensorView::F32(&[1.0; 1000]), TensorView::F64(&[-2.0; 24])],
    )
    .unwrap();
    path
}

/// Moves the model to a fixed value, so the right answer is known exactly.
struct Fixed {
    value: f32,
    weight: u64,
    rounds: Arc<std::sync::Mutex<Vec<u32>>>,
}

impl Fixed {
    fn new(value: f32, weight: u64) -> Box<Fixed> {
        Box::new(Fixed {
            value,
            weight,
            rounds: Arc::default(),
        })
    }
}

impl Trainer for Fixed {
    fn train(&mut self, ctx: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        self.rounds.lock().unwrap().push(ctx.round);
        let w = vec![self.value; 1000];
        let b = vec![self.value as f64; 24];
        submit(&[TensorView::F32(&w), TensorView::F64(&b)], self.weight)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

struct Running {
    addr: String,
    aggregator: JoinHandle<Result<fedtally_core::replay::Replay, NodeError>>,
    helpers: Vec<JoinHandle<Result<(), NodeError>>>,
}

async fn start(fed: &LocalFederation, hook: Option<Hook>) -> Running {
    let (tx, rx) = oneshot::channel();
    let mut config = fed.aggregator_config("127.0.0.1:0");
    config.hook = hook;
    config.on_ready = Some(Box::new(move |addr| {
        let _ = tx.send(addr);
    }));
    let aggregator = tokio::spawn(run_aggregator(config));
    let addr = rx.await.unwrap();
    let helpers = (0..fed.helpers.len())
        .map(|h| tokio::spawn(run_helper(fed.helper_config(h, &addr))))
        .collect();
    Running {
        addr,
        aggregator,
        helpers,
    }
}

fn options(sites: usize, helpers: usize, rounds: u32) -> Options {
    let mut options = Options::new(sites, helpers, rounds);
    options.chunk_coords = 64;
    options.purpose = "test".into();
    options
}

async fn run_all(
    fed: &LocalFederation,
    trainers: Vec<Box<dyn Trainer>>,
    hook: Option<Hook>,
) -> (Running, Vec<Result<SiteSummary, NodeError>>) {
    let running = start(fed, hook).await;
    let tasks: Vec<_> = trainers
        .into_iter()
        .enumerate()
        .map(|(i, t)| tokio::spawn(run_site(fed.site_config(i, &running.addr), t)))
        .collect();
    let mut results = Vec::new();
    for task in tasks {
        results.push(task.await.unwrap());
    }
    (running, results)
}

/// Runs every site, and lets `after` go once the sites in `first` have finished.
async fn run_all_then(
    fed: &LocalFederation,
    trainers: Vec<Box<dyn Trainer>>,
    first: &[usize],
    after: impl FnOnce(),
) -> (Running, Vec<Result<SiteSummary, NodeError>>) {
    let running = start(fed, None).await;
    let mut tasks: Vec<_> = trainers
        .into_iter()
        .enumerate()
        .map(|(i, t)| Some(tokio::spawn(run_site(fed.site_config(i, &running.addr), t))))
        .collect();
    let mut results: Vec<_> = tasks.iter().map(|_| None).collect();
    for i in first {
        results[*i] = Some(tasks[*i].take().unwrap().await.unwrap());
    }
    after();
    for (i, task) in tasks.into_iter().enumerate() {
        if let Some(task) = task {
            results[i] = Some(task.await.unwrap());
        }
    }
    (running, results.into_iter().map(Option::unwrap).collect())
}

fn read_tensor_mean(path: &Path) -> f64 {
    let _ = ModelFile::open(path, &layout()).unwrap();
    let bytes = std::fs::read(path).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let data = &bytes[8 + len..];
    let mut total = 0.0;
    for c in data[..4000].as_chunks::<4>().0 {
        total += f32::from_le_bytes(*c) as f64;
    }
    total / 1000.0
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-5
}

async fn honest_run(sites: usize, helpers: usize, rounds: u32) {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(
        dir.path(),
        layout(),
        &init,
        &options(sites, helpers, rounds),
    )
    .unwrap();
    let trainers: Vec<Box<dyn Trainer>> = (0..sites)
        .map(|i| Fixed::new(i as f32, (i + 1) as u64) as Box<dyn Trainer>)
        .collect();
    let (running, results) = run_all(&fed, trainers, None).await;
    running.aggregator.await.unwrap().unwrap();
    for helper in running.helpers {
        helper.await.unwrap().unwrap();
    }

    let total: f64 = (1..=sites as u64).sum::<u64>() as f64;
    let mean: f64 = (0..sites).map(|i| i as f64 * (i + 1) as f64).sum::<f64>() / total;
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    for summary in &summaries {
        assert_eq!(summary.final_digest, summaries[0].final_digest);
        assert!(close(read_tensor_mean(&summary.final_model), mean));
        assert_eq!(summary.rounds_contributed, (1..=rounds).collect::<Vec<_>>());
    }

    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");
    assert_eq!(report.rounds.len(), rounds as usize);
    for i in 0..sites {
        let copy = fed.site_dir(i).join("log.jsonl");
        assert_eq!(std::fs::read(&copy).unwrap(), std::fs::read(&log).unwrap());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn plain_run_without_helpers() {
    honest_run(3, 0, 2).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn masked_run_with_one_helper() {
    honest_run(4, 1, 2).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn masked_run_with_two_helpers() {
    honest_run(3, 2, 1).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_aggregator_only_ever_holds_noise_when_masked() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options(2, 1, 1)).unwrap();
    let seen = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    let copy = seen.clone();
    let hook: Hook = Arc::new(move |stage, round_dir| {
        if let Stage::BeforeSum { .. } = stage {
            let bytes = std::fs::read(round_dir.join("site-0.bin")).unwrap();
            copy.lock().unwrap().extend(
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| u64::from_le_bytes(*c)),
            );
        }
    });
    // Without masking the first coordinate would be weight * 0 = 0 and all 1000 alike.
    let (running, results) = run_all(
        &fed,
        vec![Fixed::new(1.0, 1), Fixed::new(1.0, 1)],
        Some(hook),
    )
    .await;
    running.aggregator.await.unwrap().unwrap();
    results.into_iter().for_each(|r| {
        r.unwrap();
    });
    let seen = seen.lock().unwrap();
    let distinct: std::collections::BTreeSet<_> = seen.iter().collect();
    assert_eq!(seen.len(), 1026);
    assert!(
        distinct.len() > 1020,
        "masked vector repeats values: {}",
        distinct.len()
    );
}

fn flip(path: &Path, byte: usize) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes[byte] ^= 4;
    std::fs::write(path, bytes).unwrap();
}

fn no_one_acknowledged(fed: &LocalFederation, results: &[Result<SiteSummary, NodeError>]) {
    for result in results {
        assert!(
            result.is_err(),
            "a site finished with a bad round: {result:?}"
        );
    }
    let report = verify_log(&fed.aggregator_dir().join("log.jsonl"), None).unwrap();
    assert!(!report.finished);
    let round = &report.rounds[0];
    assert!(
        round
            .included
            .iter()
            .all(|c| round.unacked.contains(&c.site))
    );
}

fn quick(sites: usize, helpers: usize) -> Options {
    let mut opts = options(sites, helpers, 1);
    opts.timeouts.attest_secs = 3;
    opts.timeouts.ack_secs = 3;
    opts
}

fn fixed_trainers(n: usize) -> Vec<Box<dyn Trainer>> {
    (0..n)
        .map(|_| Fixed::new(1.0, 1) as Box<dyn Trainer>)
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tampered_upload_stops_the_run_before_anyone_takes_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &quick(3, 1)).unwrap();
    let hook: Hook = Arc::new(|stage, round_dir| {
        if let Stage::BeforeSum { .. } = stage {
            flip(&round_dir.join("site-1.bin"), 16);
        }
    });
    let (running, results) = run_all(&fed, fixed_trainers(3), Some(hook)).await;
    running.aggregator.await.unwrap().unwrap();
    no_one_acknowledged(&fed, &results);
    let report = verify_log(&fed.aggregator_dir().join("log.jsonl"), None).unwrap();
    assert!(!report.rounds[0].sealed);
    assert!(report.aborted.unwrap().contains(&fed.plan.sites[1]));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_changed_aggregate_is_refused_by_every_site() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &quick(3, 1)).unwrap();
    let hook: Hook = Arc::new(|stage, round_dir| {
        if let Stage::BeforeOutcome { .. } = stage {
            flip(&round_dir.join("aggregate.bin"), 0);
        }
    });
    let (running, results) = run_all(&fed, fixed_trainers(3), Some(hook)).await;
    running.aggregator.abort();
    for result in &results {
        assert!(matches!(result, Err(NodeError::Check(_))), "{result:?}");
    }
    no_one_acknowledged(&fed, &results);
}

/// Adds to the first coordinate, so the same change to two vectors stays
/// consistent in the sum.
fn bump(path: &Path) {
    let mut bytes = std::fs::read(path).unwrap();
    let first = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let moved = fedtally_core::field::Fp::new(first) + fedtally_core::field::Fp::new(4);
    bytes[..8].copy_from_slice(&moved.value().to_le_bytes());
    std::fs::write(path, bytes).unwrap();
}

fn colluding_hook(colluder: PathBuf) -> Hook {
    Arc::new(move |stage, round_dir| {
        if let Stage::BeforeOutcome { round } = stage {
            bump(&round_dir.join("aggregate.bin"));
            bump(&round_dir.join("site-0.bin"));
            bump(&colluder.join(format!("round-{round}/masked.bin")));
        }
    })
}

/// The aggregator changes the sum and a site on its side attests to match. Each
/// honest site's own numbers still add up, so only the audit can catch this.
#[tokio::test(flavor = "multi_thread")]
async fn an_aggregator_and_one_site_changing_the_sum_are_caught_by_the_audit() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &quick(4, 1)).unwrap();
    let hook = colluding_hook(fed.site_dir(0));
    let (running, results) = run_all(&fed, fixed_trainers(4), Some(hook)).await;
    running.aggregator.await.unwrap().unwrap();
    let caught = results
        .iter()
        .filter(|r| matches!(r, Err(NodeError::Check(_))))
        .count();
    assert_eq!(
        caught, 1,
        "exactly one site audits the colluder: {results:?}"
    );
    no_one_acknowledged(&fed, &results);
}

/// The same attack with the audits turned off, to show what they buy.
#[tokio::test(flavor = "multi_thread")]
async fn without_audits_that_collusion_goes_unnoticed() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut opts = quick(4, 1);
    opts.audits = 0;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &opts).unwrap();
    let hook = colluding_hook(fed.site_dir(0));
    let (running, results) = run_all(&fed, fixed_trainers(4), Some(hook)).await;
    running.aggregator.await.unwrap().unwrap();
    assert!(results.iter().all(Result::is_ok), "{results:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_helper_lying_about_a_share_is_caught_by_that_site() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &quick(3, 1)).unwrap();

    let (tx, rx) = oneshot::channel();
    let mut config = fed.aggregator_config("127.0.0.1:0");
    config.on_ready = Some(Box::new(move |addr| {
        let _ = tx.send(addr);
    }));
    let aggregator = tokio::spawn(run_aggregator(config));
    let addr = rx.await.unwrap();
    let mut helper = fed.helper_config(0, &addr);
    helper.share_hook = Some(Arc::new(|shares| shares[2] = shares[2] + shares[0]));
    tokio::spawn(run_helper(helper));
    let sites: Vec<_> = (0..3)
        .map(|i| tokio::spawn(run_site(fed.site_config(i, &addr), Fixed::new(1.0, 1))))
        .collect();
    let mut results = Vec::new();
    for site in sites {
        results.push(site.await.unwrap());
    }
    aggregator.await.unwrap().unwrap();

    assert!(
        matches!(&results[2], Err(NodeError::Check(why)) if why.contains("helper")),
        "{:?}",
        results[2]
    );
    assert!(
        matches!(&results[0], Err(NodeError::Aborted(_))),
        "{:?}",
        results[0]
    );
    no_one_acknowledged(&fed, &results);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_slow_site_is_left_out_and_the_round_goes_on() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut opts = options(3, 1, 1);
    opts.min_cohort = 2;
    opts.timeouts.collect_secs = 6;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &opts).unwrap();
    let (release, blocked) = std::sync::mpsc::channel::<()>();
    let (running, results) = run_all_then(
        &fed,
        vec![
            Fixed::new(0.0, 1),
            Fixed::new(2.0, 1),
            Box::new(Blocked(blocked)),
        ],
        &[0, 1],
        move || drop(release),
    )
    .await;
    running.aggregator.await.unwrap().unwrap();
    assert!(close(
        read_tensor_mean(&results[0].as_ref().unwrap().final_model),
        1.0
    ));
    assert!(matches!(results[2], Err(NodeError::Trainer(_))));
    let report = verify_log(&fed.aggregator_dir().join("log.jsonl"), None).unwrap();
    assert_eq!(report.rounds[0].included.len(), 2);
    assert!(report.finished);
}

#[tokio::test(flavor = "multi_thread")]
async fn too_few_sites_abort_the_run_instead_of_leaking_one_update() {
    struct Silent;
    impl Trainer for Silent {
        fn train(&mut self, _: &TrainContext, _: &mut Submit<'_>) -> Result<(), String> {
            Err("no data today".into())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut opts = options(2, 1, 1);
    opts.timeouts.collect_secs = 1;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &opts).unwrap();
    let (running, _) = run_all(&fed, vec![Fixed::new(1.0, 1), Box::new(Silent)], None).await;
    running.aggregator.await.unwrap().unwrap();
    for helper in running.helpers {
        assert!(matches!(helper.await.unwrap(), Err(NodeError::Aborted(_))));
    }
    let report = verify_log(&fed.aggregator_dir().join("log.jsonl"), None).unwrap();
    assert!(report.aborted.is_some());
    assert!(report.rounds[0].included.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_can_refuse_a_plan_outside_its_grant() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options(2, 1, 1)).unwrap();
    let other = LocalFederation::create(
        &dir.path().join("other"),
        layout(),
        &init,
        &options(2, 1, 1),
    )
    .unwrap();
    let mut config = fed.site_config(0, "127.0.0.1:1");
    config.invocation = other.grants[0].invocation.clone();
    let result = run_site(config, Fixed::new(1.0, 1)).await;
    assert!(matches!(result, Err(NodeError::Refused(_))), "{result:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_party_outside_the_plan_is_turned_away() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options(2, 0, 1)).unwrap();
    let running = start(&fed, None).await;
    let stranger =
        fedtally_core::keys::Identity::generate(fedtally_core::keys::SigAlg::Ed25519).unwrap();
    let client = fedtally_node::net::Client::connect(
        &running.addr,
        fed.plan.aggregator.tls_pin,
        fed.plan_id,
        &stranger,
        1 << 20,
    )
    .await;
    if let Ok(mut client) = client {
        use fedtally_node::wire::Msg;
        let _ = client
            .send(&Msg::Subscribe {
                from: 0,
                head: fedtally_core::log::GENESIS,
            })
            .await;
        assert!(client.entries().await.is_err());
    }
    running.aggregator.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_wrong_certificate_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options(2, 0, 1)).unwrap();
    let running = start(&fed, None).await;
    let result = fedtally_node::net::Client::connect(
        &running.addr,
        Digest::of(b"another certificate"),
        fed.plan_id,
        &fed.sites[0],
        1 << 20,
    )
    .await;
    assert!(
        matches!(result, Err(NodeError::Tls(_))),
        "{:?}",
        result.err()
    );
    running.aggregator.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn connections_settle_on_a_post_quantum_key_exchange() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options(2, 0, 1)).unwrap();
    let running = start(&fed, None).await;
    let client = fedtally_node::net::Client::connect(
        &running.addr,
        fed.plan.aggregator.tls_pin,
        fed.plan_id,
        &fed.sites[0],
        1 << 20,
    )
    .await
    .unwrap();
    assert!(
        client.key_exchange().unwrap().contains("MLKEM"),
        "{:?}",
        client.key_exchange()
    );
    running.aggregator.abort();
}

/// Trains normally, and flips the site's revocation switch after one round.
struct Withdraws {
    switch: Arc<std::sync::atomic::AtomicBool>,
}

impl Trainer for Withdraws {
    fn train(&mut self, ctx: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        let (w, b) = (vec![5.0f32; 1000], vec![5.0f64; 24]);
        let result = submit(&[TensorView::F32(&w), TensorView::F64(&b)], 1);
        if ctx.round == 1 {
            self.switch.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        result.map(|_| ()).map_err(|e| e.to_string())
    }
}

struct Switch(Arc<std::sync::atomic::AtomicBool>);

impl RevocationStore for Switch {
    fn is_revoked(&self, _: &Cid, _: &[&Did]) -> bool {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_revoked_grant_stops_the_next_round_and_shows_in_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut opts = options(3, 1, 3);
    opts.min_cohort = 2;
    opts.timeouts.collect_secs = 6;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &opts).unwrap();
    let running = start(&fed, None).await;
    let switch = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let mut config = fed.site_config(2, &running.addr);
    config.revocations = Some(Arc::new(Switch(switch.clone())));
    let leaving = tokio::spawn(run_site(config, Box::new(Withdraws { switch })));
    let others: Vec<_> = (0..2)
        .map(|i| {
            tokio::spawn(run_site(
                fed.site_config(i, &running.addr),
                Fixed::new(1.0, 1),
            ))
        })
        .collect();

    let summary = leaving.await.unwrap().unwrap();
    assert_eq!(summary.rounds_contributed, vec![1]);
    assert_eq!(summary.rounds_refused, vec![2, 3]);
    let mut digests = vec![summary.final_digest];
    for other in others {
        digests.push(other.await.unwrap().unwrap().final_digest);
    }
    assert!(
        digests.windows(2).all(|d| d[0] == d[1]),
        "the site that left no longer follows the model"
    );

    running.aggregator.await.unwrap().unwrap();
    let report = verify_log(&fed.aggregator_dir().join("log.jsonl"), None).unwrap();
    assert_eq!(report.rounds_with(&fed.plan.sites[2]), vec![1]);
    assert_eq!(report.rounds_with(&fed.plan.sites[0]), vec![1, 2, 3]);
    assert!(report.finished);
}

async fn connected_site(fed: &LocalFederation, addr: &str) -> fedtally_node::net::Client {
    fedtally_node::net::Client::connect(
        addr,
        fed.plan.aggregator.tls_pin,
        fed.plan_id,
        &fed.sites[0],
        1 << 20,
    )
    .await
    .unwrap()
}

/// Holds its round open until the test lets go.
struct Blocked(std::sync::mpsc::Receiver<()>);

impl Trainer for Blocked {
    fn train(&mut self, _: &TrainContext, _: &mut Submit<'_>) -> Result<(), String> {
        let _ = self.0.recv();
        Err("released".into())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_upload_that_was_never_signed_closes_the_connection() {
    use fedtally_core::vector::Manifest;
    use fedtally_node::wire::{Msg, Object};

    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options(2, 0, 1)).unwrap();
    let running = start(&fed, None).await;
    let (release, blocked) = std::sync::mpsc::channel();
    let sites = [
        tokio::spawn(run_site(
            fed.site_config(0, &running.addr),
            Fixed::new(1.0, 1),
        )),
        tokio::spawn(run_site(
            fed.site_config(1, &running.addr),
            Box::new(Blocked(blocked)),
        )),
    ];

    let log = fed.aggregator_dir().join("log.jsonl");
    while !std::fs::read_to_string(&log).is_ok_and(|text| text.contains(r#""kind":"open""#)) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Site 1 has signed nothing this round, so nothing can be uploaded for it.
    let mut rogue = fedtally_node::net::Client::connect(
        &running.addr,
        fed.plan.aggregator.tls_pin,
        fed.plan_id,
        &fed.sites[1],
        1 << 20,
    )
    .await
    .unwrap();
    let chunks = vec![Digest::of(b"chunk"); fed.plan.shape().chunks() as usize];
    rogue
        .send(&Msg::PutManifest {
            object: Object::Upload { round: 1, site: 1 },
            manifest: Manifest::new(chunks),
        })
        .await
        .unwrap();
    assert!(rogue.reply().await.is_err());

    release.send(()).unwrap();
    running.aggregator.abort();
    for site in sites {
        site.abort();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_past_the_last_chunk_closes_the_connection_only() {
    use fedtally_node::wire::{Msg, Object};

    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options(2, 0, 1)).unwrap();
    let running = start(&fed, None).await;
    let mut client = connected_site(&fed, &running.addr).await;
    client
        .send(&Msg::GetChunk {
            object: Object::Aggregate { round: 1 },
            chunk: u64::MAX / 8,
        })
        .await
        .unwrap();
    assert!(client.reply().await.is_err());
    // The aggregator is still serving.
    connected_site(&fed, &running.addr).await;
    assert!(!running.aggregator.is_finished());
    running.aggregator.abort();
}

#[test]
fn chunk_sums_match_for_large_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let shape = vector::Shape {
        coords: 10_000,
        chunk_coords: 777,
    };
    let path = dir.path().join("v");
    let manifest = vector::build(&path, shape, |chunk, buf, _| {
        vector::apply_mask(&[3u8; 32], chunk, buf);
        Ok(())
    })
    .unwrap();
    assert_eq!(manifest, vector::manifest_of(&path, shape).unwrap());
}

fn private_options(sites: usize, rounds: u32) -> Options {
    let mut options = options(sites, 1, rounds);
    options.privacy = Some(Request {
        epsilon: 8.0,
        delta: 1e-5,
        clip: 1.0,
        colluding: 0,
    });
    options
}

/// The 1000 values of the first tensor.
fn read_weights(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    bytes[8 + len..][..4000]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c) as f64)
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_private_run_adds_the_noise_its_plan_names_and_stays_in_budget() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let (sites, rounds) = (4usize, 2u32);
    let fed = LocalFederation::create(dir.path(), layout(), &init, &private_options(sites, rounds))
        .unwrap();
    let privacy = fed.plan.privacy.unwrap();
    assert_eq!(fed.plan.max_weight, 1);

    // Every site leaves the first tensor alone, so whatever moves it is noise.
    // The weights are ignored: a private run counts each site once.
    let trainers: Vec<Box<dyn Trainer>> = (0..sites)
        .map(|i| {
            Box::new(Still {
                weight: 10 + i as u64,
            }) as Box<dyn Trainer>
        })
        .collect();
    let (running, results) = run_all(&fed, trainers, None).await;
    running.aggregator.await.unwrap().unwrap();
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    for summary in &summaries {
        assert_eq!(summary.final_digest, summaries[0].final_digest);
    }

    let moved: Vec<f64> = read_weights(&summaries[0].final_model)
        .iter()
        .map(|w| w - 1.0)
        .collect();
    let spread = (moved.iter().map(|m| m * m).sum::<f64>() / moved.len() as f64).sqrt();
    // Each round averages four sites' noise: its deviation is sqrt(4 * variance)
    // grid units, over four sites and the grid scale. Two rounds add up.
    let per_round =
        (sites as f64 * privacy.noise as f64).sqrt() / (sites as f64 * fed.plan.encoding.scale());
    let want = per_round * (rounds as f64).sqrt();
    assert!(
        (spread / want - 1.0).abs() < 0.15,
        "spread {spread}, expected {want}"
    );
    assert!(want > 0.05, "the noise should be plain to see: {want}");

    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");
    assert!(report.rounds.iter().all(|r| r.total_weight == Some(4)));
    let spent = report.privacy.as_ref().unwrap();
    assert!(spent.rho_spent > 0.0 && spent.rho_spent <= spent.rho_budget);
    assert!(spent.epsilon_spent <= 8.0 + 1e-6, "{spent:?}");
    // The plan budgets for the smallest cohort, and all four sites took part.
    assert!(spent.epsilon_spent < 7.5, "{spent:?}");
}

/// Sends the model back unchanged.
struct Still {
    weight: u64,
}

impl Trainer for Still {
    fn train(&mut self, _: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        submit(
            &[TensorView::F32(&[1.0; 1000]), TensorView::F64(&[-2.0; 24])],
            self.weight,
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_private_run_stops_when_its_budget_cannot_cover_a_round() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create_with(
        dir.path(),
        layout(),
        &init,
        &private_options(4, 3),
        |plan: &mut Plan| {
            // Room for one round and a half at the smallest cohort.
            let one = plan.round_cost(plan.min_cohort as usize).unwrap();
            plan.privacy.as_mut().unwrap().budget = one + one / 2;
        },
    )
    .unwrap();
    let trainers: Vec<Box<dyn Trainer>> = (0..4)
        .map(|_| Box::new(Still { weight: 1 }) as Box<dyn Trainer>)
        .collect();
    let (running, results) = run_all(&fed, trainers, None).await;
    let _ = running.aggregator.await.unwrap();
    for result in results {
        assert!(matches!(result, Err(NodeError::Aborted(_))), "{result:?}");
    }

    let report = verify_log(&fed.aggregator_dir().join("log.jsonl"), None).unwrap();
    assert!(!report.complete());
    assert!(
        report
            .aborted
            .as_deref()
            .is_some_and(|r| r.contains("privacy budget")),
        "{:?}",
        report.aborted
    );
    assert_eq!(report.rounds.len(), 1);
    assert!(report.rounds[0].sealed && report.rounds[0].unacked.is_empty());
    let spent = report.privacy.unwrap();
    assert!(spent.rho_spent <= spent.rho_budget);
}

fn gated_options(sites: usize, rounds: u32) -> Options {
    let mut options = options(sites, 1, rounds);
    options.credentials = true;
    options.min_cohort = 2;
    options.timeouts.join_secs = 3;
    options
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_gated_on_credentials_admits_sites_that_hold_them() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &gated_options(3, 1)).unwrap();
    let trainers: Vec<Box<dyn Trainer>> = (0..3)
        .map(|_| Fixed::new(2.0, 1) as Box<dyn Trainer>)
        .collect();
    let (running, results) = run_all(&fed, trainers, None).await;
    running.aggregator.await.unwrap().unwrap();
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();

    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");
    assert!(report.revoked.is_empty());
    for (i, join) in report.joined.iter().enumerate() {
        assert_eq!(join.dataset, fed.standing[i].dataset);
        let kinds: Vec<&str> = join.credentials.iter().map(|c| c.kind.as_str()).collect();
        assert_eq!(
            kinds,
            [
                "AffiliationAndRole",
                "EthicsApproval",
                "ControlledAccessGrants",
                "Provenance"
            ]
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_that_lacks_a_credential_is_kept_out_and_the_rest_carry_on() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &gated_options(3, 1)).unwrap();
    let running = start(&fed, None).await;
    let tasks: Vec<_> = (0..3)
        .map(|i| {
            let mut config = fed.site_config(i, &running.addr);
            if i == 2 {
                // Everything but the access grant.
                config.standing.credentials.remove(2);
            }
            tokio::spawn(run_site(config, Fixed::new(2.0, 1)))
        })
        .collect();
    let mut results = Vec::new();
    for task in tasks {
        results.push(task.await.unwrap());
    }
    running.aggregator.await.unwrap().unwrap();

    let refusal = results[2].as_ref().unwrap_err().to_string();
    assert!(refusal.contains("ControlledAccessGrants"), "{refusal}");
    let done = results[0].as_ref().unwrap();
    assert!(close(read_tensor_mean(&done.final_model), 2.0));
    let report = verify_log(&fed.aggregator_dir().join("log.jsonl"), None).unwrap();
    assert_eq!(report.joined.len(), 2);
    assert_eq!(report.rounds[0].included.len(), 2);
}

/// Runs something once, just before training in one round.
struct Before {
    round: u32,
    act: Box<dyn Fn() + Send>,
    then: Box<Fixed>,
}

impl Trainer for Before {
    fn train(&mut self, ctx: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        if ctx.round == self.round {
            (self.act)();
        }
        self.then.train(ctx, submit)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn revoking_a_grant_takes_the_site_out_of_the_next_round() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = Arc::new(
        LocalFederation::create(dir.path(), layout(), &init, &gated_options(3, 2)).unwrap(),
    );
    // The committee takes back site 2's grant while round 1 is still open.
    // Nothing reads the list again until round 2 is about to open.
    let committee = fed.clone();
    let trainers: Vec<Box<dyn Trainer>> = vec![
        Box::new(Before {
            round: 1,
            act: Box::new(move || committee.revoke_access(2).unwrap()),
            then: Fixed::new(1.0, 1),
        }),
        Fixed::new(1.0, 1),
        Fixed::new(4.0, 1),
    ];
    let (running, results) = run_all(&fed, trainers, None).await;
    running.aggregator.await.unwrap().unwrap();
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();

    assert_eq!(summaries[2].rounds_contributed, [1]);
    assert_eq!(summaries[2].rounds_refused, [2]);
    assert_eq!(summaries[0].rounds_contributed, [1, 2]);
    // Round 1 averaged all three, round 2 only the two still admitted.
    assert!(close(read_tensor_mean(&summaries[0].final_model), 1.0));
    assert_eq!(summaries[2].final_digest, summaries[0].final_digest);

    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert_eq!(report.rounds[0].included.len(), 3);
    assert_eq!(report.rounds[1].included.len(), 2);
    assert_eq!(report.revoked.len(), 1);
    assert_eq!(report.revoked[0].site, fed.sites[2].did_string());
    assert!(
        report.revoked[0].reason.contains("revoked"),
        "{:?}",
        report.revoked
    );
    assert!(report.complete(), "a revoked site is not a gap: {report:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn sampled_rounds_take_only_the_sites_drawn() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut options = options(4, 1, 3);
    options.sample = Some(2);
    options.min_cohort = 2;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options).unwrap();
    let trained: Vec<_> = (0..4)
        .map(|_| Arc::<std::sync::Mutex<Vec<u32>>>::default())
        .collect();
    let trainers: Vec<Box<dyn Trainer>> = trained
        .iter()
        .map(|rounds| {
            Box::new(Fixed {
                value: 3.0,
                weight: 1,
                rounds: rounds.clone(),
            }) as Box<dyn Trainer>
        })
        .collect();
    let (running, results) = run_all(&fed, trainers, None).await;
    running.aggregator.await.unwrap().unwrap();
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    for summary in &summaries {
        assert_eq!(summary.final_digest, summaries[0].final_digest);
        assert!(close(read_tensor_mean(&summary.final_model), 3.0));
    }

    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");
    for round in &report.rounds {
        let invited = round.invited.as_ref().unwrap();
        assert_eq!(invited.len(), 2);
        let included: Vec<&String> = round.included.iter().map(|c| &c.site).collect();
        assert_eq!(included, invited.iter().collect::<Vec<_>>());
    }
    for (i, rounds) in trained.iter().enumerate() {
        let did = fed.sites[i].did_string();
        assert_eq!(
            *rounds.lock().unwrap(),
            report.rounds_with(&did),
            "site {i}"
        );
        assert_eq!(summaries[i].rounds_contributed, report.rounds_with(&did));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_between_did_bio_parties_completes() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut options = options(3, 1, 1);
    options.did_bio = Some("devnet".into());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options).unwrap();
    assert!(
        fed.plan
            .sites
            .iter()
            .all(|s| s.starts_with("did:bio:devnet:"))
    );
    let trainers: Vec<Box<dyn Trainer>> = (0..3)
        .map(|_| Fixed::new(2.0, 1) as Box<dyn Trainer>)
        .collect();
    let (running, results) = run_all(&fed, trainers, None).await;
    running.aggregator.await.unwrap().unwrap();
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    let log = fed.aggregator_dir().join("log.jsonl");
    assert!(
        verify_log(&log, Some(&summaries[0].final_model))
            .unwrap()
            .complete()
    );
}

/// A registry that holds whatever a test puts at an address.
#[derive(Default)]
struct Ledger {
    accounts: std::collections::BTreeMap<[u8; 32], fedtally_core::registry::Account>,
    down: bool,
}

impl fedtally_node::registry::Registry for Ledger {
    fn account(
        &self,
        address: &[u8; 32],
    ) -> Result<Option<fedtally_core::registry::Account>, String> {
        if self.down {
            return Err("the endpoint is down".into());
        }
        Ok(self.accounts.get(address).cloned())
    }
}

/// The registry's record of a deactivated identifier, as devnet stores one,
/// with this identifier's key in place of the original's.
fn tombstone(did: &str) -> ([u8; 32], fedtally_core::registry::Account) {
    use base64::Engine;
    let image = "TVjvjfsd7fMvAAAAAAAAAPt33uRUXs12r5aXnY/MgROqmHNIgyZSyvvtpJKzO83qwQHEf6FqAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
    let mut data = base64::engine::general_purpose::STANDARD
        .decode(image)
        .unwrap();
    let key = bs58::decode(did.rsplit(':').next().unwrap())
        .into_vec()
        .unwrap();
    data[17..49].copy_from_slice(&key);
    let owner = bs58::decode("H1gnV4GjNT3UV7AgGNUCkSaciuVVtM7hKb8JhPV3Xxy6")
        .into_vec()
        .unwrap()
        .try_into()
        .unwrap();
    let address = fedtally_core::registry::account_address(did).unwrap();
    (address, fedtally_core::registry::Account { owner, data })
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_sits_out_when_the_registry_says_an_operator_is_gone() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut options = options(4, 1, 1);
    options.did_bio = Some("devnet".into());
    options.min_cohort = 2;
    options.timeouts.collect_secs = 6;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options).unwrap();

    let (address, account) = tombstone(&fed.plan.helpers[0].did);
    let mut sees_tombstone = Ledger::default();
    sees_tombstone.accounts.insert(address, account);
    let registries: Vec<Option<Ledger>> = vec![
        Some(sees_tombstone),
        Some(Ledger {
            down: true,
            ..Ledger::default()
        }),
        Some(Ledger::default()),
        None,
    ];

    let running = start(&fed, None).await;
    let tasks: Vec<_> = registries
        .into_iter()
        .enumerate()
        .map(|(i, registry)| {
            let mut config = fed.site_config(i, &running.addr);
            config.registry = registry.map(|r| Arc::new(r) as _);
            tokio::spawn(run_site(config, Fixed::new(2.0, 1)))
        })
        .collect();
    let mut summaries = Vec::new();
    for task in tasks {
        summaries.push(task.await.unwrap().unwrap());
    }
    running.aggregator.await.unwrap().unwrap();

    // One site saw the helper deactivated, one could not ask at all. Both
    // stay out. A site that asked and found nothing registered carries on,
    // and so does one that was never told to ask.
    assert_eq!(summaries[0].rounds_refused, [1]);
    assert_eq!(summaries[1].rounds_refused, [1]);
    assert_eq!(summaries[2].rounds_contributed, [1]);
    assert_eq!(summaries[3].rounds_contributed, [1]);
    let report = verify_log(&fed.aggregator_dir().join("log.jsonl"), None).unwrap();
    assert_eq!(report.rounds[0].included.len(), 2);
}

fn bounded_options(sites: usize, rounds: u32) -> Options {
    let mut options = private_options(sites, rounds);
    options.bounded = true;
    options.timeouts.collect_secs = 20;
    options
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bounded_run_proves_every_update_short_and_completes() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &bounded_options(4, 2)).unwrap();
    assert!(fed.plan.norm().is_some());
    let trainers: Vec<Box<dyn Trainer>> = (0..4)
        .map(|_| Box::new(Still { weight: 1 }) as Box<dyn Trainer>)
        .collect();
    let (running, results) = run_all(&fed, trainers, None).await;
    running.aggregator.await.unwrap().unwrap();
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    assert!(
        summaries
            .iter()
            .all(|s| s.final_digest == summaries[0].final_digest)
    );
    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");
    assert!(report.rounds.iter().all(|r| r.total_weight == Some(4)));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_that_breaks_the_norm_bound_is_left_out_of_the_round() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &bounded_options(4, 2)).unwrap();
    let running = start(&fed, None).await;

    // The cheat keeps the plan's noise and clips sixty-four times too far out,
    // then pushes every weight as far as that lets it.
    let mut config = fed.site_config(3, &running.addr);
    let mut terms = fed.plan.privacy.unwrap();
    terms.clip = 64 * fed.plan.norm().unwrap().enforced();
    config.cheat = Some(terms);
    let cheat = tokio::spawn(run_site(config, Fixed::new(1.0e6, 1)));
    let honest: Vec<_> = (0..3)
        .map(|i| {
            tokio::spawn(run_site(
                fed.site_config(i, &running.addr),
                Box::new(Still { weight: 1 }),
            ))
        })
        .collect();

    let mut summaries = Vec::new();
    for site in honest {
        summaries.push(site.await.unwrap().unwrap());
    }
    let _ = cheat.await.unwrap();
    running.aggregator.await.unwrap().unwrap();

    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.finished, "{report:?}");
    for round in &report.rounds {
        assert_eq!(round.total_weight, Some(3), "{round:?}");
        assert_eq!(round.excluded, vec![fed.plan.sites[3].clone()]);
    }
    // Nothing of the cheat's push reached the model: what moved is noise.
    let moved = read_weights(&summaries[0].final_model)
        .iter()
        .map(|w| (w - 1.0).abs())
        .fold(0.0, f64::max);
    assert!(moved < 2.0, "the model moved by {moved}");
}

#[tokio::test(flavor = "multi_thread")]
async fn without_norm_proofs_the_same_cheat_moves_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &private_options(4, 1)).unwrap();
    let running = start(&fed, None).await;
    let mut config = fed.site_config(3, &running.addr);
    let mut terms = fed.plan.privacy.unwrap();
    terms.clip *= 1 << 12;
    config.cheat = Some(terms);
    let cheat = tokio::spawn(run_site(config, Fixed::new(1.0e6, 1)));
    let honest: Vec<_> = (0..3)
        .map(|i| {
            tokio::spawn(run_site(
                fed.site_config(i, &running.addr),
                Box::new(Still { weight: 1 }),
            ))
        })
        .collect();
    let mut summaries = Vec::new();
    for site in honest {
        summaries.push(site.await.unwrap().unwrap());
    }
    cheat.await.unwrap().unwrap();
    running.aggregator.await.unwrap().unwrap();
    let moved = read_weights(&summaries[0].final_model)
        .iter()
        .map(|w| (w - 1.0).abs())
        .fold(0.0, f64::max);
    assert!(moved > 10.0, "the model moved by {moved}");
}

/// The aggregator sees the helpers' messages before it settles its own, so it
/// can write one that makes any proof pass. Only the site that recomputes
/// that message from the committed vector can tell.
#[tokio::test(flavor = "multi_thread")]
async fn an_aggregator_passing_a_long_update_is_caught_by_the_audit() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut opts = bounded_options(4, 1);
    opts.timeouts.attest_secs = 3;
    opts.timeouts.ack_secs = 3;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &opts).unwrap();
    let size = fed.plan.norm().unwrap().share_bytes();
    let hook: Hook = Arc::new(move |stage, round_dir| {
        if let Stage::BeforeVerdict { .. } = stage {
            let helper = std::fs::read(round_dir.join("check-0.bin")).unwrap();
            let path = round_dir.join("screened.bin");
            let mut own = std::fs::read(&path).unwrap();
            let cheat = 3 * size..4 * size;
            for (mine, theirs) in own[cheat.clone()]
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(helper[cheat].as_chunks::<8>().0)
            {
                let v = u64::from_le_bytes(*theirs);
                let negated = -fedtally_core::field::Fp::new(v);
                mine.copy_from_slice(&negated.value().to_le_bytes());
            }
            std::fs::write(path, own).unwrap();
        }
    });
    let running = start(&fed, Some(hook)).await;
    let mut config = fed.site_config(3, &running.addr);
    let mut terms = fed.plan.privacy.unwrap();
    terms.clip = 64 * fed.plan.norm().unwrap().enforced();
    config.cheat = Some(terms);
    let cheat = tokio::spawn(run_site(config, Fixed::new(1.0e6, 1)));
    let honest: Vec<_> = (0..3)
        .map(|i| {
            tokio::spawn(run_site(
                fed.site_config(i, &running.addr),
                Box::new(Still { weight: 1 }),
            ))
        })
        .collect();
    let mut results = Vec::new();
    for site in honest {
        results.push(site.await.unwrap());
    }
    let _ = cheat.await.unwrap();
    running.aggregator.await.unwrap().unwrap();

    let caught = results
        .iter()
        .filter(|r| matches!(r, Err(NodeError::Check(why)) if why.contains("norm proof")))
        .count();
    assert_eq!(caught, 1, "one honest site audits the cheat: {results:?}");
    assert!(results.iter().all(Result::is_err), "{results:?}");
    let report = verify_log(&fed.aggregator_dir().join("log.jsonl"), None).unwrap();
    assert!(!report.finished);
    assert_eq!(report.rounds[0].total_weight, Some(4));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_gated_run_works_with_post_quantum_keys_and_credentials() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut opts = gated_options(3, 1);
    opts.sig_alg = fedtally_core::keys::SigAlg::MlDsa65;
    // These signatures are slow in a debug build, and every site joins anyway.
    opts.timeouts.join_secs = 120;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &opts).unwrap();
    let (running, results) = run_all(&fed, fixed_trainers(3), None).await;
    running.aggregator.await.unwrap().unwrap();
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");
    let text = std::fs::read_to_string(log).unwrap();
    assert!(text.contains("mldsa65-jcs-2024"));
    assert!(!text.contains("eddsa-jcs-2022"));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_aggregator_waits_for_a_named_watcher_to_fetch_the_last_round() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options(2, 1, 1)).unwrap();
    let watcher = Arc::new(Identity::generate(SigAlg::Ed25519).unwrap());
    let (tx, rx) = oneshot::channel();
    let mut config = fed.aggregator_config("127.0.0.1:0");
    config.keep_spool = true;
    config.watchers = vec![watcher.did_string()];
    config.on_ready = Some(Box::new(move |addr| {
        let _ = tx.send(addr);
    }));
    let aggregator = tokio::spawn(run_aggregator(config));
    let addr = rx.await.unwrap();
    let helper = tokio::spawn(run_helper(fed.helper_config(0, &addr)));
    let sites: Vec<_> = fixed_trainers(2)
        .into_iter()
        .enumerate()
        .map(|(i, t)| tokio::spawn(run_site(fed.site_config(i, &addr), t)))
        .collect();
    for site in sites {
        site.await.unwrap().unwrap();
    }
    helper.await.unwrap().unwrap();

    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!aggregator.is_finished(), "it left before the watcher came");

    let mut replay = Replay::new();
    let log = std::fs::read(fed.aggregator_dir().join("log.jsonl")).unwrap();
    for line in log.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        replay.apply_line(line).unwrap();
    }
    let root = replay.rounds[0].outcome.as_ref().unwrap().aggregate;
    let source = ObserverConfig {
        identity: watcher,
        plan_line: fed.plan_line.clone(),
        addr,
        dir: dir.path().join("watcher"),
        give_up_after: Duration::from_secs(10),
    };
    let out = dir.path().join("aggregate.bin");
    let object = Object::Aggregate { round: 1 };
    fetch_vector(&source, object, root, &out).await.unwrap();

    // Well short of what it would wait for a watcher that never comes.
    tokio::time::timeout(Duration::from_secs(10), aggregator)
        .await
        .expect("it stayed after the watcher had what it came for")
        .unwrap()
        .unwrap();
}

fn retrying_options(sites: usize, rounds: u32, retries: u32) -> Options {
    let mut options = private_options(sites, rounds);
    options.min_cohort = 2;
    options.retries = retries;
    options.timeouts.attest_secs = 6;
    options
}

/// A run in which a hook stops a site's task at a given stage, as if the
/// machine had gone away.
struct Losing {
    addr: String,
    aggregator: JoinHandle<Result<Replay, NodeError>>,
    sites: Vec<JoinHandle<Result<SiteSummary, NodeError>>>,
}

async fn start_losing(
    fed: &LocalFederation,
    trainers: Vec<Box<dyn Trainer>>,
    losses: Vec<(Stage, usize)>,
) -> Losing {
    start_losing_with(fed, trainers, losses, None).await
}

async fn start_losing_with(
    fed: &LocalFederation,
    trainers: Vec<Box<dyn Trainer>>,
    losses: Vec<(Stage, usize)>,
    share_hook: Option<ShareHook>,
) -> Losing {
    let handles: Arc<Mutex<BTreeMap<usize, AbortHandle>>> = Arc::default();
    let hook: Hook = Arc::new({
        let handles = handles.clone();
        move |stage, _| {
            for (when, site) in &losses {
                if *when == stage
                    && let Some(handle) = handles.lock().unwrap().remove(site)
                {
                    handle.abort();
                    // An abort lands when the task next yields, which may be
                    // after it has read what the aggregator sends next.
                    std::thread::sleep(Duration::from_millis(300));
                }
            }
        }
    });
    let (tx, rx) = oneshot::channel();
    let mut config = fed.aggregator_config("127.0.0.1:0");
    config.hook = Some(hook);
    config.keep_spool = true;
    config.on_ready = Some(Box::new(move |addr| {
        let _ = tx.send(addr);
    }));
    let aggregator = tokio::spawn(run_aggregator(config));
    let addr = rx.await.unwrap();
    for h in 0..fed.helpers.len() {
        let mut helper = fed.helper_config(h, &addr);
        helper.share_hook = share_hook.clone();
        tokio::spawn(run_helper(helper));
    }
    let sites: Vec<_> = trainers
        .into_iter()
        .enumerate()
        .map(|(i, t)| tokio::spawn(run_site(fed.site_config(i, &addr), t)))
        .collect();
    let mut held = handles.lock().unwrap();
    for (i, site) in sites.iter().enumerate() {
        held.insert(i, site.abort_handle());
    }
    drop(held);
    Losing {
        addr,
        aggregator,
        sites,
    }
}

impl Losing {
    /// The final log state, and each site's result unless it was stopped.
    async fn finish(self) -> (Replay, Vec<Option<Result<SiteSummary, NodeError>>>) {
        let mut results = Vec::new();
        for site in self.sites {
            results.push(site.await.ok());
        }
        (self.aggregator.await.unwrap().unwrap(), results)
    }
}

fn stage_outcome(round: u32) -> Stage {
    Stage::BeforeOutcome { round }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_lost_before_it_attests_costs_an_attempt_and_not_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed =
        LocalFederation::create(dir.path(), layout(), &init, &retrying_options(4, 1, 1)).unwrap();
    let staying = Fixed::new(1.0, 1);
    let trained = staying.rounds.clone();
    let mut trainers: Vec<Box<dyn Trainer>> = vec![staying];
    trainers.extend(fixed_trainers(3));
    let losing = start_losing(&fed, trainers, vec![(stage_outcome(1), 3)]).await;
    let (replay, results) = losing.finish().await;

    assert_eq!(replay.phase, Phase::Done);
    assert_eq!(replay.retried, 1);
    assert!(results[3].is_none());
    let summary = results[0].as_ref().unwrap().as_ref().unwrap();
    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summary.final_model)).unwrap();
    assert!(report.finished);
    assert!(!report.complete(), "a retry is worth a second look");
    assert!(report.rounds[0].voided && !report.rounds[0].sealed);
    assert!(report.rounds[1].sealed && !report.rounds[1].voided);
    assert_eq!(report.rounds[1].included.len(), 3);
    assert_eq!(
        replay.spent,
        fed.plan.round_cost(4).unwrap() + fed.plan.round_cost(3).unwrap()
    );

    // The same step is trained again, into new masks and new noise.
    assert_eq!(*trained.lock().unwrap(), [1, 1]);
    let (first, second) = (&replay.rounds[0], &replay.rounds[1]);
    for site in 0..3 {
        let (old, new) = (&first.contributions[&site], &second.contributions[&site]);
        assert_ne!(old.masked, new.masked);
        assert_ne!(old.sealed, new.sealed);
    }
    assert_eq!(first.base, second.base);
    assert!(
        !fed.site_dir(0).join("round-1").exists(),
        "the seeds of the given up round are still on disk"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plan_with_no_noise_cannot_ask_for_retries_and_still_stops() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut asking = options(3, 1, 1);
    asking.min_cohort = 2;
    asking.retries = 1;
    assert!(LocalFederation::create(dir.path(), layout(), &init, &asking).is_err());

    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut plain = options(3, 1, 1);
    plain.min_cohort = 2;
    plain.timeouts.attest_secs = 3;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &plain).unwrap();
    let losing = start_losing(&fed, fixed_trainers(3), vec![(stage_outcome(1), 2)]).await;
    let (replay, _) = losing.finish().await;
    assert_eq!(replay.phase, Phase::Aborted);
    assert_eq!(replay.rounds.len(), 1);
    let reason = replay.abort.unwrap();
    assert!(reason.contains("no valid attestation"), "{reason}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_lost_after_the_seal_still_stops_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut options = retrying_options(4, 1, 1);
    options.timeouts.ack_secs = 3;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options).unwrap();
    let lost = vec![(Stage::BeforeSeal { round: 1 }, 3)];
    let losing = start_losing(&fed, fixed_trainers(4), lost).await;
    let (replay, _) = losing.finish().await;
    assert_eq!(replay.phase, Phase::Aborted);
    assert_eq!(replay.retried, 0);
    let reason = replay.abort.unwrap();
    assert!(reason.contains("no acknowledgement"), "{reason}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_stops_once_its_retries_are_spent() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed =
        LocalFederation::create(dir.path(), layout(), &init, &retrying_options(4, 1, 1)).unwrap();
    let lost = vec![(stage_outcome(1), 3), (stage_outcome(2), 2)];
    let losing = start_losing(&fed, fixed_trainers(4), lost).await;
    let (replay, _) = losing.finish().await;
    assert_eq!(replay.phase, Phase::Aborted);
    assert_eq!(replay.retried, 1);
    assert_eq!(replay.rounds.len(), 2);
    assert_eq!(
        replay.spent,
        fed.plan.round_cost(4).unwrap() + fed.plan.round_cost(3).unwrap()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn too_few_sites_left_stops_the_run_even_with_retries_left() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut options = retrying_options(3, 1, 1);
    options.min_cohort = 2;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options).unwrap();
    let lost = vec![(stage_outcome(1), 2), (stage_outcome(1), 1)];
    let losing = start_losing(&fed, fixed_trainers(3), lost).await;
    let (replay, _) = losing.finish().await;
    assert_eq!(replay.phase, Phase::Aborted);
    assert_eq!(replay.retried, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_retry_in_a_bounded_plan_proves_its_updates_afresh() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut options = retrying_options(4, 1, 1);
    options.bounded = true;
    options.timeouts.collect_secs = 20;
    let fed = LocalFederation::create(dir.path(), layout(), &init, &options).unwrap();
    let losing = start_losing(&fed, fixed_trainers(4), vec![(stage_outcome(1), 3)]).await;
    let (replay, _) = losing.finish().await;
    assert_eq!(replay.phase, Phase::Done);
    let (first, second) = (&replay.rounds[0], &replay.rounds[1]);
    for site in &second.included {
        assert_ne!(
            first.contributions[site].proof,
            second.contributions[site].proof
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_lost_site_that_comes_back_sits_out_the_retry_and_rejoins_after_it() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed =
        LocalFederation::create(dir.path(), layout(), &init, &retrying_options(4, 2, 1)).unwrap();
    let mut losing = start_losing(&fed, fixed_trainers(4), vec![(stage_outcome(1), 3)]).await;
    let log = fed.aggregator_dir().join("log.jsonl");
    let waited = tokio::time::timeout(Duration::from_secs(120), async {
        while !std::fs::read_to_string(&log).is_ok_and(|text| text.contains(r#""kind":"retry""#)) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });
    waited.await.expect("no retry in the log");

    let back = Fixed::new(1.0, 1);
    let trained = back.rounds.clone();
    let config = fed.site_config(3, &losing.addr);
    losing.sites[3] = tokio::spawn(run_site(config, back));
    let (replay, results) = losing.finish().await;

    assert_eq!(replay.phase, Phase::Done);
    assert_eq!(replay.rounds.len(), 3);
    assert_eq!(replay.rounds[1].included, [0, 1, 2]);
    assert_eq!(replay.rounds[2].included, [0, 1, 2, 3]);
    assert_eq!(
        *trained.lock().unwrap(),
        [2],
        "it trains the next step only"
    );
    let summary = results[3].as_ref().unwrap().as_ref().unwrap();
    verify_log(&log, Some(&summary.final_model)).unwrap();
    assert!(!fed.site_dir(3).join("round-1").exists());
}

/// The coordinates of a stored aggregate, as signed values in grid units.
fn read_aggregate(fed: &LocalFederation, round: u32) -> Vec<i128> {
    let path = fed
        .aggregator_dir()
        .join(format!("round-{round}/aggregate.bin"));
    let file = std::fs::File::open(path).unwrap();
    let shape = fed.plan.shape();
    let mut all = Vec::new();
    let mut chunk = Vec::new();
    for at in 0..shape.chunks() {
        vector::read_chunk(&file, shape, at, &mut chunk).unwrap();
        all.extend(chunk.iter().map(|v| match *v > P / 2 {
            true => i128::from(*v) - i128::from(P),
            false => i128::from(*v),
        }));
    }
    all
}

/// Two sums over cohorts of 4 and 3 differ by the lost site's update and the
/// noise of 7 sites' worth: its own, and two draws from each of the others.
#[tokio::test(flavor = "multi_thread")]
async fn a_retry_hides_the_lost_update_behind_noise_alone() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed =
        LocalFederation::create(dir.path(), layout(), &init, &retrying_options(4, 1, 1)).unwrap();
    let mut trainers = fixed_trainers(3);
    trainers.push(Fixed::new(2.0, 1));
    let losing = start_losing(&fed, trainers, vec![(stage_outcome(1), 3)]).await;
    let (replay, _) = losing.finish().await;
    assert_eq!(replay.phase, Phase::Done);

    let (first, second) = (read_aggregate(&fed, 1), read_aggregate(&fed, 2));
    // The first tensor, where every other site's update is exactly zero.
    let gaps: Vec<f64> = first[..1000]
        .iter()
        .zip(&second[..1000])
        .map(|(a, b)| (a - b) as f64)
        .collect();
    let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
    let variance = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / (gaps.len() - 1) as f64;
    let noise = fed.plan.privacy.as_ref().unwrap().noise as f64;
    let ratio = variance / (7.0 * noise);
    eprintln!("variance over 7 sites' noise: {ratio:.3}");
    // Over 1000 samples the estimate has a spread of about 4.5%.
    assert!((0.75..1.25).contains(&ratio), "{ratio}");
}

#[tokio::test(flavor = "multi_thread")]
async fn without_noise_two_sums_give_the_missing_update_away_exactly() {
    let sum_of = |values: Vec<f32>| async move {
        let dir = tempfile::tempdir().unwrap();
        let init = initial_model(dir.path());
        let mut options = options(values.len(), 1, 1);
        options.min_cohort = 2;
        let fed = LocalFederation::create(dir.path(), layout(), &init, &options).unwrap();
        let trainers = values
            .into_iter()
            .map(|value| Fixed::new(value, 1) as Box<dyn Trainer>)
            .collect();
        let (replay, _) = start_losing(&fed, trainers, Vec::new())
            .await
            .finish()
            .await;
        assert_eq!(replay.phase, Phase::Done);
        (read_aggregate(&fed, 1), options.encoding.frac_bits)
    };
    let (with, frac_bits) = sum_of(vec![1.0, 1.5, 3.0]).await;
    let (without, _) = sum_of(vec![1.0, 1.5]).await;
    // The base is 1.0, so the third site's update is 2.0 in every coordinate.
    let update = 2i128 << frac_bits;
    for (a, b) in with[..1000].iter().zip(&without[..1000]) {
        assert_eq!(a - b, update);
    }
}

/// Trains towards a different value each time it is asked.
struct Changing(Vec<f32>);

impl Trainer for Changing {
    fn train(&mut self, _: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        let value = self.0.remove(0);
        let w = vec![value; 1000];
        let b = vec![f64::from(value); 24];
        submit(&[TensorView::F32(&w), TensorView::F64(&b)], 1)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

/// This is a limit, not a feature: nothing in the log can tell an honest
/// retrain from a site that read the first sum and changed its mind.
#[tokio::test(flavor = "multi_thread")]
async fn a_site_in_a_retry_can_send_a_different_update() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed =
        LocalFederation::create(dir.path(), layout(), &init, &retrying_options(4, 1, 1)).unwrap();
    let mut trainers = fixed_trainers(3);
    trainers.insert(0, Box::new(Changing(vec![1.0, -5.0])));
    let losing = start_losing(&fed, trainers, vec![(stage_outcome(1), 3)]).await;
    let (replay, _) = losing.finish().await;
    assert_eq!(replay.phase, Phase::Done);
    assert_eq!(replay.rounds[1].included, [0, 1, 2]);
}

/// Also a limit: a helper can push a site of its choosing out of a run that
/// then carries on. Without retries the same lie stops the run.
#[tokio::test(flavor = "multi_thread")]
async fn a_lying_helper_can_force_a_site_out_of_a_plan_that_retries() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let fed =
        LocalFederation::create(dir.path(), layout(), &init, &retrying_options(4, 1, 1)).unwrap();
    let lied = AtomicBool::new(false);
    let hook: ShareHook = Arc::new(move |shares| {
        if !lied.swap(true, Ordering::SeqCst) {
            shares[3] = shares[3] + shares[0];
        }
    });
    let losing = start_losing_with(&fed, fixed_trainers(4), Vec::new(), Some(hook)).await;
    let (replay, results) = losing.finish().await;
    assert_eq!(replay.phase, Phase::Done);
    assert_eq!(replay.retried, 1);
    assert!(
        matches!(&results[3], Some(Err(NodeError::Check(why))) if why.contains("helper")),
        "{:?}",
        results[3]
    );
    assert_eq!(replay.rounds[1].included, [0, 1, 2]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_scoring_run_pays_the_group_that_moved_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut opts = options(6, 1, 2);
    opts.scoring = Some(2);
    let fed = LocalFederation::create(dir.path(), layout(), &init, &opts).unwrap();
    // Five sites pull the weights the same way and the last pulls against
    // them. Whichever pair it lands in agrees less with the other two.
    let mut trainers: Vec<Box<dyn Trainer>> = (0..5)
        .map(|_| Fixed::new(2.0, 1) as Box<dyn Trainer>)
        .collect();
    trainers.push(Fixed::new(0.0, 1));
    let (running, results) = run_all(&fed, trainers, None).await;
    running.aggregator.await.unwrap().unwrap();
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");

    let against = &fed.plan.sites[5];
    let round = &report.rounds[0];
    assert_eq!(round.groups.len(), 3);
    let (with, without): (Vec<_>, Vec<_>) =
        round.groups.iter().partition(|g| g.sites.contains(against));
    assert!(without.iter().all(|g| with[0].share < g.share), "{round:?}");
    assert_eq!(
        round.groups.iter().filter_map(|g| g.share).sum::<u32>(),
        1_000_000
    );
    let total: f64 = report.rewards.iter().map(|r| r.share).sum();
    assert!((total - 1.0).abs() < 1e-9, "{total}");
    let earned = |site: &String| {
        report
            .rewards
            .iter()
            .find(|r| &r.site == site)
            .map_or(0.0, |r| r.share)
    };
    assert!(earned(against) < 1.0 / 6.0, "{:?}", report.rewards);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_private_scoring_run_pays_for_its_groups_in_the_budget() {
    let dir = tempfile::tempdir().unwrap();
    let init = initial_model(dir.path());
    let mut opts = private_options(4, 2);
    opts.scoring = Some(2);
    let fed = LocalFederation::create(dir.path(), layout(), &init, &opts).unwrap();
    let trainers: Vec<Box<dyn Trainer>> = (0..4)
        .map(|_| Box::new(Still { weight: 1 }) as Box<dyn Trainer>)
        .collect();
    let (running, results) = run_all(&fed, trainers, None).await;
    running.aggregator.await.unwrap().unwrap();
    let summaries: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    let log = fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");
    let privacy = fed.plan.privacy.unwrap();
    let pair = privacy.round_cost(fed.plan.layout.coords(), 2).unwrap();
    assert_eq!(fed.plan.round_cost(4), Some(pair));
    let spent = report.privacy.unwrap();
    assert!(spent.rho_spent <= spent.rho_budget);
    assert!(spent.epsilon_spent > 6.0, "{spent:?}");
    for round in &report.rounds {
        assert_eq!(
            round.groups.iter().filter_map(|g| g.share).sum::<u32>(),
            1_000_000
        );
    }
}
