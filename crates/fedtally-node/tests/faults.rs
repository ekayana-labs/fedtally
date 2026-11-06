//! Runs through a proxy that can cut any party off, and restarts after the damage
//! a power cut can do. All on one machine: this shows the protocol copes, and says
//! nothing about a real network or real disks.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fedtally_core::model::{Dtype, Layout, TensorSpec, TensorView, write_model};
use fedtally_core::replay::Phase;
use fedtally_core::report::verify_log;
use fedtally_node::NodeError;
use fedtally_node::aggregator::{AggregatorConfig, Hook, Stage, run_aggregator};
use fedtally_node::helper::run_helper;
use fedtally_node::local::{LocalFederation, Options};
use fedtally_node::site::{SiteSummary, Submit, TrainContext, Trainer, run_site};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, watch};
use tokio::task::JoinHandle;

const COORDS: usize = 4096;

type SiteTask = JoinHandle<Result<SiteSummary, NodeError>>;
type HelperTask = JoinHandle<Result<(), NodeError>>;

fn layout() -> Layout {
    Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![COORDS as u64],
        }],
    }
}

struct Toward(f32);

impl Trainer for Toward {
    fn train(&mut self, _: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        submit(&[TensorView::F32(&vec![self.0; COORDS])], 1)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

/// Doesn't start on the first round until it's told to.
struct Held(Arc<AtomicBool>);

impl Trainer for Held {
    fn train(&mut self, ctx: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        while ctx.round == 1 && !self.0.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(5));
        }
        Toward(3.0).train(ctx, submit)
    }
}

/// One party's connection to the aggregator. Cutting it drops what's open and
/// refuses anything new. The aggregator can move behind it, as after a restart.
#[derive(Clone)]
struct Link {
    addr: String,
    up: watch::Sender<bool>,
}

impl Link {
    async fn open(target: Arc<Mutex<String>>) -> Link {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let (up, _) = watch::channel(true);
        let link = Link {
            addr,
            up: up.clone(),
        };
        tokio::spawn(async move {
            while let Ok((mut peer, _)) = listener.accept().await {
                let mut state = up.subscribe();
                if !*state.borrow() {
                    continue;
                }
                let target = target.lock().unwrap().clone();
                tokio::spawn(async move {
                    let Ok(mut server) = TcpStream::connect(target).await else {
                        return;
                    };
                    tokio::select! {
                        _ = tokio::io::copy_bidirectional(&mut peer, &mut server) => {}
                        _ = state.wait_for(|up| !*up) => {}
                    }
                });
            }
        });
        link
    }

    fn cut(&self) {
        self.up.send_replace(false);
    }

    fn heal(&self) {
        self.up.send_replace(true);
    }
}

struct Net {
    fed: LocalFederation,
    sites: Vec<Link>,
    helpers: Vec<Link>,
}

fn federation(root: &Path, options: &Options) -> LocalFederation {
    let init = root.join("init.safetensors");
    write_model(&init, &layout(), &[TensorView::F32(&vec![0.0; COORDS])]).unwrap();
    LocalFederation::create(root, layout(), &init, options).unwrap()
}

fn options(sites: usize, rounds: u32) -> Options {
    let mut options = Options::new(sites, 1, rounds);
    options.chunk_coords = 256;
    options.min_cohort = 2;
    options.timeouts.collect_secs = 30;
    options.timeouts.attest_secs = 30;
    options.timeouts.ack_secs = 30;
    options
}

fn aggregator_config(fed: &LocalFederation, hook: Option<Hook>) -> AggregatorConfig {
    let mut config = fed.aggregator_config("127.0.0.1:0");
    config.hook = hook;
    config
}

async fn start_aggregator(
    mut config: AggregatorConfig,
) -> (
    String,
    JoinHandle<Result<fedtally_core::replay::Replay, NodeError>>,
) {
    let (tx, rx) = oneshot::channel();
    config.on_ready = Some(Box::new(move |addr| {
        let _ = tx.send(addr);
    }));
    let aggregator = tokio::spawn(run_aggregator(config));
    (rx.await.unwrap(), aggregator)
}

impl Net {
    async fn over(fed: LocalFederation, aggregator_addr: String) -> Net {
        let target = Arc::new(Mutex::new(aggregator_addr));
        let mut sites = Vec::new();
        for _ in 0..fed.sites.len() {
            sites.push(Link::open(target.clone()).await);
        }
        let mut helpers = Vec::new();
        for _ in 0..fed.helpers.len() {
            helpers.push(Link::open(target.clone()).await);
        }
        Net {
            fed,
            sites,
            helpers,
        }
    }

    fn all(&self) -> impl Iterator<Item = &Link> {
        self.sites.iter().chain(&self.helpers)
    }

    fn run(&self) -> (Vec<SiteTask>, Vec<HelperTask>) {
        self.run_with(|site| Box::new(Toward(1.0 + site as f32)))
    }

    fn run_with(
        &self,
        trainer: impl Fn(usize) -> Box<dyn Trainer>,
    ) -> (Vec<SiteTask>, Vec<HelperTask>) {
        let helpers = self
            .helpers
            .iter()
            .enumerate()
            .map(|(h, link)| tokio::spawn(run_helper(self.fed.helper_config(h, &link.addr))))
            .collect();
        let sites = self
            .sites
            .iter()
            .enumerate()
            .map(|(i, link)| {
                tokio::spawn(run_site(self.fed.site_config(i, &link.addr), trainer(i)))
            })
            .collect();
        (sites, helpers)
    }
}

/// A hook that runs `act` the first time the aggregator reaches `when`.
fn at(when: Stage, act: impl Fn() + Send + Sync + 'static) -> Hook {
    let fired = AtomicBool::new(false);
    Arc::new(move |stage, _| {
        if stage == when && !fired.swap(true, Ordering::SeqCst) {
            act();
        }
    })
}

async fn wait_in_log(log: &Path, kind: &str) {
    let want = format!(r#""kind":"{kind}""#);
    let waited = tokio::time::timeout(Duration::from_secs(120), async {
        while !std::fs::read_to_string(log).is_ok_and(|text| text.contains(&want)) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
    waited.await.expect("the record never reached the log");
}

/// The run finished, every site has the same model, and the log says so.
async fn all_finish(net: &Net, sites: Vec<SiteTask>) {
    let mut models = Vec::new();
    for site in sites {
        models.push(site.await.unwrap().unwrap());
    }
    let log = net.fed.aggregator_dir().join("log.jsonl");
    let report = verify_log(&log, Some(&models[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");
    assert!(
        models
            .iter()
            .all(|m| m.final_digest == models[0].final_digest)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_cut_off_for_a_moment_while_uploading_carries_on() {
    let dir = tempfile::tempdir().unwrap();
    let fed = federation(dir.path(), &options(3, 2));
    let (addr, aggregator) = start_aggregator(aggregator_config(&fed, None)).await;
    let net = Net::over(fed, addr).await;
    let (sites, _helpers) = net.run();
    wait_in_log(&net.fed.aggregator_dir().join("log.jsonl"), "contribute").await;
    net.sites[0].cut();
    tokio::time::sleep(Duration::from_secs(2)).await;
    net.sites[0].heal();
    all_finish(&net, sites).await;
    aggregator.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_helper_cut_off_while_it_masks_carries_on() {
    let dir = tempfile::tempdir().unwrap();
    let fed = federation(dir.path(), &options(3, 1));
    let links: Arc<Mutex<Option<Link>>> = Arc::default();
    let cut = links.clone();
    let hook = at(Stage::BeforeSum { round: 1 }, move || {
        if let Some(link) = cut.lock().unwrap().as_ref() {
            link.cut();
        }
    });
    let (addr, aggregator) = start_aggregator(aggregator_config(&fed, Some(hook))).await;
    let net = Net::over(fed, addr).await;
    *links.lock().unwrap() = Some(net.helpers[0].clone());
    let (sites, _helpers) = net.run();
    wait_in_log(&net.fed.aggregator_dir().join("log.jsonl"), "close").await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    net.helpers[0].heal();
    all_finish(&net, sites).await;
    aggregator.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_whole_network_going_down_mid_round_costs_nothing_but_time() {
    let dir = tempfile::tempdir().unwrap();
    let fed = federation(dir.path(), &options(3, 2));
    let (addr, aggregator) = start_aggregator(aggregator_config(&fed, None)).await;
    let net = Net::over(fed, addr).await;
    let (sites, _helpers) = net.run();
    wait_in_log(&net.fed.aggregator_dir().join("log.jsonl"), "outcome").await;
    net.all().for_each(Link::cut);
    tokio::time::sleep(Duration::from_secs(3)).await;
    net.all().for_each(Link::heal);
    all_finish(&net, sites).await;
    aggregator.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_link_that_keeps_dropping_still_gets_the_run_through() {
    let dir = tempfile::tempdir().unwrap();
    let fed = federation(dir.path(), &options(3, 2));
    let (addr, aggregator) = start_aggregator(aggregator_config(&fed, None)).await;
    let net = Net::over(fed, addr).await;
    let (sites, _helpers) = net.run();
    let flapping = net.sites[1].clone();
    let flap = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(400)).await;
            flapping.cut();
            tokio::time::sleep(Duration::from_millis(100)).await;
            flapping.heal();
        }
    });
    all_finish(&net, sites).await;
    flap.abort();
    aggregator.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_cut_off_past_the_attest_deadline_stops_the_run_with_no_model() {
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options(3, 1);
    opts.timeouts.attest_secs = 2;
    let fed = federation(dir.path(), &opts);
    let links: Arc<Mutex<Option<Link>>> = Arc::default();
    let cut = links.clone();
    let hook = at(Stage::BeforeOutcome { round: 1 }, move || {
        if let Some(link) = cut.lock().unwrap().as_ref() {
            link.cut();
        }
    });
    let (addr, aggregator) = start_aggregator(aggregator_config(&fed, Some(hook))).await;
    let net = Net::over(fed, addr).await;
    *links.lock().unwrap() = Some(net.sites[2].clone());
    let (sites, _helpers) = net.run();
    let replay = aggregator.await.unwrap().unwrap();
    assert_eq!(replay.phase, Phase::Aborted);
    assert!(replay.rounds[0].seal.is_none());
    for site in sites.into_iter().take(2) {
        assert!(matches!(site.await.unwrap(), Err(NodeError::Aborted(_))));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_cut_off_for_a_whole_round_is_left_out_and_rejoins() {
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options(3, 4);
    opts.timeouts.collect_secs = 3;
    let fed = federation(dir.path(), &opts);
    let (addr, aggregator) = start_aggregator(aggregator_config(&fed, None)).await;
    let net = Net::over(fed, addr).await;
    let log = net.fed.aggregator_dir().join("log.jsonl");
    let go = Arc::new(AtomicBool::new(false));
    let gate = go.clone();
    let (sites, _helpers) = net.run_with(|site| match site {
        2 => Box::new(Held(gate.clone())),
        _ => Box::new(Toward(1.0 + site as f32)),
    });
    // It joins, and goes before it can contribute.
    wait_in_log(&log, "open").await;
    net.sites[2].cut();
    go.store(true, Ordering::SeqCst);
    wait_in_log(&log, "seal").await;
    net.sites[2].heal();
    let mut models = Vec::new();
    for site in sites {
        models.push(site.await.unwrap().unwrap());
    }
    let replay = aggregator.await.unwrap().unwrap();
    assert_eq!(replay.phase, Phase::Done);
    assert_eq!(replay.rounds[0].included, [0, 1]);
    // Reconnecting backs off, so it may miss another round before it's back.
    assert!(
        replay
            .rounds
            .iter()
            .skip(1)
            .any(|r| r.included.contains(&2))
    );
    assert!(
        models
            .iter()
            .all(|m| m.final_digest == models[0].final_digest)
    );
}

/// What a power cut can leave behind: a log whose last line was half written.
fn tear(log: &Path) {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().append(true).open(log).unwrap();
    // The sector with the newline can land without the one before it.
    file.write_all(b"{\"body\":{\"kind\":\"ack\",\0\0\0\n")
        .unwrap();
    file.write_all(br#"{"body":{"kind":"ack","pl"#).unwrap();
}

/// Held at a round's end, a site that has contributed comes back without the
/// model the round started from. It must fetch it again rather than fail.
#[tokio::test(flavor = "multi_thread")]
async fn a_site_past_contributing_fetches_a_lost_base_model_again() {
    let dir = tempfile::tempdir().unwrap();
    let fed = federation(dir.path(), &options(3, 2));
    let reached = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let (held, go) = (reached.clone(), release.clone());
    let hook: Hook = Arc::new(move |stage, _| {
        if stage == (Stage::BeforeOutcome { round: 2 }) {
            held.store(true, Ordering::SeqCst);
            while !go.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    });
    let (addr, aggregator) = start_aggregator(aggregator_config(&fed, Some(hook))).await;
    let net = Net::over(fed, addr).await;
    let (sites, _helpers) = net.run();
    while !reached.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    sites[0].abort();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let base = net.fed.site_dir(0).join("models/round-1.safetensors");
    assert!(base.exists(), "the site finished the first round");
    std::fs::write(&base, b"").unwrap();

    let trainer = Box::new(Toward(1.0));
    let back = tokio::spawn(run_site(
        net.fed.site_config(0, &net.sites[0].addr),
        trainer,
    ));
    release.store(true, Ordering::SeqCst);
    let summary = back.await.unwrap().unwrap();
    assert_eq!(summary.rounds_contributed, [1, 2]);
    let replay = aggregator.await.unwrap().unwrap();
    assert_eq!(replay.phase, Phase::Done);
    assert_eq!(summary.final_digest, replay.model);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_site_restarts_on_a_torn_log_and_a_lost_model() {
    let dir = tempfile::tempdir().unwrap();
    let fed = federation(dir.path(), &options(3, 2));
    let (addr, aggregator) = start_aggregator(aggregator_config(&fed, None)).await;
    let net = Net::over(fed, addr).await;
    let log = net.fed.aggregator_dir().join("log.jsonl");
    let (mut sites, _helpers) = net.run();
    wait_in_log(&log, "seal").await;
    sites[0].abort();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let own = net.fed.site_dir(0);
    tear(&own.join("log.jsonl"));
    // Fetched without a sync, so it can be missing or damaged.
    if let Ok(entries) = std::fs::read_dir(own.join("models")) {
        for entry in entries.flatten() {
            std::fs::write(entry.path(), b"damaged").unwrap();
        }
    }
    let back = Box::new(Toward(1.0));
    sites[0] = tokio::spawn(run_site(net.fed.site_config(0, &net.sites[0].addr), back));
    all_finish(&net, sites).await;
    aggregator.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_helper_restarts_on_a_torn_log() {
    let dir = tempfile::tempdir().unwrap();
    let fed = federation(dir.path(), &options(3, 2));
    let (addr, aggregator) = start_aggregator(aggregator_config(&fed, None)).await;
    let net = Net::over(fed, addr).await;
    let log = net.fed.aggregator_dir().join("log.jsonl");
    let (sites, mut helpers) = net.run();
    wait_in_log(&log, "seal").await;
    helpers[0].abort();
    tokio::time::sleep(Duration::from_millis(200)).await;
    tear(&net.fed.root.join("helper-0/log.jsonl"));
    helpers[0] = tokio::spawn(run_helper(net.fed.helper_config(0, &net.helpers[0].addr)));
    all_finish(&net, sites).await;
    aggregator.await.unwrap().unwrap();
}
