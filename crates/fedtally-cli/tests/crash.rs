//! Kills the aggregator process with SIGKILL during a run and restarts it.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fedtally_core::model::{Dtype, Layout, TensorSpec, TensorView, write_model};
use fedtally_core::privacy::Request;
use fedtally_core::report::verify_log;
use fedtally_node::helper::run_helper;
use fedtally_node::local::{LocalFederation, Options};
use fedtally_node::site::{Submit, TrainContext, Trainer, run_site};

const COORDS: usize = 400_000;

fn layout() -> Layout {
    Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![COORDS as u64],
        }],
    }
}

struct Counting {
    value: f32,
    rounds: Arc<Mutex<Vec<u32>>>,
}

impl Trainer for Counting {
    fn train(&mut self, ctx: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        self.rounds.lock().unwrap().push(ctx.round);
        let w = vec![self.value; COORDS];
        submit(&[TensorView::F32(&w)], 1)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

fn free_port() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().to_string()
}

fn spawn_aggregator(root: &Path, init: &Path, addr: &str) -> Child {
    Command::new(env!("CARGO_BIN_EXE_fedtally"))
        .args(["aggregator", "--key"])
        .arg(root.join("aggregator.key"))
        .arg("--plan")
        .arg(root.join("plan.jsonl"))
        .arg("--model")
        .arg(init)
        .arg("--dir")
        .arg(root.join("aggregator"))
        .args(["--listen", addr])
        .stdout(Stdio::null())
        .stderr(File::create(root.join("aggregator.err")).unwrap())
        .spawn()
        .unwrap()
}

fn damage(path: &Path) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes[100] ^= 1;
    std::fs::write(path, bytes).unwrap();
}

fn log_kinds(root: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(root.join("aggregator/log.jsonl")) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).ok()?;
            Some(value["body"]["kind"].as_str()?.to_owned())
        })
        .collect()
}

/// Waits for `kind` to appear `count` times, then kills and restarts.
async fn kill_when(
    child: &mut Child,
    root: &Path,
    init: &Path,
    addr: &str,
    kind: &str,
    count: usize,
) -> Result<(), String> {
    let started = Instant::now();
    while log_kinds(root).iter().filter(|k| *k == kind).count() < count {
        if started.elapsed() > Duration::from_secs(60) {
            return Err(format!(
                "never saw {count} of {kind}: {:?}",
                log_kinds(root)
            ));
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    child.kill().map_err(|e| e.to_string())?;
    child.wait().map_err(|e| e.to_string())?;
    *child = spawn_aggregator(root, init, addr);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_killed_aggregator_resumes_and_nobody_trains_twice() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let init: PathBuf = root.join("init.safetensors");
    write_model(&init, &layout(), &[TensorView::F32(&vec![0.0; COORDS])]).unwrap();

    let mut options = Options::new(4, 1, 3);
    options.chunk_coords = 4096;
    let fed = LocalFederation::create(root, layout(), &init, &options).unwrap();
    fed.save().unwrap();

    let addr = free_port();
    let mut aggregator = spawn_aggregator(root, &init, &addr);

    let helper = tokio::spawn(run_helper(fed.helper_config(0, &addr)));
    let counters: Vec<Arc<Mutex<Vec<u32>>>> = (0..4).map(|_| Arc::default()).collect();
    let sites: Vec<_> = (0..4)
        .map(|i| {
            let trainer = Counting {
                value: i as f32,
                rounds: counters[i].clone(),
            };
            tokio::spawn(run_site(fed.site_config(i, &addr), Box::new(trainer)))
        })
        .collect();

    // During upload, after the sum is fixed, once the challenge exists and
    // while a later round is collecting.
    for (kind, count) in [
        ("contribute", 1),
        ("close", 1),
        ("outcome", 1),
        ("attest", 1),
        ("contribute", 6),
        ("seal", 2),
        ("open", 3),
    ] {
        kill_when(&mut aggregator, root, &init, &addr, kind, count)
            .await
            .unwrap();
    }

    let mut summaries = Vec::new();
    for site in sites {
        summaries.push(site.await.unwrap().unwrap());
    }
    helper.await.unwrap().unwrap();
    let status = tokio::task::spawn_blocking(move || aggregator.wait().unwrap())
        .await
        .unwrap();
    assert!(status.success());

    for counter in &counters {
        assert_eq!(
            *counter.lock().unwrap(),
            vec![1, 2, 3],
            "a site trained more than once"
        );
    }
    let report = verify_log(
        &root.join("aggregator/log.jsonl"),
        Some(&summaries[0].final_model),
    )
    .unwrap();
    assert!(report.finished);
    assert_eq!(report.rounds.len(), 3);
    assert!(report.complete(), "{report:?}");

    // Each round moves every coordinate to the mean of the four values, 1.5.
    let bytes = std::fs::read(&summaries[0].final_model).unwrap();
    let header = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let first = f32::from_le_bytes(bytes[8 + header..8 + header + 4].try_into().unwrap());
    assert!((first - 1.5).abs() < 1e-5, "{first}");
}

struct Small;

impl Trainer for Small {
    fn train(&mut self, _: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        submit(&[TensorView::F32(&[0.5; 256])], 1)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

/// A federation on a small model, with its aggregator running and its sites
/// started, but no helper yet, so a round waits after it closes.
async fn waiting_for_a_mask(
    root: &Path,
    helpers: usize,
) -> (
    LocalFederation,
    PathBuf,
    String,
    Child,
    Vec<
        tokio::task::JoinHandle<Result<fedtally_node::site::SiteSummary, fedtally_node::NodeError>>,
    >,
) {
    let init: PathBuf = root.join("init.safetensors");
    let small = Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![256],
        }],
    };
    write_model(&init, &small, &[TensorView::F32(&[0.0; 256])]).unwrap();
    let mut options = Options::new(3, helpers, 1);
    options.chunk_coords = 64;
    let fed = LocalFederation::create(root, small, &init, &options).unwrap();
    fed.save().unwrap();
    let addr = free_port();
    let aggregator = spawn_aggregator(root, &init, &addr);
    let sites = (0..3)
        .map(|i| tokio::spawn(run_site(fed.site_config(i, &addr), Box::new(Small))))
        .collect();
    (fed, init, addr, aggregator, sites)
}

async fn until_in_log(root: &Path, kind: &str) {
    let started = Instant::now();
    while !log_kinds(root).iter().any(|k| k == kind) {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "no {kind} in the log"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// What a power cut can leave: a half written last line, and a file that was
/// never synced because it can be rebuilt.
#[tokio::test(flavor = "multi_thread")]
async fn an_aggregator_restarts_on_a_torn_log_and_a_lost_sum() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let (fed, init, addr, mut aggregator, sites) = waiting_for_a_mask(root, 1).await;
    until_in_log(root, "close").await;
    aggregator.kill().unwrap();
    aggregator.wait().unwrap();

    let log = root.join("aggregator/log.jsonl");
    let mut file = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
    file.write_all(b"{\"body\":{\"kind\":\"ack\",\0\0\0\n")
        .unwrap();
    std::fs::remove_file(root.join("aggregator/round-1/sum.bin")).unwrap();

    let mut aggregator = spawn_aggregator(root, &init, &addr);
    let helper = tokio::spawn(run_helper(fed.helper_config(0, &addr)));
    let mut summaries = Vec::new();
    for site in sites {
        summaries.push(site.await.unwrap().unwrap());
    }
    helper.await.unwrap().unwrap();
    let status = tokio::task::spawn_blocking(move || aggregator.wait().unwrap())
        .await
        .unwrap();
    assert!(status.success());
    let report = verify_log(&log, Some(&summaries[0].final_model)).unwrap();
    assert!(report.complete(), "{report:?}");
}

/// The run ended in an Abort naming the file, with no Outcome before it.
fn stopped_on(root: &Path, why: &str) {
    let kinds = log_kinds(root);
    assert!(!kinds.iter().any(|k| k == "outcome"));
    assert_eq!(kinds.last().map(String::as_str), Some("abort"));
    let log = std::fs::read_to_string(root.join("aggregator/log.jsonl")).unwrap();
    assert!(log.lines().last().unwrap().contains(why), "{log}");
}

/// A bit gone wrong in an upload and in the sum made from it must not reach a model.
#[tokio::test(flavor = "multi_thread")]
async fn a_damaged_upload_stops_the_aggregator_rather_than_reaching_a_model() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let (fed, init, addr, mut aggregator, sites) = waiting_for_a_mask(root, 1).await;
    until_in_log(root, "close").await;
    aggregator.kill().unwrap();
    aggregator.wait().unwrap();

    for name in ["sum.bin", "site-0.bin"] {
        damage(&root.join("aggregator/round-1").join(name));
    }

    let mut aggregator = spawn_aggregator(root, &init, &addr);
    let _helper = tokio::spawn(run_helper(fed.helper_config(0, &addr)));
    let status = tokio::task::spawn_blocking(move || aggregator.wait().unwrap())
        .await
        .unwrap();
    assert!(status.success(), "it ends the run in the log");
    stopped_on(root, "sum.bin no longer matches");
    for site in sites {
        let ended = site.await.unwrap().err().unwrap().to_string();
        assert!(ended.contains("no longer matches"), "{ended}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_damaged_mask_stops_the_aggregator_rather_than_reaching_a_model() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let (fed, init, addr, mut aggregator, sites) = waiting_for_a_mask(root, 2).await;
    let first = tokio::spawn(run_helper(fed.helper_config(0, &addr)));
    until_in_log(root, "mask_sum").await;
    aggregator.kill().unwrap();
    aggregator.wait().unwrap();
    damage(&root.join("aggregator/round-1/mask-0.bin"));

    let mut aggregator = spawn_aggregator(root, &init, &addr);
    let second = tokio::spawn(run_helper(fed.helper_config(1, &addr)));
    let status = tokio::task::spawn_blocking(move || aggregator.wait().unwrap())
        .await
        .unwrap();
    assert!(status.success(), "it ends the run in the log");
    stopped_on(root, "mask-0.bin no longer matches");
    for party in [first, second] {
        assert!(party.await.unwrap().is_err());
    }
    for site in sites {
        let ended = site.await.unwrap().err().unwrap().to_string();
        assert!(ended.contains("no longer matches"), "{ended}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_redo_survives_every_party_being_killed_around_it() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let init: PathBuf = root.join("init.safetensors");
    let small = Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![256],
        }],
    };
    write_model(&init, &small, &[TensorView::F32(&[0.0; 256])]).unwrap();

    let mut options = Options::new(3, 3, 1);
    options.chunk_coords = 64;
    options.min_cohort = 2;
    options.min_helpers = 2;
    options.timeouts.collect_secs = 4;
    options.timeouts.attest_secs = 4;
    let fed = LocalFederation::create(root, small, &init, &options).unwrap();
    fed.save().unwrap();

    let addr = free_port();
    let mut aggregator = spawn_aggregator(root, &init, &addr);
    let mut helpers: Vec<_> = (0..3)
        .map(|h| tokio::spawn(run_helper(fed.helper_config(h, &addr))))
        .collect();
    let mut sites: Vec<_> = (0..3)
        .map(|i| tokio::spawn(run_site(fed.site_config(i, &addr), Box::new(Small))))
        .collect();

    // Helper 2 goes while the cohort is being fixed, so it never gives a mask.
    let started = Instant::now();
    while !log_kinds(root).iter().any(|k| k == "contribute") {
        assert!(started.elapsed() < Duration::from_secs(60));
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    helpers[2].abort();

    kill_when(&mut aggregator, root, &init, &addr, "remove_helper", 1)
        .await
        .unwrap();
    // A site dies while it may be masking again, and so does a helper.
    sites[0].abort();
    helpers[0].abort();
    tokio::time::sleep(Duration::from_millis(50)).await;
    sites[0] = tokio::spawn(run_site(fed.site_config(0, &addr), Box::new(Small)));
    helpers[0] = tokio::spawn(run_helper(fed.helper_config(0, &addr)));
    // The removed helper comes back too, and has no part left to play.
    helpers[2] = tokio::spawn(run_helper(fed.helper_config(2, &addr)));
    kill_when(&mut aggregator, root, &init, &addr, "outcome", 1)
        .await
        .unwrap();

    let mut summaries = Vec::new();
    for site in sites {
        summaries.push(site.await.unwrap().unwrap());
    }
    for helper in helpers {
        helper.await.unwrap().unwrap();
    }
    let status = tokio::task::spawn_blocking(move || aggregator.wait().unwrap())
        .await
        .unwrap();
    assert!(status.success());

    let report = verify_log(
        &root.join("aggregator/log.jsonl"),
        Some(&summaries[0].final_model),
    )
    .unwrap();
    assert!(report.finished, "{report:?}");
    assert_eq!(report.rounds.len(), 2);
    assert!(report.rounds[0].voided);
    assert_eq!(report.removed_helpers.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_retry_survives_the_aggregator_and_a_site_being_killed_around_it() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let init: PathBuf = root.join("init.safetensors");
    let small = Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![256],
        }],
    };
    write_model(&init, &small, &[TensorView::F32(&[0.0; 256])]).unwrap();

    let mut options = Options::new(4, 1, 1);
    options.chunk_coords = 64;
    options.min_cohort = 2;
    options.retries = 1;
    options.timeouts.attest_secs = 8;
    options.privacy = Some(Request {
        epsilon: 8.0,
        delta: 1e-5,
        clip: 1.0,
        colluding: 0,
    });
    let fed = LocalFederation::create(root, small, &init, &options).unwrap();
    fed.save().unwrap();

    let addr = free_port();
    let mut aggregator = spawn_aggregator(root, &init, &addr);
    let helper = tokio::spawn(run_helper(fed.helper_config(0, &addr)));
    let mut sites: Vec<_> = (0..4)
        .map(|i| tokio::spawn(run_site(fed.site_config(i, &addr), Box::new(Small))))
        .collect();

    // Site 3 is gone before it can attest, so the round is given up.
    let started = Instant::now();
    while !log_kinds(root).iter().any(|k| k == "close") {
        assert!(started.elapsed() < Duration::from_secs(60));
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    sites[3].abort();

    kill_when(&mut aggregator, root, &init, &addr, "retry", 1)
        .await
        .unwrap();
    // A site dies while it may be retraining, and comes back.
    sites[0].abort();
    tokio::time::sleep(Duration::from_millis(50)).await;
    sites[0] = tokio::spawn(run_site(fed.site_config(0, &addr), Box::new(Small)));
    kill_when(&mut aggregator, root, &init, &addr, "outcome", 2)
        .await
        .unwrap();

    let lost = sites.pop().unwrap();
    assert!(lost.await.unwrap_err().is_cancelled());
    let mut summaries = Vec::new();
    for site in sites {
        summaries.push(site.await.unwrap().unwrap());
    }
    helper.await.unwrap().unwrap();
    let status = tokio::task::spawn_blocking(move || aggregator.wait().unwrap())
        .await
        .unwrap();
    assert!(status.success());

    let report = verify_log(
        &root.join("aggregator/log.jsonl"),
        Some(&summaries[0].final_model),
    )
    .unwrap();
    assert!(report.finished);
    assert_eq!(report.rounds.len(), 2);
    assert!(report.rounds[0].voided);
    assert_eq!(report.rounds[1].included.len(), 3);
}

#[test]
fn verify_exit_codes_tell_failure_from_gaps() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let init = root.join("init.safetensors");
    write_model(&init, &layout(), &[TensorView::F32(&vec![0.0; COORDS])]).unwrap();
    let fed = LocalFederation::create(root, layout(), &init, &Options::new(2, 0, 1)).unwrap();
    let log = root.join("log.jsonl");
    std::fs::write(&log, [fed.plan_line.as_slice(), b"\n"].concat()).unwrap();
    let verify = |path: &Path| {
        Command::new(env!("CARGO_BIN_EXE_fedtally"))
            .arg("verify")
            .arg(path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .code()
    };
    assert_eq!(verify(&log), Some(2));
    let mut text = std::fs::read(&log).unwrap();
    text[20] ^= 1;
    std::fs::write(&log, text).unwrap();
    assert_eq!(verify(&log), Some(1));
}
