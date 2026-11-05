//! Kills the aggregator process with SIGKILL during a run and restarts it.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fedtally_core::model::{Dtype, Layout, TensorSpec, TensorView, write_model};
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
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
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
