//! The commands a real deployment uses to set up a run, one party at a time.

use std::path::Path;
use std::process::{Command, Output};

use fedtally_core::admission;
use fedtally_core::model::{Dtype, Layout, TensorSpec, TensorView, write_model};
use fedtally_core::pq_ucan::did::Did;
use fedtally_core::pq_ucan::time::Timestamp;
use fedtally_core::record::{Body, Record};
use fedtally_node::local::GrantFile;
use serde_json::{Value, json};

fn fedtally(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fedtally"))
        .args(args)
        .output()
        .unwrap()
}

fn ok(args: &[&str]) -> String {
    let output = fedtally(args);
    assert!(
        output.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn key(dir: &Path, name: &str, extra: &[&str]) -> (String, Value) {
    let path = dir.join(name).to_str().unwrap().to_owned();
    let mut args = vec!["keygen", "--out", &path];
    args.extend_from_slice(extra);
    let details: Value = serde_json::from_str(&ok(&args)).unwrap();
    (path, details)
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[test]
fn each_party_can_set_up_a_run_with_its_own_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = |name: &str| dir.path().join(name).to_str().unwrap().to_owned();

    let (sponsor_key, sponsor) = key(dir.path(), "sponsor.key", &["--alg", "ml-dsa-65"]);
    let (_, aggregator) = key(dir.path(), "aggregator.key", &["--tls"]);
    let (_, helper) = key(dir.path(), "helper.key", &[]);
    let (site_key, site) = key(dir.path(), "site-0.key", &[]);
    let (_, other_site) = key(dir.path(), "site-1.key", &[]);
    let (did, helper_did) = (
        site["did"].as_str().unwrap().to_owned(),
        helper["did"].as_str().unwrap().to_owned(),
    );

    let layout = Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![4],
        }],
    };
    write_model(
        dir.path().join("init").as_path(),
        &layout,
        &[TensorView::F32(&[0.0; 4])],
    )
    .unwrap();

    let spec = json!({
        "purpose": "cardiac risk",
        "aggregator": {"did": aggregator["did"], "tls_pin": aggregator["tls_pin"]},
        "helpers": [{"did": helper["did"], "seal": helper["seal"], "seal_proof": helper["seal_proof"]}],
        "sites": [site["did"], other_site["did"]],
        "layout": layout,
        "rounds": 2,
        "min_cohort": 2,
        "chunk_coords": 2,
        "encoding": {"frac_bits": 24, "clip": 128.0},
        "max_weight": 1000,
        "timeouts": {"join_secs": 60, "collect_secs": 60, "attest_secs": 60, "ack_secs": 60},
    });
    std::fs::write(path("spec.json"), spec.to_string()).unwrap();
    ok(&[
        "plan",
        "--key",
        &sponsor_key,
        "--spec",
        &path("spec.json"),
        "--model",
        &path("init"),
        "--out",
        &path("plan.jsonl"),
    ]);

    ok(&[
        "grant",
        "--key",
        &site_key,
        "--to",
        sponsor["did"].as_str().unwrap(),
        "--purpose",
        "cardiac risk",
        "--trust-helper",
        &helper_did,
        "--min-cohort",
        "2",
        "--max-rounds",
        "5",
        "--out",
        &path("grant.ucan"),
    ]);
    ok(&[
        "invoke",
        "--key",
        &sponsor_key,
        "--plan",
        &path("plan.jsonl"),
        "--site",
        &did,
        "--grant",
        &path("grant.ucan"),
        "--out",
        &path("site-0.grant.json"),
    ]);

    let record = Record::parse(&std::fs::read(path("plan.jsonl")).unwrap()).unwrap();
    let Body::Plan(plan) = &record.body else {
        panic!("not a plan");
    };
    let grant = GrantFile::load(Path::new(&path("site-0.grant.json"))).unwrap();
    admission::check(
        &Did::parse(&did).unwrap(),
        &record.body_hash,
        plan,
        &grant.invocation,
        &grant.proofs,
        Timestamp::from_unix(now()).unwrap(),
        None,
    )
    .unwrap();
}

#[test]
fn a_site_has_to_say_what_it_will_accept() {
    let dir = tempfile::tempdir().unwrap();
    let (site_key, _) = key(dir.path(), "site.key", &[]);
    let (_, sponsor) = key(dir.path(), "sponsor.key", &[]);
    let out = dir.path().join("grant.ucan");
    let out = out.to_str().unwrap();
    let base = [
        "grant",
        "--key",
        &site_key,
        "--to",
        sponsor["did"].as_str().unwrap(),
        "--out",
        out,
    ];
    let with = |extra: &[&str]| {
        let mut args = base.to_vec();
        args.extend_from_slice(extra);
        fedtally(&args).status.success()
    };
    assert!(!with(&[]));
    assert!(!with(&["--min-cohort", "3"]));
    assert!(!with(&["--any-helper"]));
    assert!(with(&["--any-helper", "--min-cohort", "3"]));
    assert!(with(&["--allow-plain"]));
}
