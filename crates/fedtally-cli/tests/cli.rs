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

#[test]
fn an_issuer_signs_credentials_and_revocations_from_the_command_line() {
    use fedtally_core::credential::Credential;
    use fedtally_core::status::StatusList;
    use fedtally_node::local::StandingFile;

    let dir = tempfile::tempdir().unwrap();
    let path = |name: &str| dir.path().join(name).to_str().unwrap().to_owned();
    let (dac_key, dac) = key(dir.path(), "dac.key", &[]);
    let (_, site) = key(dir.path(), "site.key", &[]);
    let (dac_did, site_did) = (dac["did"].as_str().unwrap(), site["did"].as_str().unwrap());
    let dataset = "did:example:cohort-7";

    let claims = json!({
        "ga4gh_visa_v1": {"type": "ControlledAccessGrants", "value": dataset, "by": "dac"},
        "data_use": {"permission": "DUO:0000006"},
    });
    std::fs::write(path("claims.json"), claims.to_string()).unwrap();
    let digest = ok(&[
        "issue",
        "--key",
        &dac_key,
        "--type",
        "ControlledAccessGrants",
        "--subject",
        site_did,
        "--claims",
        &path("claims.json"),
        "--list",
        "urn:dac:list",
        "--index",
        "7",
        "--days",
        "30",
        "--out",
        &path("grant.json"),
    ]);
    let credential =
        Credential::parse(&std::fs::read_to_string(path("grant.json")).unwrap()).unwrap();
    assert_eq!(credential.digest.to_string(), digest.trim());
    assert_eq!(
        (credential.issuer.as_str(), credential.subject.as_str()),
        (dac_did, site_did)
    );
    assert_eq!(credential.status.as_ref().unwrap().index, 7);
    assert!(credential.valid_at(now()) && !credential.valid_at(now() + 31 * 86_400));

    let shown: Value = serde_json::from_str(&ok(&["show", &path("grant.json")])).unwrap();
    assert_eq!(shown["claims"]["ga4gh_visa_v1"]["value"], dataset);
    assert_eq!(shown["valid_now"], true);
    assert_eq!(shown["status"]["list"], "urn:dac:list");

    ok(&[
        "standing",
        "--dataset",
        dataset,
        "--credential",
        &path("grant.json"),
        "--out",
        &path("standing.json"),
    ]);
    let standing = StandingFile::load(Path::new(&path("standing.json")), None).unwrap();
    assert_eq!(standing.dataset.as_deref(), Some(dataset));
    assert_eq!(standing.credentials, std::slice::from_ref(&credential.text));

    ok(&[
        "revoke",
        "--key",
        &dac_key,
        "--list",
        "urn:dac:list",
        "--index",
        "7",
        "--index",
        "9",
        "--out",
        &path("list.json"),
    ]);
    let list = StatusList::parse(&std::fs::read_to_string(path("list.json")).unwrap()).unwrap();
    assert_eq!(
        (list.id.as_str(), list.issuer.as_str()),
        ("urn:dac:list", dac_did)
    );
    assert!(list.is_set(7) && list.is_set(9) && !list.is_set(8));
    let shown: Value = serde_json::from_str(&ok(&["show", &path("list.json")])).unwrap();
    assert_eq!(shown["list"]["id"], "urn:dac:list");

    // A file someone edited no longer verifies.
    let edited = credential.text.replace("cohort-7", "cohort-8");
    std::fs::write(path("edited.json"), edited).unwrap();
    assert!(!fedtally(&["show", &path("edited.json")]).status.success());
    assert!(
        !fedtally(&[
            "standing",
            "--credential",
            &path("edited.json"),
            "--out",
            &path("s.json"),
        ])
        .status
        .success()
    );
    let (pq_key, _) = key(dir.path(), "pq.key", &["--alg", "ml-dsa-65"]);
    let issued = fedtally(&[
        "issue",
        "--key",
        &pq_key,
        "--type",
        "T",
        "--subject",
        site_did,
        "--out",
        &path("x.json"),
    ]);
    assert!(issued.status.success(), "{issued:?}");
    let text = std::fs::read_to_string(path("x.json")).unwrap();
    assert!(text.contains("mldsa65-jcs-2024"));
}

#[test]
fn a_site_can_train_with_a_command_of_its_own() {
    use fedtally_core::model::{Dtype, Layout, TensorSpec, TensorView, write_model};
    use fedtally_core::report::verify_log;
    use fedtally_node::local::{LocalFederation, Options};

    let dir = tempfile::tempdir().unwrap();
    let layout = Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![64],
        }],
    };
    let init = dir.path().join("start.safetensors");
    write_model(&init, &layout, &[TensorView::F32(&[0.5; 64])]).unwrap();
    let mut options = Options::new(2, 1, 2);
    options.chunk_coords = 32;
    let fed = LocalFederation::create(dir.path(), layout, &init, &options).unwrap();
    fed.save().unwrap();

    let path = |name: &str| dir.path().join(name).to_str().unwrap().to_owned();
    let addr = {
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        free.local_addr().unwrap().to_string()
    };
    let spawn = |args: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_fedtally"))
            .args(args)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap()
    };
    let mut parties = vec![
        spawn(&[
            "aggregator",
            "--key",
            &path("aggregator.key"),
            "--plan",
            &path("plan.jsonl"),
            "--model",
            init.to_str().unwrap(),
            "--dir",
            &path("aggregator"),
            "--listen",
            &addr,
        ]),
        spawn(&[
            "helper",
            "--key",
            &path("helper-0.key"),
            "--plan",
            &path("plan.jsonl"),
            "--dir",
            &path("helper-0"),
            "--connect",
            &addr,
        ]),
    ];
    for site in 0..2 {
        parties.push(spawn(&[
            "site",
            "--key",
            &path(&format!("site-{site}.key")),
            "--plan",
            &path("plan.jsonl"),
            "--grant",
            &path(&format!("site-{site}.grant.json")),
            "--connect",
            &addr,
            "--dir",
            &path(&format!("site-{site}")),
            "--",
            "sh",
            "-c",
            "cp \"$FEDTALLY_BASE\" \"$FEDTALLY_OUT\"; echo 3",
        ]));
    }
    for party in &mut parties {
        assert!(party.wait().unwrap().success());
    }
    let report = verify_log(Path::new(&path("aggregator/log.jsonl")), None).unwrap();
    assert!(report.complete(), "{report:?}");
    assert!(report.rounds.iter().all(|r| r.total_weight == Some(6)));
}
