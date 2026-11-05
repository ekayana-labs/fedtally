//! Every rule in the replay, tried against a record that breaks it.

use std::collections::BTreeMap;
use std::sync::Arc;

use fedtally_core::admission::{self, AdmissionError, Terms};
use fedtally_core::credential::{Credential, Draft, StatusEntry};
use fedtally_core::field::{Fp, Fp2};
use fedtally_core::keys::{Identity, SigAlg};
use fedtally_core::log::{GENESIS, chain};
use fedtally_core::model::{Dtype, Encoding, Layout, TensorSpec};
use fedtally_core::plan::{AggregatorSpec, HelperSpec, Plan, PlanError, Timeouts};
use fedtally_core::policy::{About, Atom, Policy, Rule};
use fedtally_core::pq_ucan::Ipld;
use fedtally_core::pq_ucan::command::Command;
use fedtally_core::pq_ucan::crypto::Signer;
use fedtally_core::pq_ucan::delegation::{Delegation, Subject};
use fedtally_core::pq_ucan::invocation::Invocation;
use fedtally_core::pq_ucan::nonce::Nonce;
use fedtally_core::pq_ucan::time::Timestamp;
use fedtally_core::privacy::{Privacy, rho_units};
use fedtally_core::record::{Body, Included, Record, encode_token};
use fedtally_core::replay::{Fault, Phase, Replay};
use fedtally_core::report::{VerifyError, verify_log};
use fedtally_core::sampling;
use fedtally_core::status::StatusList;
use fedtally_core::vector::Digest;

const NOW: i64 = 1_800_000_000;

type Id = Arc<Identity>;

fn fp(a: u64, b: u64) -> Fp2 {
    Fp2 {
        a: Fp::new(a),
        b: Fp::new(b),
    }
}

fn identity() -> Id {
    Arc::new(Identity::generate(SigAlg::Ed25519).unwrap())
}

fn timestamp(at: i64) -> Timestamp {
    Timestamp::from_unix(at).unwrap()
}

struct Run {
    sponsor: Id,
    aggregator: Id,
    helper: Id,
    sites: Vec<Id>,
    plan: Plan,
    plan_id: Digest,
    replay: Replay,
    lines: Vec<Vec<u8>>,
}

/// Fingerprints that agree with each other, as an honest round's would.
struct Numbers {
    masked: [Fp2; 3],
    shares: [Fp2; 3],
    sum: Fp2,
    aggregate: Fp2,
}

fn numbers() -> Numbers {
    let masked = [fp(10, 1), fp(20, 2), fp(30, 3)];
    let shares = [fp(1, 1), fp(2, 0), fp(3, 1)];
    let sum: Fp2 = masked.iter().copied().sum();
    let total: Fp2 = shares.iter().copied().sum();
    Numbers {
        masked,
        shares,
        sum,
        aggregate: sum - total,
    }
}

impl Run {
    fn new() -> Run {
        Run::with(|_| {})
    }

    fn with(edit: impl FnOnce(&mut Plan)) -> Run {
        Run::of(identity, edit)
    }

    fn of(identity: fn() -> Id, edit: impl FnOnce(&mut Plan)) -> Run {
        let (sponsor, aggregator, helper) = (identity(), identity(), identity());
        let sites: Vec<Id> = (0..3).map(|_| identity()).collect();
        let mut plan = Plan {
            version: 1,
            nonce: "n".into(),
            purpose: "cardiology".into(),
            sponsor: sponsor.did_string(),
            aggregator: AggregatorSpec {
                did: aggregator.did_string(),
                tls_pin: Digest::of(b"cert"),
                credentials: Vec::new(),
            },
            helpers: vec![HelperSpec {
                did: helper.did_string(),
                seal: helper.seal_public().clone(),
                seal_proof: helper.seal_proof().unwrap(),
                credentials: Vec::new(),
            }],
            sites: sites.iter().map(|s| s.did_string()).collect(),
            layout: Layout {
                tensors: vec![TensorSpec {
                    name: "w".into(),
                    dtype: Dtype::F32,
                    shape: vec![8],
                }],
            },
            init_model: Digest::of(b"model 0"),
            rounds: 2,
            min_cohort: 2,
            chunk_coords: 4,
            encoding: Encoding {
                frac_bits: 24,
                clip: 128.0,
            },
            max_weight: 1000,
            audits: 1,
            timeouts: Timeouts {
                join_secs: 1,
                collect_secs: 1,
                attest_secs: 1,
                ack_secs: 1,
            },
            privacy: None,
            admission: None,
            sample: None,
        };
        edit(&mut plan);
        let record = Record::sign(&sponsor, &Body::Plan(Box::new(plan.clone()))).unwrap();
        let mut replay = Replay::new();
        replay.apply(&record).unwrap();
        Run {
            plan_id: record.body_hash,
            sponsor,
            aggregator,
            helper,
            sites,
            plan,
            replay,
            lines: vec![record.line],
        }
    }

    fn apply(&mut self, who: &Id, body: Body) -> Digest {
        let record = Record::sign(who, &body).unwrap();
        self.replay.apply(&record).unwrap();
        self.lines.push(record.line);
        record.body_hash
    }

    /// The record must be refused and must change nothing.
    fn refuse(&mut self, who: &Id, body: Body) -> Fault {
        let (head, entries) = (self.replay.head, self.replay.entries);
        let record = Record::sign(who, &body).unwrap();
        let fault = self.replay.apply(&record).unwrap_err();
        assert_eq!(self.replay.head, head, "a refused record moved the head");
        assert_eq!(self.replay.entries, entries);
        fault
    }

    fn site(&self, i: usize) -> Id {
        self.sites[i].clone()
    }

    fn terms(&self) -> Terms {
        Terms {
            purpose: Some("cardiology".into()),
            trusted_helpers: vec![self.helper.did_string()],
            min_cohort: Some(2),
            max_rounds: Some(5),
            ..Terms::default()
        }
    }

    fn join_body_from(&self, invocation: &Invocation, chain: &[&Delegation]) -> Body {
        Body::Join {
            plan: self.plan_id,
            at: NOW,
            invocation: encode_token(invocation.bytes()),
            proofs: chain.iter().map(|d| encode_token(d.bytes())).collect(),
            dataset: None,
            credentials: Vec::new(),
        }
    }

    fn join_body(&self, site: usize, terms: &Terms) -> Body {
        let id = &self.sites[site];
        let expires = timestamp(NOW + 3600);
        let grant = admission::grant(id.as_ref(), &self.sponsor.did(), terms, expires).unwrap();
        let invocation = admission::invoke(
            self.sponsor.as_ref(),
            &self.plan_id,
            &self.plan,
            &id.did(),
            &[&grant],
            expires,
        )
        .unwrap();
        self.join_body_from(&invocation, &[&grant])
    }

    fn join(&mut self, sites: std::ops::Range<usize>) {
        let terms = self.terms();
        for i in sites {
            let body = self.join_body(i, &terms);
            let site = self.site(i);
            self.apply(&site, body);
        }
    }

    fn open_body(&self, round: u32) -> Body {
        Body::Open {
            plan: self.plan_id,
            round,
            prev: self.replay.head,
            base: self.replay.model,
        }
    }

    fn open(&mut self, round: u32) {
        let body = self.open_body(round);
        let aggregator = self.aggregator.clone();
        self.apply(&aggregator, body);
    }

    fn contribute_body(&self, round: u32, weight: u64) -> Body {
        Body::Contribute {
            plan: self.plan_id,
            round,
            base: self.replay.model,
            masked: Digest::of(b"masked"),
            weight,
            sealed: vec![Digest::of(b"sealed")],
        }
    }

    fn contribute(&mut self, site: usize, round: u32, weight: u64) -> (usize, Digest) {
        let mut body = self.contribute_body(round, weight);
        if let Body::Contribute { masked, .. } = &mut body {
            *masked = Digest::of(format!("masked by {site}").as_bytes());
        }
        let id = self.site(site);
        (site, self.apply(&id, body))
    }

    fn included(&self, list: &[(usize, Digest)]) -> Vec<Included> {
        list.iter()
            .map(|(site, hash)| Included {
                site: self.sites[*site].did_string(),
                contribution: *hash,
            })
            .collect()
    }

    fn close_body(&self, round: u32, list: &[(usize, Digest)]) -> Body {
        Body::Close {
            plan: self.plan_id,
            round,
            prev: self.replay.head,
            included: self.included(list),
            sum: Digest::of(b"sum"),
        }
    }

    fn close(&mut self, round: u32, list: &[(usize, Digest)]) {
        let body = self.close_body(round, list);
        let aggregator = self.aggregator.clone();
        self.apply(&aggregator, body);
    }

    fn open_and_close_all(&mut self, round: u32) -> Vec<(usize, Digest)> {
        self.open_and_close_all_at(round, 10)
    }

    fn open_and_close_all_at(&mut self, round: u32, weight: u64) -> Vec<(usize, Digest)> {
        self.open(round);
        let list: Vec<_> = (0..3).map(|s| self.contribute(s, round, weight)).collect();
        self.close(round, &list);
        list
    }

    fn mask_sum_body(&self, round: u32) -> Body {
        Body::MaskSum {
            plan: self.plan_id,
            round,
            close: self.replay.current().unwrap().close_head,
            mask: Digest::of(b"mask"),
        }
    }

    fn mask_sum(&mut self, round: u32) {
        let body = self.mask_sum_body(round);
        let helper = self.helper.clone();
        self.apply(&helper, body);
    }

    fn outcome_body(&self, round: u32, total_weight: u64) -> Body {
        Body::Outcome {
            plan: self.plan_id,
            round,
            prev: self.replay.head,
            total_weight,
            aggregate: Digest::of(b"aggregate"),
            model: Digest::of(format!("model {round}").as_bytes()),
        }
    }

    fn outcome(&mut self, round: u32, total_weight: u64) {
        let body = self.outcome_body(round, total_weight);
        let aggregator = self.aggregator.clone();
        self.apply(&aggregator, body);
    }

    fn reach_attesting(&mut self, round: u32) {
        self.open_and_close_all(round);
        self.mask_sum(round);
        self.outcome(round, 30);
    }

    fn challenge(&self) -> Digest {
        self.replay
            .current()
            .unwrap()
            .outcome
            .as_ref()
            .unwrap()
            .challenge
    }

    fn helper_attest_body(&self, round: u32, shares: &[Fp2]) -> Body {
        Body::HelperAttest {
            plan: self.plan_id,
            round,
            challenge: self.challenge(),
            total: shares.iter().copied().sum(),
            per_site: shares.to_vec(),
        }
    }

    fn helper_attest(&mut self, round: u32, shares: &[Fp2]) {
        let body = self.helper_attest_body(round, shares);
        let helper = self.helper.clone();
        self.apply(&helper, body);
    }

    fn attest_body(&self, round: u32, masked: Fp2) -> Body {
        Body::Attest {
            plan: self.plan_id,
            round,
            challenge: self.challenge(),
            masked,
        }
    }

    fn seal_body(&self, round: u32, per_site: &[Fp2], sum: Fp2, aggregate: Fp2) -> Body {
        Body::Seal {
            plan: self.plan_id,
            round,
            prev: self.replay.head,
            per_site: per_site.to_vec(),
            sum,
            aggregate,
        }
    }

    fn seal(&mut self, round: u32) {
        let n = numbers();
        self.helper_attest(round, &n.shares);
        for i in 0..3 {
            let body = self.attest_body(round, n.masked[i]);
            let site = self.site(i);
            self.apply(&site, body);
        }
        let body = self.seal_body(round, &n.masked, n.sum, n.aggregate);
        let aggregator = self.aggregator.clone();
        self.apply(&aggregator, body);
    }

    fn ack_body(&self, round: u32) -> Body {
        Body::Ack {
            plan: self.plan_id,
            round,
            head: self.replay.current().unwrap().seal.as_ref().unwrap().head,
        }
    }

    fn finish_body(&self) -> Body {
        Body::Finish {
            plan: self.plan_id,
            prev: self.replay.head,
        }
    }

    fn complete_round(&mut self, round: u32) {
        self.complete_round_at(round, 10);
    }

    fn complete_round_at(&mut self, round: u32, weight: u64) {
        self.open_and_close_all_at(round, weight);
        self.mask_sum(round);
        self.outcome(round, 3 * weight);
        self.seal(round);
        for i in 0..3 {
            let body = self.ack_body(round);
            let site = self.site(i);
            self.apply(&site, body);
        }
    }

    fn complete_run() -> Run {
        let mut run = Run::new();
        run.join(0..3);
        run.complete_round(1);
        run.complete_round(2);
        let body = run.finish_body();
        let aggregator = run.aggregator.clone();
        run.apply(&aggregator, body);
        run
    }
}

/// Noise multiplier one over three sites, so a full round costs rho = 1/2.
fn private(budget: u64) -> impl FnOnce(&mut Plan) {
    move |plan| {
        plan.max_weight = 1;
        plan.privacy = Some(Privacy {
            clip: 1 << 18,
            noise: (1 << 36) / 3,
            colluding: 0,
            budget,
            epsilon: 8.0,
            delta: 1e-5,
        });
    }
}

fn replay_lines(lines: &[Vec<u8>]) -> Result<Replay, Fault> {
    Replay::from_lines(lines.iter().map(Vec::as_slice))
}

#[test]
fn an_honest_run_reads_as_complete() {
    let run = Run::complete_run();
    assert_eq!(run.replay.phase, Phase::Done);
    let report = run.replay.report().unwrap();
    assert!(report.complete(), "{report:?}");
    assert_eq!(report.model, Digest::of(b"model 2"));
    assert_eq!(report.rounds_with(&run.sites[1].did_string()), vec![1, 2]);
    assert_eq!(report.joined[0].admission.chain_len, 1);
}

#[test]
fn a_log_file_verifies_against_its_model_and_no_other() {
    let run = Run::complete_run();
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log.jsonl");
    let framed: Vec<u8> = run
        .lines
        .iter()
        .flat_map(|l| [l.as_slice(), b"\n"].concat())
        .collect();
    std::fs::write(&log, framed).unwrap();
    let (right, wrong) = (dir.path().join("right"), dir.path().join("wrong"));
    std::fs::write(&right, b"model 2").unwrap();
    std::fs::write(&wrong, b"model 1").unwrap();

    assert!(verify_log(&log, None).unwrap().complete());
    assert!(verify_log(&log, Some(&right)).unwrap().complete());
    assert!(matches!(
        verify_log(&log, Some(&wrong)),
        Err(VerifyError::Model { .. })
    ));

    std::fs::write(&log, b"").unwrap();
    assert!(matches!(verify_log(&log, None), Err(VerifyError::Empty)));
    std::fs::write(&log, [run.lines[1].as_slice(), b"\n"].concat()).unwrap();
    assert!(matches!(
        verify_log(&log, None),
        Err(VerifyError::Fault { entry: 0, .. })
    ));
}

#[test]
fn each_gap_on_its_own_makes_a_report_incomplete() {
    let whole = || Run::complete_run().replay.report().unwrap();
    let site = "did:key:someone".to_owned();

    assert!(whole().complete());
    let mut report = whole();
    report.finished = false;
    assert!(!report.complete());
    let mut report = whole();
    report.rounds[1].sealed = false;
    assert!(!report.complete());
    let mut report = whole();
    report.rounds[1].excluded.push(site.clone());
    assert!(!report.complete());
    let mut report = whole();
    report.rounds[1].absent.push(site.clone());
    assert!(!report.complete());
    let mut report = whole();
    report.rounds[1].unattested.push(site.clone());
    assert!(!report.complete());
    let mut report = whole();
    report.rounds[1].unacked.push(site);
    assert!(!report.complete());
}

#[test]
fn the_report_lists_only_the_rounds_a_site_was_in() {
    let mut run = Run::new();
    run.join(0..3);
    run.complete_round(1);
    run.open(2);
    let a = run.contribute(0, 2, 10);
    let b = run.contribute(1, 2, 10);
    run.close(2, &[a, b]);
    let report = run.replay.report().unwrap();
    assert_eq!(report.rounds_with(&run.sites[0].did_string()), vec![1, 2]);
    assert_eq!(report.rounds_with(&run.sites[2].did_string()), vec![1]);
    assert!(report.rounds_with("did:key:nobody").is_empty());
}

#[test]
fn a_round_remembers_what_it_started_from() {
    let mut run = Run::new();
    run.join(0..3);
    run.complete_round(1);
    run.open(2);
    assert_eq!(run.replay.current().unwrap().base, Digest::of(b"model 1"));
}

#[test]
fn refusals_name_the_kind_of_record() {
    let mut run = Run::new();
    let (aggregator, body) = (run.aggregator.clone(), run.close_body(1, &[]));
    let fault = run.refuse(&aggregator, body);
    assert!(fault.to_string().starts_with("close "), "{fault}");
}

#[test]
fn replaying_the_lines_gives_the_same_state() {
    let run = Run::complete_run();
    let replayed = replay_lines(&run.lines).unwrap();
    assert_eq!(replayed.head, run.replay.head);
    assert_eq!(replayed.entries, run.lines.len() as u64);
    let expected = run
        .lines
        .iter()
        .fold(GENESIS, |head, line| chain(&head, line));
    assert_eq!(expected, run.replay.head);
}

#[test]
fn removing_any_line_is_caught() {
    let run = Run::complete_run();
    for i in 0..run.lines.len() - 1 {
        let mut lines = run.lines.clone();
        lines.remove(i);
        assert!(
            replay_lines(&lines).is_err(),
            "removed line {i} and it still replayed"
        );
    }
}

#[test]
fn swapping_neighbouring_lines_from_different_parties_changes_the_head_or_fails() {
    let run = Run::complete_run();
    for i in 0..run.lines.len() - 1 {
        let mut lines = run.lines.clone();
        lines.swap(i, i + 1);
        if let Ok(replay) = replay_lines(&lines) {
            assert_ne!(
                replay.head, run.replay.head,
                "swapped {i} and the head stayed"
            );
        }
    }
}

#[test]
fn editing_any_line_is_caught() {
    let run = Run::complete_run();
    for (i, line) in run.lines.iter().enumerate() {
        let mut lines = run.lines.clone();
        let middle = line.len() / 2;
        lines[i][middle] ^= 1;
        assert!(
            replay_lines(&lines).is_err(),
            "edited line {i} and it still replayed"
        );
    }
}

#[test]
fn a_cut_short_log_is_valid_but_not_complete() {
    let run = Run::complete_run();
    let lines = &run.lines[..run.lines.len() - 2];
    let report = replay_lines(lines).unwrap().report().unwrap();
    assert!(!report.finished);
    assert!(!report.complete());
}

#[test]
fn the_first_record_must_be_a_plan_from_its_sponsor() {
    let run = Run::new();
    let mut fresh = Replay::new();
    let by_stranger =
        Record::sign(&run.aggregator, &Body::Plan(Box::new(run.plan.clone()))).unwrap();
    assert!(matches!(fresh.apply(&by_stranger), Err(Fault::NoPlan)));
    let not_plan = Record::sign(&run.aggregator, &run.finish_body()).unwrap();
    assert!(matches!(fresh.apply(&not_plan), Err(Fault::NoPlan)));
}

#[test]
fn a_second_plan_is_refused() {
    let mut run = Run::new();
    let (sponsor, plan) = (run.sponsor.clone(), run.plan.clone());
    let fault = run.refuse(&sponsor, Body::Plan(Box::new(plan)));
    assert!(matches!(fault, Fault::NoPlan));
}

#[test]
fn the_log_refuses_a_plan_that_cannot_work() {
    let run = Run::new();
    let mut plan = run.plan.clone();
    plan.rounds = 0;
    let record = Record::sign(&run.sponsor, &Body::Plan(Box::new(plan))).unwrap();
    assert!(matches!(
        Replay::new().apply(&record),
        Err(Fault::Plan(PlanError::NoRounds))
    ));
}

#[test]
fn plans_that_cannot_work_are_refused() {
    let run = Run::new();
    let check = |change: fn(&mut Plan), want: PlanError| {
        let mut plan = run.plan.clone();
        change(&mut plan);
        assert_eq!(plan.validate().unwrap_err(), want);
    };
    check(|p| p.version = 2, PlanError::Version(2));
    check(|p| p.rounds = 0, PlanError::NoRounds);
    check(|p| p.sites.clear(), PlanError::NoSites);
    check(
        |p| p.min_cohort = 1,
        PlanError::MinCohort { low: 2, got: 1 },
    );
    check(
        |p| p.min_cohort = 9,
        PlanError::MinCohort { low: 2, got: 9 },
    );
    check(|p| p.chunk_coords = 0, PlanError::ChunkSize);
    check(|p| p.layout.tensors.clear(), PlanError::Layout);
    check(|p| p.max_weight = 0, PlanError::MaxWeight);
    check(|p| p.encoding.clip = f64::NAN, PlanError::Encoding);
    check(
        |p| p.sites[0] = "not a did".into(),
        PlanError::BadDid("not a did".into()),
    );

    let mut plan = run.plan.clone();
    plan.helpers[0].seal = run.sponsor.seal_public().clone();
    assert!(matches!(plan.validate(), Err(PlanError::BadSealProof(_))));

    let mut plan = run.plan.clone();
    plan.sites[1] = plan.sites[0].clone();
    assert!(matches!(plan.validate(), Err(PlanError::DuplicateParty(_))));
    let mut plan = run.plan.clone();
    plan.helpers[0].did = plan.aggregator.did.clone();
    assert!(matches!(plan.validate(), Err(PlanError::DuplicateParty(_))));
    let mut plan = run.plan.clone();
    plan.max_weight = 1 << 40;
    assert!(matches!(plan.validate(), Err(PlanError::FieldOverflow(_))));

    let mut plain = run.plan.clone();
    plain.helpers.clear();
    plain.min_cohort = 1;
    plain.validate().unwrap();
    assert!(!plain.is_masked());
    assert!(run.plan.is_masked());
}

#[test]
fn plan_limits_are_inclusive() {
    let run = Run::new();
    let accepts = |change: fn(&mut Plan)| {
        let mut plan = run.plan.clone();
        change(&mut plan);
        plan.validate()
    };
    accepts(|p| p.min_cohort = 3).unwrap();
    accepts(|p| {
        p.encoding.frac_bits = 48;
        p.encoding.clip = 1.0;
        p.max_weight = 1;
    })
    .unwrap();
    assert_eq!(
        accepts(|p| p.encoding.frac_bits = 0),
        Err(PlanError::Encoding)
    );
    assert_eq!(
        accepts(|p| p.encoding.frac_bits = 49),
        Err(PlanError::Encoding)
    );
    assert_eq!(accepts(|p| p.encoding.clip = 0.0), Err(PlanError::Encoding));

    // 24 fractional bits and a clip of 128 take 31 bits, leaving 31 for the total weight.
    accepts(|p| p.max_weight = (1 << 30) / 3).unwrap();
    assert!(matches!(
        accepts(|p| p.max_weight = 1 << 31),
        Err(PlanError::FieldOverflow(_))
    ));
}

#[test]
fn only_listed_sites_with_their_own_grant_can_join() {
    let mut run = Run::new();
    let terms = run.terms();
    let body = run.join_body(0, &terms);
    let fault = run.refuse(&identity(), body.clone());
    assert!(matches!(fault, Fault::WrongSigner { .. }));
    let other_site = run.site(1);
    assert!(matches!(
        run.refuse(&other_site, body.clone()),
        Fault::Admission(..)
    ));
    let site = run.site(0);
    run.apply(&site, body.clone());
    assert!(matches!(run.refuse(&site, body), Fault::Duplicate(..)));
}

fn refused_by_terms(change: fn(&mut Terms)) {
    let mut run = Run::new();
    let mut terms = run.terms();
    change(&mut terms);
    let body = run.join_body(0, &terms);
    let site = run.site(0);
    assert!(matches!(run.refuse(&site, body), Fault::Admission(..)));
}

fn invocation_for(
    run: &Run,
    subject: &Id,
    command: &str,
    args: BTreeMap<String, Ipld>,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let expires = timestamp(NOW + 3600);
    let grant = admission::grant(
        subject.as_ref(),
        &run.sponsor.did(),
        &Terms::default(),
        expires,
    )
    .unwrap();
    let invocation = Invocation::builder(subject.did(), Command::parse(command).unwrap())
        .args(args)
        .proof(*grant.cid())
        .nonce(Nonce::random(&mut fedtally_core::pq_ucan::rng::system()))
        .expires_at(expires)
        .sign(run.sponsor.as_ref())
        .unwrap();
    (invocation.bytes().to_vec(), vec![grant.bytes().to_vec()])
}

#[test]
fn an_invocation_must_name_this_site_this_command_and_this_plan() {
    let run = Run::new();
    let (site, other) = (run.site(0), run.site(1));
    let args = admission::arguments(&run.plan_id, &run.plan);
    let check = |subject: &Id, command: &str, args: BTreeMap<String, Ipld>| {
        let (invocation, proofs) = invocation_for(&run, subject, command, args);
        admission::check(
            &site.did(),
            &run.plan_id,
            &run.plan,
            &invocation,
            &proofs,
            timestamp(NOW),
            None,
        )
    };

    assert!(check(&site, admission::COMMAND, args.clone()).is_ok());
    assert!(matches!(
        check(&other, admission::COMMAND, args.clone()),
        Err(AdmissionError::Mismatch)
    ));
    assert!(matches!(
        check(&site, "/fedtally/other", args.clone()),
        Err(AdmissionError::Mismatch)
    ));
    let mut other_plan = args;
    other_plan.insert(
        "plan".to_owned(),
        Ipld::String(Digest::of(b"another").hex()),
    );
    assert!(matches!(
        check(&site, admission::COMMAND, other_plan),
        Err(AdmissionError::Mismatch)
    ));
}

#[test]
fn a_grant_for_another_purpose_does_not_cover_the_plan() {
    refused_by_terms(|t| t.purpose = Some("marketing".into()));
}

#[test]
fn a_grant_trusting_other_helpers_does_not_cover_the_plan() {
    refused_by_terms(|t| t.trusted_helpers = vec![identity().did_string()]);
}

#[test]
fn a_grant_demanding_a_larger_cohort_does_not_cover_the_plan() {
    refused_by_terms(|t| t.min_cohort = Some(3));
}

#[test]
fn a_default_grant_refuses_a_plan_with_no_helpers() {
    let mut run = Run::new();
    run.plan.helpers.clear();
    let record = Record::sign(&run.sponsor, &Body::Plan(Box::new(run.plan.clone()))).unwrap();
    let mut replay = Replay::new();
    replay.apply(&record).unwrap();
    run.replay = replay;
    run.plan_id = record.body_hash;
    let site = run.site(0);

    let body = run.join_body(0, &Terms::default());
    assert!(matches!(run.refuse(&site, body), Fault::Admission(..)));

    let allowing = Terms {
        allow_plain: true,
        ..Terms::default()
    };
    let body = run.join_body(0, &allowing);
    run.apply(&site, body);
}

#[test]
fn a_default_grant_refuses_a_masked_plan_with_no_audits() {
    let mut run = Run::new();
    run.plan.audits = 0;
    let record = Record::sign(&run.sponsor, &Body::Plan(Box::new(run.plan.clone()))).unwrap();
    let mut replay = Replay::new();
    replay.apply(&record).unwrap();
    run.replay = replay;
    run.plan_id = record.body_hash;
    let site = run.site(0);

    let body = run.join_body(0, &Terms::default());
    assert!(matches!(run.refuse(&site, body), Fault::Admission(..)));

    let allowing = Terms {
        min_audits: Some(0),
        ..Terms::default()
    };
    let body = run.join_body(0, &allowing);
    run.apply(&site, body);
}

#[test]
fn a_grant_can_limit_who_a_site_is_averaged_with() {
    let mut run = Run::new();
    let everyone: Vec<String> = run.sites.iter().map(|s| s.did_string()).collect();
    let mut terms = run.terms();
    terms.allowed_sites = everyone[..2].to_vec();
    let body = run.join_body(0, &terms);
    let site = run.site(0);
    assert!(matches!(run.refuse(&site, body), Fault::Admission(..)));
    terms.allowed_sites = everyone;
    let body = run.join_body(0, &terms);
    run.apply(&site, body);
}

#[test]
fn a_grant_for_fewer_rounds_does_not_cover_the_plan() {
    refused_by_terms(|t| t.max_rounds = Some(1));
}

#[test]
fn an_expired_grant_is_refused() {
    let mut run = Run::new();
    let site = run.site(0);
    let grant = admission::grant(
        site.as_ref(),
        &run.sponsor.did(),
        &run.terms(),
        timestamp(NOW - 3600),
    )
    .unwrap();
    let invocation = admission::invoke(
        run.sponsor.as_ref(),
        &run.plan_id,
        &run.plan,
        &site.did(),
        &[&grant],
        timestamp(NOW + 60),
    )
    .unwrap();
    let body = run.join_body_from(&invocation, &[&grant]);
    assert!(matches!(run.refuse(&site, body), Fault::Admission(..)));
}

#[test]
fn a_grant_from_another_site_does_not_cover_this_one() {
    let mut run = Run::new();
    let (site, other) = (run.site(0), run.site(1));
    let expires = timestamp(NOW + 3600);
    let grant =
        admission::grant(other.as_ref(), &run.sponsor.did(), &run.terms(), expires).unwrap();
    let invocation = admission::invoke(
        run.sponsor.as_ref(),
        &run.plan_id,
        &run.plan,
        &site.did(),
        &[&grant],
        expires,
    )
    .unwrap();
    let body = run.join_body_from(&invocation, &[&grant]);
    assert!(matches!(run.refuse(&site, body), Fault::Admission(..)));
}

#[test]
fn a_sponsor_can_delegate_but_the_chain_must_be_whole() {
    let mut run = Run::new();
    let site = run.site(0);
    let contract_org = identity();
    let expires = timestamp(NOW + 3600);
    let first = admission::grant(site.as_ref(), &run.sponsor.did(), &run.terms(), expires).unwrap();
    let second = Delegation::builder(
        contract_org.did(),
        Subject::Did(site.did()),
        Command::parse(admission::COMMAND).unwrap(),
    )
    .nonce(Nonce::random(&mut fedtally_core::pq_ucan::rng::system()))
    .expires_at(expires)
    .sign(run.sponsor.as_ref())
    .unwrap();

    let broken = admission::invoke(
        contract_org.as_ref(),
        &run.plan_id,
        &run.plan,
        &site.did(),
        &[&first],
        expires,
    )
    .unwrap();
    let body = run.join_body_from(&broken, &[&first]);
    assert!(matches!(run.refuse(&site, body), Fault::Admission(..)));

    let whole = admission::invoke(
        contract_org.as_ref(),
        &run.plan_id,
        &run.plan,
        &site.did(),
        &[&first, &second],
        expires,
    )
    .unwrap();
    let body = run.join_body_from(&whole, &[&first, &second]);
    run.apply(&site, body);
    let report = run.replay.report().unwrap();
    assert_eq!(report.joined[0].admission.chain_len, 2);
    assert_eq!(
        report.joined[0].admission.invoked_by,
        contract_org.did_string()
    );
}

#[test]
fn opening_needs_the_aggregator_the_head_and_the_current_model() {
    let mut run = Run::new();
    run.join(0..3);
    let aggregator = run.aggregator.clone();
    let helper = run.helper.clone();

    let body = run.open_body(1);
    assert!(matches!(
        run.refuse(&helper, body),
        Fault::WrongSigner { .. }
    ));

    let mut body = run.open_body(1);
    if let Body::Open { prev, .. } = &mut body {
        *prev = Digest::of(b"elsewhere");
    }
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::BrokenChain(_)
    ));

    let mut body = run.open_body(1);
    if let Body::Open { base, .. } = &mut body {
        *base = Digest::of(b"swapped model");
    }
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::WrongBase { .. }
    ));

    let body = run.open_body(2);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::WrongRound(..)
    ));

    let mut body = run.open_body(1);
    if let Body::Open { plan, .. } = &mut body {
        *plan = Digest::of(b"another run");
    }
    assert!(matches!(run.refuse(&aggregator, body), Fault::WrongPlan(_)));
}

#[test]
fn contributions_are_held_to_the_plan() {
    let mut run = Run::new();
    run.join(0..3);
    run.open(1);
    let site = run.site(0);

    let body = run.contribute_body(1, 10);
    assert!(matches!(
        run.refuse(&identity(), body),
        Fault::WrongSigner { .. }
    ));
    let body = run.contribute_body(1, 0);
    assert!(matches!(
        run.refuse(&site, body),
        Fault::BadContribution { .. }
    ));
    let body = run.contribute_body(1, 1001);
    assert!(matches!(
        run.refuse(&site, body),
        Fault::BadContribution { .. }
    ));
    let top_weight = run.contribute_body(1, 1000);
    let other = run.site(1);
    run.apply(&other, top_weight);

    let mut body = run.contribute_body(1, 10);
    if let Body::Contribute { base, .. } = &mut body {
        *base = Digest::of(b"stale");
    }
    assert!(matches!(
        run.refuse(&site, body),
        Fault::BadContribution { .. }
    ));

    let mut body = run.contribute_body(1, 10);
    if let Body::Contribute { sealed, .. } = &mut body {
        sealed.clear();
    }
    assert!(matches!(
        run.refuse(&site, body),
        Fault::BadContribution { .. }
    ));

    let body = run.contribute_body(2, 10);
    assert!(matches!(run.refuse(&site, body), Fault::WrongRound(..)));

    let body = run.contribute_body(1, 10);
    run.apply(&site, body.clone());
    assert!(matches!(run.refuse(&site, body), Fault::Duplicate(..)));
}

#[test]
fn a_site_that_never_joined_cannot_contribute() {
    let mut run = Run::new();
    run.join(0..2);
    run.open(1);
    let site = run.site(2);
    let body = run.contribute_body(1, 10);
    assert!(matches!(run.refuse(&site, body), Fault::NotJoined(_)));
}

#[test]
fn a_record_from_another_run_is_refused() {
    let mut run = Run::new();
    run.join(0..3);
    run.open(1);
    let other = Run::new();
    let site = run.site(0);
    let mut body = run.contribute_body(1, 10);
    if let Body::Contribute { plan, .. } = &mut body {
        *plan = other.plan_id;
    }
    assert!(matches!(run.refuse(&site, body), Fault::WrongPlan(_)));
}

#[test]
fn close_cannot_invent_alter_or_undercount_contributions() {
    let mut run = Run::new();
    run.join(0..3);
    run.open(1);
    let a = run.contribute(0, 1, 10);
    let b = run.contribute(1, 1, 10);
    let aggregator = run.aggregator.clone();

    for list in [
        vec![a],
        vec![a, (2, a.1)],
        vec![(0, b.1), b],
        vec![b, a],
        vec![a, a],
    ] {
        let body = run.close_body(1, &list);
        assert!(
            matches!(run.refuse(&aggregator, body), Fault::BadClose(_)),
            "{list:?}"
        );
    }

    let mut body = run.close_body(1, &[a, b]);
    if let Body::Close { included, .. } = &mut body {
        included[1].site = identity().did_string();
    }
    assert!(matches!(run.refuse(&aggregator, body), Fault::BadClose(_)));

    let mut body = run.close_body(1, &[a, b]);
    if let Body::Close { prev, .. } = &mut body {
        *prev = Digest::of(b"older head");
    }
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::BrokenChain(_)
    ));
}

#[test]
fn a_site_left_out_of_a_round_shows_in_the_report() {
    let mut run = Run::new();
    run.join(0..3);
    run.open(1);
    let a = run.contribute(0, 1, 10);
    let b = run.contribute(1, 1, 10);
    run.contribute(2, 1, 10);
    run.close(1, &[a, b]);
    let report = run.replay.report().unwrap();
    assert_eq!(report.rounds[0].excluded, vec![run.sites[2].did_string()]);
    assert!(!report.complete());
}

#[test]
fn a_site_that_never_appears_is_listed_as_absent() {
    let mut run = Run::new();
    run.join(0..3);
    run.open(1);
    let a = run.contribute(0, 1, 10);
    let b = run.contribute(1, 1, 10);
    run.close(1, &[a, b]);
    let report = run.replay.report().unwrap();
    assert!(report.rounds[0].excluded.is_empty());
    assert_eq!(report.rounds[0].absent, vec![run.sites[2].did_string()]);
    assert!(!report.complete());
}

#[test]
fn a_second_close_for_the_round_is_refused() {
    let mut run = Run::new();
    run.join(0..3);
    let list = run.open_and_close_all(1);
    let aggregator = run.aggregator.clone();
    let body = run.close_body(1, &list);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::WrongPhase { .. }
    ));
}

#[test]
fn the_helper_answers_this_close_and_only_once() {
    let mut run = Run::new();
    run.join(0..3);
    run.open_and_close_all(1);
    let (helper, aggregator) = (run.helper.clone(), run.aggregator.clone());

    let body = run.mask_sum_body(1);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::WrongSigner { .. }
    ));

    let mut body = run.mask_sum_body(1);
    if let Body::MaskSum { close, .. } = &mut body {
        *close = Digest::of(b"another close");
    }
    assert!(matches!(run.refuse(&helper, body), Fault::BrokenChain(_)));

    let body = run.mask_sum_body(1);
    run.apply(&helper, body.clone());
    assert!(matches!(run.refuse(&helper, body), Fault::Duplicate(..)));
}

#[test]
fn outcome_waits_for_every_helper_and_needs_the_right_weight() {
    let mut run = Run::new();
    run.join(0..3);
    run.open_and_close_all(1);
    let aggregator = run.aggregator.clone();

    let body = run.outcome_body(1, 30);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::HelperMissing(_)
    ));

    run.mask_sum(1);
    let body = run.outcome_body(1, 31);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::WrongWeight { got: 31, want: 30 }
    ));
}

#[test]
fn attestations_must_answer_this_challenge_after_the_helpers() {
    let mut run = Run::new();
    run.join(0..3);
    run.reach_attesting(1);
    let site = run.site(0);
    let n = numbers();

    let body = run.attest_body(1, n.masked[0]);
    assert!(matches!(run.refuse(&site, body), Fault::HelperMissing(_)));

    run.helper_attest(1, &n.shares);

    let mut body = run.attest_body(1, n.masked[0]);
    if let Body::Attest { challenge, .. } = &mut body {
        *challenge = Digest::of(b"chosen earlier");
    }
    assert!(matches!(run.refuse(&site, body), Fault::WrongChallenge(_)));

    let body = run.attest_body(1, n.masked[0]);
    run.apply(&site, body.clone());
    assert!(matches!(run.refuse(&site, body), Fault::Duplicate(..)));
}

#[test]
fn a_site_outside_the_round_cannot_attest_or_ack() {
    let mut run = Run::new();
    run.join(0..3);
    run.open(1);
    let a = run.contribute(0, 1, 10);
    let b = run.contribute(1, 1, 10);
    run.close(1, &[a, b]);
    run.mask_sum(1);
    run.outcome(1, 20);
    run.helper_attest(1, &[fp(1, 0), fp(2, 0)]);

    let outside = run.site(2);
    let body = run.attest_body(1, fp(1, 1));
    assert!(matches!(
        run.refuse(&outside, body),
        Fault::WrongSigner { .. }
    ));

    let inside = run.site(0);
    let body = run.attest_body(1, fp(5, 0));
    run.apply(&inside, body);
    let other = run.site(1);
    let body = run.attest_body(1, fp(6, 0));
    run.apply(&other, body);
    let body = run.seal_body(1, &[fp(5, 0), fp(6, 0)], fp(11, 0), fp(8, 0));
    let aggregator = run.aggregator.clone();
    run.apply(&aggregator, body);

    let wrong = Body::Ack {
        plan: run.plan_id,
        round: 1,
        head: Digest::of(b"another view of the log"),
    };
    assert!(matches!(run.refuse(&inside, wrong), Fault::WrongHead));
    let right = run.ack_body(1);
    assert!(matches!(
        run.refuse(&outside, right.clone()),
        Fault::WrongSigner { .. }
    ));
    run.apply(&inside, right.clone());
    assert!(matches!(run.refuse(&inside, right), Fault::Duplicate(..)));
}

#[test]
fn a_helper_total_must_be_the_sum_of_its_shares() {
    let mut run = Run::new();
    run.join(0..3);
    run.reach_attesting(1);
    let helper = run.helper.clone();
    let n = numbers();

    let mut body = run.helper_attest_body(1, &n.shares);
    if let Body::HelperAttest { total, .. } = &mut body {
        *total = fp(999, 999);
    }
    assert!(matches!(run.refuse(&helper, body), Fault::HelperTotal(_)));

    let body = run.helper_attest_body(1, &n.shares[..2]);
    assert!(matches!(run.refuse(&helper, body), Fault::WrongLength(_)));
}

#[test]
fn the_seal_must_add_up() {
    let mut run = Run::new();
    run.join(0..3);
    run.reach_attesting(1);
    let aggregator = run.aggregator.clone();
    let n = numbers();

    let body = run.seal_body(1, &n.masked, n.sum, n.aggregate);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::HelperMissing(_)
    ));

    run.helper_attest(1, &n.shares);
    for i in 0..3 {
        let site = run.site(i);
        let body = run.attest_body(1, n.masked[i]);
        run.apply(&site, body);
    }

    let body = run.seal_body(1, &n.masked, n.sum + fp(1, 0), n.aggregate);
    assert!(matches!(run.refuse(&aggregator, body), Fault::SealSum));
    let body = run.seal_body(1, &n.masked, n.sum, n.aggregate + fp(1, 0));
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::SealAggregate
    ));
    let body = run.seal_body(1, &n.masked[..2], n.sum, n.aggregate);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::WrongLength(_)
    ));

    let mut changed = n.masked;
    changed[0] = changed[0] + fp(1, 0);
    let sum: Fp2 = changed.iter().copied().sum();
    let total: Fp2 = n.shares.iter().copied().sum();
    let body = run.seal_body(1, &changed, sum, sum - total);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::SiteMasked(_)
    ));

    let body = run.seal_body(1, &n.masked, n.sum, n.aggregate);
    run.apply(&aggregator, body);
}

/// The attack the rule stops: with a site silent, the aggregator could solve for
/// its fingerprint and make the equations hold for any aggregate it likes.
#[test]
fn a_fingerprint_no_site_vouches_for_cannot_be_sealed() {
    let mut run = Run::new();
    run.join(0..3);
    run.reach_attesting(1);
    let n = numbers();
    run.helper_attest(1, &n.shares);
    for i in 0..2 {
        let site = run.site(i);
        let body = run.attest_body(1, n.masked[i]);
        run.apply(&site, body);
    }

    let chosen_aggregate = fp(12_345, 678);
    let total: Fp2 = n.shares.iter().copied().sum();
    let solved = chosen_aggregate + total - n.masked[0] - n.masked[1];
    let per_site = [n.masked[0], n.masked[1], solved];
    let body = run.seal_body(1, &per_site, chosen_aggregate + total, chosen_aggregate);
    let aggregator = run.aggregator.clone();
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::Unattested(_)
    ));
}

#[test]
fn the_next_round_waits_for_every_acknowledgement() {
    let mut run = Run::new();
    run.join(0..3);
    run.reach_attesting(1);
    run.seal(1);
    for i in 0..2 {
        let site = run.site(i);
        let body = run.ack_body(1);
        run.apply(&site, body);
    }
    let aggregator = run.aggregator.clone();
    let body = run.open_body(2);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::Unacknowledged(_)
    ));
    let body = run.finish_body();
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::Unacknowledged(_)
    ));
    let site = run.site(2);
    let body = run.ack_body(1);
    run.apply(&site, body);
    run.open(2);
}

#[test]
fn finish_needs_the_last_round() {
    let mut run = Run::new();
    run.join(0..3);
    run.complete_round(1);
    let aggregator = run.aggregator.clone();
    let body = run.finish_body();
    assert!(matches!(run.refuse(&aggregator, body), Fault::EarlyFinish));
}

#[test]
fn a_finished_run_takes_no_more_rounds() {
    let mut run = Run::complete_run();
    let aggregator = run.aggregator.clone();
    let body = run.open_body(3);
    assert!(matches!(
        run.refuse(&aggregator, body),
        Fault::WrongPhase { .. }
    ));
}

#[test]
fn only_the_aggregator_or_a_helper_can_abort() {
    let mut run = Run::new();
    run.join(0..3);
    let (site, helper) = (run.site(0), run.helper.clone());
    let body = Body::Abort {
        plan: run.plan_id,
        round: 0,
        reason: "no".into(),
    };
    assert!(matches!(
        run.refuse(&site, body.clone()),
        Fault::WrongSigner { .. }
    ));
    run.apply(&helper, body.clone());
    assert_eq!(run.replay.phase, Phase::Aborted);
    assert!(matches!(
        run.refuse(&helper, body),
        Fault::WrongPhase { .. }
    ));
}

#[test]
fn a_private_round_is_paid_for_when_it_closes() {
    let mut run = Run::with(private(4 << 40));
    run.join(0..3);
    run.open(1);
    assert_eq!(run.replay.spent, 0);
    let list: Vec<_> = (0..3).map(|s| run.contribute(s, 1, 1)).collect();
    assert_eq!(
        run.replay.spent, 0,
        "nothing is released until the cohort is fixed"
    );
    run.close(1, &list);
    let cost = run.plan.round_cost(3).unwrap();
    assert_eq!(run.replay.spent, cost);
    assert!((rho_units(cost) - 0.5).abs() < 1e-3);

    let report = run.replay.report().unwrap();
    let privacy = report.privacy.as_ref().unwrap();
    assert!((privacy.rho_spent - 0.5).abs() < 1e-3);
    assert_eq!(privacy.rho_budget, 4.0);
    assert!(privacy.epsilon_spent > 0.0 && privacy.epsilon_spent < privacy.epsilon_budget);
    assert!(
        (privacy.rho_unmasked - 1.5).abs() < 1e-2,
        "one site alone has a third of the noise"
    );
    assert_eq!(report.rounds[0].rho, Some(rho_units(cost)));
    assert_eq!(
        report.rho_spent_by(&run.sites[0].did_string()),
        Some(rho_units(cost))
    );
}

#[test]
fn a_run_that_stops_after_closing_has_still_spent_the_round() {
    let mut run = Run::with(private(4 << 40));
    run.join(0..3);
    run.open_and_close_all_at(1, 1);
    let aggregator = run.aggregator.clone();
    run.apply(
        &aggregator,
        Body::Abort {
            plan: run.plan_id,
            round: 1,
            reason: "a helper went away".into(),
        },
    );
    assert_eq!(run.replay.spent, run.plan.round_cost(3).unwrap());
}

#[test]
fn a_smaller_cohort_has_less_noise_and_costs_more() {
    let mut run = Run::with(private(4 << 40));
    run.join(0..3);
    run.open(1);
    let list: Vec<_> = (0..2).map(|s| run.contribute(s, 1, 1)).collect();
    run.close(1, &list);
    let (two, three) = (
        run.plan.round_cost(2).unwrap(),
        run.plan.round_cost(3).unwrap(),
    );
    assert_eq!(run.replay.spent, two);
    assert!(two > three);
    let report = run.replay.report().unwrap();
    assert_eq!(report.rho_spent_by(&run.sites[2].did_string()), Some(0.0));
}

#[test]
fn a_round_the_budget_cannot_cover_does_not_open() {
    // Enough for one round at the smallest cohort of two, and a little more.
    let one = Run::with(private(4 << 40)).plan.round_cost(2).unwrap();
    let mut run = Run::with(private(one + one / 2));
    run.join(0..3);
    run.complete_round_at(1, 1);
    let left = run.plan.privacy.unwrap().budget - run.replay.spent;
    assert!(
        left >= run.plan.round_cost(3).unwrap(),
        "a full cohort would still fit"
    );
    let fault = run.refuse(&run.aggregator.clone(), run.open_body(2));
    assert!(
        matches!(fault, Fault::OverBudget { round: 2, cost, left: l } if cost == one && l == left),
        "{fault}"
    );
}

#[test]
fn a_plan_whose_budget_is_below_one_round_is_refused() {
    let one = Run::with(private(4 << 40)).plan.round_cost(2).unwrap();
    let mut plan = Run::new().plan;
    private(one - 1)(&mut plan);
    assert!(matches!(plan.validate(), Err(PlanError::Privacy(_))));
    private(one)(&mut plan);
    plan.validate().unwrap();
}

#[test]
fn a_site_in_a_private_run_counts_once() {
    let mut run = Run::with(private(4 << 40));
    run.join(0..3);
    run.open(1);
    let fault = run.refuse(&run.site(0), run.contribute_body(1, 2));
    assert!(matches!(fault, Fault::BadContribution { .. }), "{fault}");
    run.contribute(0, 1, 1);
}

#[test]
fn a_run_without_privacy_spends_nothing_and_reports_none() {
    let run = Run::complete_run();
    assert_eq!(run.replay.spent, 0);
    let report = run.replay.report().unwrap();
    assert!(report.privacy.is_none());
    assert_eq!(report.rounds[0].rho, None);
    assert_eq!(report.rho_spent_by(&run.sites[0].did_string()), None);
}

const LIST: &str = "urn:example:dac:list";

fn holds(kind: &str, issuer: &Id) -> Atom {
    Atom {
        kind: kind.into(),
        issuers: vec![issuer.did_string()],
        about: About::Holder,
        claims: BTreeMap::new(),
        names_dataset: None,
        data_use: None,
    }
}

/// A run that admits a site on an access grant from `dac` for its dataset.
fn gated(dac: &Id) -> impl FnOnce(&mut Plan) + use<> {
    let mut grant = holds("ControlledAccessGrants", dac);
    grant.names_dataset = Some("dataset".into());
    let policy = Policy {
        purpose: Default::default(),
        sites: Rule::Holds(grant),
        operators: None,
    };
    move |plan| plan.admission = Some(policy)
}

fn credential(issuer: &Id, kind: &str, subject: &str, dataset: Option<&str>, index: u64) -> String {
    let mut claims = serde_json::Map::new();
    if let Some(dataset) = dataset {
        claims.insert("dataset".into(), dataset.into());
    }
    let draft = Draft {
        kind: kind.into(),
        subject: subject.into(),
        claims,
        valid_from: NOW - 10,
        valid_until: Some(NOW + 1000),
        status: Some(StatusEntry {
            list: LIST.into(),
            index,
        }),
    };
    Credential::issue(issuer.as_ref(), &draft, NOW - 10)
        .unwrap()
        .text
}

fn dataset(site: usize) -> String {
    format!("did:example:dataset-{site}")
}

fn list(issuer: &Id, revoked: &[u64], issued: i64) -> String {
    StatusList::issue(issuer.as_ref(), LIST, revoked, issued)
        .unwrap()
        .text
}

impl Run {
    fn status_body(&self, list: String) -> Body {
        Body::Status {
            plan: self.plan_id,
            list,
        }
    }

    fn status(&mut self, list: String) {
        let body = self.status_body(list);
        let aggregator = self.aggregator.clone();
        self.apply(&aggregator, body);
    }

    fn join_holding(&self, site: usize, names: Option<String>, held: Vec<String>) -> Body {
        let mut body = self.join_body(site, &self.terms());
        if let Body::Join {
            dataset,
            credentials,
            ..
        } = &mut body
        {
            (*dataset, *credentials) = (names, held);
        }
        body
    }

    /// Site `i` holds a grant for its own dataset at status index `i`.
    fn grant(&self, dac: &Id, site: usize) -> Vec<String> {
        let did = self.sites[site].did_string();
        vec![credential(
            dac,
            "ControlledAccessGrants",
            &did,
            Some(&dataset(site)),
            site as u64,
        )]
    }

    fn join_gated(&mut self, dac: &Id, sites: std::ops::Range<usize>) {
        for i in sites {
            let body = self.join_holding(i, Some(dataset(i)), self.grant(dac, i));
            let site = self.site(i);
            self.apply(&site, body);
        }
    }

    fn admission_refusal(&mut self, site: usize, body: Body) -> String {
        match self.refuse(&self.site(site), body) {
            Fault::Admission(_, why) => why,
            other => panic!("expected an admission refusal, got {other}"),
        }
    }
}

#[test]
fn a_site_is_admitted_on_credentials_that_meet_the_policy() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[], NOW - 5));
    run.join_gated(&dac, 0..3);
    assert_eq!(run.replay.joined.len(), 3);
    let report = run.replay.report().unwrap();
    assert_eq!(
        report.joined[1].dataset.as_deref(),
        Some("did:example:dataset-1")
    );
    let held = &report.joined[1].credentials;
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].kind, "ControlledAccessGrants");
    assert_eq!(held[0].issuer, dac.did_string());
    assert_eq!(held[0].valid_until, Some(NOW + 1000));
    assert!(report.revoked.is_empty());
}

#[test]
fn a_site_without_the_credential_is_not_admitted() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[], NOW - 5));
    let why = run.admission_refusal(0, run.join_holding(0, Some(dataset(0)), Vec::new()));
    assert!(why.contains("no ControlledAccessGrants"), "{why}");

    // A grant for another site, another dataset, or from another issuer.
    let theirs = run.grant(&dac, 1);
    let why = run.admission_refusal(0, run.join_holding(0, Some(dataset(0)), theirs));
    assert!(why.contains("someone else"), "{why}");
    let mine = run.grant(&dac, 0);
    let why = run.admission_refusal(0, run.join_holding(0, Some(dataset(2)), mine.clone()));
    assert!(why.contains("another dataset"), "{why}");
    let why = run.admission_refusal(0, run.join_holding(0, None, mine));
    assert!(why.contains("another dataset"), "{why}");
    let stranger = identity();
    run.refuse(
        &run.aggregator.clone(),
        run.status_body(list(&stranger, &[], NOW - 5)),
    );
    let unknown = run.grant(&stranger, 0);
    let why = run.admission_refusal(0, run.join_holding(0, Some(dataset(0)), unknown));
    assert!(
        why.contains(&stranger.did_string()) && why.contains("not in the log"),
        "{why}"
    );

    let garbage = vec!["{}".to_owned()];
    run.admission_refusal(0, run.join_holding(0, Some(dataset(0)), garbage));
    let many = vec![run.grant(&dac, 0)[0].clone(); 65];
    let why = run.admission_refusal(0, run.join_holding(0, Some(dataset(0)), many));
    assert!(why.contains("too many"), "{why}");
}

#[test]
fn a_credential_counts_only_once_its_issuers_list_is_in_the_log() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    let body = run.join_holding(0, Some(dataset(0)), run.grant(&dac, 0));
    let why = run.admission_refusal(0, body.clone());
    assert!(
        why.contains("status list") && why.contains("not in the log"),
        "{why}"
    );
    run.status(list(&dac, &[], NOW - 5));
    let site = run.site(0);
    run.apply(&site, body);
}

#[test]
fn a_credential_revoked_before_joining_does_not_admit() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[1], NOW - 5));
    run.join_gated(&dac, 0..1);
    let why = run.admission_refusal(1, run.join_holding(1, Some(dataset(1)), run.grant(&dac, 1)));
    assert!(why.contains("revoked"), "{why}");
}

#[test]
fn a_credential_outside_its_dates_does_not_admit() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[], NOW - 5));
    let mut body = run.join_holding(0, Some(dataset(0)), run.grant(&dac, 0));
    if let Body::Join { at, .. } = &mut body {
        *at = NOW + 1000;
    }
    let why = run.admission_refusal(0, body);
    assert!(why.contains("not valid at this time"), "{why}");
}

#[test]
fn a_plan_without_a_policy_takes_no_credentials() {
    let dac = identity();
    let mut run = Run::new();
    let why = run.admission_refusal(0, run.join_holding(0, None, run.grant(&dac, 0)));
    assert!(why.contains("asks for no credentials"), "{why}");
    let why = run.admission_refusal(0, run.join_holding(0, Some(dataset(0)), Vec::new()));
    assert!(why.contains("asks for no credentials"), "{why}");
    let fault = run.refuse(
        &run.aggregator.clone(),
        run.status_body(list(&dac, &[], NOW)),
    );
    assert!(matches!(fault, Fault::Status(_)), "{fault}");
}

#[test]
fn a_revocation_takes_a_site_out_of_later_rounds() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[], NOW - 5));
    run.join_gated(&dac, 0..3);
    run.complete_round(1);

    run.status(list(&dac, &[2], NOW));
    assert!(run.replay.revoked[&2].contains("revoked"));
    assert_eq!(run.replay.revoked.len(), 1);

    run.open(2);
    let fault = run.refuse(&run.site(2), run.contribute_body(2, 10));
    assert!(
        matches!(&fault, Fault::BadContribution { why, .. } if why.contains("no longer admitted")),
        "{fault}"
    );
    let list_of_two: Vec<_> = (0..2).map(|s| run.contribute(s, 2, 10)).collect();
    run.close(2, &list_of_two);

    let report = run.replay.report().unwrap();
    assert_eq!(report.revoked.len(), 1);
    assert_eq!(report.revoked[0].site, run.sites[2].did_string());
    assert!(
        report.rounds[1].absent.is_empty(),
        "a revoked site is not missing"
    );
    assert_eq!(report.rounds[0].included.len(), 3);
}

#[test]
fn a_site_revoked_while_a_round_collects_cannot_be_closed_into_it() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[], NOW - 5));
    run.join_gated(&dac, 0..3);
    run.open(1);
    let all: Vec<_> = (0..3).map(|s| run.contribute(s, 1, 10)).collect();
    run.status(list(&dac, &[2], NOW));
    let fault = run.refuse(&run.aggregator.clone(), run.close_body(1, &all));
    assert!(
        matches!(&fault, Fault::BadClose(why) if why.contains("no longer admitted")),
        "{fault}"
    );
    run.close(1, &all[..2]);
}

#[test]
fn a_site_already_in_a_closed_round_finishes_it() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[], NOW - 5));
    run.join_gated(&dac, 0..3);
    run.reach_attesting(1);
    run.status(list(&dac, &[2], NOW));
    run.seal(1);
    for i in 0..3 {
        let body = run.ack_body(1);
        let site = run.site(i);
        run.apply(&site, body);
    }
    assert_eq!(run.replay.phase, Phase::Acking);
}

#[test]
fn a_status_list_only_moves_forward() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[1], NOW));
    let aggregator = run.aggregator.clone();
    let refused = |run: &mut Run, text: String, needle: &str| {
        let fault = run.refuse(&aggregator, run.status_body(text));
        assert!(
            matches!(&fault, Fault::Status(why) if why.contains(needle)),
            "{fault}"
        );
    };
    refused(&mut run, list(&dac, &[1, 2], NOW - 1), "newer list");
    refused(&mut run, list(&dac, &[1], NOW), "newer list");
    refused(&mut run, list(&dac, &[2], NOW + 1), "cannot be undone");
    refused(
        &mut run,
        list(&identity(), &[1], NOW + 1),
        "accepts that issuer",
    );
    refused(&mut run, "{}".into(), "");
    run.status(list(&dac, &[1, 2], NOW + 1));
    assert!(run.replay.list(&dac.did_string(), LIST).unwrap().is_set(2));

    // Anyone in the run may bring a list, and nobody outside it.
    let site = run.site(0);
    let body = run.status_body(list(&dac, &[1, 2, 3], NOW + 2));
    run.apply(&site, body);
    let fault = run.refuse(
        &identity(),
        run.status_body(list(&dac, &[1, 2, 3, 4], NOW + 3)),
    );
    assert!(matches!(fault, Fault::WrongSigner { .. }), "{fault}");
}

#[test]
fn one_issuer_cannot_take_anothers_list() {
    let (dac, other) = (identity(), identity());
    let mut grant = holds("ControlledAccessGrants", &dac);
    grant.issuers.push(other.did_string());
    let mut run = Run::with(move |plan| {
        plan.admission = Some(Policy {
            purpose: Default::default(),
            sites: Rule::Holds(grant),
            operators: None,
        });
    });
    // The second issuer gets a list in first, under the same id.
    run.status(list(&other, &[], NOW - 6));
    run.status(list(&dac, &[], NOW - 5));
    assert!(run.replay.list(&dac.did_string(), LIST).is_some());
    assert!(run.replay.list(&other.did_string(), LIST).is_some());
    let did = run.sites[0].did_string();
    let held = vec![credential(&dac, "ControlledAccessGrants", &did, None, 0)];
    let body = run.join_holding(0, None, held);
    let site = run.site(0);
    run.apply(&site, body);

    // Each issuer's list only speaks for its own credentials.
    run.status(list(&other, &[0], NOW));
    assert!(run.replay.revoked.is_empty());
    run.status(list(&dac, &[0], NOW));
    assert_eq!(run.replay.revoked.len(), 1);
}

#[test]
fn an_issuer_may_revoke_without_changing_the_date() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[], NOW - 5));
    run.join_gated(&dac, 0..3);
    run.status(list(&dac, &[2], NOW - 5));
    assert_eq!(run.replay.revoked.len(), 1);
    let fault = run.refuse(
        &run.aggregator.clone(),
        run.status_body(list(&dac, &[2], NOW - 5)),
    );
    assert!(
        matches!(&fault, Fault::Status(why) if why.contains("newer list")),
        "{fault}"
    );
}

#[test]
fn a_join_cannot_be_dated_before_a_list_in_the_log() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[], NOW + 50));
    let why = run.admission_refusal(0, run.join_holding(0, Some(dataset(0)), run.grant(&dac, 0)));
    assert!(why.contains("before a revocation list"), "{why}");
    let mut body = run.join_holding(0, Some(dataset(0)), run.grant(&dac, 0));
    if let Body::Join { at, .. } = &mut body {
        *at = NOW + 50;
    }
    let site = run.site(0);
    run.apply(&site, body);
}

#[test]
fn two_sites_cannot_train_on_one_dataset() {
    let dac = identity();
    let mut run = Run::with(gated(&dac));
    run.status(list(&dac, &[], NOW - 5));
    run.join_gated(&dac, 0..1);
    let did = run.sites[1].did_string();
    let held = vec![credential(
        &dac,
        "ControlledAccessGrants",
        &did,
        Some(&dataset(0)),
        1,
    )];
    let why = run.admission_refusal(1, run.join_holding(1, Some(dataset(0)), held));
    assert!(why.contains("another site trains on that dataset"), "{why}");
}

#[test]
fn one_key_under_two_names_is_one_party() {
    let key = Identity::generate(SigAlg::Ed25519).unwrap();
    let plain = key.did_string();
    let bio = key.on_registry("devnet").unwrap().did_string();
    let mut plan = Run::new().plan;
    plan.sites[0] = plain.clone();
    plan.validate().unwrap();
    plan.sites[1] = bio.clone();
    assert_eq!(plan.validate(), Err(PlanError::DuplicateParty(bio.clone())));
    plan.sites[1] = Run::new().plan.sites[1].clone();
    plan.helpers[0].did = bio.clone();
    assert!(matches!(plan.validate(), Err(PlanError::DuplicateParty(_))));
}

/// The aggregator and the helper must each hold a certification from `dac`.
fn certified(dac: &Id, give_helper: bool) -> impl FnOnce(&mut Plan) + use<> {
    let dac = dac.clone();
    move |plan| {
        plan.admission = Some(Policy {
            purpose: Default::default(),
            sites: Rule::All(Vec::new()),
            operators: Some(Rule::Holds(holds("Certification", &dac))),
        });
        let aggregator = plan.aggregator.did.clone();
        plan.aggregator.credentials =
            vec![credential(&dac, "Certification", &aggregator, None, 10)];
        if give_helper {
            let helper = plan.helpers[0].did.clone();
            plan.helpers[0].credentials =
                vec![credential(&dac, "Certification", &helper, None, 11)];
        }
    }
}

#[test]
fn operators_must_meet_the_policy_before_any_site_joins() {
    let dac = identity();
    let mut run = Run::with(certified(&dac, true));
    run.status(list(&dac, &[], NOW - 5));
    run.join(0..3);
    assert_eq!(run.replay.joined.len(), 3);

    let mut bare = Run::with(certified(&dac, false));
    bare.status(list(&dac, &[], NOW - 5));
    let why = bare.admission_refusal(0, bare.join_body(0, &bare.terms()));
    assert!(
        why.contains(&bare.helper.did_string()) && why.contains("Certification"),
        "{why}"
    );
}

#[test]
fn an_operator_that_loses_its_credential_ends_the_run() {
    let dac = identity();
    let mut run = Run::with(certified(&dac, true));
    run.status(list(&dac, &[], NOW - 5));
    run.join(0..3);
    run.open(1);
    run.status(list(&dac, &[11], NOW));
    assert_eq!(run.replay.phase, Phase::Aborted);
    let reason = run.replay.abort.clone().unwrap();
    assert!(
        reason.contains(&run.helper.did_string()) && reason.contains("revoked"),
        "{reason}"
    );
    let fault = run.refuse(&run.site(0), run.contribute_body(1, 10));
    assert!(matches!(fault, Fault::WrongPhase { .. }), "{fault}");
}

#[test]
fn a_plan_cannot_list_a_credential_about_someone_else() {
    let dac = identity();
    let mut plan = Run::new().plan;
    plan.aggregator.credentials = vec![credential(
        &dac,
        "Certification",
        "did:example:other",
        None,
        0,
    )];
    assert!(
        matches!(plan.validate(), Err(PlanError::Credential(did)) if did == plan.aggregator.did)
    );
    plan.aggregator.credentials = vec!["not a credential".into()];
    assert!(matches!(plan.validate(), Err(PlanError::Credential(_))));
    let mut plan = Run::new().plan;
    plan.admission = Some(Policy {
        purpose: Default::default(),
        sites: Rule::Any(Vec::new()),
        operators: None,
    });
    assert!(matches!(plan.validate(), Err(PlanError::Policy(_))));
}

fn drawn(run: &Run, round: u32) -> Vec<usize> {
    let beacon = sampling::beacon(&run.plan_id, round, &run.replay.model);
    sampling::invited(&beacon, 3, 2).0
}

#[test]
fn only_the_sites_drawn_for_a_round_may_contribute() {
    let mut run = Run::with(|plan| plan.sample = Some(2));
    run.join(0..3);
    let invited = drawn(&run, 1);
    run.open(1);
    assert_eq!(
        run.replay.current().unwrap().invited.as_ref(),
        Some(&invited)
    );
    let outsider = (0..3).find(|s| !invited.contains(s)).unwrap();
    let fault = run.refuse(&run.site(outsider), run.contribute_body(1, 10));
    assert!(
        matches!(&fault, Fault::BadContribution { why, .. } if why.contains("not drawn")),
        "{fault}"
    );
    let list: Vec<_> = invited.iter().map(|s| run.contribute(*s, 1, 10)).collect();
    run.close(1, &list);

    let report = run.replay.report().unwrap();
    let names: Vec<String> = invited.iter().map(|s| run.sites[*s].did_string()).collect();
    assert_eq!(report.rounds[0].invited.as_ref(), Some(&names));
    assert!(
        report.rounds[0].absent.is_empty(),
        "a site not drawn is not missing"
    );
}

#[test]
fn a_drawn_site_that_stays_away_is_reported() {
    let mut run = Run::with(|plan| {
        plan.sample = Some(3);
    });
    run.join(0..3);
    run.open(1);
    let list: Vec<_> = (0..2).map(|s| run.contribute(s, 1, 10)).collect();
    let before = run.replay.report().unwrap();
    assert_eq!(before.rounds[0].absent, [run.sites[2].did_string()]);
    run.close(1, &list);
    let after = run.replay.report().unwrap();
    assert_eq!(after.rounds[0].absent, [run.sites[2].did_string()]);
    assert!(!after.complete());
}

#[test]
fn the_draw_follows_the_model_from_round_to_round() {
    let mut run = Run::with(|plan| {
        plan.sample = Some(2);
        plan.rounds = 5;
    });
    run.join(0..3);
    let mut models = std::collections::BTreeSet::new();
    for round in 1..=5 {
        assert!(
            models.insert(run.replay.model),
            "each round starts from a new model"
        );
        let invited = drawn(&run, round);
        run.open(round);
        assert_eq!(
            run.replay.current().unwrap().invited.as_ref(),
            Some(&invited)
        );
        let list: Vec<_> = invited
            .iter()
            .map(|s| run.contribute(*s, round, 10))
            .collect();
        run.close(round, &list);
        run.mask_sum(round);
        run.outcome(round, 20);
        // Finish the round by hand: two sites, so the shared numbers do not fit.
        let (masked, shares) = ([fp(5, 1), fp(7, 2)], [fp(1, 0), fp(2, 0)]);
        run.helper_attest(round, &shares);
        for (i, site) in invited.iter().enumerate() {
            let body = run.attest_body(round, masked[i]);
            let id = run.site(*site);
            run.apply(&id, body);
        }
        let sum: Fp2 = masked.iter().copied().sum();
        let total: Fp2 = shares.iter().copied().sum();
        let body = run.seal_body(round, &masked, sum, sum - total);
        let aggregator = run.aggregator.clone();
        run.apply(&aggregator, body);
        for site in &invited {
            let body = run.ack_body(round);
            let id = run.site(*site);
            run.apply(&id, body);
        }
    }
}

#[test]
fn a_sample_must_fit_between_the_smallest_cohort_and_the_sites() {
    let mut plan = Run::new().plan;
    for (sample, ok) in [(1, false), (2, true), (3, true), (4, false)] {
        plan.sample = Some(sample);
        assert_eq!(plan.validate().is_ok(), ok, "sample {sample}");
        if !ok {
            assert_eq!(plan.validate(), Err(PlanError::Sample));
        }
    }
}

#[test]
fn a_site_can_make_privacy_a_condition_of_joining() {
    let needs = |max_budget: Option<u64>, min_colluding: Option<u32>| Terms {
        max_budget,
        min_colluding,
        ..Terms::default()
    };
    let refused = |run: &mut Run, terms: &Terms| {
        let body = run.join_body(0, terms);
        match run.refuse(&run.site(0), body) {
            Fault::Admission(_, why) => assert!(why.contains("rejected"), "{why}"),
            other => panic!("{other}"),
        }
    };

    // A plan that makes no privacy promise meets neither condition.
    let mut plain = Run::new();
    refused(&mut plain, &needs(Some(u64::MAX), None));
    refused(&mut plain, &needs(None, Some(0)));

    let mut private_run = Run::with(private(4 << 40));
    refused(&mut private_run, &needs(Some((4 << 40) - 1), None));
    refused(&mut private_run, &needs(None, Some(1)));
    let body = private_run.join_body(0, &needs(Some(4 << 40), Some(0)));
    let site = private_run.site(0);
    private_run.apply(&site, body);
}

fn on_registry() -> Id {
    let key = Identity::generate(SigAlg::Ed25519).unwrap();
    Arc::new(key.on_registry("devnet").unwrap())
}

#[test]
fn a_run_between_did_bio_identities_follows_the_same_rules() {
    let mut run = Run::of(on_registry, |_| {});
    assert!(
        run.plan
            .sites
            .iter()
            .all(|s| s.starts_with("did:bio:devnet:"))
    );
    assert!(run.plan.aggregator.did.starts_with("did:bio:devnet:"));
    run.join(0..3);
    run.complete_round(1);
    run.complete_round(2);
    let body = run.finish_body();
    let aggregator = run.aggregator.clone();
    run.apply(&aggregator, body);
    let report = run.replay.report().unwrap();
    assert!(report.complete(), "{report:?}");
    assert!(
        report.joined[0]
            .admission
            .invoked_by
            .starts_with("did:bio:devnet:")
    );

    // The same key under its did:key name is another party as far as the plan goes.
    let stranger = Arc::new(Identity::generate(SigAlg::Ed25519).unwrap());
    let fault = run.refuse(&stranger, run.ack_body(2));
    assert!(
        matches!(fault, Fault::WrongSigner { .. } | Fault::WrongPhase { .. }),
        "{fault}"
    );
}
