//! Every rule in the replay, tried against a record that breaks it.

use std::collections::BTreeMap;
use std::sync::Arc;

use fedtally_core::admission::{self, AdmissionError, Terms};
use fedtally_core::field::{Fp, Fp2};
use fedtally_core::keys::{Identity, SigAlg};
use fedtally_core::log::{GENESIS, chain};
use fedtally_core::model::{Dtype, Encoding, Layout, TensorSpec};
use fedtally_core::plan::{AggregatorSpec, HelperSpec, Plan, PlanError, Timeouts};
use fedtally_core::pq_ucan::Ipld;
use fedtally_core::pq_ucan::command::Command;
use fedtally_core::pq_ucan::crypto::Signer;
use fedtally_core::pq_ucan::delegation::{Delegation, Subject};
use fedtally_core::pq_ucan::invocation::Invocation;
use fedtally_core::pq_ucan::nonce::Nonce;
use fedtally_core::pq_ucan::time::Timestamp;
use fedtally_core::record::{Body, Included, Record, encode_token};
use fedtally_core::replay::{Fault, Phase, Replay};
use fedtally_core::report::{VerifyError, verify_log};
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
        let (sponsor, aggregator, helper) = (identity(), identity(), identity());
        let sites: Vec<Id> = (0..3).map(|_| identity()).collect();
        let plan = Plan {
            version: 1,
            nonce: "n".into(),
            purpose: "cardiology".into(),
            sponsor: sponsor.did_string(),
            aggregator: AggregatorSpec {
                did: aggregator.did_string(),
                tls_pin: Digest::of(b"cert"),
            },
            helpers: vec![HelperSpec {
                did: helper.did_string(),
                seal: helper.seal_public().clone(),
                seal_proof: helper.seal_proof().unwrap(),
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
        };
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
        self.open(round);
        let list: Vec<_> = (0..3).map(|s| self.contribute(s, round, 10)).collect();
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
        self.reach_attesting(round);
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
