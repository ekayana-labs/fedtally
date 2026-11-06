//! Compute unit report. It drives one task through every instruction and
//! prints what each one consumed, with a ceiling per line so a cost
//! regression fails the test.
//!
//! The wallets are fixed, so the numbers are reproducible: an instruction
//! that creates an account searches for its bump, and every rejected
//! candidate costs a hash and a curve check.
//!
//! Run `cargo test --test compute_units -- --nocapture` to see the output.

mod common;

use common::{key, reference, vectors::Vectors, Hash, Posting, Run, World};
use fedtally_ledger::client::{self, ProofPart, StatusRef};

/// What an instruction may cost unless its line says otherwise.
const CEILING: u64 = 12_000;

#[derive(Default)]
struct Report {
    lines: Vec<(String, u64, u64)>,
}

impl Report {
    fn add(&mut self, name: &str, units: u64) {
        self.add_under(name, units, CEILING);
    }

    fn add_under(&mut self, name: &str, units: u64, ceiling: u64) {
        self.lines.push((name.to_string(), units, ceiling));
    }

    fn print(&self) {
        println!();
        println!("{:<58} {:>8}", "instruction", "CU");
        println!("{}", "-".repeat(67));
        for (name, units, _) in &self.lines {
            println!("{name:<58} {units:>8}");
        }
        println!("{}", "-".repeat(67));
        for (name, units, ceiling) in &self.lines {
            assert!(
                units < ceiling,
                "`{name}` consumed {units} CU (ceiling {ceiling})"
            );
        }
    }
}

/// A sampled task of `sites` sites with its first round open and only the
/// invited sites enrolled. Returns the cost of committing the invited.
fn sampled_commit(sites: usize, sample: u16) -> u64 {
    let mut run = Run::in_world(World::fixed(), sites, |t| {
        (t.min_cohort, t.sample) = (sample, sample);
    });
    run.publish();
    run.back();
    let number = run.open_round();
    let (threshold, invited) = run.invitation(number);
    for index in &invited {
        run.enroll(*index);
    }
    let ix = run.commit_ix(number, &invited, &Posting::filler(1, 1), &threshold);
    run.world.run_with_limit(ix, &[&run.aggregator], 400_000)
}

/// A dispute over one full leaf of vectors with 2^20 leaves, so that the
/// leaf's path has 20 levels. The trees are not built: the path is made of
/// arbitrary siblings and the roots are what it leads to.
fn deep_dispute(report: &mut Report) {
    const LEAVES: usize = 1 << 20;
    const LEAF_INDEX: usize = 0b1010_1010_1010_1010_1010;
    let members = 16;
    let mut run = Run::in_world(World::fixed(), members, |t| {
        t.coords = 128 * LEAVES as u64;
    });
    run.publish();
    run.back();
    for index in 0..members as u16 {
        run.enroll(index);
    }

    let leaf: Vec<u8> = (0..1024u32).map(|i| (i * 7) as u8).collect();
    let path: Vec<Hash> = (0..20u8).map(|level| [level; 32]).collect();
    let vector_root =
        reference::root_from_path(reference::leaf_hash(&leaf), LEAF_INDEX, LEAVES, &path);
    let mut input_roots: Vec<Hash> = (0..members as u8).map(|member| [member; 32]).collect();
    input_roots[5] = vector_root;

    let posting = Posting {
        inputs_root: reference::root(&input_roots),
        aggregate_root: vector_root,
        ..Posting::filler(1, 1)
    };
    let cohort: Vec<u16> = (0..members as u16).collect();
    let number = run.post_round(&cohort, &posting);
    let challenger = run.challenger();
    run.dispute_open(&challenger, number, LEAF_INDEX as u64);

    let aggregate = client::leaf_proof(&leaf, &path);
    let input = client::input_proof(
        &leaf,
        &vector_root,
        &reference::path(&input_roots, 5),
        &path,
    );
    assert_eq!(aggregate.len(), 1024 + 20 * 32);
    report.add_under(
        "dispute_leaf (aggregate, full leaf, 20-level path)",
        run.give(&challenger, number, ProofPart::Aggregate, &aggregate),
        20_000,
    );
    report.add_under(
        "dispute_leaf (input of 16, full leaf, 20-level path)",
        run.give(&challenger, number, ProofPart::Input(5), &input),
        20_000,
    );
}

#[test]
fn compute_unit_report() {
    let mut report = Report::default();
    let mut run = Run::in_world(World::fixed(), 16, |t| t.rounds = 3);

    report.add(
        "publish",
        run.world.run(
            client::publish(&key(&run.publisher), &run.terms),
            &[&run.publisher],
        ),
    );
    report.add(
        "back",
        run.world.run(
            client::back(&key(&run.aggregator), &run.task),
            &[&run.aggregator],
        ),
    );

    let issuer = run.world.wallet();
    report.add(
        "status_init (80000 bits)",
        run.world
            .run(client::status_init(&key(&issuer), 1, 80_000), &[&issuer]),
    );
    report.add(
        "status_init (64 bits)",
        run.world
            .run(client::status_init(&key(&issuer), 2, 64), &[&issuer]),
    );
    let lists = [1, 2].map(|list| client::status_list(&key(&issuer), list));
    let refs = [
        StatusRef {
            list: lists[0],
            bit: 79_999,
        },
        StatusRef {
            list: lists[1],
            bit: 3,
        },
    ];
    report.add(
        "enroll (16 sites, 2 status references)",
        run.world.run(run.enroll_ix(0, &refs), &[&run.sites[0]]),
    );
    for index in 1..16 {
        run.enroll(index);
    }

    let vectors = Vectors::honest(16, 1, 296, 1);
    let posting = vectors.posting(1);
    let cohort: Vec<u16> = (0..16).collect();
    report.add(
        "open_round",
        run.world.run(
            client::open_round(&key(&run.aggregator), &run.task, 1),
            &[&run.aggregator],
        ),
    );
    report.add(
        "commit_root (no sampling, cohort of 16)",
        run.world.run(
            run.commit_ix(1, &cohort, &posting, &[0; 32]),
            &[&run.aggregator],
        ),
    );
    report.add_under(
        "commit_root (sampling 8 of 16)",
        sampled_commit(16, 8),
        12_000,
    );
    report.add_under(
        "commit_root (sampling 32 of 128)",
        sampled_commit(128, 32),
        40_000,
    );
    report.add_under(
        "commit_root (sampling 64 of 1024)",
        sampled_commit(1024, 64),
        200_000,
    );
    report.add(
        "post_aggregate (1 helper)",
        run.world.run(run.post_ix(1, &posting), &[&run.aggregator]),
    );
    report.add(
        "ack",
        run.world
            .run(run.ack_ix(0, 1, &posting.seal_head), &[&run.sites[0]]),
    );
    for index in 1..16 {
        run.ack(index, 1);
    }

    // A dispute that fails, over the short last leaf.
    let challenger = run.world.wallet();
    report.add(
        "buffer_init (4000 bytes)",
        run.world.run(
            client::buffer_init(&key(&challenger), 4_000),
            &[&challenger],
        ),
    );
    report.add(
        "buffer_write (900 bytes)",
        run.world.run(
            client::buffer_write(&key(&challenger), 0, &[7; 900]),
            &[&challenger],
        ),
    );
    report.add(
        "dispute_open",
        run.world
            .run(run.dispute_open_ix(&challenger, 1, 1), &[&challenger]),
    );
    for part in vectors.parts() {
        let units = run.give(&challenger, 1, part, &vectors.proof(part, 1));
        match part {
            ProofPart::Input(0) => {
                report.add("dispute_leaf (input of 16, full leaf, 2-level path)", units)
            }
            ProofPart::Mask(0) => report.add("dispute_leaf (mask, full leaf, 2-level path)", units),
            ProofPart::Aggregate => {
                report.add("dispute_leaf (aggregate, full leaf, 2-level path)", units)
            }
            _ => {}
        }
    }
    deep_dispute(&mut report);
    report.add(
        "dispute_close (aggregate upheld)",
        run.world.run(run.dispute_close_ix(&challenger, 1), &[]),
    );
    report.add(
        "buffer_close",
        run.world
            .run(client::buffer_close(&key(&challenger)), &[&challenger]),
    );

    run.world.warp(100);
    report.add("settle", run.world.run(client::settle(&run.task, 1), &[]));
    report.add("claim", run.world.run(run.claim_ix(0, 1), &[&run.sites[0]]));

    report.add(
        "revoke",
        run.world
            .run(client::revoke(&key(&issuer), 1, 79_999), &[&issuer]),
    );
    report.add(
        "revoke_check (2 status lists)",
        run.world
            .run(client::revoke_check(&run.task, 0, &lists), &[]),
    );
    report.add(
        "withdraw (revoked site)",
        run.world.run(run.withdraw_site_ix(0), &[&run.sites[0]]),
    );

    // A second round with a wrong aggregate, and the fraud proof against it.
    let wrong = Vectors::honest(15, 1, 296, 2).with_wrong_aggregate(0);
    let cohort: Vec<u16> = (1..16).collect();
    run.post_round(&cohort, &wrong.posting(2));
    let challenger = run.challenger();
    run.dispute_open(&challenger, 2, 0);
    run.give_all(&challenger, 2, 0, &wrong);
    report.add(
        "dispute_close (fraud)",
        run.world.run(run.dispute_close_ix(&challenger, 2), &[]),
    );
    report.add(
        "withdraw (publisher)",
        run.world
            .run(run.withdraw_publisher_ix(), &[&run.publisher]),
    );

    // Another task, aborted before any round, for the remaining two.
    let mut other = Run::in_world(World::fixed(), 2, |_| {});
    other.publish();
    other.back();
    report.add(
        "abort",
        other
            .world
            .run(other.abort_ix(&other.aggregator), &[&other.aggregator]),
    );
    report.add(
        "withdraw (aggregator)",
        other
            .world
            .run(other.withdraw_aggregator_ix(), &[&other.aggregator]),
    );

    report.print();
}

#[test]
fn a_compute_unit_limit_does_not_change_what_an_instruction_costs() {
    let commit = |with_limit: bool| {
        let mut run = Run::in_world(World::fixed(), 4, |_| {});
        run.publish();
        run.back();
        for index in 0..4 {
            run.enroll(index);
        }
        let number = run.open_round();
        let ix = run.commit_ix(number, &[0, 1, 2], &Posting::filler(1, 1), &[0; 32]);
        if with_limit {
            run.world.run_with_limit(ix, &[&run.aggregator], 400_000)
        } else {
            run.world.run(ix, &[&run.aggregator])
        }
    };
    assert_eq!(commit(true), commit(false));
}

#[test]
fn demand_and_round_close_units() {
    use common::vectors::Vectors;

    let mut run = Run::in_world(World::fixed(), 3, |t| t.rounds = 3);
    run.publish();
    run.back();
    for index in 0..3 {
        run.enroll(index);
    }
    let first = run.settle_round(&[0, 1, 2]);
    for index in 0..3 {
        run.claim(index, first);
    }
    let vectors = Vectors::honest(3, 1, 296, 5);
    let number = run.post_round(&[0, 1, 2], &vectors.posting(2));
    let challenger = run.challenger();
    let who = key(&challenger);
    let part = ProofPart::Input(1);

    let open = run.world.run(
        client::demand_open(&who, &run.task, number, part, 0),
        &[&challenger],
    );
    let demand = client::demand(&run.task, &who, number, part, 0);
    run.world.upload(&challenger, &vectors.proof(part, 0));
    let answer = run.world.run(
        client::demand_answer(&who, &run.task, number, demand),
        &[&challenger],
    );
    let close = run.world.run(
        client::demand_close(&run.task, number, demand, &who, true),
        &[&challenger],
    );
    let round = run.world.run(
        client::round_close(&key(&run.aggregator), &run.task, first),
        &[],
    );
    println!(
        "demand_open {open}, demand_answer {answer}, demand_close {close}, round_close {round}"
    );
    assert!(open < 20_000 && answer < 20_000 && close < 5_000 && round < 5_000);
}
