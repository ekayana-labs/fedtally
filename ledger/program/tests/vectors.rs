//! The shared definitions as the deployed program computes them. Each
//! published vector is read back from an account the program wrote, or
//! pinned by what the program accepts and refuses, and the sampling rules
//! are exercised around them.

mod common;

use common::{hex, key, reference, unhex, vectors::Vectors, Hash, Posting, Run};
use fedtally_ledger::{
    client::{self, ProofPart},
    error::LedgerError,
    state::*,
};

const SENSITIVITY: u64 = 262146;
const NOISE: u64 = 1 << 34;

const BEACON: &str = "ceca54fea8e95b4f6e7a32645347900387e511946c06cfa23f46dab22702a892";

/// The first and last bytes of ticket(0) to ticket(3) under [`BEACON`].
const TICKET_ENDS: [(&str, &str); 4] = [
    ("86bfe570", "72a8"),
    ("4338762c", "2a61"),
    ("65cb3cd5", "3ba2"),
    ("00e856cc", "85fd"),
];

/// A task of four sites on the plan of the beacon vector, at the start of
/// round 3 with the model the vector names.
fn at_round_three(sample: u16) -> Run {
    let mut run = Run::started(4, |t| {
        t.plan_id = core::array::from_fn(|i| i as u8);
        (t.rounds, t.min_cohort, t.sample) = (3, 1, sample);
    });
    for number in 1..=2u32 {
        let posting = Posting {
            next_model: [0x07; 32],
            ..Posting::filler(number, 1)
        };
        run.open_round();
        let (_, invited) = run.invitation(number);
        run.commit(number, &invited, &posting);
        run.post(number, &posting);
        for index in invited {
            run.ack(index, number);
        }
        run.world.warp(100);
        run.settle(number);
    }
    assert_eq!(run.open_round(), 3);
    run
}

#[test]
fn the_reference_meets_the_published_vectors() {
    assert_eq!(
        hex(&reference::leaf_hash(b"")),
        "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
    );
    let leaves: [&[u8]; 3] = [&[], &[0x00], &[0x10]];
    assert_eq!(
        hex(&reference::root(&leaves)),
        "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77"
    );
    assert_eq!(
        reference::round_cost(SENSITIVITY, NOISE, 0, 4),
        Some(549764202530)
    );
    assert_eq!(
        reference::round_cost(SENSITIVITY, NOISE, 0, 3),
        Some(733018936706)
    );
    assert_eq!(
        reference::round_cost(SENSITIVITY, NOISE, 2, 8),
        Some(366509468354)
    );
    assert_eq!(
        reference::round_cost(SENSITIVITY, NOISE, 1, 2),
        Some(2199056810114)
    );
    assert_eq!(reference::round_cost(SENSITIVITY, NOISE, 2, 2), None);

    let plan_id: [u8; 32] = core::array::from_fn(|i| i as u8);
    let beacon = reference::beacon(&plan_id, 3, &[0x07; 32]);
    assert_eq!(hex(&beacon), BEACON);
    for (site, (head, tail)) in TICKET_ENDS.iter().enumerate() {
        let ticket = hex(&reference::ticket(&beacon, site as u32));
        assert!(
            ticket.starts_with(head) && ticket.ends_with(tail),
            "{ticket}"
        );
    }
}

#[test]
fn the_program_modules_agree_with_the_reference() {
    use fedtally_ledger::{field, merkle, privacy, sampling};

    assert_eq!(merkle::leaf(b""), reference::leaf_hash(b""));
    assert_eq!(
        merkle::node(&[1; 32], &[2; 32]),
        reference::node_hash(&[1; 32], &[2; 32])
    );
    let beacon = sampling::beacon(&[5; 32], 9, &[6; 32]);
    assert_eq!(beacon, reference::beacon(&[5; 32], 9, &[6; 32]));
    for site in [0, 1, 1023] {
        assert_eq!(
            sampling::ticket(&beacon, site),
            reference::ticket(&beacon, site)
        );
    }
    for (colluding, cohort) in [(0u16, 1u16), (0, 1024), (5, 6), (1023, 1024)] {
        assert_eq!(
            privacy::round_cost(SENSITIVITY, NOISE, colluding, cohort).ok(),
            reference::round_cost(SENSITIVITY, NOISE, colluding as u64, cohort as u64)
        );
    }
    assert_eq!(field::P, reference::P);
}

#[test]
fn the_beacon_vector_on_chain() {
    let run = at_round_three(0);
    assert_eq!(run.state().model(), &[0x07; 32]);
    assert_eq!(hex(run.round(3).beacon()), BEACON);
}

#[test]
fn the_ticket_vectors_on_chain() {
    let beacon = unhex(BEACON);
    let order = reference::by_ticket(&beacon, 4);
    let sites: Vec<u16> = order.iter().map(|(_, site)| *site).collect();
    assert_eq!(sites, [3, 1, 2, 0]);

    // With a sample of k, the k-th smallest ticket is the one threshold
    // that both invites its site and invites exactly k.
    for sample in 1..=4u16 {
        let mut run = at_round_three(sample);
        let posting = Posting::filler(3, 1);
        let (ticket, site) = order[sample as usize - 1];
        let (head, tail) = TICKET_ENDS[site as usize];
        assert!(hex(&ticket).starts_with(head) && hex(&ticket).ends_with(tail));

        let below = reference::step(&ticket, false);
        run.world.refuse(
            run.commit_ix(3, &[site], &posting, &below),
            &[&run.aggregator],
            LedgerError::NotInvited,
        );
        if sample > 1 {
            run.world.refuse(
                run.commit_ix(3, &[sites[0]], &posting, &below),
                &[&run.aggregator],
                LedgerError::InvalidThreshold,
            );
        }
        if sample < 4 {
            let next = order[sample as usize].0;
            run.world.refuse(
                run.commit_ix(3, &[site], &posting, &next),
                &[&run.aggregator],
                LedgerError::InvalidThreshold,
            );
        }
        run.world.run(
            run.commit_ix(3, &[site], &posting, &ticket),
            &[&run.aggregator],
        );
        assert_eq!(run.round(3).state(), ROUND_COMMITTED);
    }
}

#[test]
fn the_round_cost_vectors_on_chain() {
    let cases = [
        (4u16, 0u16, 549764202530u64),
        (3, 0, 733018936706),
        (8, 2, 366509468354),
        (2, 1, 2199056810114),
    ];
    for (cohort, colluding, cost) in cases {
        let mut run = Run::started(cohort as usize, |t| {
            (t.sensitivity, t.noise, t.colluding) = (SENSITIVITY, NOISE, colluding);
            (t.min_cohort, t.budget) = (cohort, cost);
        });
        let number = run.open_round();
        let members: Vec<u16> = (0..cohort).collect();
        run.commit(number, &members, &Posting::filler(1, 1));
        let task = run.state();
        assert_eq!((task.spent(), task.budget()), (cost, cost));
    }

    // One lamport of budget less and the task cannot be published.
    let mut run = Run::new(4, |t| {
        (t.sensitivity, t.noise, t.min_cohort) = (SENSITIVITY, NOISE, 4);
        t.budget = 549764202529;
    });
    run.world.refuse(
        client::publish(&key(&run.publisher), &run.terms),
        &[&run.publisher],
        LedgerError::BudgetTooSmall,
    );
}

/// Proves a wrong aggregate in the last leaf of vectors of `coords`
/// coordinates, whatever shape their tree has.
fn prove_fraud_in_last_leaf(coords: usize) {
    let vectors = Vectors::honest(2, 1, coords, coords as u64).with_wrong_aggregate(coords - 1);
    let mut run = Run::started(2, |t| t.coords = coords as u64);
    let number = run.post_round(&[0, 1], &vectors.posting(1));
    let challenger = run.challenger();
    let last = coords.div_ceil(128) - 1;
    run.dispute_open(&challenger, number, last as u64);
    assert_eq!(
        run.world
            .dispute(&run.task, number, &challenger)
            .leaf_coords() as usize,
        coords - 128 * last
    );
    run.give_all(&challenger, number, last, &vectors);
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);
    assert_eq!(
        run.round(number).state(),
        ROUND_FRAUD,
        "{coords} coordinates"
    );
}

#[test]
fn vector_roots_on_chain_for_trees_of_many_shapes() {
    // One short leaf, one full leaf, a second leaf of one coordinate, and
    // trees of five, seven and eight leaves.
    for coords in [4, 128, 129, 600, 800, 1024] {
        prove_fraud_in_last_leaf(coords);
    }
}

#[test]
fn inputs_roots_on_chain_for_cohorts_of_many_sizes() {
    for members in [1usize, 2, 5, 7, 11] {
        let vectors = Vectors::honest(members, 1, 200, members as u64).with_wrong_aggregate(3);
        let mut run = Run::started(members, |t| (t.coords, t.min_cohort) = (200, 1));
        let cohort: Vec<u16> = (0..members as u16).collect();
        let number = run.post_round(&cohort, &vectors.posting(1));
        let challenger = run.challenger();
        run.dispute_open(&challenger, number, 0);
        run.give_all(&challenger, number, 0, &vectors);
        run.world
            .run(run.dispute_close_ix(&challenger, number), &[]);
        assert_eq!(run.round(number).state(), ROUND_FRAUD, "{members} members");
    }
}

#[test]
fn a_cohort_that_skips_sites_is_listed_in_ascending_site_index() {
    let vectors = Vectors::honest(3, 1, 296, 8).with_wrong_aggregate(130);
    let mut run = Run::started(9, |_| {});
    let number = run.post_round(&[1, 4, 8], &vectors.posting(1));
    let challenger = run.challenger();
    run.dispute_open(&challenger, number, 1);
    run.world.refuse(
        run.dispute_leaf_ix(&challenger, number, ProofPart::Input(3)),
        &[&challenger],
        LedgerError::InvalidProofPart,
    );
    run.give_all(&challenger, number, 1, &vectors);
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);
    assert_eq!(run.round(number).state(), ROUND_FRAUD);
}

/// A sampled task of sixteen sites with its first round open, the round's
/// threshold, and the sites it invites.
fn sampled() -> (Run, Hash, Vec<u16>) {
    let mut run = Run::started(16, |t| (t.min_cohort, t.sample) = (3, 6));
    run.open_round();
    let (threshold, invited) = run.invitation(1);
    (run, threshold, invited)
}

#[test]
fn sampling_invites_the_sites_with_the_smallest_tickets() {
    let (mut run, threshold, invited) = sampled();
    let posting = Posting::filler(1, 1);
    let beacon = *run.round(1).beacon();
    let order = reference::by_ticket(&beacon, 16);
    assert_eq!(invited.len(), 6);
    assert_eq!(threshold, order[5].0);
    let outsider = order[6].1;

    // A threshold that invites five or seven, or everyone, or no one.
    let wrong = [
        (order[4].0, LedgerError::NotInvited),
        (reference::step(&threshold, false), LedgerError::NotInvited),
        (order[6].0, LedgerError::InvalidThreshold),
        ([0xff; 32], LedgerError::InvalidThreshold),
        ([0x00; 32], LedgerError::NotInvited),
    ];
    for (threshold, why) in wrong {
        run.world.refuse(
            run.commit_ix(1, &invited, &posting, &threshold),
            &[&run.aggregator],
            why,
        );
    }
    let smallest: Vec<u16> = order[..3].iter().map(|(_, site)| *site).collect();
    run.world.refuse(
        run.commit_ix(1, &smallest, &posting, &order[4].0),
        &[&run.aggregator],
        LedgerError::InvalidThreshold,
    );

    // The right threshold with a member that was not invited.
    let mut with_outsider = invited[..3].to_vec();
    with_outsider.push(outsider);
    run.world.refuse(
        run.commit_ix(1, &with_outsider, &posting, &threshold),
        &[&run.aggregator],
        LedgerError::NotInvited,
    );
    run.world.refuse(
        run.commit_ix(1, &invited[..2], &posting, &threshold),
        &[&run.aggregator],
        LedgerError::InvalidCohortSize,
    );

    // Any threshold from the sixth ticket up to just below the seventh
    // invites the same six, and not all of them have to show up.
    let below_next = reference::step(&order[6].0, false);
    run.world.run(
        run.commit_ix(1, &invited[1..5], &posting, &below_next),
        &[&run.aggregator],
    );
    let round = run.round(1);
    assert_eq!(round.cohort_count(), 4);
    assert_eq!(
        round.cohort(),
        &client::site_bitmap(invited[1..5].iter().copied())
    );
}

#[test]
fn each_round_draws_its_own_cohort() {
    let (mut run, first_threshold, first) = sampled();
    let posting = Posting::filler(1, 1);
    run.commit(1, &first, &posting);
    run.post(1, &posting);
    for index in &first {
        run.ack(*index, 1);
    }
    run.world.warp(100);
    run.settle(1);

    run.open_round();
    let (threshold, invited) = run.invitation(2);
    assert_ne!(threshold, first_threshold);
    assert_ne!(invited, first);
    let posting = Posting::filler(2, 1);
    run.world.refuse(
        run.commit_ix(2, &first, &posting, &first_threshold),
        &[&run.aggregator],
        LedgerError::NotInvited,
    );
    run.commit(2, &invited, &posting);
    assert_eq!(run.round(2).cohort_count(), 6);
}

#[test]
fn a_task_without_sampling_ignores_the_threshold() {
    let mut run = Run::started(4, |_| {});
    let number = run.open_round();
    let posting = Posting::filler(1, 1);
    run.world.run(
        run.commit_ix(number, &[0, 3], &posting, &[0; 32]),
        &[&run.aggregator],
    );
    assert_eq!(run.round(number).cohort_count(), 2);
}
