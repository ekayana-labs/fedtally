//! A task from publish to the last withdrawal, and the rules of the
//! instructions that carry an honest run: for each rule one transaction
//! that satisfies it and one that breaks it and is refused.

mod common;

use common::{admission, key, reference, Posting, Run, World, WALLET};
use fedtally_ledger::{
    client::{self, StatusRef, TaskTerms},
    error::LedgerError,
    state::*,
};

fn bits(sites: &[u16]) -> [u8; 128] {
    client::site_bitmap(sites.iter().copied())
}

#[test]
fn honest_task_from_publish_to_last_withdrawal() {
    let mut run = Run::new(5, |_| {});
    let terms = run.terms.clone();
    let (task_rent, site_rent, round_rent) = (
        run.world.rent(TASK_SPACE),
        run.world.rent(ENROLLMENT_SPACE),
        run.world.rent(ROUND_SPACE),
    );

    run.publish();
    let escrow = 2 * terms.reward_per_round;
    let task = run.state();
    assert_eq!(task.address().as_array(), &run.task);
    assert_eq!(task.state(), TASK_OPEN);
    assert!(!task.backed() && !task.fraud());
    assert_eq!(task.publisher(), &key(&run.publisher));
    assert_eq!(task.aggregator(), &terms.aggregator);
    assert_eq!(task.plan_id(), &terms.plan_id);
    assert_eq!(task.sites_root(), &terms.sites_root);
    assert_eq!(task.model(), &terms.model);
    assert_eq!((task.sites(), task.helpers()), (5, 1));
    assert_eq!((task.coords(), task.leaves()), (296, 3));
    assert_eq!((task.rounds(), task.round(), task.settled()), (2, 0, 0));
    assert_eq!((task.min_cohort(), task.sample()), (2, 0));
    assert_eq!(
        (task.sensitivity(), task.noise(), task.colluding()),
        (0, 0, 0)
    );
    assert_eq!((task.budget(), task.spent(), task.owed()), (0, 0, 0));
    assert_eq!(task.site_stake(), terms.site_stake);
    assert_eq!(task.aggregator_stake(), terms.aggregator_stake);
    assert_eq!(task.stake_held(), 0);
    assert_eq!(task.reward_per_round(), terms.reward_per_round);
    assert_eq!(task.dispute_bond(), terms.dispute_bond);
    assert_eq!((task.challenge_slots(), task.stall_slots()), (100, 1_000));
    assert_eq!(task.last_change(), 0);
    assert_eq!((task.enrolled(), task.revoked()), (&[0; 128], &[0; 128]));
    assert_eq!(run.world.balance(&run.task), task_rent + escrow);
    assert_eq!(
        run.world.balance(&key(&run.publisher)),
        WALLET - task_rent - escrow
    );

    run.world.warp(3);
    run.back();
    let task = run.state();
    assert!(task.backed());
    assert_eq!(task.stake_held(), terms.aggregator_stake);
    assert_eq!(task.last_change(), 3);
    assert_eq!(
        run.world.balance(&run.task),
        task_rent + escrow + terms.aggregator_stake
    );

    for index in 0..5 {
        run.enroll(index);
        let site = run.world.enrollment(&run.task, index);
        assert_eq!(
            site.address().as_array(),
            &client::enrollment(&run.task, index)
        );
        assert_eq!(site.status(), SITE_ACTIVE);
        assert_eq!(site.index(), index);
        assert_eq!(site.task(), &run.task);
        assert_eq!(site.wallet(), &key(&run.sites[index as usize]));
        assert_eq!(site.stake(), terms.site_stake);
        assert_eq!(site.admission(), &admission(index));
        assert_eq!(site.status_ref_count(), 0);
        assert_eq!(
            run.world.balance(&client::enrollment(&run.task, index)),
            site_rent + terms.site_stake
        );
    }
    assert_eq!(run.state().enrolled(), &bits(&[0, 1, 2, 3, 4]));

    // Round 1 with three of the five sites.
    run.world.warp(7);
    let first = Posting::filler(1, 1);
    assert_eq!(run.open_round(), 1);
    let round = run.round(1);
    assert_eq!(round.address().as_array(), &client::round(&run.task, 1));
    assert_eq!(round.state(), ROUND_COLLECTING);
    assert_eq!((round.round(), round.task()), (1, &run.task));
    assert_eq!(
        round.beacon(),
        &reference::beacon(&terms.plan_id, 1, &terms.model)
    );
    assert_eq!((run.state().round(), run.state().last_change()), (1, 10));

    run.world.warp(1);
    run.commit(1, &[0, 1, 2], &first);
    let round = run.round(1);
    assert_eq!(round.state(), ROUND_COMMITTED);
    assert_eq!(
        (round.cohort(), round.cohort_count()),
        (&bits(&[0, 1, 2]), 3)
    );
    assert_eq!(round.inputs_root(), &first.inputs_root);
    assert_eq!(round.close_head(), &first.close_head);
    assert_eq!(run.state().last_change(), 11);

    run.world.warp(1);
    run.post(1, &first);
    let round = run.round(1);
    assert_eq!(round.state(), ROUND_POSTED);
    assert_eq!(round.mask_root(0), &first.mask_roots[0]);
    assert_eq!(round.mask_roots()[32..], [0; 96]);
    assert_eq!(round.aggregate_root(), &first.aggregate_root);
    assert_eq!(round.next_model(), &first.next_model);
    assert_eq!(round.seal_head(), &first.seal_head);
    assert_eq!(round.posted_slot(), 12);
    assert_eq!(round.disputes(), 0);
    assert_eq!(run.state().last_change(), 12);

    for index in [2, 0, 1] {
        run.ack(index, 1);
    }
    assert_eq!(run.round(1).acked(), &bits(&[0, 1, 2]));

    run.world.warp(100);
    run.settle(1);
    let share_one = terms.reward_per_round / 3;
    let (task, round) = (run.state(), run.round(1));
    assert_eq!((round.state(), round.share()), (ROUND_FINAL, share_one));
    assert_eq!(task.state(), TASK_OPEN);
    assert_eq!(task.model(), &first.next_model);
    assert_eq!((task.settled(), task.owed()), (1, 3 * share_one));
    assert_eq!(task.last_change(), 112);

    // Round 2 with another cohort, starting from the model round 1 left.
    let second = Posting::filler(2, 1);
    assert_eq!(run.settle_round_with(&[1, 2, 3, 4], &second), 2);
    assert_eq!(
        run.round(2).beacon(),
        &reference::beacon(&terms.plan_id, 2, &first.next_model)
    );
    let share_two = terms.reward_per_round / 4;
    let task = run.state();
    assert_eq!(task.state(), TASK_FINISHED);
    assert_eq!(task.model(), &second.next_model);
    assert_eq!((task.round(), task.settled()), (2, 2));
    assert_eq!(task.owed(), 3 * share_one + 4 * share_two);

    for index in [0, 1, 2] {
        run.claim(index, 1);
    }
    for index in [1, 2, 3, 4] {
        run.claim(index, 2);
    }
    assert_eq!(run.round(1).claimed(), &bits(&[0, 1, 2]));
    assert_eq!(run.round(2).claimed(), &bits(&[1, 2, 3, 4]));
    assert_eq!(run.state().owed(), 0);

    for index in 0..5 {
        let ix = run.withdraw_site_ix(index);
        run.world.run(ix, &[&run.sites[index as usize]]);
        assert_eq!(
            run.world.enrollment(&run.task, index).status(),
            SITE_WITHDRAWN
        );
        assert_eq!(
            run.world.balance(&client::enrollment(&run.task, index)),
            site_rent
        );
    }
    let ix = run.withdraw_aggregator_ix();
    run.world.run(ix, &[&run.aggregator]);
    let ix = run.withdraw_publisher_ix();
    run.world.run(ix, &[&run.publisher]);

    // Everyone is whole again, less the rent of the accounts they opened,
    // and the rewards went from the publisher to the sites.
    let rewards = [
        share_one,
        share_one + share_two,
        share_one + share_two,
        share_two,
        share_two,
    ];
    for (index, reward) in rewards.iter().enumerate() {
        assert_eq!(
            run.world.balance(&key(&run.sites[index])),
            WALLET - site_rent + reward
        );
    }
    let paid: u64 = rewards.iter().sum();
    assert_eq!(paid, 19_999_999);
    assert_eq!(
        run.world.balance(&key(&run.publisher)),
        WALLET - task_rent - paid
    );
    assert_eq!(
        run.world.balance(&key(&run.aggregator)),
        WALLET - 2 * round_rent
    );
    assert_eq!(run.world.balance(&run.task), task_rent);
    assert_eq!(run.state().stake_held(), 0);
}

/// Terms that `publish` accepts, each under a plan of its own.
fn terms(plan: u8, adjust: impl FnOnce(&mut TaskTerms)) -> TaskTerms {
    let mut terms = TaskTerms {
        aggregator: [2; 32],
        plan_id: [plan; 32],
        sites: 8,
        helpers: 2,
        coords: 1_000,
        rounds: 3,
        min_cohort: 3,
        reward_per_round: 1_000,
        challenge_slots: 10,
        stall_slots: 100,
        ..TaskTerms::default()
    };
    adjust(&mut terms);
    terms
}

const SENSITIVITY: u64 = 262146;
const NOISE: u64 = 1 << 34;

#[test]
fn publish_checks_the_terms() {
    let mut world = World::new();
    let publisher = world.wallet();
    let who = key(&publisher);
    let mut plan = 0;
    let mut accepted = |world: &mut World, adjust: fn(&mut TaskTerms)| {
        plan += 1;
        world.run(client::publish(&who, &terms(plan, adjust)), &[&publisher]);
    };
    let refused = |world: &mut World, adjust: fn(&mut TaskTerms), why: LedgerError| {
        world.refuse(
            client::publish(&who, &terms(200, adjust)),
            &[&publisher],
            why,
        );
        assert!(!world.exists(&client::task(&who, &[200; 32])));
    };

    accepted(&mut world, |t| t.sites = 1024);
    accepted(&mut world, |t| (t.sites, t.min_cohort) = (1, 1));
    refused(&mut world, |t| t.sites = 0, LedgerError::InvalidSites);
    refused(&mut world, |t| t.sites = 1025, LedgerError::InvalidSites);

    refused(
        &mut world,
        |t| (t.challenge_slots, t.stall_slots) = (100, 100),
        LedgerError::InvalidWindows,
    );
    accepted(&mut world, |t| {
        (t.challenge_slots, t.stall_slots) = (100, 101)
    });

    accepted(&mut world, |t| t.helpers = 4);
    accepted(&mut world, |t| t.helpers = 0);
    refused(&mut world, |t| t.helpers = 5, LedgerError::InvalidHelpers);

    accepted(&mut world, |t| t.min_cohort = 8);
    refused(
        &mut world,
        |t| t.min_cohort = 0,
        LedgerError::InvalidMinCohort,
    );
    refused(
        &mut world,
        |t| t.min_cohort = 9,
        LedgerError::InvalidMinCohort,
    );

    accepted(&mut world, |t| t.sample = 3);
    accepted(&mut world, |t| t.sample = 8);
    refused(&mut world, |t| t.sample = 2, LedgerError::InvalidSample);
    refused(&mut world, |t| t.sample = 9, LedgerError::InvalidSample);

    accepted(&mut world, |t| t.rounds = 1);
    refused(&mut world, |t| t.rounds = 0, LedgerError::InvalidRounds);

    // With noise the cohort has to outnumber the colluding sites, and the
    // budget has to cover a round at the minimum cohort.
    let private = |t: &mut TaskTerms| {
        (t.sensitivity, t.noise) = (SENSITIVITY, NOISE);
        t.budget = 733018936706;
    };
    accepted(&mut world, private);
    accepted(&mut world, |t| t.colluding = 3);
    refused(
        &mut world,
        |t| {
            (t.sensitivity, t.noise, t.budget) = (SENSITIVITY, NOISE, u64::MAX);
            t.colluding = 3;
        },
        LedgerError::InvalidCollusion,
    );
    refused(
        &mut world,
        |t| {
            (t.sensitivity, t.noise) = (SENSITIVITY, NOISE);
            t.budget = 733018936705;
        },
        LedgerError::BudgetTooSmall,
    );
    refused(
        &mut world,
        |t| (t.sensitivity, t.noise, t.budget) = (1 << 44, NOISE, u64::MAX),
        LedgerError::CostOverflow,
    );
}

#[test]
fn publish_checks_its_accounts_and_encoding() {
    let mut world = World::new();
    let publisher = world.wallet();
    let who = key(&publisher);
    let good = terms(1, |_| {});

    let mut unsigned = client::publish(&who, &good);
    unsigned.accounts[0].is_signer = false;
    world.refuse(unsigned, &[], "MissingRequiredSignature");

    let mut elsewhere = client::publish(&who, &good);
    elsewhere.accounts[1].address = client::task(&who, &[9; 32]);
    world.refuse(elsewhere, &[&publisher], "InvalidSeeds");

    let mut readonly = client::publish(&who, &good);
    readonly.accounts[1].is_writable = false;
    world.refuse(readonly, &[&publisher], "Immutable");

    let mut not_system = client::publish(&who, &good);
    not_system.accounts[2].address = who;
    world.refuse(not_system, &[&publisher], "IncorrectProgramId");

    let mut trailing = client::publish(&who, &good);
    trailing.data.push(0);
    world.refuse(trailing, &[&publisher], "InvalidInstructionData");

    let mut short = client::publish(&who, &good);
    short.data.pop();
    world.refuse(short, &[&publisher], "InvalidInstructionData");

    let mut unknown = client::publish(&who, &good);
    unknown.data[0] ^= 1;
    world.refuse(unknown, &[&publisher], "InvalidInstructionData");

    let mut empty = client::publish(&who, &good);
    empty.data.clear();
    world.refuse(empty, &[&publisher], "InvalidInstructionData");

    let too_rich = terms(1, |t| (t.rounds, t.reward_per_round) = (3, u64::MAX / 2));
    world.refuse(
        client::publish(&who, &too_rich),
        &[&publisher],
        "ArithmeticOverflow",
    );

    world.run(client::publish(&who, &good), &[&publisher]);
    world.refuse(
        client::publish(&who, &good),
        &[&publisher],
        "AccountAlreadyInitialized",
    );
}

#[test]
fn publish_arguments_are_laid_out_in_specification_order() {
    let terms = TaskTerms {
        aggregator: [1; 32],
        plan_id: [2; 32],
        sites_root: [3; 32],
        model: [4; 32],
        sites: 0x0105,
        helpers: 6,
        coords: 7,
        rounds: 8,
        min_cohort: 9,
        sample: 10,
        sensitivity: 11,
        noise: 12,
        colluding: 13,
        budget: 14,
        site_stake: 15,
        aggregator_stake: 16,
        reward_per_round: 17,
        dispute_bond: 18,
        challenge_slots: 19,
        stall_slots: 20,
    };
    let mut expected = vec![129, 177, 182, 160, 184, 224, 219, 5];
    for fill in 1..=4u8 {
        expected.extend([fill; 32]);
    }
    expected.extend([0x05, 0x01, 6]);
    expected.extend(7u64.to_le_bytes());
    expected.extend(8u32.to_le_bytes());
    expected.extend([9, 0, 10, 0]);
    expected.extend(11u64.to_le_bytes());
    expected.extend(12u64.to_le_bytes());
    expected.extend([13, 0]);
    for value in 14..=20u64 {
        expected.extend(value.to_le_bytes());
    }
    assert_eq!(client::publish(&[0; 32], &terms).data, expected);
}

#[test]
fn back_is_the_aggregators_and_happens_once() {
    let mut run = Run::new(3, |_| {});
    run.publish();
    let back = client::back(&key(&run.aggregator), &run.task);

    let stranger = run.world.wallet();
    run.world.refuse(
        client::back(&key(&stranger), &run.task),
        &[&stranger],
        LedgerError::Unauthorized,
    );
    let mut unsigned = back.clone();
    unsigned.accounts[0].is_signer = false;
    run.world.refuse(unsigned, &[], "MissingRequiredSignature");
    let mut trailing = back.clone();
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&run.aggregator], "InvalidInstructionData");

    let before = run.world.balance(&key(&run.aggregator));
    run.world.run(back.clone(), &[&run.aggregator]);
    assert_eq!(
        run.world.balance(&key(&run.aggregator)),
        before - run.terms.aggregator_stake
    );
    run.world
        .refuse(back, &[&run.aggregator], LedgerError::AlreadyBacked);
}

#[test]
fn back_needs_an_open_task() {
    let mut run = Run::new(3, |_| {});
    run.publish();
    let ix = run.abort_ix(&run.publisher);
    run.world.run(ix, &[&run.publisher]);
    run.world.refuse(
        client::back(&key(&run.aggregator), &run.task),
        &[&run.aggregator],
        LedgerError::TaskNotOpen,
    );
}

#[test]
fn enroll_checks_the_wallets_place_in_the_sites_root() {
    let mut run = Run::new(5, |_| {});
    run.publish();
    let wallet = |index: usize| key(&run.sites[index]);
    let enroll = |wallet: [u8; 32], index: u16, path: &[[u8; 32]]| {
        client::enroll(&wallet, &run.task, index, path, &admission(index), &[])
    };
    let path = run.site_path(2);

    // The path of another leaf, another wallet's leaf, a path one hash
    // short or long, and an index past the last site.
    let cases = [
        (enroll(wallet(2), 2, &run.site_path(3)), 2),
        (enroll(wallet(3), 2, &path), 3),
        (enroll(wallet(2), 2, &path[..path.len() - 1]), 2),
        (
            enroll(wallet(2), 2, &[path.clone(), vec![[0; 32]]].concat()),
            2,
        ),
        (enroll(wallet(2), 5, &path), 2),
        (enroll(wallet(2), 3, &path), 2),
    ];
    for (ix, signer) in cases {
        run.world
            .refuse(ix, &[&run.sites[signer]], LedgerError::InvalidSiteProof);
    }

    run.world.run(enroll(wallet(2), 2, &path), &[&run.sites[2]]);
    assert_eq!(run.state().enrolled(), &bits(&[2]));
    run.world.refuse(
        enroll(wallet(2), 2, &path),
        &[&run.sites[2]],
        "AccountAlreadyInitialized",
    );

    let mut unsigned = run.enroll_ix(4, &[]);
    unsigned.accounts[0].is_signer = false;
    run.world.refuse(unsigned, &[], "MissingRequiredSignature");
    let mut wrong_account = run.enroll_ix(4, &[]);
    wrong_account.accounts[2].address = client::enrollment(&run.task, 3);
    run.world
        .refuse(wrong_account, &[&run.sites[4]], "InvalidSeeds");
    let mut trailing = run.enroll_ix(4, &[]);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&run.sites[4]], "InvalidInstructionData");
}

#[test]
fn enroll_paths_verify_in_trees_of_many_shapes() {
    for sites in [1, 2, 3, 4, 5, 6, 7, 9, 13] {
        let mut run = Run::new(sites, |t| t.min_cohort = 1);
        run.publish();
        for index in 0..sites as u16 {
            let other = (index + 1) % sites as u16;
            if sites > 1 {
                let mut ix = run.enroll_ix(index, &[]);
                ix.accounts[2].address = client::enrollment(&run.task, other);
                ix.data[8..10].copy_from_slice(&other.to_le_bytes());
                run.world.refuse(
                    ix,
                    &[&run.sites[index as usize]],
                    LedgerError::InvalidSiteProof,
                );
            }
            run.enroll(index);
        }
        assert_eq!(bitmap_count(run.state().enrolled()), sites);
    }
}

fn bitmap_count(bits: &[u8]) -> usize {
    bits.iter().map(|byte| byte.count_ones() as usize).sum()
}

#[test]
fn enroll_records_up_to_four_status_references() {
    let mut run = Run::new(3, |_| {});
    run.publish();
    let refs: Vec<StatusRef> = (0..5)
        .map(|i| StatusRef {
            list: [i + 1; 32],
            bit: 1000 + i as u32,
        })
        .collect();

    let five = run.enroll_ix(0, &refs);
    run.world
        .refuse(five, &[&run.sites[0]], LedgerError::TooManyStatusRefs);

    run.enroll_with(0, &refs[..4]);
    let site = run.world.enrollment(&run.task, 0);
    assert_eq!(site.status_ref_count(), 4);
    for (i, status_ref) in refs[..4].iter().enumerate() {
        assert_eq!(site.status_ref(i), (&status_ref.list, status_ref.bit));
    }

    run.enroll_with(1, &refs[..1]);
    let site = run.world.enrollment(&run.task, 1);
    assert_eq!(site.status_ref_count(), 1);
    assert_eq!(site.status_refs()[36..], [0; 108]);
}

#[test]
fn enroll_needs_an_open_task() {
    let mut run = Run::new(3, |_| {});
    run.publish();
    run.enroll(0);
    let ix = run.abort_ix(&run.publisher);
    run.world.run(ix, &[&run.publisher]);
    let ix = run.enroll_ix(1, &[]);
    run.world
        .refuse(ix, &[&run.sites[1]], LedgerError::TaskNotOpen);
}

#[test]
fn open_round_needs_a_backed_task_and_a_settled_previous_round() {
    let mut run = Run::new(3, |_| {});
    run.publish();
    for index in 0..3 {
        run.enroll(index);
    }
    let aggregator = key(&run.aggregator);
    let task = run.task;
    let open = |number: u32| client::open_round(&aggregator, &task, number);

    run.world
        .refuse(open(1), &[&run.aggregator], LedgerError::NotBacked);
    run.back();

    let stranger = run.world.wallet();
    run.world.refuse(
        client::open_round(&key(&stranger), &run.task, 1),
        &[&stranger],
        LedgerError::Unauthorized,
    );
    run.world
        .refuse(open(2), &[&run.aggregator], "InvalidSeeds");
    let mut trailing = open(1);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&run.aggregator], "InvalidInstructionData");

    run.world.run(open(1), &[&run.aggregator]);
    run.world.refuse(
        open(2),
        &[&run.aggregator],
        LedgerError::PreviousRoundNotFinal,
    );

    let posting = Posting::filler(1, 1);
    run.commit(1, &[0, 1], &posting);
    run.post(1, &posting);
    run.world.refuse(
        open(2),
        &[&run.aggregator],
        LedgerError::PreviousRoundNotFinal,
    );
    run.ack(0, 1);
    run.ack(1, 1);
    run.world.warp(100);
    run.settle(1);

    // Round 2 is the task's last. A third cannot open while it runs, nor
    // once it has settled and finished the task.
    run.world.run(open(2), &[&run.aggregator]);
    run.world
        .refuse(open(3), &[&run.aggregator], LedgerError::NoRoundsLeft);
    let posting = Posting::filler(2, 1);
    run.commit(2, &[1, 2], &posting);
    run.post(2, &posting);
    run.ack(1, 2);
    run.ack(2, 2);
    run.world.warp(100);
    run.settle(2);
    run.world
        .refuse(open(3), &[&run.aggregator], LedgerError::TaskNotOpen);
}

#[test]
fn commit_root_checks_the_cohort() {
    let mut run = Run::new(6, |t| t.min_cohort = 3);
    run.publish();
    run.back();
    for index in 0..5 {
        run.enroll(index);
    }
    let number = run.open_round();
    let posting = Posting::filler(1, 1);
    let commit = |run: &Run, cohort: &[u16]| run.commit_ix(number, cohort, &posting, &[0; 32]);

    // Too small, a site that never enrolled, and bits past the last site.
    run.world.refuse(
        commit(&run, &[0, 1]),
        &[&run.aggregator],
        LedgerError::InvalidCohortSize,
    );
    run.world.refuse(
        commit(&run, &[]),
        &[&run.aggregator],
        LedgerError::InvalidCohortSize,
    );
    run.world.refuse(
        commit(&run, &[0, 1, 5]),
        &[&run.aggregator],
        LedgerError::InvalidCohortMember,
    );
    run.world.refuse(
        commit(&run, &[0, 1, 6]),
        &[&run.aggregator],
        LedgerError::InvalidCohortMember,
    );
    run.world.refuse(
        commit(&run, &[0, 1, 2, 1023]),
        &[&run.aggregator],
        LedgerError::InvalidCohortMember,
    );
    run.world.refuse(
        commit(&run, &[0, 1, 2, 3, 4, 5, 6]),
        &[&run.aggregator],
        LedgerError::InvalidCohortSize,
    );

    let stranger = run.world.wallet();
    let mut by_stranger = commit(&run, &[0, 1, 2]);
    by_stranger.accounts[0].address = key(&stranger);
    run.world
        .refuse(by_stranger, &[&stranger], LedgerError::Unauthorized);
    let mut unsigned = commit(&run, &[0, 1, 2]);
    unsigned.accounts[0].is_signer = false;
    run.world.refuse(unsigned, &[], "MissingRequiredSignature");
    let mut trailing = commit(&run, &[0, 1, 2]);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&run.aggregator], "InvalidInstructionData");
    let mut readonly = commit(&run, &[0, 1, 2]);
    readonly.accounts[2].is_writable = false;
    run.world.refuse(readonly, &[&run.aggregator], "Immutable");

    run.world.run(commit(&run, &[0, 1, 2]), &[&run.aggregator]);
    run.world.refuse(
        commit(&run, &[0, 1, 2, 3]),
        &[&run.aggregator],
        LedgerError::InvalidRoundState,
    );
}

/// A second task of the same aggregator in the same world, with a round
/// open, to pass where the first task's accounts belong.
fn other_task(run: &mut Run) -> [u8; 32] {
    let terms = TaskTerms {
        plan_id: [0x52; 32],
        ..run.terms.clone()
    };
    let task = client::task(&key(&run.publisher), &terms.plan_id);
    let aggregator = key(&run.aggregator);
    run.world.run(
        client::publish(&key(&run.publisher), &terms),
        &[&run.publisher],
    );
    run.world
        .run(client::back(&aggregator, &task), &[&run.aggregator]);
    run.world.run(
        client::open_round(&aggregator, &task, 1),
        &[&run.aggregator],
    );
    task
}

#[test]
fn accounts_of_another_task_are_refused() {
    let mut run = Run::started(3, |_| {});
    let number = run.open_round();
    let other = other_task(&mut run);
    let posting = Posting::filler(1, 1);

    // The other task's round under this task, and an enrollment where a
    // round belongs.
    let mut commit = run.commit_ix(number, &[0, 1], &posting, &[0; 32]);
    commit.accounts[2].address = client::round(&other, 1);
    run.world
        .refuse(commit, &[&run.aggregator], LedgerError::AccountMismatch);
    let mut commit = run.commit_ix(number, &[0, 1], &posting, &[0; 32]);
    commit.accounts[2].address = client::enrollment(&run.task, 0);
    run.world
        .refuse(commit, &[&run.aggregator], "InvalidAccountData");
    let mut commit = run.commit_ix(number, &[0, 1], &posting, &[0; 32]);
    commit.accounts[1].address = key(&run.publisher);
    run.world
        .refuse(commit, &[&run.aggregator], "InvalidAccountOwner");

    run.commit(number, &[0, 1], &posting);
    let mut post = run.post_ix(number, &posting);
    post.accounts[2].address = client::round(&other, 1);
    run.world
        .refuse(post, &[&run.aggregator], LedgerError::AccountMismatch);
    run.post(number, &posting);

    let mut settle = client::settle(&run.task, number);
    settle.accounts[1].address = client::round(&other, 1);
    run.world.refuse(settle, &[], LedgerError::AccountMismatch);

    let mut abort = run.abort_ix(&run.publisher);
    abort.accounts[2].address = client::round(&other, 1);
    run.world
        .refuse(abort, &[&run.publisher], LedgerError::AccountMismatch);
}

#[test]
fn post_aggregate_takes_one_mask_root_per_helper() {
    let mut run = Run::started(3, |t| t.helpers = 2);
    let number = run.open_round();
    let posting = Posting::filler(1, 2);

    run.world.refuse(
        run.post_ix(number, &posting),
        &[&run.aggregator],
        LedgerError::InvalidRoundState,
    );
    run.commit(number, &[0, 2], &posting);

    for helpers in [0, 1, 3] {
        let wrong = Posting {
            mask_roots: vec![[7; 32]; helpers],
            ..posting.clone()
        };
        run.world.refuse(
            run.post_ix(number, &wrong),
            &[&run.aggregator],
            LedgerError::InvalidMaskRoots,
        );
    }
    let stranger = run.world.wallet();
    let mut by_stranger = run.post_ix(number, &posting);
    by_stranger.accounts[0].address = key(&stranger);
    run.world
        .refuse(by_stranger, &[&stranger], LedgerError::Unauthorized);
    let mut trailing = run.post_ix(number, &posting);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&run.aggregator], "InvalidInstructionData");

    run.world.warp(5);
    run.post(number, &posting);
    let round = run.round(number);
    assert_eq!(round.posted_slot(), 5);
    assert_eq!(round.mask_root(0), &posting.mask_roots[0]);
    assert_eq!(round.mask_root(1), &posting.mask_roots[1]);
    run.world.refuse(
        run.post_ix(number, &posting),
        &[&run.aggregator],
        LedgerError::InvalidRoundState,
    );
}

#[test]
fn ack_is_for_cohort_members_who_saw_the_posted_seal() {
    let mut run = Run::started(4, |_| {});
    let number = run.open_round();
    let posting = Posting::filler(1, 1);
    run.commit(number, &[0, 1, 2], &posting);
    let seal = posting.seal_head;

    run.world.refuse(
        run.ack_ix(0, number, &seal),
        &[&run.sites[0]],
        LedgerError::InvalidRoundState,
    );
    run.post(number, &posting);

    run.world.refuse(
        run.ack_ix(0, number, &posting.close_head),
        &[&run.sites[0]],
        LedgerError::SealMismatch,
    );
    run.world.refuse(
        run.ack_ix(3, number, &seal),
        &[&run.sites[3]],
        LedgerError::NotInCohort,
    );
    let mut wrong_number = run.ack_ix(0, number, &seal);
    wrong_number.data[8..12].copy_from_slice(&2u32.to_le_bytes());
    run.world
        .refuse(wrong_number, &[&run.sites[0]], LedgerError::AccountMismatch);
    let mut for_another = run.ack_ix(0, number, &seal);
    for_another.accounts[0].address = key(&run.sites[1]);
    run.world
        .refuse(for_another, &[&run.sites[1]], LedgerError::Unauthorized);
    let mut unsigned = run.ack_ix(0, number, &seal);
    unsigned.accounts[0].is_signer = false;
    run.world.refuse(unsigned, &[], "MissingRequiredSignature");
    let mut trailing = run.ack_ix(0, number, &seal);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&run.sites[0]], "InvalidInstructionData");

    run.world
        .run(run.ack_ix(0, number, &seal), &[&run.sites[0]]);
    assert_eq!(run.round(number).acked(), &bits(&[0]));
    run.world.refuse(
        run.ack_ix(0, number, &seal),
        &[&run.sites[0]],
        LedgerError::AlreadyAcked,
    );
}

#[test]
fn settle_waits_for_every_ack_and_the_challenge_window() {
    let mut run = Run::started(3, |_| {});
    let posting = Posting::filler(1, 1);
    let number = run.open_round();
    let settle = client::settle(&run.task, number);

    run.commit(number, &[0, 1, 2], &posting);
    run.world
        .refuse(settle.clone(), &[], LedgerError::InvalidRoundState);
    run.world.warp(10);
    run.post(number, &posting);

    run.ack(0, number);
    run.ack(1, number);
    run.world.warp(100);
    run.world
        .refuse(settle.clone(), &[], LedgerError::AcksMissing);
    run.ack(2, number);
    run.settle(number);
    run.world
        .refuse(settle, &[], LedgerError::InvalidRoundState);

    // The window closes exactly `challenge_slots` after the post.
    let posting = Posting::filler(2, 1);
    let number = run.post_round(&[0, 1], &posting);
    let settle = client::settle(&run.task, number);
    run.ack(0, number);
    run.ack(1, number);
    run.world.warp(99);
    run.world
        .refuse(settle.clone(), &[], LedgerError::ChallengeWindowOpen);
    let mut trailing = settle.clone();
    trailing.data.push(0);
    run.world.refuse(trailing, &[], "InvalidInstructionData");
    run.world.warp(1);
    run.world.run(settle, &[]);
    assert_eq!(run.round(number).share(), 5_000_000);
    assert_eq!(run.state().state(), TASK_FINISHED);
}

#[test]
fn settle_needs_an_open_task() {
    let mut run = Run::started(3, |_| {});
    let posting = Posting::filler(1, 1);
    let number = run.post_round(&[0, 1], &posting);
    run.ack(0, number);
    run.ack(1, number);
    run.world.warp(100);
    let ix = run.abort_ix(&run.aggregator);
    run.world.run(ix, &[&run.aggregator]);
    run.world.refuse(
        client::settle(&run.task, number),
        &[],
        LedgerError::TaskNotOpen,
    );
}

#[test]
fn claim_pays_each_cohort_member_once() {
    let mut run = Run::started(4, |_| {});
    let posting = Posting::filler(1, 1);
    let number = run.post_round(&[0, 1, 2], &posting);
    for index in 0..3 {
        run.ack(index, number);
    }
    run.world.refuse(
        run.claim_ix(0, number),
        &[&run.sites[0]],
        LedgerError::InvalidRoundState,
    );
    run.world.warp(100);
    run.settle(number);

    run.world.refuse(
        run.claim_ix(3, number),
        &[&run.sites[3]],
        LedgerError::NotInCohort,
    );
    let mut for_another = run.claim_ix(0, number);
    for_another.accounts[0].address = key(&run.sites[1]);
    run.world
        .refuse(for_another, &[&run.sites[1]], LedgerError::Unauthorized);
    let mut unsigned = run.claim_ix(0, number);
    unsigned.accounts[0].is_signer = false;
    run.world.refuse(unsigned, &[], "MissingRequiredSignature");
    let mut trailing = run.claim_ix(0, number);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&run.sites[0]], "InvalidInstructionData");
    let other = other_task(&mut run);
    let mut other_round = run.claim_ix(0, number);
    other_round.accounts[3].address = client::round(&other, 1);
    run.world
        .refuse(other_round, &[&run.sites[0]], LedgerError::AccountMismatch);

    let share = run.round(number).share();
    let (before, held) = (
        run.world.balance(&key(&run.sites[0])),
        run.world.balance(&run.task),
    );
    run.claim(0, number);
    assert_eq!(run.world.balance(&key(&run.sites[0])), before + share);
    assert_eq!(run.world.balance(&run.task), held - share);
    assert_eq!(run.state().owed(), 2 * share);
    run.world.refuse(
        run.claim_ix(0, number),
        &[&run.sites[0]],
        LedgerError::AlreadyClaimed,
    );
}

#[test]
fn the_privacy_budget_runs_out() {
    let cost = |cohort| reference::round_cost(SENSITIVITY, NOISE, 0, cohort).unwrap();
    let mut run = Run::started(4, |t| {
        (t.sensitivity, t.noise, t.min_cohort, t.rounds) = (SENSITIVITY, NOISE, 3, 5);
        t.budget = cost(4) + cost(3) + cost(3) - 1;
    });

    // A round is charged for the cohort that showed up.
    run.settle_round(&[0, 1, 2, 3]);
    assert_eq!(run.state().spent(), cost(4));
    run.settle_round(&[1, 2, 3]);
    assert_eq!(run.state().spent(), cost(4) + cost(3));

    // What is left is one short of a round at the minimum cohort.
    let open = client::open_round(&key(&run.aggregator), &run.task, 3);
    run.world
        .refuse(open, &[&run.aggregator], LedgerError::BudgetExhausted);
    assert_eq!(run.state().round(), 2);
}

#[test]
fn a_budget_that_covers_the_rounds_exactly_is_enough() {
    let cost = |cohort| reference::round_cost(SENSITIVITY, NOISE, 1, cohort).unwrap();
    let mut run = Run::started(4, |t| {
        (t.sensitivity, t.noise, t.colluding) = (SENSITIVITY, NOISE, 1);
        (t.min_cohort, t.rounds) = (2, 2);
        t.budget = 2 * cost(2);
    });
    run.settle_round(&[0, 1]);
    run.settle_round(&[2, 3]);
    let task = run.state();
    assert_eq!((task.spent(), task.budget()), (2 * cost(2), 2 * cost(2)));
    assert_eq!(task.state(), TASK_FINISHED);
}

#[test]
fn abort_is_for_the_aggregator_the_publisher_or_anyone_after_a_stall() {
    for by_aggregator in [true, false] {
        let mut run = Run::started(3, |_| {});
        let stranger = run.world.wallet();
        run.world.warp(1_000);
        run.world.refuse(
            run.abort_ix(&stranger),
            &[&stranger],
            LedgerError::NotStalled,
        );
        let mut unsigned = run.abort_ix(&run.publisher);
        unsigned.accounts[0].is_signer = false;
        run.world.refuse(unsigned, &[], "MissingRequiredSignature");
        let mut trailing = run.abort_ix(&run.publisher);
        trailing.data.push(0);
        run.world
            .refuse(trailing, &[&run.publisher], "InvalidInstructionData");

        let caller = if by_aggregator {
            &run.aggregator
        } else {
            &run.publisher
        };
        run.world.run(run.abort_ix(caller), &[caller]);
        let task = run.state();
        assert_eq!((task.state(), task.fraud()), (TASK_ABORTED, false));
        assert_eq!(task.last_change(), 1_000);
        run.world
            .refuse(run.abort_ix(caller), &[caller], LedgerError::TaskNotOpen);
    }
}

#[test]
fn abort_looks_at_the_latest_round() {
    let mut run = Run::started(3, |_| {});
    run.settle_round(&[0, 1]);
    run.open_round();

    let mut without_round = run.abort_ix(&run.publisher);
    without_round.accounts.pop();
    run.world
        .refuse(without_round, &[&run.publisher], "NotEnoughAccountKeys");
    let mut older_round = run.abort_ix(&run.publisher);
    older_round.accounts[2].address = client::round(&run.task, 1);
    run.world
        .refuse(older_round, &[&run.publisher], LedgerError::AccountMismatch);
    run.world
        .run(run.abort_ix(&run.publisher), &[&run.publisher]);
}

#[test]
fn a_stalled_task_can_be_aborted_by_anyone_and_pays_everyone_back() {
    let mut run = Run::started(3, |_| {});
    run.settle_round(&[0, 1, 2]);
    let share = run.round(1).share();

    // Round 2 is posted, and site 2 never acknowledges it.
    let posting = Posting::filler(2, 1);
    let number = run.post_round(&[0, 1, 2], &posting);
    run.ack(0, number);
    run.ack(1, number);
    let posted = run.round(number).posted_slot();
    assert_eq!(run.state().last_change(), posted);

    let stranger = run.world.wallet();
    run.world.warp(1_000);
    run.world.refuse(
        run.abort_ix(&stranger),
        &[&stranger],
        LedgerError::NotStalled,
    );
    run.world.warp(1);
    run.world.run(run.abort_ix(&stranger), &[&stranger]);
    assert_eq!(run.state().state(), TASK_ABORTED);
    assert_eq!(run.round(number).state(), ROUND_POSTED);

    // The unfinished round pays nothing. The first round's shares are still
    // owed after the publisher has taken the rest of the escrow back.
    run.world.refuse(
        run.claim_ix(0, number),
        &[&run.sites[0]],
        LedgerError::InvalidRoundState,
    );
    let task_rent = run.world.rent(TASK_SPACE);
    run.world
        .run(run.withdraw_publisher_ix(), &[&run.publisher]);
    assert_eq!(
        run.world.balance(&run.task),
        task_rent + 3 * share + run.terms.aggregator_stake
    );
    run.world
        .run(run.withdraw_aggregator_ix(), &[&run.aggregator]);
    for index in 0..3 {
        run.world
            .run(run.withdraw_site_ix(index), &[&run.sites[index as usize]]);
        run.claim(index, 1);
        assert_eq!(
            run.world.balance(&key(&run.sites[index as usize])),
            WALLET - run.world.rent(ENROLLMENT_SPACE) + share
        );
    }
    assert_eq!(run.world.balance(&run.task), task_rent);
    assert_eq!(
        run.world.balance(&key(&run.publisher)),
        WALLET - task_rent - 3 * share
    );
    assert_eq!(
        run.world.balance(&key(&run.aggregator)),
        WALLET - 2 * run.world.rent(ROUND_SPACE)
    );
}

#[test]
fn withdrawals_wait_for_the_task_to_end() {
    let mut run = Run::started(3, |t| t.rounds = 1);
    let site = &run.sites[0];
    run.world
        .refuse(run.withdraw_site_ix(0), &[site], LedgerError::FundsLocked);
    run.world.refuse(
        run.withdraw_aggregator_ix(),
        &[&run.aggregator],
        LedgerError::FundsLocked,
    );
    run.world.refuse(
        run.withdraw_publisher_ix(),
        &[&run.publisher],
        LedgerError::FundsLocked,
    );

    run.settle_round(&[0, 1, 2]);
    assert_eq!(run.state().state(), TASK_FINISHED);

    // Each withdrawal is for its own wallet and role.
    let stranger = run.world.wallet();
    let mut for_another = run.withdraw_site_ix(0);
    for_another.accounts[0].address = key(&stranger);
    run.world
        .refuse(for_another, &[&stranger], LedgerError::Unauthorized);
    let mut for_another = run.withdraw_aggregator_ix();
    for_another.accounts[0].address = key(&stranger);
    run.world
        .refuse(for_another, &[&stranger], LedgerError::Unauthorized);
    let mut for_another = run.withdraw_publisher_ix();
    for_another.accounts[0].address = key(&stranger);
    run.world
        .refuse(for_another, &[&stranger], LedgerError::Unauthorized);
    let mut unknown_role = run.withdraw_publisher_ix();
    *unknown_role.data.last_mut().unwrap() = 3;
    run.world
        .refuse(unknown_role, &[&run.publisher], LedgerError::InvalidRole);
    let mut trailing = run.withdraw_publisher_ix();
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&run.publisher], "InvalidInstructionData");
    let mut unsigned = run.withdraw_publisher_ix();
    unsigned.accounts[0].is_signer = false;
    run.world.refuse(unsigned, &[], "MissingRequiredSignature");
    let mut wrong_site = run.withdraw_site_ix(0);
    wrong_site.accounts[2].address = client::enrollment(&run.task, 1);
    run.world
        .refuse(wrong_site, &[&run.sites[0]], LedgerError::Unauthorized);

    let site = &run.sites[0];
    run.world.run(run.withdraw_site_ix(0), &[site]);
    run.world.refuse(
        run.withdraw_site_ix(0),
        &[site],
        LedgerError::AlreadyWithdrawn,
    );
    run.world
        .run(run.withdraw_aggregator_ix(), &[&run.aggregator]);
    run.world.refuse(
        run.withdraw_aggregator_ix(),
        &[&run.aggregator],
        LedgerError::NothingToWithdraw,
    );

    // The publisher gets the indivisible remainder, and the shares stay.
    let before = run.world.balance(&key(&run.publisher));
    run.world
        .run(run.withdraw_publisher_ix(), &[&run.publisher]);
    assert_eq!(run.world.balance(&key(&run.publisher)), before + 1);
    run.world.refuse(
        run.withdraw_publisher_ix(),
        &[&run.publisher],
        LedgerError::NothingToWithdraw,
    );
    assert_eq!(
        run.world.balance(&run.task),
        run.world.rent(TASK_SPACE) + 9_999_999
    );
    for index in 0..3 {
        run.claim(index, 1);
    }
}

#[test]
fn an_aborted_task_before_any_round_returns_everything() {
    let mut run = Run::started(2, |_| {});
    run.world
        .run(run.abort_ix(&run.aggregator), &[&run.aggregator]);
    run.world
        .run(run.withdraw_aggregator_ix(), &[&run.aggregator]);
    run.world
        .run(run.withdraw_publisher_ix(), &[&run.publisher]);
    run.world.run(run.withdraw_site_ix(1), &[&run.sites[1]]);
    assert_eq!(run.world.balance(&run.task), run.world.rent(TASK_SPACE));
    assert_eq!(run.world.balance(&key(&run.aggregator)), WALLET);
    assert_eq!(
        run.world.balance(&key(&run.publisher)),
        WALLET - run.world.rent(TASK_SPACE)
    );
}

#[test]
fn an_enrollment_of_another_task_is_refused() {
    let mut run = Run::started(3, |t| t.rounds = 1);
    let other = other_task(&mut run);
    let site = &run.sites[0].insecure_clone();
    let elsewhere = client::enrollment(&other, 0);
    let enroll = client::enroll(&key(site), &other, 0, &run.site_path(0), &admission(0), &[]);
    run.world.run(enroll, &[site]);

    let posting = Posting::filler(1, 1);
    let number = run.post_round(&[0, 1, 2], &posting);
    let mut ack = run.ack_ix(0, number, &posting.seal_head);
    ack.accounts[1].address = elsewhere;
    run.world.refuse(ack, &[site], LedgerError::AccountMismatch);
    for index in 0..3 {
        run.ack(index, number);
    }
    run.world.warp(100);
    run.settle(number);

    let mut claim = run.claim_ix(0, number);
    claim.accounts[1].address = elsewhere;
    run.world
        .refuse(claim, &[site], LedgerError::AccountMismatch);
    let mut withdraw = run.withdraw_site_ix(0);
    withdraw.accounts[2].address = elsewhere;
    run.world
        .refuse(withdraw, &[site], LedgerError::AccountMismatch);
    let mut check = client::revoke_check(&run.task, 0, &[]);
    check.accounts[1].address = elsewhere;
    run.world.refuse(check, &[], LedgerError::AccountMismatch);
}

#[test]
fn the_signer_of_each_step_is_required() {
    let mut run = Run::started(3, |t| t.rounds = 1);
    let unsigned = |mut ix: client::Instruction| {
        ix.accounts[0].is_signer = false;
        ix
    };
    let open = client::open_round(&key(&run.aggregator), &run.task, 1);
    run.world
        .refuse(unsigned(open), &[], "MissingRequiredSignature");

    let posting = Posting::filler(1, 1);
    let number = run.open_round();
    run.commit(number, &[0, 1, 2], &posting);
    run.world.refuse(
        unsigned(run.post_ix(number, &posting)),
        &[],
        "MissingRequiredSignature",
    );
    run.post(number, &posting);
    for index in 0..3 {
        run.ack(index, number);
    }
    run.world.warp(100);
    run.settle(number);

    for ix in [run.withdraw_site_ix(0), run.withdraw_aggregator_ix()] {
        run.world
            .refuse(unsigned(ix), &[], "MissingRequiredSignature");
    }
}

#[test]
fn an_address_that_was_funded_ahead_still_becomes_the_account() {
    let mut run = Run::new(3, |_| {});
    let rent = run.world.rent(ENROLLMENT_SPACE);
    let stake = run.terms.site_stake;
    assert!(rent > 1_000_000);
    run.publish();

    // Someone sends lamports to the addresses two enrollments will have,
    // less than the rent to one and more to the other. Neither gift counts
    // towards the stake.
    for (index, gift) in [(0u16, 1_000_000u64), (1, rent + 5)] {
        let address = client::enrollment(&run.task, index);
        run.world
            .svm
            .airdrop(&solana_pubkey::Pubkey::new_from_array(address), gift)
            .unwrap();
        let before = run.world.balance(&key(&run.sites[index as usize]));
        run.enroll(index);
        let held = run.world.balance(&address);
        assert_eq!(held, rent.max(gift) + stake);
        assert_eq!(
            run.world.balance(&key(&run.sites[index as usize])),
            before - (held - gift)
        );
        let site = run.world.enrollment(&run.task, index);
        assert_eq!((site.status(), site.stake()), (SITE_ACTIVE, stake));
    }
    assert_eq!(run.state().enrolled(), &bits(&[0, 1]));
}

#[test]
fn a_plan_id_cannot_be_taken_by_another_publisher() {
    let mut run = Run::new(3, |_| {});
    let squatter = run.world.wallet();
    let theirs = TaskTerms {
        aggregator: key(&squatter),
        reward_per_round: 0,
        ..run.terms.clone()
    };
    run.world
        .run(client::publish(&key(&squatter), &theirs), &[&squatter]);

    // The publisher's task sits at an address of its own.
    run.publish();
    assert_ne!(run.task, client::task(&key(&squatter), &run.terms.plan_id));
    assert_eq!(run.state().publisher(), &key(&run.publisher));
    assert_eq!(run.state().aggregator(), &key(&run.aggregator));

    let mut into_theirs = client::publish(&key(&run.publisher), &run.terms);
    into_theirs.accounts[1].address = client::task(&key(&squatter), &run.terms.plan_id);
    run.world
        .refuse(into_theirs, &[&run.publisher], "InvalidSeeds");
}
