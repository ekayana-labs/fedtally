//! The dispute path, over real vectors and Merkle trees built in the test:
//! fraud proofs that slash the aggregator, disputes that fail and cost the
//! challenger the bond, and the rules of the dispute and buffer
//! instructions.

mod common;

use common::{key, reference, vectors::Vectors, Run, WALLET};
use fedtally_ledger::{
    client::{self, ProofPart},
    error::LedgerError,
    state::*,
};
use solana_keypair::Keypair;

/// Vectors of three leaves: two full ones and a last one of 40 coordinates.
const COORDS: usize = 296;

/// A task whose first round is posted with `vectors` by the whole cohort.
fn posted(members: usize, helpers: u8, vectors: &Vectors) -> (Run, u32) {
    let mut run = Run::started(members, |t| t.helpers = helpers);
    let cohort: Vec<u16> = (0..members as u16).collect();
    let number = run.post_round(&cohort, &vectors.posting(1));
    (run, number)
}

fn close_buffer(run: &mut Run, challenger: &Keypair) {
    run.world
        .run(client::buffer_close(&key(challenger)), &[challenger]);
}

/// Posts an aggregate that is wrong in one coordinate of leaf `leaf_index`
/// and proves it.
fn prove_fraud(members: usize, helpers: u8, leaf_index: usize, coordinate: usize) {
    let vectors = Vectors::honest(members, helpers as usize, COORDS, 77 + members as u64)
        .with_wrong_aggregate(coordinate);
    let (mut run, number) = posted(members, helpers, &vectors);
    for index in 0..members as u16 {
        run.ack(index, number);
    }
    let challenger = run.challenger();
    let task_rent = run.world.rent(TASK_SPACE);

    run.world.warp(40);
    run.dispute_open(&challenger, number, leaf_index as u64);
    run.give_all(&challenger, number, leaf_index, &vectors);
    let dispute = run.world.dispute(&run.task, number, &challenger);
    assert!(dispute.is_complete(members as u16, helpers));
    assert!(!dispute.sums_match());

    run.world.warp(5);
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);
    close_buffer(&mut run, &challenger);

    let (task, round) = (run.state(), run.round(number));
    assert_eq!((round.state(), round.disputes()), (ROUND_FRAUD, 0));
    assert_eq!((task.state(), task.fraud()), (TASK_ABORTED, true));
    assert_eq!((task.stake_held(), task.last_change()), (0, 45));
    assert!(!run
        .world
        .exists(&client::dispute(&run.task, number, &key(&challenger))));

    // The challenger has the bond back and the aggregator's stake on top.
    assert_eq!(
        run.world.balance(&key(&challenger)),
        WALLET + run.terms.aggregator_stake
    );
    assert_eq!(
        run.world.balance(&key(&run.aggregator)),
        WALLET - run.terms.aggregator_stake - run.world.rent(ROUND_SPACE)
    );
    run.world.refuse(
        run.withdraw_aggregator_ix(),
        &[&run.aggregator],
        LedgerError::FundsLocked,
    );

    // The round pays nothing, and everyone else gets their own back.
    run.world.refuse(
        client::settle(&run.task, number),
        &[],
        LedgerError::TaskNotOpen,
    );
    run.world.refuse(
        run.claim_ix(0, number),
        &[&run.sites[0]],
        LedgerError::InvalidRoundState,
    );
    run.world.run(run.withdraw_site_ix(0), &[&run.sites[0]]);
    run.world
        .run(run.withdraw_publisher_ix(), &[&run.publisher]);
    assert_eq!(run.world.balance(&run.task), task_rent);
    assert_eq!(run.world.balance(&key(&run.publisher)), WALLET - task_rent);
}

#[test]
fn fraud_in_the_first_leaf_with_three_members() {
    prove_fraud(3, 1, 0, 0);
}

#[test]
fn fraud_in_a_middle_leaf_with_three_members() {
    prove_fraud(3, 1, 1, 128 + 127);
}

#[test]
fn fraud_in_the_short_last_leaf_with_three_members() {
    prove_fraud(3, 1, 2, 256 + 39);
}

#[test]
fn fraud_in_the_first_leaf_with_sixteen_members() {
    prove_fraud(16, 2, 0, 64);
}

#[test]
fn fraud_in_a_middle_leaf_with_sixteen_members() {
    prove_fraud(16, 2, 1, 128);
}

#[test]
fn fraud_in_the_short_last_leaf_with_sixteen_members() {
    prove_fraud(16, 2, 2, 256);
}

#[test]
fn fraud_is_proven_without_helpers_and_in_any_order() {
    let vectors = Vectors::honest(5, 0, COORDS, 3).with_wrong_aggregate(200);
    let (mut run, number) = posted(5, 0, &vectors);
    let challenger = run.challenger();
    run.dispute_open(&challenger, number, 1);
    let mut parts = vectors.parts();
    parts.reverse();
    for part in parts {
        run.give(&challenger, number, part, &vectors.proof(part, 1));
    }
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);
    assert_eq!(run.round(number).state(), ROUND_FRAUD);
}

#[test]
fn a_dispute_against_an_honest_aggregate_costs_the_bond() {
    for (members, helpers) in [(3, 1), (16, 2)] {
        let vectors = Vectors::honest(members, helpers as usize, COORDS, 5);
        let (mut run, number) = posted(members, helpers, &vectors);
        for index in 0..members as u16 {
            run.ack(index, number);
        }
        let challenger = run.challenger();
        let (bond, dispute_rent) = (run.terms.dispute_bond, run.world.rent(DISPUTE_SPACE));

        run.world.warp(99);
        let before = run.world.balance(&key(&challenger));
        run.dispute_open(&challenger, number, 2);
        assert_eq!(
            run.world.balance(&key(&challenger)),
            before - bond - dispute_rent
        );
        assert_eq!(
            run.world
                .balance(&client::dispute(&run.task, number, &key(&challenger))),
            bond + dispute_rent
        );
        run.give_all(&challenger, number, 2, &vectors);
        let dispute = run.world.dispute(&run.task, number, &challenger);
        assert!(dispute.is_complete(members as u16, helpers) && dispute.sums_match());

        // Complete and equal, so it closes before its deadline.
        run.world.warp(10);
        let aggregator_before = run.world.balance(&key(&run.aggregator));
        run.world
            .run(run.dispute_close_ix(&challenger, number), &[]);
        close_buffer(&mut run, &challenger);
        assert_eq!(run.world.balance(&key(&challenger)), WALLET - bond);
        assert_eq!(
            run.world.balance(&key(&run.aggregator)),
            aggregator_before + bond
        );
        assert!(!run
            .world
            .exists(&client::dispute(&run.task, number, &key(&challenger))));

        // The round carries on from where it was.
        let (task, round) = (run.state(), run.round(number));
        assert_eq!((round.state(), round.disputes()), (ROUND_POSTED, 0));
        assert_eq!(round.posted_slot(), 0);
        assert_eq!((task.state(), task.fraud()), (TASK_OPEN, false));
        assert_eq!(task.stake_held(), run.terms.aggregator_stake);
        assert_eq!(task.last_change(), 109);
        run.settle(number);
        assert_eq!(run.round(number).state(), ROUND_FINAL);
    }
}

#[test]
fn an_incomplete_dispute_closes_only_after_its_deadline() {
    let vectors = Vectors::honest(3, 1, COORDS, 9).with_wrong_aggregate(1);
    let (mut run, number) = posted(3, 1, &vectors);
    let challenger = run.challenger();
    run.world.warp(30);
    run.dispute_open(&challenger, number, 0);
    assert_eq!(
        run.world.dispute(&run.task, number, &challenger).deadline(),
        130
    );

    // Everything but one input is given.
    for part in vectors.parts().into_iter().skip(1) {
        run.give(&challenger, number, part, &vectors.proof(part, 0));
    }
    let close = run.dispute_close_ix(&challenger, number);
    run.world.warp(100);
    run.world
        .refuse(close.clone(), &[], LedgerError::DisputeNotOver);
    run.world.warp(1);
    run.world.run(close, &[]);

    let (task, round) = (run.state(), run.round(number));
    assert_eq!((round.state(), round.disputes()), (ROUND_POSTED, 0));
    assert_eq!((task.state(), task.last_change()), (TASK_OPEN, 131));

    // The bond is lost, but not to the aggregator: it stays in the task.
    assert_eq!(
        run.world.balance(&key(&run.aggregator)),
        WALLET - run.terms.aggregator_stake - run.world.rent(ROUND_SPACE)
    );
    assert_eq!(
        run.world.balance(&key(&challenger)),
        WALLET - run.terms.dispute_bond - run.world.rent(buffer_space(4_000))
    );
    assert_eq!(
        run.world.balance(&run.task),
        run.world.rent(TASK_SPACE)
            + 2 * run.terms.reward_per_round
            + run.terms.aggregator_stake
            + run.terms.dispute_bond
    );
}

#[test]
fn a_proof_completed_after_the_deadline_still_counts() {
    let vectors = Vectors::honest(3, 1, COORDS, 9).with_wrong_aggregate(1);
    let (mut run, number) = posted(3, 1, &vectors);
    let challenger = run.challenger();
    run.dispute_open(&challenger, number, 0);
    run.world.warp(500);
    run.give_all(&challenger, number, 0, &vectors);
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);
    assert_eq!(run.round(number).state(), ROUND_FRAUD);
}

#[test]
fn dispute_open_needs_a_posted_round_inside_its_challenge_window() {
    let vectors = Vectors::honest(3, 1, COORDS, 1);
    let posting = vectors.posting(1);
    let mut run = Run::started(3, |_| {});
    let challenger = run.challenger();
    let number = run.open_round();
    run.commit(number, &[0, 1, 2], &posting);
    run.world.refuse(
        run.dispute_open_ix(&challenger, number, 0),
        &[&challenger],
        LedgerError::InvalidRoundState,
    );
    run.world.warp(20);
    run.post(number, &posting);

    run.world.refuse(
        run.dispute_open_ix(&challenger, number, 3),
        &[&challenger],
        LedgerError::InvalidLeafIndex,
    );
    let mut wrong_number = run.dispute_open_ix(&challenger, number, 0);
    wrong_number.data[8..12].copy_from_slice(&2u32.to_le_bytes());
    run.world
        .refuse(wrong_number, &[&challenger], LedgerError::AccountMismatch);
    let mut wrong_account = run.dispute_open_ix(&challenger, number, 0);
    wrong_account.accounts[3].address = client::dispute(&run.task, 2, &key(&challenger));
    run.world
        .refuse(wrong_account, &[&challenger], "InvalidSeeds");
    let mut unsigned = run.dispute_open_ix(&challenger, number, 0);
    unsigned.accounts[0].is_signer = false;
    run.world.refuse(unsigned, &[], "MissingRequiredSignature");
    let mut trailing = run.dispute_open_ix(&challenger, number, 0);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&challenger], "InvalidInstructionData");

    // The last slot of the window is 119, and the dispute then runs to 219.
    run.world.warp(99);
    run.dispute_open(&challenger, number, 2);

    let dispute = run.world.dispute(&run.task, number, &challenger);
    assert_eq!(
        dispute.address().as_array(),
        &client::dispute(&run.task, number, &key(&challenger))
    );
    assert_eq!((dispute.task(), dispute.round()), (&run.task, number));
    assert_eq!(dispute.challenger(), &key(&challenger));
    assert_eq!((dispute.leaf_index(), dispute.leaf_coords()), (2, 40));
    assert_eq!(dispute.deadline(), 219);
    assert_eq!((dispute.inputs_seen(), dispute.seen_masks()), (0, 0));
    assert!(!dispute.aggregate_given());
    assert_eq!(dispute.acc(), &[0; 1024]);
    assert_eq!(run.round(number).disputes(), 1);

    // Another challenger opens a dispute of their own, and the round and
    // the task wait for both.
    let rival = run.challenger();
    run.dispute_open(&rival, number, 0);
    assert_eq!(run.round(number).disputes(), 2);
    assert_eq!(run.world.dispute(&run.task, number, &rival).leaf_index(), 0);
    for index in 0..3 {
        run.ack(index, number);
    }
    run.world.warp(50);
    run.world.refuse(
        client::settle(&run.task, number),
        &[],
        LedgerError::RoundDisputed,
    );
    run.world.refuse(
        run.abort_ix(&run.aggregator),
        &[&run.aggregator],
        LedgerError::RoundDisputed,
    );
}

#[test]
fn a_round_can_be_disputed_again_inside_its_window() {
    let vectors = Vectors::honest(3, 1, COORDS, 21).with_wrong_aggregate(290);
    let (mut run, number) = posted(3, 1, &vectors);
    let challenger = run.challenger();

    // The first dispute picks an honest leaf and loses.
    run.dispute_open(&challenger, number, 0);
    run.give_all(&challenger, number, 0, &vectors);
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);
    assert_eq!(run.round(number).state(), ROUND_POSTED);

    run.world.warp(50);
    run.dispute_open(&challenger, number, 2);
    run.give_all(&challenger, number, 2, &vectors);
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);
    assert_eq!(run.round(number).state(), ROUND_FRAUD);
    assert_eq!(
        run.world.balance(&key(&challenger)),
        WALLET + run.terms.aggregator_stake
            - run.terms.dispute_bond
            - run.world.rent(buffer_space(4_000))
    );
}

#[test]
fn dispute_leaf_takes_each_part_once_from_the_challenger() {
    let vectors = Vectors::honest(3, 2, COORDS, 13);
    let (mut run, number) = posted(3, 2, &vectors);
    let challenger = run.challenger();
    run.dispute_open(&challenger, number, 1);
    let leaf = |part| vectors.proof(part, 1);

    // Parts the dispute does not have.
    run.world.upload(&challenger, &leaf(ProofPart::Aggregate));
    for part in [ProofPart::Input(3), ProofPart::Mask(2)] {
        run.world.refuse(
            run.dispute_leaf_ix(&challenger, number, part),
            &[&challenger],
            LedgerError::InvalidProofPart,
        );
    }
    let mut second_aggregate = run.dispute_leaf_ix(&challenger, number, ProofPart::Aggregate);
    second_aggregate.data[9] = 1;
    run.world.refuse(
        second_aggregate,
        &[&challenger],
        LedgerError::InvalidProofPart,
    );
    let mut unknown_kind = run.dispute_leaf_ix(&challenger, number, ProofPart::Aggregate);
    unknown_kind.data[8] = 3;
    run.world
        .refuse(unknown_kind, &[&challenger], LedgerError::InvalidProofPart);
    let mut trailing = run.dispute_leaf_ix(&challenger, number, ProofPart::Aggregate);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&challenger], "InvalidInstructionData");
    let mut unsigned = run.dispute_leaf_ix(&challenger, number, ProofPart::Aggregate);
    unsigned.accounts[0].is_signer = false;
    run.world.refuse(unsigned, &[], "MissingRequiredSignature");

    // Someone else, with a buffer of their own or with the challenger's.
    let rival = run.challenger();
    run.world.upload(&rival, &leaf(ProofPart::Aggregate));
    run.world.refuse(
        run.dispute_leaf_ix(&rival, number, ProofPart::Aggregate),
        &[&rival],
        "InvalidAccountOwner",
    );
    let mut into_anothers = run.dispute_leaf_ix(&rival, number, ProofPart::Aggregate);
    into_anothers.accounts[3].address = client::dispute(&run.task, number, &key(&challenger));
    run.world
        .refuse(into_anothers, &[&rival], LedgerError::Unauthorized);
    let mut borrowed = run.dispute_leaf_ix(&challenger, number, ProofPart::Aggregate);
    borrowed.accounts[4].address = client::buffer(&key(&rival));
    run.world
        .refuse(borrowed, &[&challenger], LedgerError::AccountMismatch);

    for part in [
        ProofPart::Aggregate,
        ProofPart::Mask(1),
        ProofPart::Input(2),
    ] {
        run.give(&challenger, number, part, &leaf(part));
        run.world.refuse(
            run.dispute_leaf_ix(&challenger, number, part),
            &[&challenger],
            LedgerError::ProofPartRepeated,
        );
    }
    let dispute = run.world.dispute(&run.task, number, &challenger);
    assert_eq!((dispute.inputs_seen(), dispute.seen_masks()), (1, 0b10));
    assert_eq!(dispute.seen_inputs()[0], 0b100);
    assert!(dispute.aggregate_given());
    assert!(!dispute.is_complete(3, 2));
}

#[test]
fn dispute_leaf_keeps_a_running_sum_in_the_field() {
    let vectors = Vectors::honest(3, 1, COORDS, 17);
    let (mut run, number) = posted(3, 1, &vectors);
    let challenger = run.challenger();
    run.dispute_open(&challenger, number, 2);

    let coordinates = |bytes: &[u8]| -> Vec<u64> {
        bytes
            .chunks(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let leaf = |vector: &[u64]| vector[256..].to_vec();
    let mut expected = vec![0u64; 128];
    let mut check = |run: &Run, add: bool, vector: &[u64]| {
        for (sum, value) in expected.iter_mut().zip(leaf(vector)) {
            *sum = if add {
                reference::field_add(*sum, value)
            } else {
                reference::field_sub(*sum, value)
            };
        }
        let dispute = run.world.dispute(&run.task, number, &challenger);
        assert_eq!(coordinates(dispute.acc()), expected);
    };

    for (position, input) in vectors.inputs.iter().enumerate() {
        let part = ProofPart::Input(position as u16);
        run.give(&challenger, number, part, &vectors.proof(part, 2));
        check(&run, true, input);
    }
    run.give(
        &challenger,
        number,
        ProofPart::Mask(0),
        &vectors.proof(ProofPart::Mask(0), 2),
    );
    check(&run, false, &vectors.masks[0]);

    // The aggregate leaf is kept reduced, and only its 40 coordinates.
    run.give(
        &challenger,
        number,
        ProofPart::Aggregate,
        &vectors.proof(ProofPart::Aggregate, 2),
    );
    let dispute = run.world.dispute(&run.task, number, &challenger);
    let stored = coordinates(dispute.aggregate());
    let given = leaf(&vectors.aggregate);
    assert!(given.iter().any(|value| *value >= reference::P));
    for (i, value) in stored.iter().enumerate() {
        let wanted = given.get(i).map_or(0, |v| reference::field_add(*v, 0));
        assert_eq!(*value, wanted, "coordinate {i}");
    }
    assert!(dispute.sums_match());
}

#[test]
fn dispute_leaf_refuses_proofs_that_do_not_reach_the_posted_roots() {
    let vectors = Vectors::honest(4, 2, COORDS, 29);
    let (mut run, number) = posted(4, 2, &vectors);
    let challenger = run.challenger();
    run.dispute_open(&challenger, number, 1);
    let refuse = |run: &mut Run, part: ProofPart, proof: &[u8], why: LedgerError| {
        run.world.upload(&challenger, proof);
        run.world.refuse(
            run.dispute_leaf_ix(&challenger, number, part),
            &[&challenger],
            why,
        );
    };
    let input = ProofPart::Input(1);
    let good = vectors.proof(input, 1);
    let flipped = |at: usize| {
        let mut proof = good.clone();
        proof[at] ^= 1;
        proof
    };

    // A changed coordinate, vector root, inputs path or leaf path.
    for at in [0, 1023, 1024, 1056, 1056 + 64, good.len() - 1] {
        refuse(&mut run, input, &flipped(at), LedgerError::InvalidProof);
    }
    // The right proof for another member, helper, leaf or part.
    refuse(
        &mut run,
        ProofPart::Input(2),
        &good,
        LedgerError::InvalidProof,
    );
    refuse(
        &mut run,
        input,
        &vectors.proof(input, 0),
        LedgerError::InvalidProof,
    );
    refuse(
        &mut run,
        ProofPart::Mask(0),
        &vectors.proof(ProofPart::Mask(1), 1),
        LedgerError::InvalidProof,
    );
    refuse(
        &mut run,
        ProofPart::Aggregate,
        &vectors.proof(ProofPart::Mask(0), 1),
        LedgerError::InvalidProof,
    );

    // A leaf of the wrong length, or a path with a hash missing or added.
    refuse(&mut run, input, &good[8..], LedgerError::InvalidProofLength);
    refuse(
        &mut run,
        input,
        &good[..good.len() - 32],
        LedgerError::InvalidProofLength,
    );
    refuse(
        &mut run,
        input,
        &[good.clone(), vec![0; 32]].concat(),
        LedgerError::InvalidProofLength,
    );
    refuse(
        &mut run,
        ProofPart::Aggregate,
        &good,
        LedgerError::InvalidProofLength,
    );
    refuse(
        &mut run,
        input,
        &vectors.proof(input, 2),
        LedgerError::InvalidProofLength,
    );

    assert_eq!(
        run.world
            .dispute(&run.task, number, &challenger)
            .inputs_seen(),
        0
    );
    run.give(&challenger, number, input, &good);
}

#[test]
fn dispute_close_pays_the_wallets_the_dispute_names() {
    let vectors = Vectors::honest(3, 1, COORDS, 31);
    let (mut run, number) = posted(3, 1, &vectors);
    let challenger = run.challenger();
    run.dispute_open(&challenger, number, 0);
    run.give_all(&challenger, number, 0, &vectors);
    let stranger = run.world.wallet();

    let mut to_stranger = run.dispute_close_ix(&challenger, number);
    to_stranger.accounts[3].address = key(&stranger);
    run.world
        .refuse(to_stranger, &[], LedgerError::AccountMismatch);
    let mut for_stranger = run.dispute_close_ix(&challenger, number);
    for_stranger.accounts[4].address = key(&stranger);
    run.world
        .refuse(for_stranger, &[], LedgerError::AccountMismatch);
    let mut readonly = run.dispute_close_ix(&challenger, number);
    readonly.accounts[4].is_writable = false;
    run.world.refuse(readonly, &[], "Immutable");
    let mut trailing = run.dispute_close_ix(&challenger, number);
    trailing.data.push(0);
    run.world.refuse(trailing, &[], "InvalidInstructionData");

    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);
    run.world.refuse(
        run.dispute_close_ix(&challenger, number),
        &[],
        "InvalidAccountOwner",
    );
}

#[test]
fn aborting_does_not_save_the_stake_from_a_fraud_proof() {
    let vectors = Vectors::honest(3, 1, COORDS, 37).with_wrong_aggregate(5);
    let (mut run, number) = posted(3, 1, &vectors);
    let challenger = run.challenger();

    // The aggregator aborts ahead of the dispute and reaches for the stake.
    run.world.warp(10);
    run.world
        .run(run.abort_ix(&run.aggregator), &[&run.aggregator]);
    run.world.refuse(
        run.withdraw_aggregator_ix(),
        &[&run.aggregator],
        LedgerError::FundsLocked,
    );
    let mut without_round = run.withdraw_aggregator_ix();
    without_round.accounts.pop();
    run.world
        .refuse(without_round, &[&run.aggregator], "NotEnoughAccountKeys");

    run.world.warp(89);
    run.dispute_open(&challenger, number, 0);
    run.world.warp(50);
    run.world.refuse(
        run.withdraw_aggregator_ix(),
        &[&run.aggregator],
        LedgerError::FundsLocked,
    );
    run.give_all(&challenger, number, 0, &vectors);
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);

    let task = run.state();
    assert_eq!((task.state(), task.fraud()), (TASK_ABORTED, true));
    assert_eq!(run.round(number).state(), ROUND_FRAUD);
    assert_eq!(
        run.world.balance(&key(&challenger)),
        WALLET + run.terms.aggregator_stake - run.world.rent(buffer_space(4_000))
    );
    run.world.refuse(
        run.withdraw_aggregator_ix(),
        &[&run.aggregator],
        LedgerError::FundsLocked,
    );
}

#[test]
fn an_aborted_tasks_stake_is_released_once_the_window_has_passed() {
    let vectors = Vectors::honest(3, 1, COORDS, 41);
    let (mut run, number) = posted(3, 1, &vectors);
    run.world
        .run(run.abort_ix(&run.publisher), &[&run.publisher]);
    run.world.warp(99);
    run.world.refuse(
        run.withdraw_aggregator_ix(),
        &[&run.aggregator],
        LedgerError::FundsLocked,
    );
    run.world.warp(1);
    let challenger = run.challenger();
    run.world.refuse(
        run.dispute_open_ix(&challenger, number, 0),
        &[&challenger],
        LedgerError::ChallengeWindowClosed,
    );
    run.world
        .run(run.withdraw_aggregator_ix(), &[&run.aggregator]);
    assert_eq!(
        run.world.balance(&key(&run.aggregator)),
        WALLET - run.world.rent(ROUND_SPACE)
    );
}

#[test]
fn a_buffer_is_written_in_order_by_its_owner() {
    let mut run = Run::new(2, |_| {});
    let owner = run.world.wallet();
    let who = key(&owner);
    let buffer = client::buffer(&who);
    let contents = |run: &Run| {
        Buffer::parse(run.world.data(&buffer))
            .unwrap()
            .contents()
            .to_vec()
    };

    for capacity in [0, 10_001] {
        run.world.refuse(
            client::buffer_init(&who, capacity),
            &[&owner],
            LedgerError::InvalidBufferSize,
        );
    }
    let mut trailing = client::buffer_init(&who, 100);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&owner], "InvalidInstructionData");
    let mut unsigned = client::buffer_init(&who, 100);
    unsigned.accounts[0].is_signer = false;
    run.world.refuse(unsigned, &[], "MissingRequiredSignature");

    let before = run.world.balance(&who);
    run.world.run(client::buffer_init(&who, 100), &[&owner]);
    let rent = run.world.rent(buffer_space(100));
    assert_eq!(run.world.balance(&who), before - rent);
    let view = Buffer::parse(run.world.data(&buffer)).unwrap();
    assert_eq!(view.address().as_array(), &buffer);
    assert_eq!((view.owner(), view.used(), view.capacity()), (&who, 0, 100));
    run.world.refuse(
        client::buffer_init(&who, 100),
        &[&owner],
        "AccountAlreadyInitialized",
    );

    // Writes continue the contents or start them over, within the size.
    let write = |offset: u32, chunk: &[u8]| client::buffer_write(&who, offset, chunk);
    run.world.refuse(
        write(1, &[1; 10]),
        &[&owner],
        LedgerError::InvalidBufferWrite,
    );
    run.world
        .refuse(write(0, &[]), &[&owner], LedgerError::InvalidBufferWrite);
    run.world.refuse(
        write(0, &[1; 101]),
        &[&owner],
        LedgerError::InvalidBufferWrite,
    );
    run.world.run(write(0, &[1; 60]), &[&owner]);
    run.world.refuse(
        write(59, &[2; 10]),
        &[&owner],
        LedgerError::InvalidBufferWrite,
    );
    run.world.refuse(
        write(60, &[2; 41]),
        &[&owner],
        LedgerError::InvalidBufferWrite,
    );
    run.world.run(write(60, &[2; 40]), &[&owner]);
    assert_eq!(contents(&run), [vec![1; 60], vec![2; 40]].concat());
    run.world.run(write(0, &[3; 7]), &[&owner]);
    assert_eq!(contents(&run), vec![3; 7]);

    let mut trailing = write(7, &[4; 4]);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&owner], "InvalidInstructionData");
    let stranger = run.world.wallet();
    let mut by_stranger = write(7, &[4; 4]);
    by_stranger.accounts[0].address = key(&stranger);
    run.world
        .refuse(by_stranger, &[&stranger], LedgerError::Unauthorized);
    let mut close_by_stranger = client::buffer_close(&who);
    close_by_stranger.accounts[0].address = key(&stranger);
    run.world
        .refuse(close_by_stranger, &[&stranger], LedgerError::Unauthorized);
    let mut trailing = client::buffer_close(&who);
    trailing.data.push(0);
    run.world
        .refuse(trailing, &[&owner], "InvalidInstructionData");

    for mut unsigned in [write(7, &[4; 4]), client::buffer_close(&who)] {
        unsigned.accounts[0].is_signer = false;
        run.world.refuse(unsigned, &[], "MissingRequiredSignature");
    }

    run.world.run(client::buffer_close(&who), &[&owner]);
    assert_eq!(run.world.balance(&who), before);
    assert!(!run.world.exists(&buffer));
    run.world
        .refuse(write(0, &[1; 4]), &[&owner], "InvalidAccountOwner");
    run.world.run(client::buffer_init(&who, 10_000), &[&owner]);
}

#[test]
fn an_empty_dispute_by_the_aggregators_puppet_does_not_block_a_fraud_proof() {
    let vectors = Vectors::honest(3, 1, COORDS, 43).with_wrong_aggregate(7);
    let (mut run, number) = posted(3, 1, &vectors);
    for index in 0..3 {
        run.ack(index, number);
    }

    // In the slot of the post, a wallet the aggregator controls opens a
    // dispute and gives nothing. Its deadline is the end of the window.
    let puppet = run.challenger();
    run.dispute_open(&puppet, number, 0);

    // An honest challenger is not kept out, and proves the fraud while the
    // puppet's dispute sits open.
    let challenger = run.challenger();
    run.world.warp(99);
    run.dispute_open(&challenger, number, 0);
    assert_eq!(run.round(number).disputes(), 2);
    run.world.warp(2);
    run.world.refuse(
        client::settle(&run.task, number),
        &[],
        LedgerError::RoundDisputed,
    );
    run.give_all(&challenger, number, 0, &vectors);
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);

    let (task, round) = (run.state(), run.round(number));
    assert_eq!((round.state(), round.disputes()), (ROUND_FRAUD, 1));
    assert_eq!((task.state(), task.fraud()), (TASK_ABORTED, true));
    assert_eq!(
        run.world.balance(&key(&challenger)),
        WALLET + run.terms.aggregator_stake - run.world.rent(buffer_space(4_000))
    );

    // The puppet's dispute closes with its bond returned and nothing more.
    run.world.run(run.dispute_close_ix(&puppet, number), &[]);
    assert_eq!(run.round(number).disputes(), 0);
    assert_eq!(
        run.world.balance(&key(&puppet)),
        WALLET - run.world.rent(buffer_space(4_000))
    );
    assert_eq!(
        run.world.balance(&key(&run.aggregator)),
        WALLET - run.terms.aggregator_stake - run.world.rent(ROUND_SPACE)
    );
    run.world.refuse(
        run.withdraw_aggregator_ix(),
        &[&run.aggregator],
        LedgerError::FundsLocked,
    );
}

#[test]
fn an_abandoned_dispute_delays_the_round_and_leaves_its_bond_to_the_publisher() {
    let vectors = Vectors::honest(3, 1, COORDS, 47);
    let mut run = Run::started(3, |t| t.rounds = 1);
    let number = run.post_round(&[0, 1, 2], &vectors.posting(1));
    for index in 0..3 {
        run.ack(index, number);
    }
    let puppet = run.challenger();
    run.dispute_open(&puppet, number, 0);

    // The round cannot settle until the dispute is closed, and the dispute
    // cannot be closed before its deadline.
    run.world.warp(100);
    run.world.refuse(
        client::settle(&run.task, number),
        &[],
        LedgerError::RoundDisputed,
    );
    run.world.refuse(
        run.dispute_close_ix(&puppet, number),
        &[],
        LedgerError::DisputeNotOver,
    );
    run.world.warp(1);
    let aggregator_before = run.world.balance(&key(&run.aggregator));
    run.world.run(run.dispute_close_ix(&puppet, number), &[]);
    assert_eq!(run.world.balance(&key(&run.aggregator)), aggregator_before);

    run.settle(number);
    assert_eq!(run.state().state(), TASK_FINISHED);
    let before = run.world.balance(&key(&run.publisher));
    run.world
        .run(run.withdraw_publisher_ix(), &[&run.publisher]);
    assert_eq!(
        run.world.balance(&key(&run.publisher)),
        before + run.terms.dispute_bond + 1
    );
}
