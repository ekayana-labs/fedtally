//! Demands for a leaf, over real vectors: what an answer pays and reveals, what
//! silence costs the aggregator, and closing rounds that are paid out.

mod common;

use common::{key, vectors::Vectors, Run, WALLET};
use fedtally_ledger::{
    client::{self, ProofPart},
    error::LedgerError,
    instructions::demand,
    state::*,
};
use solana_keypair::Keypair;

const COORDS: usize = 296;
const WINDOW: u64 = 100;

fn posted(vectors: &Vectors) -> (Run, u32) {
    let mut run = Run::started(3, |terms| {
        (terms.challenge_slots, terms.stall_slots) = (WINDOW, 2 * WINDOW);
    });
    let number = run.post_round(&[0, 1, 2], &vectors.posting(1));
    (run, number)
}

fn honest() -> Vectors {
    Vectors::honest(3, 1, COORDS, 5)
}

fn demand_of(run: &Run, who: &Keypair, number: u32, part: ProofPart, leaf: u64) -> [u8; 32] {
    client::demand(&run.task, &key(who), number, part, leaf)
}

fn open(run: &mut Run, who: &Keypair, number: u32, part: ProofPart, leaf: u64) -> [u8; 32] {
    let ix = client::demand_open(&key(who), &run.task, number, part, leaf);
    run.world.run(ix, &[who]);
    demand_of(run, who, number, part, leaf)
}

fn answer(run: &mut Run, who: &Keypair, number: u32, demand: [u8; 32], proof: &[u8]) {
    run.world.upload(who, proof);
    let ix = client::demand_answer(&key(who), &run.task, number, demand);
    run.world.run(ix, &[who]);
}

#[test]
fn silence_past_the_deadline_costs_the_aggregator_its_stake() {
    let vectors = honest();
    let (mut run, number) = posted(&vectors);
    let challenger = run.challenger();
    let demand = open(&mut run, &challenger, number, ProofPart::Input(1), 2);
    assert_eq!(run.round(number).disputes(), 1);

    let early = client::demand_close(&run.task, number, demand, &key(&challenger), true);
    run.world
        .refuse(early, &[&challenger], LedgerError::DemandNotOver);

    // A second demand queues behind the first, and is moot once that one lands.
    let queued = open(&mut run, &challenger, number, ProofPart::Input(0), 2);
    run.world.warp(WINDOW + 1);
    let late = client::demand_close(&run.task, number, demand, &key(&challenger), false);
    run.world.run(late, &[]);
    let moot = client::demand_close(&run.task, number, queued, &key(&challenger), false);
    run.world.run(moot, &[]);

    let (task, round) = (run.state(), run.round(number));
    assert_eq!((round.state(), round.disputes()), (ROUND_FRAUD, 0));
    assert_eq!(
        (task.state(), task.fraud(), task.stake_held()),
        (TASK_ABORTED, true, 0)
    );
    assert!(!run.world.exists(&demand));
    assert_eq!(
        run.world.balance(&key(&challenger)),
        WALLET - run.world.balance(&client::buffer(&key(&challenger))) + run.terms.aggregator_stake
    );
}

#[test]
fn an_answer_pays_the_one_who_gave_it_and_stays_readable() {
    let vectors = honest();
    let (mut run, number) = posted(&vectors);
    let challenger = run.challenger();
    let part = ProofPart::Input(1);
    let demand = open(&mut run, &challenger, number, part, 0);

    // Anyone holding the leaf can answer, not only the aggregator.
    let helper = run.challenger();
    let proof = vectors.proof(part, 0);
    answer(&mut run, &helper, number, demand, &proof);

    let data = run.world.data(&demand);
    let stored = Demand::parse(data.as_slice()).unwrap();
    assert!(stored.answered());
    assert_eq!(stored.proof(), proof.as_slice());

    let fee = run.terms.dispute_bond.min(demand::ANSWER_FEE);
    let buffer_rent = run.world.balance(&client::buffer(&key(&helper)));
    assert_eq!(run.world.balance(&key(&helper)), WALLET - buffer_rent + fee);

    let close = client::demand_close(&run.task, number, demand, &key(&challenger), true);
    run.world.run(close, &[&challenger]);
    assert_eq!(
        run.world.balance(&key(&challenger)),
        WALLET - run.world.balance(&client::buffer(&key(&challenger))) - fee
    );
    assert_eq!(run.round(number).disputes(), 0);
    assert_eq!(run.round(number).state(), ROUND_POSTED);
    assert_eq!(run.state().stake_held(), run.terms.aggregator_stake);
}

#[test]
fn a_demand_holds_the_round_open_until_it_is_closed() {
    let vectors = honest();
    let (mut run, number) = posted(&vectors);
    for index in 0..3 {
        run.ack(index, number);
    }
    let challenger = run.challenger();
    let part = ProofPart::Aggregate;
    let demand = open(&mut run, &challenger, number, part, 1);
    let aggregator = run.challenger();
    answer(
        &mut run,
        &aggregator,
        number,
        demand,
        &vectors.proof(part, 1),
    );

    run.world.warp(WINDOW + 1);
    run.world.refuse(
        client::settle(&run.task, number),
        &[],
        LedgerError::RoundDisputed,
    );
    let close = client::demand_close(&run.task, number, demand, &key(&challenger), false);
    run.world.run(close, &[]);
    run.settle(number);
    assert_eq!(run.round(number).state(), ROUND_FINAL);
}

#[test]
fn an_answered_demand_lets_its_challenger_prove_fraud_past_the_window() {
    let vectors = Vectors::honest(3, 1, COORDS, 5).with_wrong_aggregate(3);
    let (mut run, number) = posted(&vectors);
    let challenger = run.challenger();
    let part = ProofPart::Aggregate;
    let demand = open(&mut run, &challenger, number, part, 0);

    run.world.warp(WINDOW - 1);
    let aggregator = run.challenger();
    answer(
        &mut run,
        &aggregator,
        number,
        demand,
        &vectors.proof(part, 0),
    );

    run.world.warp(2);
    let plain = client::dispute_open(&key(&challenger), &run.task, number, 0);
    run.world
        .refuse(plain, &[&challenger], LedgerError::ChallengeWindowClosed);

    let wrong_leaf = client::dispute_open(&key(&challenger), &run.task, number, 1);
    let mut wrong_leaf = wrong_leaf;
    wrong_leaf
        .accounts
        .push(client::AccountMeta::readonly(demand, false));
    run.world.refuse(
        wrong_leaf,
        &[&challenger],
        LedgerError::ChallengeWindowClosed,
    );

    let late = client::dispute_open_after_demand(&key(&challenger), &run.task, number, part, 0);
    run.world.run(late, &[&challenger]);
    run.give_all(&challenger, number, 0, &vectors);
    run.world
        .run(run.dispute_close_ix(&challenger, number), &[]);
    assert_eq!(run.round(number).state(), ROUND_FRAUD);
}

#[test]
fn an_unanswered_demand_opens_no_late_dispute() {
    let vectors = honest();
    let (mut run, number) = posted(&vectors);
    let challenger = run.challenger();
    let part = ProofPart::Mask(0);
    open(&mut run, &challenger, number, part, 0);
    run.world.warp(WINDOW + 1);
    let late = client::dispute_open_after_demand(&key(&challenger), &run.task, number, part, 0);
    run.world
        .refuse(late, &[&challenger], LedgerError::ChallengeWindowClosed);
}

#[test]
fn an_answer_must_prove_the_leaf_demanded() {
    let vectors = honest();
    let (mut run, number) = posted(&vectors);
    let challenger = run.challenger();
    let part = ProofPart::Input(0);
    let demand = open(&mut run, &challenger, number, part, 0);
    let responder = run.challenger();

    let another_leaf = vectors.proof(part, 2);
    run.world.upload(&responder, &another_leaf);
    let ix = client::demand_answer(&key(&responder), &run.task, number, demand);
    run.world
        .refuse(ix, &[&responder], LedgerError::InvalidProofLength);

    let mut tampered = vectors.proof(part, 0);
    tampered[3] ^= 1;
    run.world.upload(&responder, &tampered);
    let ix = client::demand_answer(&key(&responder), &run.task, number, demand);
    run.world
        .refuse(ix, &[&responder], LedgerError::InvalidProof);

    answer(
        &mut run,
        &responder,
        number,
        demand,
        &vectors.proof(part, 0),
    );
    let again = client::demand_answer(&key(&responder), &run.task, number, demand);
    run.world
        .refuse(again, &[&responder], LedgerError::DemandAnswered);
}

#[test]
fn an_answer_after_the_deadline_is_too_late() {
    let vectors = honest();
    let (mut run, number) = posted(&vectors);
    let challenger = run.challenger();
    let part = ProofPart::Aggregate;
    let demand = open(&mut run, &challenger, number, part, 0);
    run.world.warp(WINDOW + 1);
    let responder = run.challenger();
    run.world.upload(&responder, &vectors.proof(part, 0));
    let ix = client::demand_answer(&key(&responder), &run.task, number, demand);
    run.world
        .refuse(ix, &[&responder], LedgerError::DemandExpired);
}

#[test]
fn a_demand_needs_a_posted_round_inside_its_window_and_a_real_part() {
    let vectors = honest();
    let (mut run, number) = posted(&vectors);
    let challenger = run.challenger();
    let who = key(&challenger);
    for (part, leaf, error) in [
        (ProofPart::Input(3), 0, LedgerError::InvalidProofPart),
        (ProofPart::Mask(1), 0, LedgerError::InvalidProofPart),
        (ProofPart::Aggregate, 3, LedgerError::InvalidLeafIndex),
    ] {
        let ix = client::demand_open(&who, &run.task, number, part, leaf);
        run.world.refuse(ix, &[&challenger], error);
    }
    run.world.warp(WINDOW);
    let ix = client::demand_open(&who, &run.task, number, ProofPart::Aggregate, 0);
    run.world
        .refuse(ix, &[&challenger], LedgerError::ChallengeWindowClosed);
}

#[test]
fn a_paid_out_round_closes_and_returns_its_rent_to_the_aggregator() {
    let mut run = Run::started(2, |_| {});
    let first = run.settle_round(&[0, 1]);
    let second = run.settle_round(&[0, 1]);
    let aggregator = key(&run.aggregator);
    let close = |run: &Run, number| client::round_close(&aggregator, &run.task, number);

    run.world
        .refuse(close(&run, first), &[], LedgerError::SharesUnclaimed);
    run.claim(0, first);
    run.world
        .refuse(close(&run, first), &[], LedgerError::SharesUnclaimed);
    run.claim(1, first);

    let before = run.world.balance(&aggregator);
    run.world.run(close(&run, first), &[]);
    assert_eq!(
        run.world.balance(&aggregator),
        before + run.world.rent(ROUND_SPACE)
    );
    assert!(!run.world.exists(&client::round(&run.task, first)));

    run.claim(0, second);
    run.claim(1, second);
    run.world
        .refuse(close(&run, second), &[], LedgerError::LatestRound);
}

#[test]
fn a_burst_of_demands_cannot_outrun_an_honest_aggregator() {
    let vectors = honest();
    let (mut run, number) = posted(&vectors);
    let challenger = run.challenger();
    let mut opened = Vec::new();
    for leaf in 0..3 {
        for position in 0..3 {
            let part = ProofPart::Input(position);
            opened.push((part, leaf, open(&mut run, &challenger, number, part, leaf)));
        }
    }
    let deadlines: Vec<u64> = opened
        .iter()
        .map(|(.., address)| {
            let data = run.world.data(address);
            Demand::parse(data.as_slice()).unwrap().deadline()
        })
        .collect();
    assert!(deadlines
        .windows(2)
        .all(|pair| pair[1] >= pair[0] + demand::ANSWER_SLOTS));

    // Answered one after another, each at the last slot it may, all stand.
    let responder = run.challenger();
    let mut now = run.world.slot;
    for ((part, leaf, address), deadline) in opened.iter().zip(deadlines) {
        run.world.warp(deadline - now);
        now = deadline;
        answer(
            &mut run,
            &responder,
            number,
            *address,
            &vectors.proof(*part, *leaf as usize),
        );
    }
    assert_eq!(run.round(number).state(), ROUND_POSTED);
    assert_eq!(run.state().stake_held(), run.terms.aggregator_stake);
}

#[test]
fn an_unclaimed_share_expires_and_goes_back_to_the_publisher() {
    let mut run = Run::started(2, |_| {});
    let first = run.settle_round(&[0, 1]);
    run.settle_round(&[0, 1]);
    run.claim(0, first);
    let aggregator = key(&run.aggregator);
    let close = client::round_close(&aggregator, &run.task, first);

    let time_to_claim = run.terms.challenge_slots * CLAIM_WINDOWS;
    run.world
        .warp(time_to_claim - 2 * run.terms.challenge_slots);
    run.world
        .refuse(close.clone(), &[], LedgerError::SharesUnclaimed);
    run.world.warp(2 * run.terms.challenge_slots);

    let before = run.state().owed();
    let share = run.terms.reward_per_round / 2;
    run.world.run(close, &[]);
    assert_eq!(run.state().owed(), before - share);
    assert!(!run.world.exists(&client::round(&run.task, first)));
}

#[test]
fn the_latest_round_closes_once_the_task_is_over_and_the_stake_is_gone() {
    let mut run = Run::started(2, |terms| terms.rounds = 1);
    let number = run.settle_round(&[0, 1]);
    run.claim(0, number);
    run.claim(1, number);
    let aggregator = key(&run.aggregator);
    let close = client::round_close(&aggregator, &run.task, number);
    run.world
        .refuse(close.clone(), &[], LedgerError::LatestRound);

    let withdraw = run.withdraw_aggregator_ix();
    run.world.run(withdraw, &[&run.aggregator]);
    run.world.run(close, &[]);
    assert!(!run.world.exists(&client::round(&run.task, number)));
}
