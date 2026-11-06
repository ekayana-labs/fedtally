//! Giving up a round: for a silent site, or with a helper removed.

mod common;

use common::*;
use fedtally_ledger::client::{self, TaskTerms};
use fedtally_ledger::error::LedgerError;
use fedtally_ledger::state::*;

const SENSITIVITY: u64 = 262146;
const NOISE: u64 = 1 << 34;

fn private(terms: &mut TaskTerms) {
    (terms.sensitivity, terms.noise) = (SENSITIVITY, NOISE);
    terms.budget = 40 << 40;
}

fn retrying(retries: u32) -> impl FnOnce(&mut TaskTerms) {
    move |terms| {
        private(terms);
        terms.retries = retries;
    }
}

fn losing_a_helper(terms: &mut TaskTerms) {
    (terms.helpers, terms.min_helpers) = (3, 2);
}

fn void_ix(run: &Run, number: u32, remove: bool) -> client::Instruction {
    client::void_round(&key(&run.aggregator), &run.task, number, remove)
}

fn committed(run: &mut Run, cohort: &[u16]) -> u32 {
    let number = run.open_round();
    let helpers = run.state().active_helpers();
    run.commit(number, cohort, &Posting::filler(number, helpers));
    number
}

#[test]
fn a_voided_round_stays_charged_and_the_next_starts_from_the_same_model() {
    let mut run = Run::started(3, retrying(1));
    let first = committed(&mut run, &[0, 1, 2]);
    let (spent, beacon) = (run.state().spent(), *run.round(first).beacon());
    assert!(spent > 0);

    let aggregator = run.aggregator.insecure_clone();
    let units = run.world.run(void_ix(&run, first, false), &[&aggregator]);
    println!("void_round {units}");
    assert!(units < 5_000, "{units}");
    assert_eq!(run.round(first).state(), ROUND_VOID);
    let task = run.state();
    assert_eq!(
        (task.voided(), task.settled(), task.spent()),
        (1, first, spent)
    );
    assert_eq!(task.model(), &run.terms.model);

    let second = run.open_round();
    assert_eq!(second, first + 1);
    assert_eq!(run.round(second).beacon(), &beacon, "a redo draws the same");
}

#[test]
fn only_the_aggregator_can_void_and_only_a_committed_round() {
    let mut run = Run::started(3, retrying(2));
    let aggregator = run.aggregator.insecure_clone();
    let number = run.open_round();
    run.world.refuse(
        void_ix(&run, number, false),
        &[&aggregator],
        LedgerError::InvalidRoundState,
    );
    run.commit(number, &[0, 1, 2], &Posting::filler(number, 1));

    let stranger = run.challenger();
    let forged = client::void_round(&key(&stranger), &run.task, number, false);
    assert!(run.world.send(forged, &[&stranger]).is_err());
    assert_eq!(run.round(number).state(), ROUND_COMMITTED);

    // Once posted it can be acknowledged and settled, so voiding it could fork
    // the model from the log's.
    run.post(number, &Posting::filler(number, 1));
    run.world.refuse(
        void_ix(&run, number, false),
        &[&aggregator],
        LedgerError::InvalidRoundState,
    );
}

#[test]
fn a_task_gets_only_the_retries_it_published() {
    let mut run = Run::started(3, private);
    let aggregator = run.aggregator.insecure_clone();
    let number = committed(&mut run, &[0, 1, 2]);
    run.world.refuse(
        void_ix(&run, number, false),
        &[&aggregator],
        LedgerError::NoRetriesLeft,
    );

    let mut run = Run::started(3, retrying(1));
    let aggregator = run.aggregator.insecure_clone();
    let first = committed(&mut run, &[0, 1, 2]);
    run.world.run(void_ix(&run, first, false), &[&aggregator]);
    let second = committed(&mut run, &[0, 1]);
    run.world.refuse(
        void_ix(&run, second, false),
        &[&aggregator],
        LedgerError::NoRetriesLeft,
    );
}

#[test]
fn nothing_can_be_acknowledged_or_claimed_on_a_voided_round() {
    let mut run = Run::started(3, retrying(1));
    let aggregator = run.aggregator.insecure_clone();
    let number = committed(&mut run, &[0, 1, 2]);
    run.world.run(void_ix(&run, number, false), &[&aggregator]);
    let seal = Posting::filler(number, 1).seal_head;
    let site = run.sites[0].insecure_clone();
    let ack = run.ack_ix(0, number, &seal);
    run.world
        .refuse(ack, &[&site], LedgerError::InvalidRoundState);
    let claim = run.claim_ix(0, number);
    assert!(run.world.send(claim, &[&site]).is_err());
}

#[test]
fn a_run_with_a_void_finishes_after_the_steps_it_asked_for_and_pays_for_those() {
    let mut run = Run::started(3, |terms| {
        retrying(1)(terms);
        terms.rounds = 2;
    });
    let aggregator = run.aggregator.insecure_clone();
    let held = run.world.balance(&run.task);
    let first = committed(&mut run, &[0, 1, 2]);
    run.world.run(void_ix(&run, first, false), &[&aggregator]);

    let second = run.settle_round(&[0, 1]);
    assert_eq!(run.state().state(), TASK_OPEN, "one step is still to come");
    let third = run.settle_round(&[0, 1, 2]);
    assert_eq!((second, third), (2, 3));
    assert_eq!(run.state().state(), TASK_FINISHED);
    let owed = run.state().owed();
    let reward = run.terms.reward_per_round;
    assert!(owed <= 2 * reward && owed >= 2 * reward - 2);
    assert!(run.world.balance(&run.task) >= held - run.world.rent(0));

    // The voided round's rent comes back like any other round's.
    let before = run.world.balance(&key(&run.aggregator));
    let close = client::round_close(&key(&run.aggregator), &run.task, first);
    run.world.run(close, &[]);
    assert!(!run.world.exists(&client::round(&run.task, first)));
    assert!(run.world.balance(&key(&run.aggregator)) > before);
}

#[test]
fn a_removed_helper_means_one_mask_fewer_down_to_the_floor() {
    let mut run = Run::started(3, losing_a_helper);
    let aggregator = run.aggregator.insecure_clone();
    let first = committed(&mut run, &[0, 1, 2]);
    // Without a retry budget, only a removal gives up a round.
    run.world.refuse(
        void_ix(&run, first, false),
        &[&aggregator],
        LedgerError::NoRetriesLeft,
    );
    run.world.run(void_ix(&run, first, true), &[&aggregator]);
    let task = run.state();
    assert_eq!((task.helpers(), task.removed_helpers()), (3, 1));
    assert_eq!(task.active_helpers(), 2);

    let second = run.open_round();
    run.commit(second, &[0, 1, 2], &Posting::filler(second, 2));
    let too_many = run.post_ix(second, &Posting::filler(second, 3));
    run.world
        .refuse(too_many, &[&aggregator], LedgerError::InvalidMaskRoots);
    run.world.refuse(
        void_ix(&run, second, true),
        &[&aggregator],
        LedgerError::NoRetriesLeft,
    );
    run.post(second, &Posting::filler(second, 2));
    assert_eq!(run.round(second).state(), ROUND_POSTED);
}

#[test]
fn publish_refuses_redos_that_cannot_be_honoured() {
    let refused = |adjust: fn(&mut TaskTerms)| {
        let mut run = Run::new(3, adjust);
        let publisher = run.publisher.insecure_clone();
        let ix = client::publish(&key(&publisher), &run.terms);
        run.world
            .refuse(ix, &[&publisher], LedgerError::InvalidRetries);
    };
    // No noise, so two sums would give an update away.
    refused(|t| t.retries = 1);
    // Out of range floors.
    refused(|t| (t.helpers, t.min_helpers) = (3, 1));
    refused(|t| (t.helpers, t.min_helpers) = (3, 3));
    // Both at once isn't offered.
    refused(|t| {
        private(t);
        (t.helpers, t.min_helpers, t.retries) = (3, 2, 1);
    });
    // The budget has to cover every redo at the dearest cohort.
    refused(|t| {
        private(t);
        t.retries = 1000;
    });
}
