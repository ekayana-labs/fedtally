//! Rounds whose reward follows scored groups.

mod common;

use common::{key, Hash, Posting, Run};
use fedtally_core::merkle;
use fedtally_ledger::{client, error::LedgerError, privacy, state::*};

const REWARD: u64 = 1_000_003;
/// Two groups of two, with sites 0 and 3 in the first. Four sites at a group
/// size of 2 is what the log splits into, and the split here is the test's own.
const SPLIT: [(u32, u16); 4] = [(750_000, 2), (250_000, 2), (250_000, 2), (750_000, 2)];

fn scored(adjust: impl FnOnce(&mut client::TaskTerms)) -> Run {
    Run::started(4, |t| {
        (t.score_group, t.rounds, t.reward_per_round) = (2, 2, REWARD);
        adjust(t);
    })
}

/// What the program reads a site's part from: site, share, group size.
fn leaf(site: u16, (share, members): (u32, u16)) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[..2].copy_from_slice(&site.to_le_bytes());
    bytes[2..6].copy_from_slice(&share.to_le_bytes());
    bytes[6..].copy_from_slice(&members.to_le_bytes());
    bytes
}

fn tree(cohort: &[u16], split: &[(u32, u16)]) -> Vec<Hash> {
    cohort
        .iter()
        .zip(split)
        .map(|(site, part)| merkle::leaf(&leaf(*site, *part)))
        .collect()
}

fn posting(number: u32, cohort: &[u16], split: &[(u32, u16)]) -> Posting {
    Posting {
        shares_root: merkle::root(tree(cohort, split)),
        ..Posting::filler(number, 1)
    }
}

fn claim_ix(
    run: &Run,
    number: u32,
    cohort: &[u16],
    split: &[(u32, u16)],
    position: usize,
    part: (u32, u16),
) -> client::Instruction {
    let site = cohort[position];
    let wallet = key(&run.sites[site as usize]);
    let path = merkle::path(tree(cohort, split), position);
    client::claim_scored(&wallet, &run.task, site, number, part.0, part.1, &path)
}

fn pay(part: (u32, u16)) -> u64 {
    REWARD * u64::from(part.0) / 1_000_000 / u64::from(part.1)
}

#[test]
fn a_site_is_paid_its_groups_part_and_what_does_not_divide_goes_back() {
    let cohort = [0, 1, 2, 3];
    let mut run = scored(|_| {});
    let number = run.settle_round_with(&cohort, &posting(1, &cohort, &SPLIT));
    assert_eq!(run.state().owed(), REWARD);
    assert_eq!(run.round(number).share(), REWARD);

    let mut paid = 0;
    for (position, part) in SPLIT.iter().enumerate() {
        let site = &run.sites[position];
        let wallet = key(site);
        let before = run.world.balance(&wallet);
        let ix = claim_ix(&run, number, &cohort, &SPLIT, position, *part);
        let signer = site.insecure_clone();
        run.world.run(ix, &[&signer]);
        assert_eq!(run.world.balance(&wallet), before + pay(*part));
        paid += pay(*part);
    }
    assert!(paid < REWARD);
    assert_eq!(run.round(number).share(), REWARD - paid);
    assert_eq!(run.state().owed(), REWARD - paid);

    // Every share is claimed, so the round closes with no wait and the rest
    // stops being owed.
    run.settle_round_with(&cohort, &posting(2, &cohort, &SPLIT));
    let aggregator = key(&run.aggregator);
    run.world
        .run(client::round_close(&aggregator, &run.task, number), &[]);
    assert_eq!(run.state().owed(), REWARD);
}

#[test]
fn a_claim_has_to_show_what_the_cohort_acknowledged() {
    let cohort = [0, 1, 2, 3];
    let mut run = scored(|_| {});
    let number = run.settle_round_with(&cohort, &posting(1, &cohort, &SPLIT));
    let signer = run.sites[1].insecure_clone();

    // The other group's share, with this site's true path.
    let greedy = claim_ix(&run, number, &cohort, &SPLIT, 1, (750_000, 2));
    run.world
        .refuse(greedy, &[&signer], LedgerError::InvalidShareProof);
    // The true share as if the group were smaller.
    let alone = claim_ix(&run, number, &cohort, &SPLIT, 1, (250_000, 1));
    run.world
        .refuse(alone, &[&signer], LedgerError::InvalidShareProof);
    // Another site's leaf and path, in this site's name.
    let mut borrowed = claim_ix(&run, number, &cohort, &SPLIT, 0, SPLIT[0]);
    borrowed.accounts[0] = client::AccountMeta::writable(key(&signer), true);
    borrowed.accounts[1] = client::AccountMeta::readonly(client::enrollment(&run.task, 1), false);
    run.world
        .refuse(borrowed, &[&signer], LedgerError::InvalidShareProof);
    // No proof at all.
    let bare = client::claim(&key(&signer), &run.task, 1, number);
    assert!(run.world.send(bare, &[&signer]).is_err());

    let honest = claim_ix(&run, number, &cohort, &SPLIT, 1, SPLIT[1]);
    run.world.run(honest, &[&signer]);
    let again = claim_ix(&run, number, &cohort, &SPLIT, 1, SPLIT[1]);
    run.world
        .refuse(again, &[&signer], LedgerError::AlreadyClaimed);
}

#[test]
fn a_part_of_a_cohort_can_be_anywhere_in_the_task() {
    let cohort = [1, 4, 6];
    let split = [(500_000, 3), (500_000, 3), (0, 3)];
    let mut run = Run::started(7, |t| {
        (t.score_group, t.rounds, t.reward_per_round) = (2, 1, REWARD);
    });
    let number = run.settle_round_with(&cohort, &posting(1, &cohort, &split));
    for (position, part) in split.iter().enumerate() {
        let signer = run.sites[cohort[position] as usize].insecure_clone();
        let ix = claim_ix(&run, number, &cohort, &split, position, *part);
        run.world.run(ix, &[&signer]);
    }
    assert_eq!(run.round(number).claimed(), &client::site_bitmap(cohort));
}

#[test]
fn a_root_that_gives_away_too_much_still_pays_one_rounds_reward() {
    let cohort = [0, 1, 2, 3];
    let split = [(1_000_000, 1); 4];
    let mut run = scored(|_| {});
    let number = run.settle_round_with(&cohort, &posting(1, &cohort, &split));

    let first = run.sites[0].insecure_clone();
    let ix = claim_ix(&run, number, &cohort, &split, 0, split[0]);
    run.world.run(ix, &[&first]);
    assert_eq!(run.round(number).share(), 0);

    let second = run.sites[1].insecure_clone();
    let ix = claim_ix(&run, number, &cohort, &split, 1, split[1]);
    run.world.refuse(ix, &[&second], LedgerError::ShareTooLarge);
    assert_eq!(run.state().owed(), 0);
    // The second round's reward is still in the task.
    let task = run.state();
    assert!(run.world.balance(&run.task) >= task.reward_per_round());
}

#[test]
fn what_nobody_claims_stops_being_owed_once_the_time_to_claim_is_over() {
    let cohort = [0, 1, 2, 3];
    let mut run = scored(|_| {});
    let number = run.settle_round_with(&cohort, &posting(1, &cohort, &SPLIT));
    let signer = run.sites[0].insecure_clone();
    let ix = claim_ix(&run, number, &cohort, &SPLIT, 0, SPLIT[0]);
    run.world.run(ix, &[&signer]);

    run.settle_round_with(&cohort, &posting(2, &cohort, &SPLIT));
    assert_eq!(run.state().owed(), 2 * REWARD - pay(SPLIT[0]));

    let aggregator = key(&run.aggregator);
    let close = client::round_close(&aggregator, &run.task, number);
    run.world
        .refuse(close.clone(), &[], LedgerError::SharesUnclaimed);
    run.world.warp(run.terms.challenge_slots * CLAIM_WINDOWS);
    run.world.run(close, &[]);
    assert_eq!(run.state().owed(), REWARD);
}

#[test]
fn a_root_goes_with_scoring_and_only_with_scoring() {
    let cohort = [0, 1, 2, 3];
    let mut run = scored(|_| {});
    let number = run.open_round();
    let bare = Posting::filler(number, 1);
    run.commit(number, &cohort, &bare);
    let ix = run.post_ix(number, &bare);
    let signer = run.aggregator.insecure_clone();
    run.world
        .refuse(ix, &[&signer], LedgerError::InvalidSharesRoot);

    let mut equal = Run::started(4, |_| {});
    let number = equal.open_round();
    let rooted = posting(number, &cohort, &SPLIT);
    equal.commit(number, &cohort, &rooted);
    let ix = equal.post_ix(number, &rooted);
    let signer = equal.aggregator.insecure_clone();
    equal
        .world
        .refuse(ix, &[&signer], LedgerError::InvalidSharesRoot);
}

#[test]
fn a_task_cannot_score_what_its_log_would_refuse() {
    let refused = |adjust: fn(&mut client::TaskTerms), why: LedgerError| {
        let mut run = Run::new(4, |t| {
            (t.score_group, t.min_cohort) = (2, 2);
            adjust(t);
        });
        let publisher = run.publisher.insecure_clone();
        let ix = client::publish(&key(&publisher), &run.terms);
        run.world.refuse(ix, &[&publisher], why);
    };
    refused(|t| t.score_group = 1, LedgerError::InvalidScoring);
    refused(|t| t.score_group = 3, LedgerError::InvalidScoring);
    refused(|t| t.helpers = 0, LedgerError::InvalidScoring);
    refused(
        |t| {
            t.min_helpers = 2;
            t.helpers = 3;
        },
        LedgerError::InvalidScoring,
    );
    refused(
        |t| {
            (t.retries, t.noise, t.sensitivity, t.budget) = (1, 1 << 34, 1 << 18, u64::MAX);
        },
        LedgerError::InvalidScoring,
    );
    // A group has to hold more sites than may collude, not just a cohort.
    refused(
        |t| {
            t.min_cohort = 4;
            (t.colluding, t.noise, t.sensitivity, t.budget) = (2, 1 << 34, 1 << 18, u64::MAX);
        },
        LedgerError::InvalidCollusion,
    );
}

#[test]
fn a_scored_round_is_charged_at_its_smallest_group() {
    const SENSITIVITY: u64 = 262146;
    const NOISE: u64 = 1 << 34;
    let cost = |sites| privacy::round_cost(SENSITIVITY, NOISE, 0, sites).unwrap();
    let mut run = Run::started(5, |t| {
        (t.score_group, t.min_cohort, t.rounds) = (2, 4, 2);
        (t.sensitivity, t.noise, t.budget) = (SENSITIVITY, NOISE, 2 * cost(2));
    });

    // Five sites split into a group of two and a group of three.
    let number = run.open_round();
    let cohort = [0, 1, 2, 3, 4];
    let split = [(500_000, 2); 5];
    run.commit(number, &cohort, &posting(number, &cohort, &split));
    assert_eq!(run.state().spent(), cost(2));
    assert_eq!(privacy::smallest_group(5, 2), 2);
    assert_eq!(privacy::smallest_group(7, 3), 3);
    assert_eq!(privacy::smallest_group(5, 3), 5);

    // The budget has to cover the dearest group from the start.
    let mut poor = Run::new(4, |t| {
        (t.score_group, t.min_cohort) = (2, 4);
        (t.sensitivity, t.noise, t.budget) = (SENSITIVITY, NOISE, cost(4));
    });
    let publisher = poor.publisher.insecure_clone();
    let ix = client::publish(&key(&publisher), &poor.terms);
    poor.world
        .refuse(ix, &[&publisher], LedgerError::BudgetTooSmall);
}

#[test]
fn the_program_and_the_log_split_a_cohort_alike() {
    for group in 2..9u16 {
        for cohort in group..300 {
            let scoring = fedtally_core::scoring::Scoring {
                group: u32::from(group),
            };
            assert_eq!(
                usize::from(privacy::smallest_group(cohort, group)),
                scoring.smallest(usize::from(cohort))
            );
        }
    }
}
