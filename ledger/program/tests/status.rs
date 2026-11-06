//! Status lists, and what a set bit does to an enrollment that names it.

mod common;

use common::{key, Posting, Run, World, WALLET};
use fedtally_ledger::{
    client::{self, StatusRef},
    error::LedgerError,
    state::*,
};

#[test]
fn status_init_creates_a_clear_list_of_up_to_80000_bits() {
    let mut world = World::new();
    let issuer = world.wallet();
    let who = key(&issuer);

    for bits in [0, 80_001] {
        world.refuse(
            client::status_init(&who, 1, bits),
            &[&issuer],
            LedgerError::InvalidStatusList,
        );
    }
    let mut unsigned = client::status_init(&who, 1, 100);
    unsigned.accounts[0].is_signer = false;
    world.refuse(unsigned, &[], "MissingRequiredSignature");
    let mut trailing = client::status_init(&who, 1, 100);
    trailing.data.push(0);
    world.refuse(trailing, &[&issuer], "InvalidInstructionData");
    let mut wrong_account = client::status_init(&who, 1, 100);
    wrong_account.accounts[1].address = client::status_list(&who, 2);
    world.refuse(wrong_account, &[&issuer], "InvalidSeeds");

    world.run(client::status_init(&who, 1, 100), &[&issuer]);
    let list = world.status_list(&who, 1);
    assert_eq!(list.address().as_array(), &client::status_list(&who, 1));
    assert_eq!((list.issuer(), list.list()), (&who, 1));
    assert_eq!((list.bits(), list.sequence()), (100, 0));
    assert_eq!(list.bitmap(), [0; 13]);
    assert_eq!(
        world.balance(&who),
        WALLET - world.rent(status_list_space(100))
    );
    world.refuse(
        client::status_init(&who, 1, 100),
        &[&issuer],
        "AccountAlreadyInitialized",
    );

    // An issuer has as many lists as it numbers, up to the largest size.
    world.run(client::status_init(&who, 2, 80_000), &[&issuer]);
    assert_eq!(world.status_list(&who, 2).bitmap().len(), 10_000);
    world.run(client::status_init(&who, 3, 1), &[&issuer]);
    assert_eq!(world.status_list(&who, 3).bitmap().len(), 1);
}

#[test]
fn revoke_sets_a_bit_once_and_counts_the_change() {
    let mut world = World::new();
    let issuer = world.wallet();
    let who = key(&issuer);
    world.run(client::status_init(&who, 7, 80_000), &[&issuer]);

    world.refuse(
        client::revoke(&who, 7, 80_000),
        &[&issuer],
        LedgerError::InvalidBit,
    );
    let stranger = world.wallet();
    let mut by_stranger = client::revoke(&who, 7, 3);
    by_stranger.accounts[0].address = key(&stranger);
    world.refuse(by_stranger, &[&stranger], LedgerError::Unauthorized);
    let mut unsigned = client::revoke(&who, 7, 3);
    unsigned.accounts[0].is_signer = false;
    world.refuse(unsigned, &[], "MissingRequiredSignature");
    let mut trailing = client::revoke(&who, 7, 3);
    trailing.data.push(0);
    world.refuse(trailing, &[&issuer], "InvalidInstructionData");
    world.refuse(
        client::revoke(&who, 8, 3),
        &[&issuer],
        "InvalidAccountOwner",
    );

    for (count, bit) in [3u32, 12, 79_999].into_iter().enumerate() {
        world.run(client::revoke(&who, 7, bit), &[&issuer]);
        let list = world.status_list(&who, 7);
        assert!(list.is_set(bit));
        assert_eq!(list.sequence(), count as u64 + 1);
        world.refuse(
            client::revoke(&who, 7, bit),
            &[&issuer],
            LedgerError::AlreadyRevoked,
        );
    }
    let list = world.status_list(&who, 7);
    assert_eq!(list.bitmap()[..2], [0b0000_1000, 0b0001_0000]);
    assert_eq!(list.bitmap()[9_999], 0b1000_0000);
    assert_eq!(list.bitmap().iter().map(|b| b.count_ones()).sum::<u32>(), 3);
    assert_eq!(list.sequence(), 3);
}

/// A task whose site 2 names bit 5 of one list and bit 9 of another, the
/// second of which exists. Returns the issuer and the two list addresses.
fn with_references() -> (Run, solana_keypair::Keypair, [[u8; 32]; 2]) {
    let mut run = Run::new(4, |t| t.rounds = 3);
    run.publish();
    run.back();
    let issuer = run.world.wallet();
    let who = key(&issuer);
    let lists = [client::status_list(&who, 1), client::status_list(&who, 2)];
    run.world.run(client::status_init(&who, 2, 64), &[&issuer]);
    for index in [0, 1, 3] {
        run.enroll(index);
    }
    let refs = [
        StatusRef {
            list: lists[0],
            bit: 5,
        },
        StatusRef {
            list: lists[1],
            bit: 9,
        },
    ];
    run.enroll_with(2, &refs);
    (run, issuer, lists)
}

#[test]
fn revoke_check_takes_the_lists_the_enrollment_names_in_order() {
    let (mut run, issuer, lists) = with_references();
    let who = key(&issuer);
    let check = |lists: &[[u8; 32]]| client::revoke_check(&run.task, 2, lists);

    let mismatched = [
        check(&[lists[1], lists[0]]),
        check(&[lists[0]]),
        check(&[]),
        check(&[lists[0], lists[1], lists[1]]),
        check(&[lists[0], client::status_list(&who, 3)]),
    ];
    for ix in mismatched {
        run.world.refuse(ix, &[], LedgerError::StatusListMismatch);
    }
    let mut trailing = check(&lists);
    trailing.data.push(0);
    run.world.refuse(trailing, &[], "InvalidInstructionData");
    let mut other_site = check(&lists);
    other_site.accounts[1].address = client::enrollment(&run.task, 1);
    run.world
        .refuse(other_site, &[], LedgerError::StatusListMismatch);
    let mut readonly = check(&lists);
    readonly.accounts[0].is_writable = false;
    run.world.refuse(readonly, &[], "Immutable");

    // No bit is set yet, so the check passes and changes nothing. Bits the
    // enrollment does not name do not count.
    run.world.run(client::revoke(&who, 2, 8), &[&issuer]);
    let ix = check(&lists);
    run.world.run(ix, &[]);
    assert_eq!(run.world.enrollment(&run.task, 2).status(), SITE_ACTIVE);
    assert_eq!(run.state().revoked(), &[0; 128]);

    // A site without references is checked with no lists at all.
    run.world.run(client::revoke_check(&run.task, 0, &[]), &[]);
    assert_eq!(run.world.enrollment(&run.task, 0).status(), SITE_ACTIVE);
}

#[test]
fn a_set_bit_revokes_the_site_and_keeps_it_out_of_the_next_cohort() {
    let (mut run, issuer, lists) = with_references();
    let who = key(&issuer);
    run.settle_round(&[1, 2, 3]);

    // Round 2 is committed with site 2 before its bit is set.
    let posting = Posting::filler(2, 1);
    let number = run.post_round(&[0, 2], &posting);
    run.world.run(client::revoke(&who, 2, 9), &[&issuer]);
    assert_eq!(run.world.enrollment(&run.task, 2).status(), SITE_ACTIVE);

    let enrollment = client::enrollment(&run.task, 2);
    let staked = run.world.balance(&enrollment);
    let check = client::revoke_check(&run.task, 2, &lists);
    run.world.run(check.clone(), &[]);
    let site = run.world.enrollment(&run.task, 2);
    assert_eq!(site.status(), SITE_REVOKED);
    assert_eq!(run.state().revoked(), &client::site_bitmap([2]));
    assert_eq!(run.world.balance(&enrollment), staked);
    assert_eq!(staked, run.world.rent(ENROLLMENT_SPACE) + site.stake());
    run.world.run(check.clone(), &[]);
    assert_eq!(run.world.enrollment(&run.task, 2).status(), SITE_REVOKED);

    // The site finishes the round it was committed to and is paid for it.
    run.ack(0, number);
    run.ack(2, number);
    run.world.warp(100);
    run.settle(number);
    run.claim(2, number);

    // The log may have closed the next round before this was applied, so the
    // ledger takes the log's cohort, and keeping the site out is the log's job.
    let next = run.open_round();
    let posting = Posting::filler(3, 1);
    run.commit(next, &[1, 2, 3], &posting);

    // Revoked, it takes its stake back while the task is still open, and
    // an active site does not.
    run.world.refuse(
        run.withdraw_site_ix(1),
        &[&run.sites[1]],
        LedgerError::FundsLocked,
    );
    run.world.run(run.withdraw_site_ix(2), &[&run.sites[2]]);
    assert_eq!(run.world.enrollment(&run.task, 2).status(), SITE_WITHDRAWN);
    assert_eq!(
        run.world.balance(&enrollment),
        run.world.rent(ENROLLMENT_SPACE)
    );
    run.claim(2, 1);
    let shares = run.round(1).share() + run.round(2).share();
    assert_eq!(
        run.world.balance(&key(&run.sites[2])),
        WALLET - run.world.rent(ENROLLMENT_SPACE) + shares
    );

    // A later check leaves the withdrawn enrollment as it is.
    run.world.run(check, &[]);
    assert_eq!(run.world.enrollment(&run.task, 2).status(), SITE_WITHDRAWN);
    assert_eq!(run.state().revoked(), &client::site_bitmap([2]));
}

#[test]
fn a_reference_to_a_list_that_was_never_created_shields_nothing() {
    let (mut run, issuer, lists) = with_references();
    let who = key(&issuer);
    run.world.run(client::revoke(&who, 2, 9), &[&issuer]);
    run.world
        .run(client::revoke_check(&run.task, 2, &lists), &[]);
    assert_eq!(run.world.enrollment(&run.task, 2).status(), SITE_REVOKED);
}

#[test]
fn a_bit_in_the_first_list_revokes_as_well() {
    let (mut run, issuer, lists) = with_references();
    let who = key(&issuer);
    run.world.run(client::status_init(&who, 1, 6), &[&issuer]);
    run.world
        .run(client::revoke_check(&run.task, 2, &lists), &[]);
    assert_eq!(run.world.enrollment(&run.task, 2).status(), SITE_ACTIVE);
    run.world.run(client::revoke(&who, 1, 5), &[&issuer]);
    run.world
        .run(client::revoke_check(&run.task, 2, &lists), &[]);
    assert_eq!(run.world.enrollment(&run.task, 2).status(), SITE_REVOKED);
}
