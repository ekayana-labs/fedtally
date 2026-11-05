//! The registry check, against every account the deployed devnet program had
//! written when the fixture was taken with `getProgramAccounts`.

use base64::Engine;
use fedtally_core::registry::{Account, Standing, Use, account_address, standing};

const PROGRAM: &str = "H1gnV4GjNT3UV7AgGNUCkSaciuVVtM7hKb8JhPV3Xxy6";

fn decode32(text: &str) -> [u8; 32] {
    bs58::decode(text).into_vec().unwrap().try_into().unwrap()
}

struct Image {
    address: [u8; 32],
    did: String,
    account: Account,
}

fn images() -> Vec<Image> {
    let accounts: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("fixtures/devnet-accounts.json")).unwrap();
    accounts
        .iter()
        .map(|entry| {
            let data = base64::engine::general_purpose::STANDARD
                .decode(entry["data"].as_str().unwrap())
                .unwrap();
            // The subject follows the 8 byte tag, the version and the bump.
            let subject = bs58::encode(&data[17..49]).into_string();
            Image {
                address: decode32(entry["address"].as_str().unwrap()),
                did: format!("did:bio:devnet:{subject}"),
                account: Account {
                    owner: decode32(PROGRAM),
                    data,
                },
            }
        })
        .collect()
}

#[test]
fn every_devnet_account_is_found_at_its_address_and_classified() {
    let images = images();
    assert_eq!(images.len(), 21);
    let (mut active, mut rotated, mut deactivated) = (0, 0, 0);
    for image in &images {
        assert_eq!(
            account_address(&image.did).unwrap(),
            image.address,
            "{}",
            image.did
        );
        match standing(&image.did, Some(&image.account), Use::Assertion).unwrap() {
            Standing::Active { version } => {
                assert!(version >= 1);
                active += 1;
            }
            Standing::Rotated { .. } => rotated += 1,
            Standing::Deactivated => deactivated += 1,
            Standing::Unregistered => panic!("an account exists for {}", image.did),
        }
    }
    assert_eq!(active + rotated + deactivated, 21);
    assert!(active > 0, "no account still lists its own key");
    assert_eq!((active, rotated, deactivated), (11, 7, 3));
}

#[test]
fn only_a_live_document_that_lists_the_key_lets_it_act() {
    let images = images();
    let find = |want: fn(&Standing) -> bool| {
        images
            .iter()
            .find(|i| want(&standing(&i.did, Some(&i.account), Use::Assertion).unwrap()))
    };
    let live = find(|s| matches!(s, Standing::Active { .. })).unwrap();
    let dead = find(|s| matches!(s, Standing::Deactivated)).unwrap();
    assert!(
        standing(&live.did, Some(&live.account), Use::Assertion)
            .unwrap()
            .acts()
    );
    assert!(
        !standing(&dead.did, Some(&dead.account), Use::Assertion)
            .unwrap()
            .acts()
    );
    assert!(!Standing::Rotated { version: 3 }.acts());

    // With no account the identifier's own key speaks for it.
    assert_eq!(
        standing(&live.did, None, Use::Assertion),
        Ok(Standing::Unregistered)
    );
    assert!(Standing::Unregistered.acts());

    // Another identifier's account, or one the registry does not own, proves nothing.
    assert!(standing(&live.did, Some(&dead.account), Use::Assertion).is_err());
    let mut foreign = live.account.clone();
    foreign.owner = [7; 32];
    assert!(standing(&live.did, Some(&foreign), Use::Assertion).is_err());
    let mut cut = live.account.clone();
    cut.data.truncate(20);
    assert!(standing(&live.did, Some(&cut), Use::Assertion).is_err());
    assert!(
        standing(
            "did:key:z6MkfuN2vWAoHermh6vY6TgAJfwhBWZCApZb9XeQ9HwLqHfz",
            None,
            Use::Assertion
        )
        .is_err()
    );
}

#[test]
fn deactivation_ends_every_use() {
    for image in &images() {
        let auth = standing(&image.did, Some(&image.account), Use::Authentication).unwrap();
        let assertion = standing(&image.did, Some(&image.account), Use::Assertion).unwrap();
        assert_eq!(
            matches!(auth, Standing::Deactivated),
            matches!(assertion, Standing::Deactivated)
        );
    }
}
