//! Reads the live devnet registry. Needs the network, so it only runs when
//! asked: `cargo test -p fedtally-node --test devnet -- --ignored`.

use fedtally_core::keys::{Identity, SigAlg};
use fedtally_core::registry::{Standing, Use, account_address, standing};
use fedtally_node::registry::{Registry, RpcRegistry};

#[test]
#[ignore = "needs the network"]
fn the_devnet_registry_answers_through_an_endpoint() {
    let registry = RpcRegistry {
        url: "https://api.devnet.solana.com".into(),
    };
    let ask = |did: &str| {
        let account = registry.account(&account_address(did).unwrap()).unwrap();
        standing(did, account.as_ref(), Use::Assertion).unwrap()
    };
    // An identifier that was deactivated on devnet, and one nobody registered.
    let tombstone = "did:bio:devnet:94veHHJWxZ8c9ifseu63pEA34MmaHnXXB22a6L8n7ZuS";
    assert_eq!(ask(tombstone), Standing::Deactivated);
    let fresh = Identity::generate(SigAlg::Ed25519)
        .unwrap()
        .on_registry("devnet")
        .unwrap();
    assert_eq!(ask(&fresh.did_string()), Standing::Unregistered);
}
