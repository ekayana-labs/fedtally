//! What the `did:bio` registry on Solana says about an identity.
//!
//! A `did:bio` identifier holds a key, and with no account on the registry
//! that key speaks for it. Once registered, its document can drop that key or
//! be deactivated for good. This module reads an account image and answers
//! one question: may the key in the identifier still act for it?
//!
//! Nothing here touches the network. Whoever asks fetches the account from an
//! endpoint they trust and passes its bytes in.

use did_bio_core::account::{DidAccountState, KeyType, PROGRAM_ID, vm_flags};
use did_bio_core::{BioDid, find_did_account_address};

/// What a key is being relied on for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    /// Signing a run's records.
    Authentication,
    /// Signing credentials and revocation lists.
    Assertion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Standing {
    /// No account. The identifier's own key speaks for it.
    Unregistered,
    /// A registered document that still lists the identifier's key for this use.
    Active { version: u64 },
    /// A registered document that no longer does.
    Rotated { version: u64 },
    /// Deactivated. Nothing speaks for it again.
    Deactivated,
}

impl Standing {
    pub fn acts(&self) -> bool {
        matches!(self, Standing::Unregistered | Standing::Active { .. })
    }
}

/// An account as an endpoint returns it: the owning program and the data.
#[derive(Clone, Debug)]
pub struct Account {
    pub owner: [u8; 32],
    pub data: Vec<u8>,
}

/// The registry address that holds `did`'s document, to fetch.
pub fn account_address(did: &str) -> Result<[u8; 32], String> {
    let did = BioDid::parse(did).map_err(|e| e.to_string())?;
    Ok(find_did_account_address(&did.subject).0)
}

/// Where `did` stands, given what its registry address holds.
pub fn standing(did: &str, account: Option<&Account>, wanted: Use) -> Result<Standing, String> {
    let did = BioDid::parse(did).map_err(|e| e.to_string())?;
    let Some(account) = account else {
        return if did.is_key_subject() {
            Ok(Standing::Unregistered)
        } else {
            Err("an owned identifier has no key until it is registered".into())
        };
    };
    if account.owner != PROGRAM_ID {
        return Err("that account does not belong to the registry".into());
    }
    let state = DidAccountState::from_account_data(&account.data).map_err(|e| e.to_string())?;
    if state.subject != did.subject {
        return Err("that account is another identifier's".into());
    }
    if state.deactivated {
        return Ok(Standing::Deactivated);
    }
    let flag = match wanted {
        Use::Authentication => vm_flags::AUTHENTICATION,
        Use::Assertion => vm_flags::ASSERTION,
    };
    let listed = state.verification_methods.iter().any(|method| {
        method.method_type == KeyType::Ed25519
            && method.key_data == did.subject
            && method.has_flag(flag)
    });
    let version = state.version;
    Ok(if listed {
        Standing::Active { version }
    } else {
        Standing::Rotated { version }
    })
}
