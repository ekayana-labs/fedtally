//! Asking the `did:bio` registry whether the parties a site relies on still
//! stand. A site asks for itself, through an endpoint it trusts, before each
//! round. Nothing in the log depends on the answer, so a replay stays offline.

use std::collections::BTreeSet;
use std::fmt;

use base64::Engine;
use fedtally_core::plan::Plan;
use fedtally_core::registry::{Account, Use, account_address, standing};
use serde_json::{Value, json};

pub trait Registry: Send + Sync {
    /// What the registry address holds, or nothing.
    fn account(&self, address: &[u8; 32]) -> Result<Option<Account>, String>;
}

#[derive(Debug)]
pub enum Unsure {
    /// The registry couldn't be asked. It may answer next time.
    Unreachable(String),
    /// It answered, and someone no longer stands.
    Refused(String),
}

impl fmt::Display for Unsure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsure::Unreachable(why) | Unsure::Refused(why) => f.write_str(why),
        }
    }
}

/// Checks every `did:bio` operator, site and issuer the plan relies on, leaving
/// out helpers already removed from the run. The first one whose own key may no
/// longer act for it is an error.
pub fn check(
    registry: &dyn Registry,
    plan: &Plan,
    removed: &BTreeSet<usize>,
) -> Result<(), Unsure> {
    let sites = plan.sites.iter().map(String::as_str);
    // The aggregator comes first, then the helpers in plan order.
    let in_run = |(at, _): &(usize, _)| *at == 0 || !removed.contains(&(at - 1));
    let operators = plan
        .operators()
        .enumerate()
        .filter(in_run)
        .map(|(_, (did, _))| did)
        .chain(sites)
        .map(|did| (did, Use::Authentication));
    let issuers = plan
        .admission
        .iter()
        .flat_map(|policy| policy.issuers())
        .map(|did| (did, Use::Assertion));
    for (did, wanted) in operators.chain(issuers) {
        if !did.starts_with("did:bio:") {
            continue;
        }
        let address = account_address(did).map_err(Unsure::Refused)?;
        let account = registry.account(&address).map_err(Unsure::Unreachable)?;
        let found = standing(did, account.as_ref(), wanted).map_err(Unsure::Refused)?;
        if !found.acts() {
            return Err(Unsure::Refused(format!(
                "{did} no longer stands on the registry: {found:?}"
            )));
        }
    }
    Ok(())
}

/// A Solana JSON-RPC endpoint.
pub struct RpcRegistry {
    pub url: String,
}

impl Registry for RpcRegistry {
    fn account(&self, address: &[u8; 32]) -> Result<Option<Account>, String> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [bs58::encode(address).into_string(), {"encoding": "base64"}],
        });
        let reply: Value = ureq::post(&self.url)
            .send_json(request)
            .map_err(|e| e.to_string())?
            .into_json()
            .map_err(|e| e.to_string())?;
        if let Some(error) = reply.get("error") {
            return Err(format!("the endpoint refused: {error}"));
        }
        let value = reply
            .pointer("/result/value")
            .ok_or("the endpoint's reply has no result")?;
        if value.is_null() {
            return Ok(None);
        }
        let field = |pointer: &str| value.pointer(pointer).and_then(Value::as_str);
        let owner = field("/owner")
            .and_then(|text| bs58::decode(text).into_vec().ok())
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or("the reply names no owner")?;
        let data = field("/data/0")
            .and_then(|text| base64::engine::general_purpose::STANDARD.decode(text).ok())
            .ok_or("the reply holds no data")?;
        Ok(Some(Account { owner, data }))
    }
}
