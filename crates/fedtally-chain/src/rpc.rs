//! The few JSON-RPC calls an anchor needs.

use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use fedtally_ledger::error::LedgerError;
use serde_json::{Value, json};
use thiserror::Error;

const TRIES: u32 = 5;
const RETRY_FIRST: Duration = Duration::from_millis(200);
const TIMEOUT: Duration = Duration::from_secs(30);
const COMMITMENT: &str = "confirmed";

/// Why the runtime turned a transaction down.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejection {
    Ledger(LedgerError),
    /// A custom code of another program, such as the system program.
    Custom(u32),
    Other(String),
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rejection::Ledger(e) => write!(f, "{e} ({})", e.code()),
            Rejection::Custom(code) => write!(f, "custom program error {code}"),
            Rejection::Other(text) => f.write_str(text),
        }
    }
}

#[derive(Debug, Error)]
pub enum RpcError {
    #[error("transport: {0}")]
    Transport(String),
    #[error("{method}: unreadable reply: {why}")]
    Reply { method: &'static str, why: String },
    #[error("{method} refused: {code} {message}")]
    Refused {
        method: &'static str,
        code: i64,
        message: String,
    },
    #[error("transaction rejected: {reason}")]
    Rejected {
        reason: Rejection,
        logs: Vec<String>,
    },
}

impl RpcError {
    pub fn is_transient(&self) -> bool {
        matches!(self, RpcError::Transport(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Account {
    pub lamports: u64,
    pub owner: [u8; 32],
    pub executable: bool,
    pub data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blockhash {
    pub hash: [u8; 32],
    pub last_valid_height: u64,
}

/// What a landed transaction cost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Landed {
    pub slot: u64,
    pub fee: u64,
    pub units: Option<u64>,
    pub failed: bool,
}

pub struct Rpc {
    url: String,
    agent: ureq::Agent,
}

fn address(text: &str) -> Option<[u8; 32]> {
    bs58::decode(text).into_vec().ok()?.try_into().ok()
}

/// Reads the reason out of a transaction error as the endpoint encodes it.
pub fn rejection(err: &Value) -> Rejection {
    let custom = err
        .pointer("/InstructionError/1/Custom")
        .and_then(Value::as_u64)
        .and_then(|code| u32::try_from(code).ok());
    match custom {
        Some(code) => {
            LedgerError::from_code(code).map_or(Rejection::Custom(code), Rejection::Ledger)
        }
        None => Rejection::Other(err.to_string()),
    }
}

fn account(value: &Value) -> Option<Account> {
    let data = value.pointer("/data/0")?.as_str()?;
    Some(Account {
        lamports: value.get("lamports")?.as_u64()?,
        owner: address(value.get("owner")?.as_str()?)?,
        executable: value.get("executable")?.as_bool()?,
        data: STANDARD.decode(data).ok()?,
    })
}

fn unreadable(method: &'static str, why: &str) -> RpcError {
    RpcError::Reply {
        method,
        why: why.to_owned(),
    }
}

impl Rpc {
    pub fn new(url: &str) -> Rpc {
        Rpc {
            url: url.to_owned(),
            agent: ureq::AgentBuilder::new().timeout(TIMEOUT).build(),
        }
    }

    fn once(&self, method: &'static str, params: &Value) -> Result<Value, RpcError> {
        let request = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
        let reply: Value = match self.agent.post(&self.url).send_json(request) {
            Ok(response) => response
                .into_json()
                .map_err(|e| RpcError::Transport(e.to_string()))?,
            // Some endpoints put a JSON-RPC error in the body of a 4xx.
            Err(ureq::Error::Status(code, response)) => response
                .into_json()
                .map_err(|_| RpcError::Transport(format!("status {code}")))?,
            Err(e) => return Err(RpcError::Transport(e.to_string())),
        };
        if let Some(error) = reply.get("error") {
            if let Some(err) = error.pointer("/data/err").filter(|e| !e.is_null()) {
                let logs = error.pointer("/data/logs").and_then(Value::as_array);
                return Err(RpcError::Rejected {
                    reason: rejection(err),
                    logs: logs
                        .into_iter()
                        .flatten()
                        .filter_map(|line| line.as_str().map(str::to_owned))
                        .collect(),
                });
            }
            return Err(RpcError::Refused {
                method,
                code: error.get("code").and_then(Value::as_i64).unwrap_or(0),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            });
        }
        reply
            .get("result")
            .cloned()
            .ok_or(unreadable(method, "no result"))
    }

    fn call(&self, method: &'static str, params: Value) -> Result<Value, RpcError> {
        let mut wait = RETRY_FIRST;
        let mut tries = 1;
        loop {
            match self.once(method, &params) {
                Err(e) if e.is_transient() && tries < TRIES => {
                    tracing::debug!("{method} failed, trying again: {e}");
                    std::thread::sleep(wait);
                    wait *= 2;
                    tries += 1;
                }
                other => return other,
            }
        }
    }

    pub fn slot(&self) -> Result<u64, RpcError> {
        let result = self.call("getSlot", json!([{"commitment": COMMITMENT}]))?;
        result.as_u64().ok_or(unreadable("getSlot", "not a number"))
    }

    pub fn block_height(&self) -> Result<u64, RpcError> {
        let result = self.call("getBlockHeight", json!([{"commitment": COMMITMENT}]))?;
        let height = result.as_u64();
        height.ok_or(unreadable("getBlockHeight", "not a number"))
    }

    pub fn latest_blockhash(&self) -> Result<Blockhash, RpcError> {
        const METHOD: &str = "getLatestBlockhash";
        let result = self.call(METHOD, json!([{"commitment": COMMITMENT}]))?;
        let hash = result.pointer("/value/blockhash").and_then(Value::as_str);
        let height = result.pointer("/value/lastValidBlockHeight");
        match (hash.and_then(address), height.and_then(Value::as_u64)) {
            (Some(hash), Some(last_valid_height)) => Ok(Blockhash {
                hash,
                last_valid_height,
            }),
            _ => Err(unreadable(METHOD, "no blockhash")),
        }
    }

    /// The accounts at `addresses`, in order. One that does not exist is `None`.
    pub fn accounts(&self, addresses: &[[u8; 32]]) -> Result<Vec<Option<Account>>, RpcError> {
        const METHOD: &str = "getMultipleAccounts";
        let mut found = Vec::with_capacity(addresses.len());
        // The endpoint takes at most 100 addresses a call.
        for batch in addresses.chunks(100) {
            let names: Vec<String> = batch
                .iter()
                .map(|a| bs58::encode(a).into_string())
                .collect();
            let options = json!({"encoding": "base64", "commitment": COMMITMENT});
            let result = self.call(METHOD, json!([names, options]))?;
            let values = result.pointer("/value").and_then(Value::as_array);
            let values = values
                .filter(|values| values.len() == batch.len())
                .ok_or(unreadable(METHOD, "wrong number of accounts"))?;
            for value in values {
                found.push(match value {
                    Value::Null => None,
                    value => Some(account(value).ok_or(unreadable(METHOD, "bad account"))?),
                });
            }
        }
        Ok(found)
    }

    pub fn account(&self, address: &[u8; 32]) -> Result<Option<Account>, RpcError> {
        Ok(self.accounts(&[*address])?.pop().flatten())
    }

    pub fn balance(&self, address: &[u8; 32]) -> Result<u64, RpcError> {
        Ok(self.account(address)?.map_or(0, |account| account.lamports))
    }

    pub fn rent_exempt_minimum(&self, bytes: usize) -> Result<u64, RpcError> {
        const METHOD: &str = "getMinimumBalanceForRentExemption";
        let result = self.call(METHOD, json!([bytes, {"commitment": COMMITMENT}]))?;
        result.as_u64().ok_or(unreadable(METHOD, "not a number"))
    }

    /// Sends a signed transaction and returns its signature. The endpoint
    /// simulates it first, so one the program refuses never lands.
    pub fn send_transaction(&self, wire: &[u8]) -> Result<String, RpcError> {
        const METHOD: &str = "sendTransaction";
        let options = json!({"encoding": "base64", "preflightCommitment": COMMITMENT});
        let result = self.call(METHOD, json!([STANDARD.encode(wire), options]))?;
        let signature = result.as_str().map(str::to_owned);
        signature.ok_or(unreadable(METHOD, "no signature"))
    }

    /// Whether a transaction is confirmed: `None` while it is not, then how
    /// it ended.
    pub fn confirmed(&self, signature: &str) -> Result<Option<Result<(), Rejection>>, RpcError> {
        const METHOD: &str = "getSignatureStatuses";
        let result = self.call(METHOD, json!([[signature]]))?;
        let status = result
            .pointer("/value/0")
            .ok_or(unreadable(METHOD, "no status"))?;
        let level = status.get("confirmationStatus").and_then(Value::as_str);
        if !matches!(level, Some("confirmed" | "finalized")) {
            return Ok(None);
        }
        Ok(Some(match status.get("err") {
            None | Some(Value::Null) => Ok(()),
            Some(err) => Err(rejection(err)),
        }))
    }

    /// What a confirmed transaction cost, once the endpoint has it.
    pub fn landed(&self, signature: &str) -> Result<Option<Landed>, RpcError> {
        const METHOD: &str = "getTransaction";
        let options = json!({
            "encoding": "json",
            "commitment": COMMITMENT,
            "maxSupportedTransactionVersion": 0,
        });
        let result = self.call(METHOD, json!([signature, options]))?;
        if result.is_null() {
            return Ok(None);
        }
        let number = |pointer: &str| result.pointer(pointer).and_then(Value::as_u64);
        match (number("/slot"), number("/meta/fee")) {
            (Some(slot), Some(fee)) => Ok(Some(Landed {
                slot,
                fee,
                units: number("/meta/computeUnitsConsumed"),
                failed: result.pointer("/meta/err").is_some_and(|e| !e.is_null()),
            })),
            _ => Err(unreadable(METHOD, "no slot or fee")),
        }
    }

    /// Asks a local validator's faucet for lamports.
    pub fn request_airdrop(&self, address: &[u8; 32], lamports: u64) -> Result<String, RpcError> {
        const METHOD: &str = "requestAirdrop";
        let name = bs58::encode(address).into_string();
        let result = self.call(METHOD, json!([name, lamports]))?;
        let signature = result.as_str().map(str::to_owned);
        signature.ok_or(unreadable(METHOD, "no signature"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ledger_error_is_read_from_the_code() {
        let err = json!({"InstructionError": [1, {"Custom": 6026}]});
        assert_eq!(
            rejection(&err),
            Rejection::Ledger(LedgerError::AlreadyAcked)
        );
        let err = json!({"InstructionError": [0, {"Custom": 1}]});
        assert_eq!(rejection(&err), Rejection::Custom(1));
        let err = json!({"InstructionError": [0, "InvalidAccountData"]});
        assert!(
            matches!(rejection(&err), Rejection::Other(text) if text.contains("InvalidAccountData"))
        );
        assert!(matches!(
            rejection(&json!("BlockhashNotFound")),
            Rejection::Other(_)
        ));
    }

    #[test]
    fn an_account_decodes_or_is_refused() {
        let owner = bs58::encode([7u8; 32]).into_string();
        let value = json!({
            "lamports": 12, "owner": owner, "executable": false,
            "data": [STANDARD.encode([1u8, 2, 3]), "base64"],
        });
        let got = account(&value).unwrap();
        assert_eq!(
            (got.lamports, got.owner, got.data),
            (12, [7; 32], vec![1, 2, 3])
        );
        let mut broken = value.clone();
        broken["owner"] = json!("not base58 !");
        assert!(account(&broken).is_none());
        broken = value;
        broken["data"] = json!(["***", "base64"]);
        assert!(account(&broken).is_none());
    }
}
