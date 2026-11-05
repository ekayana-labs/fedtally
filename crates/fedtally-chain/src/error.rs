use fedtally_core::keys::KeyError;
use fedtally_core::replay::Fault;
use fedtally_node::NodeError;
use thiserror::Error;

use crate::owed::Divergence;
use crate::rpc::RpcError;
use crate::tx::TxError;

#[derive(Debug, Error)]
pub enum ChainError {
    #[error(transparent)]
    Rpc(#[from] RpcError),
    #[error(transparent)]
    Tx(#[from] TxError),
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("the log breaks the rules: {0}")]
    Fault(#[from] Fault),
    #[error(transparent)]
    Key(#[from] KeyError),
    #[error(transparent)]
    Node(#[from] NodeError),
    #[error("the log is for another plan")]
    AnotherPlan,
    #[error("not a plan")]
    NotAPlan,
    #[error("the plan cannot be anchored: {0}")]
    Terms(String),
    #[error("this key is not the plan's {0}")]
    NotTheParty(&'static str),
    #[error("this build is for program {built}, not {asked}")]
    Program { built: String, asked: String },
    #[error("{0} holds no program")]
    NoProgram(String),
    #[error("account {address} is not a {kind} of this program")]
    Account { address: String, kind: &'static str },
    #[error("the ledger and the log disagree: {0}")]
    Divergence(#[from] Divergence),
    #[error("{step} was sent {tries} times and never went through: {last}")]
    Stuck {
        step: String,
        tries: u32,
        last: String,
    },
    #[error("the transaction was not confirmed before its blockhash expired")]
    Expired,
    #[error("the endpoint kept failing: {0}")]
    GaveUp(String),
}
