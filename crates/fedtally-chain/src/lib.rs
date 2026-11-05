//! Posts what a party owes the ledger. Each role follows the log, reads the
//! ledger's accounts, and sends the one transaction that is due next. Nothing
//! about the ledger is remembered between steps, so a restart picks up from
//! what the accounts say.

pub mod anchor;
pub mod error;
pub mod ledger;
pub mod owed;
pub mod rpc;
pub mod run;
pub mod tx;
mod watch;

pub use anchor::{Anchor, Config, Role, Stakes, Watching};
pub use error::ChainError;
pub use fedtally_ledger as program;
