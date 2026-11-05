//! Legacy transactions with one signature: the party's own key, which also
//! pays the fee.

use std::sync::Arc;

use fedtally_core::anchor;
use fedtally_core::keys::Identity;
use fedtally_core::pq_ucan::crypto::Signer;
use fedtally_ledger::client;
use solana_compute_budget_interface::ComputeBudgetInstruction;
use solana_hash::Hash;
use solana_instruction::{AccountMeta, Instruction};
use solana_message::Message;
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_transaction::Transaction;
use thiserror::Error;

/// The most bytes a transaction may take on the wire.
pub const MAX_WIRE: usize = 1232;
/// What each compute budget instruction itself consumes.
pub const BUDGET_IX_UNITS: u64 = 150;

#[derive(Debug, Error)]
pub enum TxError {
    #[error("{0}")]
    NoAddress(String),
    #[error("signing failed: {0}")]
    Sign(String),
    #[error("the instruction needs a signature other than the wallet's")]
    Signers,
    #[error("the transaction takes {0} bytes, over the limit of {MAX_WIRE}")]
    TooLarge(usize),
    #[error("encoding failed: {0}")]
    Encode(String),
}

/// A party's key, used as its Solana account.
#[derive(Clone)]
pub struct Wallet {
    identity: Arc<Identity>,
    address: [u8; 32],
}

impl Wallet {
    pub fn new(identity: Arc<Identity>) -> Result<Wallet, TxError> {
        let address = anchor::wallet(&identity.did_string()).map_err(TxError::NoAddress)?;
        Ok(Wallet { identity, address })
    }

    pub fn address(&self) -> &[u8; 32] {
        &self.address
    }

    pub fn did(&self) -> String {
        self.identity.did_string()
    }

    fn sign(&self, message: &[u8]) -> Result<[u8; 64], TxError> {
        let signature = self
            .identity
            .sign(message)
            .map_err(|e| TxError::Sign(format!("{e:?}")))?;
        signature
            .as_bytes()
            .try_into()
            .map_err(|_| TxError::Sign("not an Ed25519 signature".into()))
    }
}

/// The compute budget a transaction asks for. Left empty, the runtime gives
/// 200,000 units and charges no priority fee.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Budget {
    pub units: Option<u32>,
    /// Micro-lamports per compute unit.
    pub price: Option<u64>,
}

impl Budget {
    /// How many instructions the budget puts in front of the ledger's.
    pub fn instructions(&self) -> u64 {
        u64::from(self.units.is_some()) + u64::from(self.price.is_some())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signed {
    pub signature: String,
    pub wire: Vec<u8>,
}

fn sdk(ix: client::Instruction) -> Instruction {
    Instruction {
        program_id: Pubkey::new_from_array(ix.program_id),
        accounts: ix
            .accounts
            .into_iter()
            .map(|meta| AccountMeta {
                pubkey: Pubkey::new_from_array(meta.address),
                is_signer: meta.is_signer,
                is_writable: meta.is_writable,
            })
            .collect(),
        data: ix.data,
    }
}

/// Builds and signs a transaction that carries `ix` and is paid by `wallet`.
pub fn sign(
    wallet: &Wallet,
    ix: client::Instruction,
    budget: Budget,
    blockhash: [u8; 32],
) -> Result<Signed, TxError> {
    let mut ixs = Vec::with_capacity(3);
    if let Some(units) = budget.units {
        ixs.push(ComputeBudgetInstruction::set_compute_unit_limit(units));
    }
    if let Some(price) = budget.price {
        ixs.push(ComputeBudgetInstruction::set_compute_unit_price(price));
    }
    ixs.push(sdk(ix));
    let payer = Pubkey::new_from_array(wallet.address);
    let message = Message::new_with_blockhash(&ixs, Some(&payer), &Hash::new_from_array(blockhash));
    let mut tx = Transaction::new_unsigned(message);
    if tx.signatures.len() != 1 {
        return Err(TxError::Signers);
    }
    let signature = Signature::from(wallet.sign(&tx.message_data())?);
    tx.signatures[0] = signature;
    let wire = wincode::serialize(&tx).map_err(|e| TxError::Encode(e.to_string()))?;
    if wire.len() > MAX_WIRE {
        return Err(TxError::TooLarge(wire.len()));
    }
    Ok(Signed {
        signature: signature.to_string(),
        wire,
    })
}

#[cfg(test)]
mod tests {
    use fedtally_core::keys::{SigAlg, verify_did};
    use fedtally_core::pq_ucan::did::Did;

    use super::*;

    fn wallet() -> Wallet {
        Wallet::new(Arc::new(Identity::generate(SigAlg::Ed25519).unwrap())).unwrap()
    }

    #[test]
    fn a_transaction_is_signed_by_the_key_in_the_partys_name() {
        let wallet = wallet();
        let task = client::task(&[3; 32], &[4; 32]);
        let ix = client::ack(wallet.address(), &task, 2, 7, &[9; 32]);
        let signed = sign(&wallet, ix.clone(), Budget::default(), [5; 32]).unwrap();

        let tx: Transaction = wincode::deserialize(&signed.wire).unwrap();
        assert_eq!(tx.signatures.len(), 1);
        assert_eq!(tx.signatures[0].to_string(), signed.signature);
        assert_eq!(tx.message.account_keys[0].to_bytes(), *wallet.address());
        assert_eq!(tx.message.recent_blockhash.to_bytes(), [5; 32]);
        let header = tx.message.header;
        assert_eq!(header.num_required_signatures, 1);

        // One signature, then the message, as the wire format has it.
        assert_eq!(signed.wire[0], 1);
        let message = &signed.wire[65..];
        assert_eq!(message, tx.message_data().as_slice());
        let did = Did::parse(&wallet.did()).unwrap();
        verify_did(&did, message, &signed.wire[1..65]).unwrap();

        let compiled = &tx.message.instructions[0];
        assert_eq!(compiled.data, ix.data);
        let named: Vec<[u8; 32]> = compiled
            .accounts
            .iter()
            .map(|i| tx.message.account_keys[*i as usize].to_bytes())
            .collect();
        let wanted: Vec<[u8; 32]> = ix.accounts.iter().map(|meta| meta.address).collect();
        assert_eq!(named, wanted);
        let program = tx.message.account_keys[compiled.program_id_index as usize];
        assert_eq!(program.to_bytes(), ix.program_id);
    }

    #[test]
    fn a_budget_goes_in_front_of_the_instruction() {
        let wallet = wallet();
        let ix = client::settle(&[1; 32], 1);
        let budget = Budget {
            units: Some(300_000),
            price: Some(1_000),
        };
        assert_eq!(budget.instructions(), 2);
        let signed = sign(&wallet, ix.clone(), budget, [0; 32]).unwrap();
        let tx: Transaction = wincode::deserialize(&signed.wire).unwrap();
        let ixs = &tx.message.instructions;
        assert_eq!(ixs.len(), 3);
        // The compute budget program's own encoding: a tag, then the number.
        assert_eq!(
            ixs[0].data,
            [&[2u8][..], &300_000u32.to_le_bytes()].concat()
        );
        assert_eq!(ixs[1].data, [&[3u8][..], &1_000u64.to_le_bytes()].concat());
        assert_eq!(ixs[2].data, ix.data);
        let plain = sign(&wallet, ix, Budget::default(), [0; 32]).unwrap();
        assert!(plain.wire.len() < signed.wire.len());
    }

    #[test]
    fn a_full_buffer_write_fits_and_a_larger_one_is_refused() {
        let wallet = wallet();
        let budget = Budget {
            units: Some(20_000),
            price: Some(1),
        };
        let chunk = vec![7u8; client::CHUNK_LEN];
        let ix = client::buffer_write(wallet.address(), 0, &chunk);
        assert!(sign(&wallet, ix, budget, [0; 32]).is_ok());
        let ix = client::buffer_write(wallet.address(), 0, &vec![7u8; 1300]);
        let refused = sign(&wallet, ix, budget, [0; 32]);
        assert!(matches!(refused, Err(TxError::TooLarge(_))));
    }

    #[test]
    fn a_key_without_an_address_is_refused() {
        let pq = Arc::new(Identity::generate(SigAlg::MlDsa65).unwrap());
        assert!(matches!(Wallet::new(pq), Err(TxError::NoAddress(_))));
    }
}
