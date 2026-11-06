//! Log records. A record's signature covers the exact bytes of its body.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use pq_ucan::crypto::Signer;
use pq_ucan::did::Did;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use thiserror::Error;

use crate::field::Fp2;
use crate::keys::{self, Identity};
use crate::plan::Plan;
use crate::vector::Digest;

const SIGNING_DOMAIN: &[u8] = b"fedtally v1 record\0";

/// Well under the transport's frame limit, so every accepted record can be sent on.
pub const MAX_RECORD: usize = 1 << 22;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Included {
    pub site: String,
    pub contribution: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Body {
    /// Sponsor: the run's terms. Its body hash is the run id.
    Plan(Box<Plan>),
    /// Site: agrees to the plan, with the authority it acted on.
    Join {
        plan: Digest,
        at: i64,
        invocation: String,
        proofs: Vec<String>,
        /// The dataset this site trains on, where the plan's policy asks.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dataset: Option<String>,
        /// The credentials that meet the plan's policy.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        credentials: Vec<String>,
    },
    /// Any party: an issuer's newer revocation list.
    Status { plan: Digest, list: String },
    /// Aggregator: starts a round from `base`.
    Open {
        plan: Digest,
        round: u32,
        prev: Digest,
        base: Digest,
    },
    /// Site: commits to its uploaded vector, its weight and its sealed seeds.
    Contribute {
        plan: Digest,
        round: u32,
        base: Digest,
        masked: Digest,
        weight: u64,
        sealed: Vec<Digest>,
        /// Its share of the proof that the update is short, where the plan asks.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proof: Option<Digest>,
    },
    /// Aggregator: the contributions whose norm proofs are checked next.
    /// The log head after this record is the challenge for those proofs.
    Screen {
        plan: Digest,
        round: u32,
        prev: Digest,
        candidates: Vec<Included>,
    },
    /// Helper: the digest of its messages about the screened proofs.
    Check {
        plan: Digest,
        round: u32,
        screen: Digest,
        shares: Digest,
    },
    /// Aggregator: fixes who is in the round and the digest of their sum.
    Close {
        plan: Digest,
        round: u32,
        prev: Digest,
        included: Vec<Included>,
        sum: Digest,
        /// Its own messages about the screened proofs.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shares: Option<Digest>,
    },
    /// Helper: the digest of the mask total it computed for that Close.
    MaskSum {
        plan: Digest,
        round: u32,
        close: Digest,
        mask: Digest,
        /// Its mask total for each scored group, in a scoring run.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        groups: Vec<Digest>,
    },
    /// Aggregator: commits to the unmasked sum and the next model.
    /// The log head after this record is the challenge.
    Outcome {
        plan: Digest,
        round: u32,
        prev: Digest,
        total_weight: u64,
        aggregate: Digest,
        model: Digest,
        /// The unmasked sum of each scored group.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        groups: Vec<Digest>,
    },
    /// Site: the fingerprint of its own masked vector. An honest site signs this
    /// only after checking the helpers' shares for it against its own seeds.
    Attest {
        plan: Digest,
        round: u32,
        challenge: Digest,
        masked: Fp2,
    },
    /// Helper: the fingerprint of its total and of each site's share.
    HelperAttest {
        plan: Digest,
        round: u32,
        challenge: Digest,
        total: Fp2,
        per_site: Vec<Fp2>,
    },
    /// Aggregator: fingerprints of every masked vector, the sum and the result.
    Seal {
        plan: Digest,
        round: u32,
        prev: Digest,
        per_site: Vec<Fp2>,
        sum: Fp2,
        aggregate: Fp2,
        /// Each scored group's share of the round's reward, in millionths.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        shares: Vec<u32>,
    },
    /// Site: checked the sealed round against the bytes it holds.
    Ack {
        plan: Digest,
        round: u32,
        head: Digest,
    },
    /// Aggregator or helper: the run cannot continue.
    Abort {
        plan: Digest,
        round: u32,
        reason: String,
    },
    /// Aggregator: a site went silent before the seal, so this round number is
    /// given up and the next one starts over without it. Who was silent follows
    /// from the attestations in the log.
    Retry {
        plan: Digest,
        round: u32,
        prev: Digest,
    },
    /// Aggregator: the last round is sealed.
    Finish { plan: Digest, prev: Digest },
}

#[derive(Debug, Error)]
pub enum RecordError {
    #[error("malformed record: {0}")]
    Malformed(String),
    #[error("bad signature from {0}")]
    Signature(String),
}

#[derive(Serialize)]
struct LineOut<'a> {
    signer: &'a str,
    sig: String,
    body: &'a RawValue,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LineIn<'a> {
    signer: String,
    sig: String,
    #[serde(borrow)]
    body: &'a RawValue,
}

/// A record whose signature has been checked.
#[derive(Clone, Debug)]
pub struct Record {
    pub signer: String,
    pub body: Body,
    pub body_hash: Digest,
    pub line: Vec<u8>,
}

fn signing_bytes(body: &str) -> Vec<u8> {
    let mut out = SIGNING_DOMAIN.to_vec();
    out.extend_from_slice(body.as_bytes());
    out
}

impl Record {
    pub fn sign(identity: &Identity, body: &Body) -> Result<Record, RecordError> {
        let malformed = |e: &dyn std::fmt::Display| RecordError::Malformed(e.to_string());
        let raw = serde_json::value::to_raw_value(body).map_err(|e| malformed(&e))?;
        let sig = identity
            .sign(&signing_bytes(raw.get()))
            .map_err(|e| malformed(&format!("{e:?}")))?;
        let signer = identity.did_string();
        let line = serde_json::to_vec(&LineOut {
            signer: &signer,
            sig: B64.encode(sig.as_bytes()),
            body: &raw,
        })
        .map_err(|e| malformed(&e))?;
        Ok(Record {
            signer,
            body: body.clone(),
            body_hash: Digest::of(raw.get().as_bytes()),
            line,
        })
    }

    pub fn parse(line: &[u8]) -> Result<Record, RecordError> {
        let malformed = |e: &dyn std::fmt::Display| RecordError::Malformed(e.to_string());
        if line.len() > MAX_RECORD {
            return Err(malformed(&"record is too large"));
        }
        let parsed: LineIn<'_> = serde_json::from_slice(line).map_err(|e| malformed(&e))?;
        let did = Did::parse(&parsed.signer).map_err(|e| malformed(&e))?;
        let sig = B64.decode(&parsed.sig).map_err(|e| malformed(&e))?;
        keys::verify_did(&did, &signing_bytes(parsed.body.get()), &sig)
            .map_err(|_| RecordError::Signature(parsed.signer.clone()))?;
        let body: Body = serde_json::from_str(parsed.body.get()).map_err(|e| malformed(&e))?;

        // Each record has one encoding. Otherwise a sender could add whitespace,
        // and a newline would split the line in every replica.
        let canonical = serde_json::to_string(&body).map_err(|e| malformed(&e))?;
        let outer = serde_json::to_vec(&LineOut {
            signer: &parsed.signer,
            sig: parsed.sig.clone(),
            body: parsed.body,
        })
        .map_err(|e| malformed(&e))?;
        if canonical != parsed.body.get() || outer != line {
            return Err(malformed(&"record is not in canonical form"));
        }

        Ok(Record {
            body_hash: Digest::of(parsed.body.get().as_bytes()),
            signer: parsed.signer,
            body,
            line: line.to_vec(),
        })
    }
}

pub fn encode_token(bytes: &[u8]) -> String {
    B64.encode(bytes)
}

pub fn decode_token(text: &str) -> Result<Vec<u8>, RecordError> {
    B64.decode(text)
        .map_err(|e| RecordError::Malformed(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::SigAlg;

    fn ack() -> Body {
        Body::Ack {
            plan: Digest::of(b"p"),
            round: 1,
            head: Digest::of(b"h"),
        }
    }

    #[test]
    fn signed_records_parse_back() {
        for alg in [SigAlg::Ed25519, SigAlg::MlDsa65] {
            let id = Identity::generate(alg).unwrap();
            let record = Record::sign(&id, &ack()).unwrap();
            let parsed = Record::parse(&record.line).unwrap();
            assert_eq!(parsed.signer, id.did_string());
            assert_eq!(parsed.body, ack());
            assert_eq!(parsed.body_hash, record.body_hash);
        }
    }

    #[test]
    fn editing_the_body_breaks_the_signature() {
        let id = Identity::generate(SigAlg::Ed25519).unwrap();
        let record = Record::sign(&id, &ack()).unwrap();
        let text = String::from_utf8(record.line).unwrap();
        let edited = text.replace("\"round\":1", "\"round\":2");
        assert_ne!(edited, text);
        assert!(matches!(
            Record::parse(edited.as_bytes()),
            Err(RecordError::Signature(_))
        ));
    }

    #[test]
    fn whitespace_inside_the_body_is_covered_too() {
        let id = Identity::generate(SigAlg::Ed25519).unwrap();
        let record = Record::sign(&id, &ack()).unwrap();
        let text = String::from_utf8(record.line).unwrap();
        let edited = text.replace("\"round\":1", "\"round\": 1");
        assert!(Record::parse(edited.as_bytes()).is_err());
    }

    #[test]
    fn a_record_has_exactly_one_encoding() {
        let id = Identity::generate(SigAlg::Ed25519).unwrap();
        let record = Record::sign(&id, &ack()).unwrap();
        let text = String::from_utf8(record.line).unwrap();
        let newline = text.replacen(r#"{"signer""#, "{\n\"signer\"", 1);
        let padded = format!("{text} ");
        for variant in [newline, padded] {
            assert!(matches!(
                Record::parse(variant.as_bytes()),
                Err(RecordError::Malformed(_))
            ));
        }
    }

    #[test]
    fn a_signed_body_must_be_canonical_too() {
        let id = Identity::generate(SigAlg::Ed25519).unwrap();
        let raw = format!(
            r#"{{"kind":"finish", "plan":"{}","prev":"{}"}}"#,
            Digest::of(b"p"),
            Digest::of(b"q")
        );
        let sig = id.sign(&signing_bytes(&raw)).unwrap();
        let line = format!(
            r#"{{"signer":"{}","sig":"{}","body":{raw}}}"#,
            id.did_string(),
            B64.encode(sig.as_bytes())
        );
        assert!(matches!(
            Record::parse(line.as_bytes()),
            Err(RecordError::Malformed(_))
        ));
    }

    #[test]
    fn oversized_records_are_refused_before_parsing() {
        let refusal = |len: usize| Record::parse(&vec![b' '; len]).unwrap_err().to_string();
        assert!(refusal(MAX_RECORD + 1).contains("too large"));
        assert!(!refusal(MAX_RECORD).contains("too large"));
    }

    #[test]
    fn another_signer_cannot_claim_a_record() {
        let (a, b) = (
            Identity::generate(SigAlg::Ed25519).unwrap(),
            Identity::generate(SigAlg::Ed25519).unwrap(),
        );
        let record = Record::sign(&a, &ack()).unwrap();
        let text = String::from_utf8(record.line).unwrap();
        let edited = text.replace(&a.did_string(), &b.did_string());
        assert!(Record::parse(edited.as_bytes()).is_err());
    }

    #[test]
    fn unknown_fields_are_refused() {
        let id = Identity::generate(SigAlg::Ed25519).unwrap();
        let raw = r#"{"kind":"finish","plan":"00","prev":"00","extra":1}"#;
        let sig = id.sign(&signing_bytes(raw)).unwrap();
        let line = format!(
            r#"{{"signer":"{}","sig":"{}","body":{raw}}}"#,
            id.did_string(),
            B64.encode(sig.as_bytes())
        );
        assert!(matches!(
            Record::parse(line.as_bytes()),
            Err(RecordError::Malformed(_))
        ));
    }
}
