//! Revocation, as W3C Bitstring Status Lists.
//!
//! An issuer publishes one signed list of bits. A credential names a list and
//! an index, and is revoked once that bit is set. A list here is carried in
//! the log, so a replay needs no network, and a newer list may only add bits.

use std::io::{Read, Write};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use pq_ucan::crypto::Signer;
use serde_json::{Map, json};

use crate::credential::{BASE_CONTEXT, Credential, CredentialError, format_time, secure};

/// The specification's minimum, so that one index says little about who holds it.
pub const MIN_BITS: usize = 131_072;
const MAX_BYTES: usize = 1 << 20;
const LIST_TYPE: &str = "BitstringStatusListCredential";

#[derive(Clone, Debug, PartialEq)]
pub struct StatusList {
    /// The canonical JSON, as carried in a record.
    pub text: String,
    pub id: String,
    pub issuer: String,
    pub issued: i64,
    bits: Vec<u8>,
}

fn shape(what: &'static str) -> CredentialError {
    CredentialError::Shape(what)
}

impl StatusList {
    pub fn issue(
        issuer: &impl Signer,
        id: &str,
        revoked: &[u64],
        issued: i64,
    ) -> Result<StatusList, CredentialError> {
        let mut bits = vec![0u8; MIN_BITS / 8];
        for index in revoked {
            let byte = bits
                .get_mut((*index / 8) as usize)
                .ok_or(shape("index beyond the list"))?;
            *byte |= 0x80 >> (index % 8);
        }
        let mut gzip = GzEncoder::new(Vec::new(), Compression::default());
        let packed = gzip
            .write_all(&bits)
            .and_then(|()| gzip.finish())
            .map_err(|e| CredentialError::Json(e.to_string()))?;
        let mut document = Map::new();
        document.insert("@context".into(), json!([BASE_CONTEXT]));
        document.insert("id".into(), json!(id));
        document.insert("type".into(), json!(["VerifiableCredential", LIST_TYPE]));
        document.insert("issuer".into(), json!(issuer.did().as_str()));
        document.insert("validFrom".into(), json!(format_time(issued)));
        document.insert(
            "credentialSubject".into(),
            json!({
                "id": format!("{id}#list"),
                "type": "BitstringStatusList",
                "statusPurpose": "revocation",
                "encodedList": format!("u{}", URL_SAFE_NO_PAD.encode(packed)),
            }),
        );
        StatusList::parse(&secure(issuer, document, issued)?)
    }

    pub fn parse(text: &str) -> Result<StatusList, CredentialError> {
        let credential = Credential::parse(text)?;
        if !credential.types.iter().any(|t| t == LIST_TYPE) {
            return Err(shape("not a status list"));
        }
        let claim = |name: &str| credential.claim(name).and_then(|v| v.as_str());
        if claim("type") != Some("BitstringStatusList")
            || claim("statusPurpose") != Some("revocation")
        {
            return Err(shape("not a revocation list"));
        }
        let packed = claim("encodedList")
            .and_then(|list| list.strip_prefix('u'))
            .and_then(|list| URL_SAFE_NO_PAD.decode(list).ok())
            .ok_or(shape("the list is not base64url"))?;
        let mut bits = Vec::new();
        GzDecoder::new(packed.as_slice())
            .take(MAX_BYTES as u64 + 1)
            .read_to_end(&mut bits)
            .map_err(|_| shape("the list is not gzip"))?;
        if bits.len() * 8 < MIN_BITS || bits.len() > MAX_BYTES {
            return Err(shape("a list holds between 131,072 and 8,388,608 bits"));
        }
        Ok(StatusList {
            text: text.to_owned(),
            id: credential.id.ok_or(shape("a list needs an id"))?,
            issuer: credential.issuer,
            issued: credential.valid_from.ok_or(shape("a list needs a date"))?,
            bits,
        })
    }

    /// Bit 0 is the first bit of the first byte.
    pub fn is_set(&self, index: u64) -> bool {
        self.bits
            .get((index / 8) as usize)
            .is_some_and(|byte| byte & (0x80 >> (index % 8)) != 0)
    }

    /// True when every bit set in `earlier` is still set here.
    pub fn keeps(&self, earlier: &StatusList) -> bool {
        self.bits.len() >= earlier.bits.len()
            && earlier
                .bits
                .iter()
                .zip(&self.bits)
                .all(|(old, new)| old & new == *old)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{Identity, SigAlg};

    fn issuer() -> Identity {
        Identity::generate(SigAlg::Ed25519).unwrap()
    }

    #[test]
    fn a_list_reads_back_with_its_bits_in_spec_order() {
        let dac = issuer();
        let list =
            StatusList::issue(&dac, "urn:example:list", &[0, 9, 131_071], 1_700_000_000).unwrap();
        assert_eq!(list.id, "urn:example:list");
        assert_eq!(list.issuer, dac.did_string());
        assert_eq!(list.issued, 1_700_000_000);
        for index in [0, 9, 131_071] {
            assert!(list.is_set(index), "{index}");
        }
        for index in [1, 7, 8, 10, 131_070, 131_072, u64::MAX] {
            assert!(!list.is_set(index), "{index}");
        }
        // Index 0 is the left-most bit, and 9 is the second bit of the second byte.
        assert_eq!(&list.bits[..2], &[0x80, 0x40]);
        assert_eq!(StatusList::parse(&list.text).unwrap(), list);
        assert!(
            list.text.len() < 2_000,
            "an almost empty list compresses well"
        );
    }

    #[test]
    fn an_index_beyond_the_list_cannot_be_revoked() {
        assert!(StatusList::issue(&issuer(), "urn:l", &[131_072], 1).is_err());
    }

    #[test]
    fn a_newer_list_keeps_every_bit_of_an_older_one() {
        let dac = issuer();
        let first = StatusList::issue(&dac, "urn:l", &[3], 1).unwrap();
        let more = StatusList::issue(&dac, "urn:l", &[3, 4], 2).unwrap();
        let other = StatusList::issue(&dac, "urn:l", &[4], 3).unwrap();
        assert!(more.keeps(&first));
        assert!(first.keeps(&first));
        assert!(!first.keeps(&more));
        assert!(!other.keeps(&first), "a bit was cleared");
    }

    #[test]
    fn a_list_is_signed_and_has_one_encoding() {
        let dac = issuer();
        let list = StatusList::issue(&dac, "urn:l", &[3], 1).unwrap();
        let edited = list.text.replace("urn:l#list", "urn:m#list");
        assert!(matches!(
            StatusList::parse(&edited),
            Err(CredentialError::Proof(_))
        ));
        assert_eq!(
            StatusList::parse(&format!("{} ", list.text)),
            Err(CredentialError::NotCanonical)
        );
    }

    fn resigned(
        dac: &Identity,
        edit: impl FnOnce(&mut Map<String, serde_json::Value>),
    ) -> Result<StatusList, CredentialError> {
        let list = StatusList::issue(dac, "urn:l", &[3], 1).unwrap();
        let mut document: Map<String, serde_json::Value> =
            serde_json::from_str(&list.text).unwrap();
        document.remove("proof");
        edit(&mut document);
        StatusList::parse(&secure(dac, document, 1)?)
    }

    fn encoded(bytes: &[u8]) -> String {
        let mut gzip = GzEncoder::new(Vec::new(), Compression::default());
        gzip.write_all(bytes).unwrap();
        format!("u{}", URL_SAFE_NO_PAD.encode(gzip.finish().unwrap()))
    }

    #[test]
    fn lists_of_the_wrong_kind_or_size_are_refused() {
        let dac = issuer();
        let refused = |result: Result<StatusList, CredentialError>| {
            assert!(
                matches!(result, Err(CredentialError::Shape(_))),
                "{result:?}"
            );
        };
        refused(resigned(&dac, |d| {
            d.insert("type".into(), json!(["VerifiableCredential"]));
        }));
        refused(resigned(&dac, |d| {
            d["credentialSubject"]["statusPurpose"] = json!("suspension");
        }));
        refused(resigned(&dac, |d| {
            d["credentialSubject"]["type"] = json!("SomethingElse");
        }));
        refused(resigned(&dac, |d| {
            d["credentialSubject"]["encodedList"] = json!("not a list");
        }));
        refused(resigned(&dac, |d| {
            d["credentialSubject"]["encodedList"] =
                json!(format!("u{}", URL_SAFE_NO_PAD.encode(b"not gzip")));
        }));
        refused(resigned(&dac, |d| {
            d["credentialSubject"]["encodedList"] = json!(encoded(&vec![0u8; MIN_BITS / 8 - 1]));
        }));
        refused(resigned(&dac, |d| {
            d["credentialSubject"]["encodedList"] = json!(encoded(&vec![0u8; MAX_BYTES + 1]));
        }));
        refused(resigned(&dac, |d| {
            d.remove("id");
        }));
        refused(resigned(&dac, |d| {
            d.remove("validFrom");
        }));
        resigned(&dac, |d| {
            d["credentialSubject"]["encodedList"] = json!(encoded(&vec![0u8; MAX_BYTES]));
        })
        .unwrap();
    }
}
