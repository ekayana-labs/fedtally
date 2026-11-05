//! Verifiable credentials: W3C VC Data Model 2.0 documents secured with the
//! `eddsa-jcs-2022` Data Integrity suite.
//!
//! This reads a strict subset. A credential is JSON in canonical form (RFC
//! 8785) with whole numbers only. It has one subject with an `id`, an issuer
//! that is a DID, and a proof made with that DID's own Ed25519 key.

use pq_ucan::crypto::{Algorithm, Signer};
use pq_ucan::did::Did;
use serde_json::{Map, Value, json};
use sha2::{Digest as _, Sha256};
use thiserror::Error;

use crate::keys;
use crate::vector::Digest;

pub const BASE_CONTEXT: &str = "https://www.w3.org/ns/credentials/v2";
/// Terms this crate adds live under one vocabulary, stated inline so a
/// credential needs no document fetched from anywhere.
pub const TERMS: &str = "urn:fedtally:terms#";
const SUITE: &str = "eddsa-jcs-2022";
const STATUS_ENTRY: &str = "BitstringStatusListEntry";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CredentialError {
    #[error("not JSON: {0}")]
    Json(String),
    #[error("not in canonical form")]
    NotCanonical,
    #[error("numbers must be whole and below 2^53")]
    Number,
    #[error("{0}")]
    Shape(&'static str),
    #[error("only Ed25519 keys can issue credentials")]
    Suite,
    #[error("proof: {0}")]
    Proof(String),
}

fn write_canonical(value: &Value, out: &mut String) -> Result<(), CredentialError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            let whole = match (n.as_i64(), n.as_u64()) {
                (Some(v), _) => v.unsigned_abs() < 1 << 53,
                (None, Some(v)) => v < 1 << 53,
                (None, None) => false,
            };
            if !whole {
                return Err(CredentialError::Number);
            }
            out.push_str(&n.to_string());
        }
        Value::String(s) => {
            let quoted =
                serde_json::to_string(s).map_err(|e| CredentialError::Json(e.to_string()))?;
            out.push_str(&quoted);
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
            out.push('{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(&Value::String(key.clone()), out)?;
                out.push(':');
                write_canonical(item, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// The RFC 8785 form of a document that holds no fractions.
pub fn canonical(value: &Value) -> Result<String, CredentialError> {
    let mut out = String::new();
    write_canonical(value, &mut out)?;
    Ok(out)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// Seconds since 1970 for an XSD `dateTimeStamp` such as `2026-01-31T12:00:00Z`.
pub fn parse_time(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    let number = |from: usize, len: usize| -> Option<i64> {
        let part = bytes.get(from..from + len)?;
        part.iter()
            .all(u8::is_ascii_digit)
            .then(|| part.iter().fold(0, |n, b| n * 10 + i64::from(b - b'0')))
    };
    let is = |at: usize, byte: u8| bytes.get(at) == Some(&byte);
    if !(is(4, b'-') && is(7, b'-') && is(10, b'T') && is(13, b':') && is(16, b':')) {
        return None;
    }
    let (year, month, day) = (number(0, 4)?, number(5, 2)?, number(8, 2)?);
    let (hour, minute, second) = (number(11, 2)?, number(14, 2)?, number(17, 2)?);
    let month_ok = (1..=12).contains(&month) && day >= 1;
    if !month_ok || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    if civil_from_days(days) != (year, month, day) {
        return None;
    }
    let mut at = 19;
    if is(at, b'.') {
        at += 1;
        let digits = bytes[at..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count();
        if digits == 0 {
            return None;
        }
        at += digits;
    }
    let offset = match bytes.get(at)? {
        b'Z' if at + 1 == bytes.len() => 0,
        sign @ (b'+' | b'-') if at + 6 == bytes.len() && is(at + 3, b':') => {
            let (h, m) = (number(at + 1, 2)?, number(at + 4, 2)?);
            if h > 23 || m > 59 {
                return None;
            }
            (h * 3600 + m * 60) * if *sign == b'+' { 1 } else { -1 }
        }
        _ => return None,
    };
    Some(days * 86_400 + hour * 3600 + minute * 60 + second - offset)
}

pub fn format_time(seconds: i64) -> String {
    let (days, rest) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// What an `eddsa-jcs-2022` proof signs: the hash of the proof's own settings,
/// then the hash of the document without its proof.
fn signing_input(
    document: &Map<String, Value>,
    proof: &Map<String, Value>,
) -> Result<[u8; 64], CredentialError> {
    let mut bare = document.clone();
    bare.remove("proof");
    let mut settings = proof.clone();
    settings.remove("proofValue");
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&sha256(canonical(&Value::Object(settings))?.as_bytes()));
    input[32..].copy_from_slice(&sha256(canonical(&Value::Object(bare))?.as_bytes()));
    Ok(input)
}

fn verification_method(did: &str) -> String {
    match did.strip_prefix("did:key:") {
        Some(key) => format!("{did}#{key}"),
        None => format!("{did}#default"),
    }
}

/// Adds a proof to `document` and returns it in canonical form.
pub fn secure(
    signer: &impl Signer,
    mut document: Map<String, Value>,
    created: i64,
) -> Result<String, CredentialError> {
    if signer.public_key().algorithm() != Algorithm::Ed25519 {
        return Err(CredentialError::Suite);
    }
    let mut proof = Map::new();
    proof.insert("type".into(), json!("DataIntegrityProof"));
    proof.insert("cryptosuite".into(), json!(SUITE));
    proof.insert("created".into(), json!(format_time(created)));
    proof.insert(
        "verificationMethod".into(),
        json!(verification_method(signer.did().as_str())),
    );
    proof.insert("proofPurpose".into(), json!("assertionMethod"));
    if let Some(context) = document.get("@context") {
        proof.insert("@context".into(), context.clone());
    }
    let input = signing_input(&document, &proof)?;
    let signature = signer
        .sign(&input)
        .map_err(|e| CredentialError::Proof(format!("{e:?}")))?;
    let encoded = bs58::encode(signature.as_bytes()).into_string();
    proof.insert("proofValue".into(), json!(format!("z{encoded}")));
    document.insert("proof".into(), Value::Object(proof));
    canonical(&Value::Object(document))
}

fn text<'a>(
    map: &'a Map<String, Value>,
    key: &str,
    what: &'static str,
) -> Result<&'a str, CredentialError> {
    map.get(key)
        .and_then(Value::as_str)
        .ok_or(CredentialError::Shape(what))
}

/// Checks a document's proof and returns the DID whose key made it.
pub fn integrity(document: &Value) -> Result<String, CredentialError> {
    let document = document
        .as_object()
        .ok_or(CredentialError::Shape("a credential is a JSON object"))?;
    let proof = document
        .get("proof")
        .and_then(Value::as_object)
        .ok_or(CredentialError::Shape("needs one proof"))?;
    let wrong = |what: &str| CredentialError::Proof(what.to_owned());
    if text(proof, "type", "proof needs a type")? != "DataIntegrityProof"
        || text(proof, "cryptosuite", "proof needs a cryptosuite")? != SUITE
    {
        return Err(wrong("not an eddsa-jcs-2022 proof"));
    }
    if text(proof, "proofPurpose", "proof needs a purpose")? != "assertionMethod" {
        return Err(wrong("not an assertion"));
    }
    if let Some(context) = proof.get("@context")
        && document.get("@context") != Some(context)
    {
        return Err(wrong("the proof was made for another context"));
    }
    let method = text(
        proof,
        "verificationMethod",
        "proof needs a verification method",
    )?;
    let controller = method.split('#').next().unwrap_or_default();
    if method != verification_method(controller) {
        return Err(wrong("unknown verification method"));
    }
    let value = text(proof, "proofValue", "proof needs a value")?;
    let signature = value
        .strip_prefix('z')
        .and_then(|rest| bs58::decode(rest).into_vec().ok())
        .ok_or_else(|| wrong("the value is not base58"))?;
    let did = Did::parse(controller).map_err(|e| wrong(&e.to_string()))?;
    let input = signing_input(document, proof)?;
    keys::verify_did(&did, &input, &signature).map_err(|_| wrong("bad signature"))?;
    Ok(controller.to_owned())
}

/// Where a credential's revocation bit lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusEntry {
    pub list: String,
    pub index: u64,
}

impl StatusEntry {
    fn to_json(&self) -> Value {
        json!({
            "type": STATUS_ENTRY,
            "statusPurpose": "revocation",
            "statusListIndex": self.index.to_string(),
            "statusListCredential": self.list,
        })
    }

    fn from_json(value: &Value) -> Result<StatusEntry, CredentialError> {
        let unreadable = CredentialError::Shape("a status this crate cannot check");
        let map = value.as_object().ok_or(unreadable.clone())?;
        if text(map, "type", "status needs a type")? != STATUS_ENTRY
            || text(map, "statusPurpose", "status needs a purpose")? != "revocation"
        {
            return Err(unreadable);
        }
        let index = text(map, "statusListIndex", "status needs an index")?;
        Ok(StatusEntry {
            list: text(map, "statusListCredential", "status needs a list")?.to_owned(),
            index: index
                .parse()
                .ok()
                .filter(|i: &u64| i.to_string() == index)
                .ok_or(unreadable)?,
        })
    }
}

impl Clone for CredentialError {
    fn clone(&self) -> Self {
        match self {
            CredentialError::Json(s) => CredentialError::Json(s.clone()),
            CredentialError::NotCanonical => CredentialError::NotCanonical,
            CredentialError::Number => CredentialError::Number,
            CredentialError::Shape(s) => CredentialError::Shape(s),
            CredentialError::Suite => CredentialError::Suite,
            CredentialError::Proof(s) => CredentialError::Proof(s.clone()),
        }
    }
}

/// What an issuer says about a subject.
#[derive(Clone, Debug)]
pub struct Draft {
    pub kind: String,
    pub subject: String,
    pub claims: Map<String, Value>,
    pub valid_from: i64,
    pub valid_until: Option<i64>,
    pub status: Option<StatusEntry>,
}

/// A credential whose proof has been checked.
#[derive(Clone, Debug, PartialEq)]
pub struct Credential {
    /// The canonical JSON, as carried in a record.
    pub text: String,
    pub digest: Digest,
    /// The document's own identifier, if it has one.
    pub id: Option<String>,
    pub issuer: String,
    pub subject: String,
    /// Its types, apart from `VerifiableCredential`.
    pub types: Vec<String>,
    /// The subject's claims, apart from its `id`.
    pub claims: Map<String, Value>,
    pub valid_from: Option<i64>,
    pub valid_until: Option<i64>,
    pub status: Option<StatusEntry>,
}

fn context() -> Value {
    json!([BASE_CONTEXT, { "@vocab": TERMS }])
}

impl Credential {
    pub fn issue(
        issuer: &impl Signer,
        draft: &Draft,
        created: i64,
    ) -> Result<Credential, CredentialError> {
        let mut subject = draft.claims.clone();
        subject.insert("id".into(), json!(draft.subject));
        let mut document = Map::new();
        document.insert("@context".into(), context());
        document.insert("type".into(), json!(["VerifiableCredential", draft.kind]));
        document.insert("issuer".into(), json!(issuer.did().as_str()));
        document.insert("validFrom".into(), json!(format_time(draft.valid_from)));
        if let Some(until) = draft.valid_until {
            document.insert("validUntil".into(), json!(format_time(until)));
        }
        if let Some(status) = &draft.status {
            document.insert("credentialStatus".into(), status.to_json());
        }
        document.insert("credentialSubject".into(), Value::Object(subject));
        Credential::parse(&secure(issuer, document, created)?)
    }

    pub fn parse(text_in: &str) -> Result<Credential, CredentialError> {
        let document: Value =
            serde_json::from_str(text_in).map_err(|e| CredentialError::Json(e.to_string()))?;
        if canonical(&document)? != text_in {
            return Err(CredentialError::NotCanonical);
        }
        let signed_by = integrity(&document)?;
        let map = document
            .as_object()
            .ok_or(CredentialError::Shape("a credential is a JSON object"))?;

        let first_context = map
            .get("@context")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(Value::as_str);
        if first_context != Some(BASE_CONTEXT) {
            return Err(CredentialError::Shape(
                "the first context must be credentials/v2",
            ));
        }
        let all_types: Vec<&str> = map
            .get("type")
            .and_then(Value::as_array)
            .and_then(|t| t.iter().map(Value::as_str).collect())
            .ok_or(CredentialError::Shape("type must be a list of names"))?;
        if !all_types.contains(&"VerifiableCredential") {
            return Err(CredentialError::Shape("not a VerifiableCredential"));
        }
        let issuer = match map.get("issuer") {
            Some(Value::String(id)) => id.as_str(),
            Some(Value::Object(issuer)) => text(issuer, "id", "issuer needs an id")?,
            _ => return Err(CredentialError::Shape("needs an issuer")),
        };
        if issuer != signed_by {
            return Err(CredentialError::Proof(
                "made by a key that is not the issuer's".into(),
            ));
        }
        let mut claims = map
            .get("credentialSubject")
            .and_then(Value::as_object)
            .cloned()
            .ok_or(CredentialError::Shape("needs one subject"))?;
        let subject = match claims.remove("id") {
            Some(Value::String(id)) => id,
            _ => return Err(CredentialError::Shape("the subject needs an id")),
        };
        let time = |key: &str| -> Result<Option<i64>, CredentialError> {
            match map.get(key) {
                None => Ok(None),
                Some(value) => value
                    .as_str()
                    .and_then(parse_time)
                    .map(Some)
                    .ok_or(CredentialError::Shape("dates must be dateTimeStamps")),
            }
        };
        Ok(Credential {
            text: text_in.to_owned(),
            digest: Digest::of(text_in.as_bytes()),
            id: map.get("id").and_then(Value::as_str).map(str::to_owned),
            issuer: issuer.to_owned(),
            subject,
            types: all_types
                .into_iter()
                .filter(|t| *t != "VerifiableCredential")
                .map(str::to_owned)
                .collect(),
            claims,
            valid_from: time("validFrom")?,
            valid_until: time("validUntil")?,
            status: map
                .get("credentialStatus")
                .map(StatusEntry::from_json)
                .transpose()?,
        })
    }

    pub fn valid_at(&self, now: i64) -> bool {
        self.valid_from.is_none_or(|from| from <= now)
            && self.valid_until.is_none_or(|until| now < until)
    }

    /// A claim by dotted path, such as `ga4gh_visa_v1.by`.
    pub fn claim(&self, path: &str) -> Option<&Value> {
        let mut parts = path.split('.');
        let mut at = self.claims.get(parts.next()?)?;
        for part in parts {
            at = at.as_object()?.get(part)?;
        }
        Some(at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{Identity, SigAlg};
    use pq_ucan::crypto::ed25519::Ed25519Keypair;

    // Data Integrity EdDSA Cryptosuites v1.0, W3C Recommendation of 15 May
    // 2025, appendix B.3.
    const SPEC_SEED: &str = "c96ef9ea10c5e414c471723aff9de72c35fa5b70fae97e8832ecac7d2e2b8ed6";
    const SPEC_UNSIGNED: &str = r#"{"@context":["https://www.w3.org/ns/credentials/v2","https://www.w3.org/ns/credentials/examples/v2"],"id":"urn:uuid:58172aac-d8ba-11ed-83dd-0b3aef56cc33","type":["VerifiableCredential","AlumniCredential"],"name":"Alumni Credential","description":"A minimum viable example of an Alumni Credential.","issuer":"https://vc.example/issuers/5678","validFrom":"2023-01-01T00:00:00Z","credentialSubject":{"id":"did:example:abcdefgh","alumniOf":"The School of Examples"}}"#;
    const SPEC_CANONICAL: &str = r#"{"@context":["https://www.w3.org/ns/credentials/v2","https://www.w3.org/ns/credentials/examples/v2"],"credentialSubject":{"alumniOf":"The School of Examples","id":"did:example:abcdefgh"},"description":"A minimum viable example of an Alumni Credential.","id":"urn:uuid:58172aac-d8ba-11ed-83dd-0b3aef56cc33","issuer":"https://vc.example/issuers/5678","name":"Alumni Credential","type":["VerifiableCredential","AlumniCredential"],"validFrom":"2023-01-01T00:00:00Z"}"#;
    const SPEC_PROOF_VALUE: &str =
        "z2HnFSSPPBzR36zdDgK8PbEHeXbR56YF24jwMpt3R1eHXQzJDMWS93FCzpvJpwTWd3GAVFuUfjoJdcnTMuVor51aX";
    const SPEC_KEY: &str = "did:key:z6MkrJVnaZkeFzdQyMZu1cgjg7k1pZZ6pvBQ7XJPt4swbTQ2";

    fn spec_key() -> Ed25519Keypair {
        let mut seed = [0u8; 32];
        hex::decode_to_slice(SPEC_SEED, &mut seed).unwrap();
        Ed25519Keypair::from_seed(&seed)
    }

    #[test]
    fn the_w3c_test_vector_reproduces_byte_for_byte() {
        let unsigned: Value = serde_json::from_str(SPEC_UNSIGNED).unwrap();
        assert_eq!(canonical(&unsigned).unwrap(), SPEC_CANONICAL);
        assert_eq!(
            hex::encode(sha256(SPEC_CANONICAL.as_bytes())),
            "59b7cb6251b8991add1ce0bc83107e3db9dbbab5bd2c28f687db1a03abc92f19"
        );
        assert_eq!(spec_key().did().as_str(), SPEC_KEY);

        let created = parse_time("2023-02-24T23:36:38Z").unwrap();
        let signed = secure(&spec_key(), unsigned.as_object().unwrap().clone(), created).unwrap();
        let document: Value = serde_json::from_str(&signed).unwrap();
        assert_eq!(document["proof"]["proofValue"], SPEC_PROOF_VALUE);
        assert_eq!(integrity(&document).unwrap(), SPEC_KEY);
    }

    #[test]
    fn the_w3c_signed_example_verifies_and_any_change_breaks_it() {
        let mut document: Value = serde_json::from_str(SPEC_UNSIGNED).unwrap();
        document["proof"] = json!({
            "type": "DataIntegrityProof",
            "cryptosuite": "eddsa-jcs-2022",
            "created": "2023-02-24T23:36:38Z",
            "verificationMethod": format!("{SPEC_KEY}#{}", &SPEC_KEY[8..]),
            "proofPurpose": "assertionMethod",
            "@context": document["@context"].clone(),
            "proofValue": SPEC_PROOF_VALUE,
        });
        assert_eq!(integrity(&document).unwrap(), SPEC_KEY);

        let broken = |edit: fn(&mut Value)| {
            let mut copy = document.clone();
            edit(&mut copy);
            integrity(&copy).unwrap_err()
        };
        broken(|d| d["name"] = json!("Another Credential"));
        broken(|d| d["credentialSubject"]["id"] = json!("did:example:someone-else"));
        broken(|d| d["proof"]["created"] = json!("2024-02-24T23:36:38Z"));
        broken(|d| d["proof"]["proofPurpose"] = json!("authentication"));
        broken(|d| d["proof"]["cryptosuite"] = json!("eddsa-rdfc-2022"));
        broken(|d| d["@context"] = json!([BASE_CONTEXT]));
        broken(|d| {
            d["proof"]["verificationMethod"] = json!(format!("{SPEC_KEY}#another"));
        });
        broken(|d| {
            d.as_object_mut().unwrap().remove("proof");
        });
    }

    #[test]
    fn canonical_form_sorts_keys_and_refuses_fractions() {
        let value: Value =
            serde_json::from_str(r#"{"b":[1,{"z":null,"a":true}],"a":"x\ny","é":-3}"#).unwrap();
        assert_eq!(
            canonical(&value).unwrap(),
            r#"{"a":"x\ny","b":[1,{"a":true,"z":null}],"é":-3}"#
        );
        for bad in ["1.5", "1e30", "9007199254740992", "-9007199254740992"] {
            let value: Value = serde_json::from_str(bad).unwrap();
            assert_eq!(canonical(&value), Err(CredentialError::Number), "{bad}");
        }
        let edge: Value = serde_json::from_str("[9007199254740991,-9007199254740991]").unwrap();
        canonical(&edge).unwrap();
    }

    #[test]
    fn times_parse_to_seconds_and_print_back() {
        assert_eq!(parse_time("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_time("2023-01-01T00:00:00Z"), Some(1_672_531_200));
        assert_eq!(parse_time("2024-02-29T12:30:45.123Z"), Some(1_709_209_845));
        assert_eq!(parse_time("2023-01-01T02:00:00+02:00"), Some(1_672_531_200));
        assert_eq!(parse_time("2022-12-31T19:00:00-05:00"), Some(1_672_531_200));
        for bad in [
            "2023-02-29T00:00:00Z",
            "2023-13-01T00:00:00Z",
            "2023-01-01T24:00:00Z",
            "2023-01-01T00:00:00",
            "2023-01-01 00:00:00Z",
            "2023-01-01T00:00:00.Z",
            "2023-01-01T00:00:00Zjunk",
            "2023-01-01T00:00:00+0200",
            "23-01-01T00:00:00Z",
        ] {
            assert_eq!(parse_time(bad), None, "{bad}");
        }
        for seconds in [0, 1_672_531_200, 1_709_209_845, 4_102_444_799] {
            assert_eq!(parse_time(&format_time(seconds)), Some(seconds));
        }
        assert_eq!(format_time(1_709_209_845), "2024-02-29T12:30:45Z");
    }

    fn draft(subject: &str) -> Draft {
        let mut claims = Map::new();
        claims.insert(
            "ga4gh_visa_v1".into(),
            json!({"type": "ControlledAccessGrants", "by": "dac", "value": "did:example:dataset"}),
        );
        Draft {
            kind: "ControlledAccessGrants".into(),
            subject: subject.into(),
            claims,
            valid_from: 1_700_000_000,
            valid_until: Some(1_800_000_000),
            status: Some(StatusEntry {
                list: "urn:example:list".into(),
                index: 42,
            }),
        }
    }

    #[test]
    fn an_issued_credential_reads_back_with_its_parts() {
        let issuer = Identity::generate(SigAlg::Ed25519).unwrap();
        let credential =
            Credential::issue(&issuer, &draft("did:example:site"), 1_700_000_001).unwrap();
        assert_eq!(credential.issuer, issuer.did_string());
        assert_eq!(credential.subject, "did:example:site");
        assert_eq!(credential.types, ["ControlledAccessGrants"]);
        assert_eq!(credential.claim("ga4gh_visa_v1.by"), Some(&json!("dac")));
        assert_eq!(credential.claim("ga4gh_visa_v1.missing"), None);
        assert_eq!(credential.claim("id"), None);
        assert_eq!(credential.status.as_ref().unwrap().index, 42);
        assert!(!credential.valid_at(1_699_999_999));
        assert!(credential.valid_at(1_700_000_000));
        assert!(!credential.valid_at(1_800_000_000));
        assert_eq!(Credential::parse(&credential.text).unwrap(), credential);
        assert_eq!(credential.digest, Digest::of(credential.text.as_bytes()));
    }

    #[test]
    fn a_credential_has_one_encoding() {
        let issuer = Identity::generate(SigAlg::Ed25519).unwrap();
        let credential = Credential::issue(&issuer, &draft("did:example:site"), 1).unwrap();
        let spaced = credential.text.replacen(':', ": ", 1);
        assert_eq!(
            Credential::parse(&spaced),
            Err(CredentialError::NotCanonical)
        );
        assert!(matches!(
            Credential::parse("{"),
            Err(CredentialError::Json(_))
        ));
    }

    #[test]
    fn only_the_issuer_can_make_the_proof() {
        let (issuer, other) = (
            Identity::generate(SigAlg::Ed25519).unwrap(),
            Identity::generate(SigAlg::Ed25519).unwrap(),
        );
        let credential = Credential::issue(&issuer, &draft("did:example:site"), 1).unwrap();
        let mut document: Map<String, Value> = serde_json::from_str(&credential.text).unwrap();
        document.remove("proof");
        let forged = secure(&other, document, 1).unwrap();
        assert!(matches!(
            Credential::parse(&forged),
            Err(CredentialError::Proof(why)) if why.contains("not the issuer")
        ));
    }

    #[test]
    fn a_did_bio_issuer_signs_with_its_default_method() {
        let issuer = Identity::generate(SigAlg::Ed25519)
            .unwrap()
            .on_registry("devnet")
            .unwrap();
        let credential = Credential::issue(&issuer, &draft("did:example:site"), 1).unwrap();
        assert!(credential.issuer.starts_with("did:bio:devnet:"));
        let document: Value = serde_json::from_str(&credential.text).unwrap();
        assert_eq!(
            document["proof"]["verificationMethod"],
            format!("{}#default", credential.issuer)
        );
        Credential::parse(&credential.text).unwrap();
    }

    #[test]
    fn a_post_quantum_key_cannot_issue_yet() {
        let issuer = Identity::generate(SigAlg::MlDsa65).unwrap();
        assert_eq!(
            Credential::issue(&issuer, &draft("did:example:site"), 1),
            Err(CredentialError::Suite)
        );
    }

    #[test]
    fn a_shape_this_crate_does_not_read_is_refused() {
        let issuer = Identity::generate(SigAlg::Ed25519).unwrap();
        let good = Credential::issue(&issuer, &draft("did:example:site"), 1).unwrap();
        let resigned = |edit: fn(&mut Map<String, Value>)| {
            let mut document: Map<String, Value> = serde_json::from_str(&good.text).unwrap();
            document.remove("proof");
            edit(&mut document);
            Credential::parse(&secure(&issuer, document, 1).unwrap())
        };
        let shape = |result: Result<Credential, CredentialError>| {
            assert!(
                matches!(result, Err(CredentialError::Shape(_))),
                "{result:?}"
            );
        };
        shape(resigned(|d| {
            d.insert("@context".into(), json!(["https://example.org/other"]));
        }));
        shape(resigned(|d| {
            d.insert("type".into(), json!(["AccessGrant"]));
        }));
        shape(resigned(|d| {
            d.insert("credentialSubject".into(), json!({"name": "no id"}));
        }));
        shape(resigned(|d| {
            d.insert("validUntil".into(), json!("next year"));
        }));
        shape(resigned(|d| {
            d.insert(
                "credentialStatus".into(),
                json!({"type": "SomeOtherStatus"}),
            );
        }));
        shape(resigned(|d| {
            d["credentialStatus"]["statusListIndex"] = json!("042");
        }));
        shape(resigned(|d| {
            d["credentialStatus"]["statusPurpose"] = json!("suspension");
        }));
        resigned(|d| {
            d.remove("validUntil");
            d.remove("credentialStatus");
        })
        .unwrap();
    }
}
