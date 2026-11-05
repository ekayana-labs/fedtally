//! Signing identities and sealing keys.

use std::path::Path;

use aws_lc_rs::kem::{Ciphertext, DecapsulationKey, EncapsulationKey, ML_KEM_768};
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce as AeadNonce};
use did_bio_core::{BioDid, Network};
use pq_ucan::crypto::ed25519::Ed25519Keypair;
use pq_ucan::crypto::ml_dsa::MlDsaKeypair;
use pq_ucan::crypto::{Algorithm, CryptoError, PublicKey, Signature, Signer};
use pq_ucan::did::{Did, ResolveError, Resolver};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use x25519_dalek::{PublicKey as XPublic, StaticSecret};

#[derive(Debug, Error)]
pub enum KeyError {
    #[error("key file: {0}")]
    File(#[from] std::io::Error),
    #[error("key file format: {0}")]
    Format(String),
    #[error("signature backend: {0:?}")]
    Crypto(CryptoError),
    #[error("sealed box did not open")]
    Seal,
}

impl From<CryptoError> for KeyError {
    fn from(e: CryptoError) -> KeyError {
        KeyError::Crypto(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SigAlg {
    Ed25519,
    #[serde(alias = "ml-dsa-65")]
    MlDsa65,
}

enum SigningKey {
    Ed25519(Ed25519Keypair),
    MlDsa(MlDsaKeypair),
}

/// A party's signing key, sealing key and optional TLS credentials.
pub struct Identity {
    alg: SigAlg,
    seed: [u8; 32],
    signing: SigningKey,
    seal: SealKey,
    /// Set when the party goes by `did:bio` on this network, not `did:key`.
    network: Option<Network>,
    pub tls: Option<TlsCredentials>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TlsCredentials {
    pub cert: String,
    pub key: String,
}

#[derive(Serialize, Deserialize)]
struct KeyFile {
    alg: SigAlg,
    seed: String,
    seal_x25519: String,
    seal_mlkem: String,
    seal_mlkem_public: String,
    /// `mainnet`, `devnet`, `testnet` or `localnet`, for a `did:bio` identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    did_bio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tls: Option<TlsCredentials>,
}

fn network_name(network: Network) -> &'static str {
    network.segment().unwrap_or("mainnet")
}

fn network_named(name: &str) -> Result<Network, KeyError> {
    match name {
        "mainnet" => Ok(Network::Mainnet),
        other => Network::from_segment(other)
            .ok_or_else(|| KeyError::Format(format!("unknown network {other}"))),
    }
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut out = [0u8; N];
    if getrandom::fill(&mut out).is_err() {
        // A machine with no entropy source must not sign or mask anything.
        std::process::abort();
    }
    out
}

impl Identity {
    pub fn generate(alg: SigAlg) -> Result<Identity, KeyError> {
        let seed = random_bytes::<32>();
        let signing = signing_from_seed(alg, &seed)?;
        Ok(Identity {
            alg,
            seed,
            signing,
            seal: SealKey::generate()?,
            network: None,
            tls: None,
        })
    }

    /// Goes by `did:bio` on the named network from now on. The identifier
    /// holds the same Ed25519 key, which is also the party's Solana address.
    pub fn on_registry(mut self, network: &str) -> Result<Identity, KeyError> {
        if self.alg != SigAlg::Ed25519 {
            return Err(KeyError::Format("did:bio needs an Ed25519 key".into()));
        }
        self.network = Some(network_named(network)?);
        Ok(self)
    }

    pub fn save(&self, path: &Path) -> Result<(), KeyError> {
        let file = KeyFile {
            alg: self.alg,
            seed: hex::encode(self.seed),
            seal_x25519: hex::encode(self.seal.x25519.to_bytes()),
            seal_mlkem: hex::encode(self.seal.mlkem_bytes()?),
            seal_mlkem_public: self.seal.public.mlkem.clone(),
            did_bio: self.network.map(|n| network_name(n).to_owned()),
            tls: self.tls.clone(),
        };
        let text = serde_json::to_vec_pretty(&file).map_err(|e| KeyError::Format(e.to_string()))?;
        write_private(path, &text)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Identity, KeyError> {
        let file: KeyFile = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| KeyError::Format(e.to_string()))?;
        let seed = decode32(&file.seed)?;
        let x25519 = StaticSecret::from(decode32(&file.seal_x25519)?);
        if file.did_bio.is_some() && file.alg != SigAlg::Ed25519 {
            return Err(KeyError::Format("did:bio needs an Ed25519 key".into()));
        }
        Ok(Identity {
            alg: file.alg,
            seed,
            signing: signing_from_seed(file.alg, &seed)?,
            seal: SealKey {
                public: SealPublic {
                    x25519: hex::encode(XPublic::from(&x25519).as_bytes()),
                    mlkem: file.seal_mlkem_public,
                },
                x25519,
                mlkem: DecapsulationKey::new(
                    &ML_KEM_768,
                    &hex::decode(&file.seal_mlkem).map_err(|e| KeyError::Format(e.to_string()))?,
                )
                .map_err(|_| KeyError::Format("bad ML-KEM key".into()))?,
            },
            network: file.did_bio.as_deref().map(network_named).transpose()?,
            tls: file.tls,
        })
    }

    pub fn did_string(&self) -> String {
        self.did().as_str().to_owned()
    }

    pub fn seal_public(&self) -> &SealPublic {
        &self.seal.public
    }

    pub fn open(&self, context: &[u8], sealed: &[u8]) -> Result<[u8; 32], KeyError> {
        self.seal.open(context, sealed)
    }

    /// This identity's signature over its sealing key. A plan carries it, so a
    /// sponsor can't name an honest helper and pair it with someone else's key.
    pub fn seal_proof(&self) -> Result<String, KeyError> {
        let signature = self.sign(&seal_proof_bytes(self.seal_public()))?;
        Ok(hex::encode(signature.as_bytes()))
    }
}

impl Signer for Identity {
    fn public_key(&self) -> &PublicKey {
        match &self.signing {
            SigningKey::Ed25519(k) => k.public_key(),
            SigningKey::MlDsa(k) => k.public_key(),
        }
    }

    fn sign(&self, message: &[u8]) -> Result<Signature, CryptoError> {
        match &self.signing {
            SigningKey::Ed25519(k) => k.sign(message),
            SigningKey::MlDsa(k) => k.sign(message),
        }
    }

    fn did(&self) -> Did {
        let bio = self.network.and_then(|network| {
            let subject: [u8; 32] = self.public_key().as_bytes().try_into().ok()?;
            Did::parse(&BioDid::new(network, subject).to_string()).ok()
        });
        bio.unwrap_or_else(|| self.public_key().did())
    }
}

/// The key that speaks for a DID: the one inside a `did:key`, or the subject
/// key of a `did:bio`. Whether a `did:bio` document has since moved on is for
/// the registry to say, and each party asks it for itself.
pub fn key_of(did: &Did) -> Result<PublicKey, String> {
    if did.method() != "bio" {
        return did.key().map_err(|e| e.to_string());
    }
    let bio = BioDid::parse(did.base()).map_err(|e| e.to_string())?;
    if !bio.is_key_subject() {
        return Err("an owned did:bio identifier holds no key".into());
    }
    PublicKey::new(Algorithm::Ed25519, &bio.subject).map_err(|e| format!("{e:?}"))
}

/// Resolves the DIDs a run accepts, for checking delegations.
pub struct Principals;

impl Resolver for Principals {
    fn resolve(&self, did: &Did) -> Result<PublicKey, ResolveError> {
        key_of(did).map_err(ResolveError::Failed)
    }
}

fn signing_from_seed(alg: SigAlg, seed: &[u8; 32]) -> Result<SigningKey, KeyError> {
    Ok(match alg {
        SigAlg::Ed25519 => SigningKey::Ed25519(Ed25519Keypair::from_seed(seed)),
        SigAlg::MlDsa65 => SigningKey::MlDsa(MlDsaKeypair::from_seed(Algorithm::MlDsa65, seed)?),
    })
}

fn decode32(text: &str) -> Result<[u8; 32], KeyError> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(text, &mut out).map_err(|e| KeyError::Format(e.to_string()))?;
    Ok(out)
}

fn write_private(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(data)
}

/// Recipient keys for sealing: X25519 and ML-KEM-768 together, so a break of
/// either one alone does not open the box.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealPublic {
    pub x25519: String,
    pub mlkem: String,
}

struct SealKey {
    x25519: StaticSecret,
    mlkem: DecapsulationKey,
    public: SealPublic,
}

const X25519_LEN: usize = 32;

fn seal_key(
    x: &[u8],
    kem: &[u8],
    eph: &[u8],
    ct: &[u8],
    public: &SealPublic,
    context: &[u8],
) -> Key {
    let mut hasher = blake3::Hasher::new_derive_key("fedtally v1 seal");
    for part in [
        x,
        kem,
        eph,
        ct,
        public.x25519.as_bytes(),
        public.mlkem.as_bytes(),
        context,
    ] {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    Key::from(*hasher.finalize().as_bytes())
}

impl SealKey {
    fn generate() -> Result<SealKey, KeyError> {
        let failed = |_| KeyError::Format("ML-KEM generation failed".into());
        let x25519 = StaticSecret::from(random_bytes::<32>());
        let mlkem = DecapsulationKey::generate(&ML_KEM_768).map_err(failed)?;
        let encap = mlkem.encapsulation_key().map_err(failed)?;
        let public = SealPublic {
            x25519: hex::encode(XPublic::from(&x25519).as_bytes()),
            mlkem: hex::encode(encap.key_bytes().map_err(failed)?.as_ref()),
        };
        Ok(SealKey {
            x25519,
            mlkem,
            public,
        })
    }

    fn mlkem_bytes(&self) -> Result<Vec<u8>, KeyError> {
        let bytes = self
            .mlkem
            .key_bytes()
            .map_err(|_| KeyError::Format("ML-KEM export failed".into()))?;
        Ok(bytes.as_ref().to_vec())
    }

    fn open(&self, context: &[u8], sealed: &[u8]) -> Result<[u8; 32], KeyError> {
        let public = &self.public;
        let kem_len = ml_kem_ciphertext_len();
        if sealed.len() < X25519_LEN + kem_len {
            return Err(KeyError::Seal);
        }
        let (eph, rest) = sealed.split_at(X25519_LEN);
        let (ct, boxed) = rest.split_at(kem_len);
        let eph_arr: [u8; 32] = eph.try_into().map_err(|_| KeyError::Seal)?;
        let x_shared = self.x25519.diffie_hellman(&XPublic::from(eph_arr));
        let kem_shared = self
            .mlkem
            .decapsulate(Ciphertext::from(ct))
            .map_err(|_| KeyError::Seal)?;
        let key = seal_key(
            x_shared.as_bytes(),
            kem_shared.as_ref(),
            eph,
            ct,
            public,
            context,
        );
        let plain = ChaCha20Poly1305::new(&key)
            .decrypt(
                &AeadNonce::default(),
                Payload {
                    msg: boxed,
                    aad: context,
                },
            )
            .map_err(|_| KeyError::Seal)?;
        plain.try_into().map_err(|_| KeyError::Seal)
    }
}

const fn ml_kem_ciphertext_len() -> usize {
    1088
}

/// Encrypts a 32 byte secret to `public`. The key is fresh for every box, so a
/// fixed nonce is safe.
pub fn seal(public: &SealPublic, context: &[u8], secret: &[u8; 32]) -> Result<Vec<u8>, KeyError> {
    let bad = |what: &str| KeyError::Format(format!("bad {what} in plan"));
    let x_bytes: [u8; 32] = hex::decode(&public.x25519)
        .ok()
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| bad("x25519 key"))?;
    let kem_bytes = hex::decode(&public.mlkem).map_err(|_| bad("ML-KEM key"))?;
    let encap = EncapsulationKey::new(&ML_KEM_768, &kem_bytes).map_err(|_| bad("ML-KEM key"))?;

    let eph = StaticSecret::from(random_bytes::<32>());
    let eph_public = XPublic::from(&eph);
    let x_shared = eph.diffie_hellman(&XPublic::from(x_bytes));
    let (ct, kem_shared) = encap.encapsulate().map_err(|_| KeyError::Seal)?;
    let key = seal_key(
        x_shared.as_bytes(),
        kem_shared.as_ref(),
        eph_public.as_bytes(),
        ct.as_ref(),
        public,
        context,
    );
    let boxed = ChaCha20Poly1305::new(&key)
        .encrypt(
            &AeadNonce::default(),
            Payload {
                msg: secret,
                aad: context,
            },
        )
        .map_err(|_| KeyError::Seal)?;
    let mut out = eph_public.as_bytes().to_vec();
    out.extend_from_slice(ct.as_ref());
    out.extend_from_slice(&boxed);
    Ok(out)
}

fn seal_proof_bytes(public: &SealPublic) -> Vec<u8> {
    format!("fedtally v1 seal key\0{}\0{}", public.x25519, public.mlkem).into_bytes()
}

/// Checks that `did` vouches for `public`.
pub fn verify_seal_proof(did: &Did, public: &SealPublic, proof: &str) -> Result<(), String> {
    let signature = hex::decode(proof).map_err(|e| e.to_string())?;
    verify_did(did, &seal_proof_bytes(public), &signature)
}

/// Parses and checks a detached signature against a DID.
pub fn verify_did(did: &Did, message: &[u8], signature: &[u8]) -> Result<(), String> {
    let key = key_of(did)?;
    let signature = Signature::new(key.algorithm(), signature).map_err(|e| format!("{e:?}"))?;
    key.verify(message, &signature)
        .map_err(|e| format!("{e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_did_bio_identity_is_the_same_key_under_another_name() {
        let dir = tempfile::tempdir().unwrap();
        let plain = Identity::generate(SigAlg::Ed25519).unwrap();
        let key_form = plain.did();
        let bio = plain.on_registry("devnet").unwrap();
        assert!(
            bio.did_string().starts_with("did:bio:devnet:"),
            "{}",
            bio.did_string()
        );
        assert_eq!(key_of(&bio.did()).unwrap(), key_of(&key_form).unwrap());

        let sig = bio.sign(b"m").unwrap();
        verify_did(&bio.did(), b"m", sig.as_bytes()).unwrap();
        verify_did(&key_form, b"m", sig.as_bytes()).unwrap();
        assert!(verify_did(&bio.did(), b"n", sig.as_bytes()).is_err());
        assert_eq!(
            Principals.resolve(&bio.did()).unwrap(),
            key_of(&key_form).unwrap()
        );

        let path = dir.path().join("k");
        bio.save(&path).unwrap();
        assert_eq!(
            Identity::load(&path).unwrap().did_string(),
            bio.did_string()
        );
        let mainnet = Identity::generate(SigAlg::Ed25519)
            .unwrap()
            .on_registry("mainnet")
            .unwrap();
        assert_eq!(mainnet.did_string().matches(':').count(), 2);
    }

    #[test]
    fn did_bio_needs_a_key_it_can_hold() {
        let pq = Identity::generate(SigAlg::MlDsa65).unwrap();
        assert!(pq.on_registry("devnet").is_err());
        let ed = Identity::generate(SigAlg::Ed25519).unwrap();
        assert!(ed.on_registry("moonnet").is_err());
        // An owned identifier is an address off the curve, with no key in it.
        let owned = did_bio_core::BioDid::owned(Network::Devnet, &[3; 32], 7);
        let did = Did::parse(&owned.to_string()).unwrap();
        assert!(key_of(&did).unwrap_err().contains("owned"));
        assert!(Principals.resolve(&did).is_err());
    }

    #[test]
    fn identities_round_trip_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        for alg in [SigAlg::Ed25519, SigAlg::MlDsa65] {
            let path = dir.path().join(format!("{alg:?}"));
            let id = Identity::generate(alg).unwrap();
            id.save(&path).unwrap();
            let again = Identity::load(&path).unwrap();
            assert_eq!(id.did_string(), again.did_string());
            assert_eq!(id.seal_public(), again.seal_public());
            let sig = again.sign(b"m").unwrap();
            verify_did(&id.did(), b"m", sig.as_bytes()).unwrap();
            assert!(verify_did(&id.did(), b"n", sig.as_bytes()).is_err());
        }
    }

    #[test]
    fn a_seal_proof_binds_the_key_to_the_identity() {
        let (helper, other) = (
            Identity::generate(SigAlg::Ed25519).unwrap(),
            Identity::generate(SigAlg::Ed25519).unwrap(),
        );
        let proof = helper.seal_proof().unwrap();
        verify_seal_proof(&helper.did(), helper.seal_public(), &proof).unwrap();
        assert!(verify_seal_proof(&helper.did(), other.seal_public(), &proof).is_err());
        assert!(verify_seal_proof(&other.did(), helper.seal_public(), &proof).is_err());
    }

    #[test]
    fn key_files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k");
        Identity::generate(SigAlg::Ed25519)
            .unwrap()
            .save(&path)
            .unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    fn helper_reloaded(id: &Identity) -> Identity {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k");
        id.save(&path).unwrap();
        Identity::load(&path).unwrap()
    }

    #[test]
    fn sealed_boxes_open_only_with_the_right_key_and_context() {
        let helper = Identity::generate(SigAlg::Ed25519).unwrap();
        let other = Identity::generate(SigAlg::Ed25519).unwrap();
        let secret = [7u8; 32];
        let sealed = seal(helper.seal_public(), b"round 1", &secret).unwrap();
        assert_eq!(helper.open(b"round 1", &sealed).unwrap(), secret);
        let reloaded = helper_reloaded(&helper);
        assert_eq!(reloaded.open(b"round 1", &sealed).unwrap(), secret);
        assert!(helper.open(b"round 2", &sealed).is_err());
        assert!(other.open(b"round 1", &sealed).is_err());
        for short in [0, 31, 32, 1119, sealed.len() - 1] {
            assert!(
                helper.open(b"round 1", &sealed[..short]).is_err(),
                "{short}"
            );
        }
        let mut flipped = sealed.clone();
        *flipped.last_mut().unwrap() ^= 1;
        assert!(helper.open(b"round 1", &flipped).is_err());
    }
}
