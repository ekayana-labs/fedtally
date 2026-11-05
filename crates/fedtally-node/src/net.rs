//! TLS with a pinned aggregator certificate, and the identity handshake.
//!
//! The certificate pin comes from the signed plan, so there is no CA to run.
//! Each peer then proves its key over the TLS session itself.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use fedtally_core::keys::{Identity, TlsCredentials, verify_did};
use fedtally_core::pq_ucan::crypto::Signer;
use fedtally_core::pq_ucan::did::Did;
use fedtally_core::vector::Digest;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, aws_lc_rs, verify_tls13_signature};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use rustls::{ClientConfig, DigitallySignedStruct, ServerConfig, SignatureScheme};
use tokio::io::{AsyncRead, AsyncWrite, ReadHalf, WriteHalf, split};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::client::TlsStream as ClientTls;
use tokio_rustls::server::TlsStream as ServerTls;
use tokio_rustls::{TlsAcceptor, TlsConnector};

use crate::NodeError;
use crate::wire::{Msg, SILENCE_LIMIT, read_msg, write_msg};

const SERVER_NAME: &str = "fedtally";
const EXPORT_LABEL: &[u8] = b"fedtally v1 peer";
const HELLO_DOMAIN: &[u8] = b"fedtally v1 hello\0";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);

fn provider() -> Arc<CryptoProvider> {
    Arc::new(aws_lc_rs::default_provider())
}

/// A self-signed certificate. Its digest is what the plan pins.
pub fn generate_tls() -> Result<TlsCredentials, NodeError> {
    let certified = rcgen::generate_simple_self_signed(vec![SERVER_NAME.to_owned()])
        .map_err(|e| NodeError::Tls(e.to_string()))?;
    Ok(TlsCredentials {
        cert: hex::encode(certified.cert.der()),
        key: hex::encode(certified.signing_key.serialize_der()),
    })
}

pub fn tls_pin(credentials: &TlsCredentials) -> Result<Digest, NodeError> {
    let der = hex::decode(&credentials.cert).map_err(|e| NodeError::Tls(e.to_string()))?;
    Ok(Digest::of(&der))
}

#[derive(Debug)]
struct PinnedCert {
    pin: Digest,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for PinnedCert {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        if Digest::of(end_entity.as_ref()) == self.pin {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General(
                "certificate does not match the plan".into(),
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Err(rustls::Error::General("TLS 1.2 is not offered".into()))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn hello_bytes(plan_id: &Digest, exported: &[u8; 32]) -> Vec<u8> {
    let mut out = HELLO_DOMAIN.to_vec();
    out.extend_from_slice(&plan_id.0);
    out.extend_from_slice(exported);
    out
}

/// The accepting side of the transport.
pub struct Listener {
    tcp: TcpListener,
    acceptor: TlsAcceptor,
}

impl Listener {
    pub async fn bind(addr: &str, identity: &Identity) -> Result<Listener, NodeError> {
        let credentials = identity
            .tls
            .as_ref()
            .ok_or_else(|| NodeError::Tls("the aggregator key has no TLS credentials".into()))?;
        let decode = |text: &str| hex::decode(text).map_err(|e| NodeError::Tls(e.to_string()));
        let cert = CertificateDer::from(decode(&credentials.cert)?);
        let key = PrivateKeyDer::try_from(decode(&credentials.key)?)
            .map_err(|e| NodeError::Tls(e.to_string()))?;
        let config = ServerConfig::builder_with_provider(provider())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|e| NodeError::Tls(e.to_string()))?
            .with_no_client_auth()
            .with_single_cert(vec![cert], key)
            .map_err(|e| NodeError::Tls(e.to_string()))?;
        Ok(Listener {
            tcp: TcpListener::bind(addr).await?,
            acceptor: TlsAcceptor::from(Arc::new(config)),
        })
    }

    pub fn local_addr(&self) -> Result<String, NodeError> {
        Ok(self.tcp.local_addr()?.to_string())
    }

    pub async fn accept(&self) -> Result<TcpStream, NodeError> {
        let (stream, _) = self.tcp.accept().await?;
        stream.set_nodelay(true)?;
        Ok(stream)
    }

    pub fn acceptor(&self) -> TlsAcceptor {
        self.acceptor.clone()
    }
}

/// Finishes TLS and the identity proof on an accepted connection.
pub async fn accept_peer(
    acceptor: TlsAcceptor,
    stream: TcpStream,
    plan_id: Digest,
    frame_limit: usize,
) -> Result<(String, ServerTls<TcpStream>), NodeError> {
    tokio::time::timeout(HANDSHAKE_TIMEOUT, async {
        let mut tls = acceptor
            .accept(stream)
            .await
            .map_err(|e| NodeError::Tls(e.to_string()))?;
        let mut exported = [0u8; 32];
        tls.get_ref()
            .1
            .export_keying_material(&mut exported[..], EXPORT_LABEL, None)
            .map_err(|e| NodeError::Tls(e.to_string()))?;
        let Msg::Hello { did, sig } = read_msg(&mut tls, frame_limit).await? else {
            return Err(NodeError::Protocol("expected hello".into()));
        };
        let parsed = Did::parse(&did).map_err(|e| NodeError::Protocol(e.to_string()))?;
        verify_did(&parsed, &hello_bytes(&plan_id, &exported), &sig)
            .map_err(|_| NodeError::Protocol("bad hello signature".into()))?;
        write_msg(&mut tls, &Msg::Welcome).await?;
        Ok((did, tls))
    })
    .await
    .map_err(|_| NodeError::Protocol("handshake timed out".into()))?
}

/// A peer's connection to the aggregator. Log entries that arrive while the
/// peer is waiting for something else are kept for later.
pub struct Client {
    reader: ReadHalf<ClientTls<TcpStream>>,
    writer: WriteHalf<ClientTls<TcpStream>>,
    backlog: VecDeque<Vec<Vec<u8>>>,
    frame_limit: usize,
    key_exchange: Option<String>,
}

impl Client {
    pub async fn connect(
        addr: &str,
        pin: Digest,
        plan_id: Digest,
        identity: &Identity,
        frame_limit: usize,
    ) -> Result<Client, NodeError> {
        let provider = provider();
        let config = ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|e| NodeError::Tls(e.to_string()))?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PinnedCert { pin, provider }))
            .with_no_client_auth();
        let stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;
        let name = ServerName::try_from(SERVER_NAME).map_err(|e| NodeError::Tls(e.to_string()))?;
        let mut tls = TlsConnector::from(Arc::new(config))
            .connect(name, stream)
            .await
            .map_err(|e| NodeError::Tls(e.to_string()))?;

        let mut exported = [0u8; 32];
        tls.get_ref()
            .1
            .export_keying_material(&mut exported[..], EXPORT_LABEL, None)
            .map_err(|e| NodeError::Tls(e.to_string()))?;
        let sig = identity
            .sign(&hello_bytes(&plan_id, &exported))
            .map_err(|e| NodeError::Tls(format!("{e:?}")))?;
        let hello = Msg::Hello {
            did: identity.did_string(),
            sig: sig.as_bytes().to_vec(),
        };
        write_msg(&mut tls, &hello).await?;
        match read_msg(&mut tls, frame_limit).await? {
            Msg::Welcome => {}
            Msg::Refused { reason } => return Err(NodeError::Refused(reason)),
            other => {
                return Err(NodeError::Protocol(format!(
                    "expected welcome, got {other:?}"
                )));
            }
        }
        let key_exchange = tls
            .get_ref()
            .1
            .negotiated_key_exchange_group()
            .map(|group| format!("{:?}", group.name()));
        let (reader, writer) = split(tls);
        Ok(Client {
            reader,
            writer,
            backlog: VecDeque::new(),
            frame_limit,
            key_exchange,
        })
    }

    pub async fn send(&mut self, msg: &Msg) -> Result<(), NodeError> {
        write_msg(&mut self.writer, msg).await
    }

    async fn next(&mut self) -> Result<Msg, NodeError> {
        loop {
            let next =
                tokio::time::timeout(SILENCE_LIMIT, read_msg(&mut self.reader, self.frame_limit))
                    .await
                    .map_err(|_| NodeError::Disconnected("the aggregator went silent".into()))??;
            match next {
                Msg::Ping => continue,
                Msg::Refused { reason } => return Err(NodeError::Refused(reason)),
                other => return Ok(other),
            }
        }
    }

    /// The next reply. Log entries that arrive first are kept for `entries`.
    pub async fn reply(&mut self) -> Result<Msg, NodeError> {
        loop {
            match self.next().await? {
                Msg::Entries { lines } => self.backlog.push_back(unwrap_lines(lines)),
                other => return Ok(other),
            }
        }
    }

    /// The next batch of log entries.
    pub async fn entries(&mut self) -> Result<Vec<Vec<u8>>, NodeError> {
        if let Some(batch) = self.backlog.pop_front() {
            return Ok(batch);
        }
        match self.next().await? {
            Msg::Entries { lines } => Ok(unwrap_lines(lines)),
            other => Err(NodeError::Protocol(format!(
                "expected log entries, got {other:?}"
            ))),
        }
    }

    /// The TLS key exchange this connection settled on.
    pub fn key_exchange(&self) -> Option<&str> {
        self.key_exchange.as_deref()
    }

    pub fn has_backlog(&self) -> bool {
        !self.backlog.is_empty()
    }
}

fn unwrap_lines(lines: Vec<serde_bytes::ByteBuf>) -> Vec<Vec<u8>> {
    lines
        .into_iter()
        .map(serde_bytes::ByteBuf::into_vec)
        .collect()
}

/// Splits an accepted stream for the serving loop.
pub fn split_stream<S: AsyncRead + AsyncWrite>(stream: S) -> (ReadHalf<S>, WriteHalf<S>) {
    split(stream)
}
