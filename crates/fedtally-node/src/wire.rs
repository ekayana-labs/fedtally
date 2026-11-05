//! Messages between a peer and the aggregator.

use std::time::Duration;

use fedtally_core::vector::{Digest, Manifest};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::NodeError;

pub const PING_EVERY: Duration = Duration::from_secs(10);
pub const SILENCE_LIMIT: Duration = Duration::from_secs(45);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Object {
    InitialModel,
    Model { round: u32 },
    Upload { round: u32, site: u32 },
    Mask { round: u32, helper: u32 },
    Aggregate { round: u32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Msg {
    Hello {
        did: String,
        #[serde(with = "serde_bytes")]
        sig: Vec<u8>,
    },
    Welcome,
    Ping,

    Subscribe {
        from: u64,
        head: Digest,
    },
    Entries {
        lines: Vec<serde_bytes::ByteBuf>,
    },
    Append {
        #[serde(with = "serde_bytes")]
        line: Vec<u8>,
    },
    Refused {
        reason: String,
    },

    PutManifest {
        object: Object,
        manifest: Manifest,
    },
    Missing {
        object: Object,
        chunks: Vec<u64>,
    },
    PutChunk {
        object: Object,
        chunk: u64,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    PutDone {
        object: Object,
    },
    Stored {
        object: Object,
    },

    GetManifest {
        object: Object,
    },
    Manifest {
        object: Object,
        manifest: Manifest,
    },
    GetFile {
        object: Object,
    },
    FileChunk {
        object: Object,
        offset: u64,
        last: bool,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    GetChunk {
        object: Object,
        chunk: u64,
    },
    Chunk {
        object: Object,
        chunk: u64,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    Unavailable {
        object: Object,
    },

    PutSealed {
        round: u32,
        helper: u32,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    GetSealed {
        round: u32,
        site: u32,
    },
    Sealed {
        round: u32,
        site: u32,
        data: Option<serde_bytes::ByteBuf>,
    },
}

const LENGTH_BYTES: usize = size_of::<u32>();
/// Room for everything in a message that carries data, other than the data.
const HEADER_ROOM: usize = 64;

pub async fn read_msg<R: AsyncRead + Unpin>(
    reader: &mut R,
    limit: usize,
) -> Result<Msg, NodeError> {
    let len = reader.read_u32_le().await? as usize;
    if len > limit {
        return Err(NodeError::Protocol(format!(
            "frame of {len} bytes is over the limit"
        )));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    postcard::from_bytes(&buf).map_err(|e| NodeError::Protocol(e.to_string()))
}

pub async fn write_msg<W: AsyncWrite + Unpin>(writer: &mut W, msg: &Msg) -> Result<(), NodeError> {
    let payload = match msg {
        Msg::PutChunk { data, .. }
        | Msg::Chunk { data, .. }
        | Msg::FileChunk { data, .. }
        | Msg::PutSealed { data, .. } => data.len(),
        _ => 0,
    };
    // The length goes in front in the same buffer, sized so a chunk is
    // copied into it once and the frame leaves in one write.
    let mut frame = Vec::with_capacity(LENGTH_BYTES + payload + HEADER_ROOM);
    frame.extend_from_slice(&[0; LENGTH_BYTES]);
    let mut frame =
        postcard::to_extend(msg, frame).map_err(|e| NodeError::Protocol(e.to_string()))?;
    let len = u32::try_from(frame.len() - LENGTH_BYTES)
        .map_err(|_| NodeError::Protocol("frame too large".into()))?;
    frame[..LENGTH_BYTES].copy_from_slice(&len.to_le_bytes());
    writer.write_all(&frame).await?;
    writer.flush().await?;
    Ok(())
}
