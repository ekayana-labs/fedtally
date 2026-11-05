//! Cohort sampling. Each round has a beacon, each site a ticket under it, and
//! a task that samples `k` of its sites invites the `k` with the smallest
//! tickets. Tickets compare as 32 byte big-endian numbers, which is the
//! order of the byte arrays themselves.

use crate::hash::{sha256, Digest};

const BEACON_TAG: &[u8] = b"fedtally v1 beacon";
const TICKET_TAG: &[u8; 18] = b"fedtally v1 ticket";

/// The beacon of `round`, which starts from the model with digest `model`.
pub fn beacon(plan_id: &[u8; 32], round: u32, model: &Digest) -> Digest {
    sha256(&[BEACON_TAG, plan_id, &round.to_le_bytes(), model])
}

/// The tickets under one beacon. The hashed message is laid out once and
/// only the site index changes between tickets, so a ticket costs a single
/// hash over one slice.
pub struct Tickets {
    message: [u8; Self::MESSAGE_LEN],
}

impl Tickets {
    const INDEX: usize = TICKET_TAG.len() + 32;
    const MESSAGE_LEN: usize = Self::INDEX + 4;

    pub fn new(beacon: &Digest) -> Self {
        let mut message = [0u8; Self::MESSAGE_LEN];
        message[..TICKET_TAG.len()].copy_from_slice(TICKET_TAG);
        message[TICKET_TAG.len()..Self::INDEX].copy_from_slice(beacon);
        Self { message }
    }

    #[inline]
    pub fn of(&mut self, site: u32) -> Digest {
        self.message[Self::INDEX..].copy_from_slice(&site.to_le_bytes());
        sha256(&[&self.message])
    }
}

/// The ticket of site `site` under `beacon`.
pub fn ticket(beacon: &Digest, site: u32) -> Digest {
    Tickets::new(beacon).of(site)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn published_beacon() -> Digest {
        let plan_id: [u8; 32] = core::array::from_fn(|i| i as u8);
        beacon(&plan_id, 3, &[0x07; 32])
    }

    #[test]
    fn published_beacon_vector() {
        assert_eq!(
            hex(&published_beacon()),
            "ceca54fea8e95b4f6e7a32645347900387e511946c06cfa23f46dab22702a892"
        );
    }

    #[test]
    fn published_ticket_vectors() {
        let beacon = published_beacon();
        let ends = [
            ("86bfe570", "72a8"),
            ("4338762c", "2a61"),
            ("65cb3cd5", "3ba2"),
            ("00e856cc", "85fd"),
        ];
        for (site, (head, tail)) in ends.iter().enumerate() {
            let ticket = hex(&ticket(&beacon, site as u32));
            assert!(ticket.starts_with(head), "ticket({site}) = {ticket}");
            assert!(ticket.ends_with(tail), "ticket({site}) = {ticket}");
        }
    }

    #[test]
    fn tickets_are_the_hash_of_tag_beacon_and_index() {
        let beacon = published_beacon();
        let mut tickets = Tickets::new(&beacon);
        for site in [0u32, 1, 255, 256, 1023] {
            let expected = sha256(&[b"fedtally v1 ticket", &beacon, &site.to_le_bytes()]);
            assert_eq!(tickets.of(site), expected);
            assert_eq!(ticket(&beacon, site), expected);
        }
    }
}
