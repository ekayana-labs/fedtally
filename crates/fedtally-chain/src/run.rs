//! A run as its log tells it, reduced to what the ledger is owed. The log is
//! read from the file a node keeps, so an anchor sees exactly what its node
//! has replayed and accepted.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use fedtally_core::anchor::{self, Committed, Posted, Terms};
use fedtally_core::credential::Credential;
use fedtally_core::record::{Body, Record, RecordError};
use fedtally_core::replay::{Fault, Phase, Replay};
use fedtally_core::vector::{Digest, Shape};
use fedtally_ledger::client::StatusRef;
use fedtally_ledger::state::{MAX_STATUS_BITS, MAX_STATUS_REFS};

use crate::error::ChainError;

/// One round of the log.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RoundLog {
    pub committed: Option<Committed>,
    pub posted: Option<Posted>,
    /// The root each site committed to for its vector.
    pub vectors: BTreeMap<u16, Digest>,
    /// One root per helper, once every helper has given its own.
    pub masks: Vec<Digest>,
    /// The plan's index of each active helper, in the order of `masks`.
    pub active: Vec<usize>,
    pub aggregate: Option<Digest>,
    pub acks: BTreeSet<u16>,
    /// Given up before its seal.
    pub voided: bool,
    /// Given up because a helper was removed.
    pub removed_helper: bool,
}

impl RoundLog {
    /// Nothing the log can still add would change this round.
    fn is_settled(&self) -> bool {
        if self.voided {
            return true;
        }
        match (&self.committed, &self.posted) {
            (Some(committed), Some(_)) => committed
                .cohort
                .iter()
                .all(|site| self.acks.contains(&(*site as u16))),
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ending {
    #[default]
    Running,
    Finished,
    Aborted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunView {
    pub terms: Terms,
    pub publisher: [u8; 32],
    pub aggregator: [u8; 32],
    /// The wallet of each site, in plan order.
    pub wallets: Vec<[u8; 32]>,
    /// The digest of each site's Join record.
    pub joins: BTreeMap<u16, Digest>,
    pub status_refs: BTreeMap<u16, Vec<StatusRef>>,
    pub rounds: Vec<RoundLog>,
    pub revoked: BTreeSet<u16>,
    pub ending: Ending,
}

impl RunView {
    pub fn round(&self, number: u32) -> Option<&RoundLog> {
        self.rounds.get(number.checked_sub(1)? as usize)
    }
}

/// The number a status list has on the ledger: the first four bytes of the
/// digest of the id its credentials name it by.
pub fn list_number(id: &str) -> u32 {
    let digest = Digest::of(id.as_bytes());
    u32::from_le_bytes([digest.0[0], digest.0[1], digest.0[2], digest.0[3]])
}

/// The ledger bits that revoke a site holding these credentials. The ledger
/// keeps four of them. An issuer without an Ed25519 key has no list there.
pub fn status_refs(credentials: &[Credential]) -> Vec<StatusRef> {
    credentials
        .iter()
        .filter_map(|credential| {
            let entry = credential.status.as_ref()?;
            let issuer = anchor::wallet(&credential.issuer).ok()?;
            let bit = u32::try_from(entry.index).ok()?;
            (bit < MAX_STATUS_BITS).then(|| StatusRef {
                list: fedtally_ledger::client::status_list(&issuer, list_number(&entry.list)),
                bit,
            })
        })
        .take(MAX_STATUS_REFS)
        .collect()
}

pub struct Run {
    path: PathBuf,
    offset: u64,
    plan_line: Vec<u8>,
    replay: Replay,
    view: RunView,
    shape: Shape,
}

impl Run {
    /// Starts from the signed plan. The log at `path` may not exist yet.
    pub fn open(plan_line: &[u8], path: &Path) -> Result<Run, ChainError> {
        let record = Record::parse(plan_line).map_err(|_| ChainError::NotAPlan)?;
        let Body::Plan(plan) = &record.body else {
            return Err(ChainError::NotAPlan);
        };
        let terms = plan
            .ledger_terms(record.body_hash)
            .map_err(ChainError::Terms)?;
        let wallet = |did: &str| anchor::wallet(did).map_err(ChainError::Terms);
        let view = RunView {
            terms,
            publisher: wallet(&plan.sponsor)?,
            aggregator: wallet(&plan.aggregator.did)?,
            wallets: plan
                .sites
                .iter()
                .map(|site| wallet(site))
                .collect::<Result<_, _>>()?,
            joins: BTreeMap::new(),
            status_refs: BTreeMap::new(),
            rounds: Vec::new(),
            revoked: BTreeSet::new(),
            ending: Ending::Running,
        };
        let mut replay = Replay::new();
        replay.apply(&record)?;
        let mut run = Run {
            path: path.to_owned(),
            offset: 0,
            plan_line: plan_line.to_owned(),
            replay,
            view,
            shape: plan.shape(),
        };
        run.refresh()?;
        Ok(run)
    }

    pub fn view(&self) -> &RunView {
        &self.view
    }

    pub fn replay(&self) -> &Replay {
        &self.replay
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Takes in the lines the log has gained. Returns whether there were any.
    pub fn refresh(&mut self) -> Result<bool, ChainError> {
        let mut file = match File::open(&self.path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        file.seek(SeekFrom::Start(self.offset))?;
        let mut fresh = Vec::new();
        file.read_to_end(&mut fresh)?;
        // A line still being written has no end yet and waits for the next read.
        let Some(end) = fresh.iter().rposition(|byte| *byte == b'\n') else {
            return Ok(false);
        };
        let lines: Vec<&[u8]> = fresh[..end].split(|byte| *byte == b'\n').collect();
        let mut took = false;
        for (at, line) in lines.iter().enumerate() {
            let record = match Record::parse(line) {
                // A power cut tore this one, and its party cuts it on restart.
                Err(RecordError::Malformed(_)) if at + 1 == lines.len() => break,
                parsed => parsed.map_err(Fault::from)?,
            };
            if self.offset == 0 {
                if *line != self.plan_line {
                    return Err(ChainError::AnotherPlan);
                }
            } else {
                self.replay.apply(&record)?;
                self.note(&record);
            }
            self.offset += line.len() as u64 + 1;
            took = true;
        }
        self.review();
        Ok(took)
    }

    fn note(&mut self, record: &Record) {
        if !matches!(record.body, Body::Join { .. }) {
            return;
        }
        let site = self
            .replay
            .plan()
            .and_then(|p| p.site_index(&record.signer));
        if let Some(site) = site {
            self.view.joins.insert(site as u16, record.body_hash);
        }
    }

    fn review(&mut self) {
        let replay = &self.replay;
        let view = &mut self.view;
        for (site, joined) in &replay.joined {
            view.status_refs
                .entry(*site as u16)
                .or_insert_with(|| status_refs(&joined.credentials));
        }
        view.revoked = replay.revoked.keys().map(|site| *site as u16).collect();
        view.ending = match replay.phase {
            Phase::Done => Ending::Finished,
            Phase::Aborted => Ending::Aborted,
            _ => Ending::Running,
        };
        let settled = view.rounds.iter().take_while(|r| r.is_settled()).count();
        view.rounds.truncate(settled);
        for (at, state) in replay.rounds.iter().enumerate().skip(settled) {
            let narrow = |site: &usize| *site as u16;
            let next = replay.rounds.get(at + 1);
            view.rounds.push(RoundLog {
                committed: replay.committed(state.round),
                posted: replay.posted(state.round),
                vectors: state
                    .contributions
                    .iter()
                    .map(|(site, c)| (narrow(site), c.masked))
                    .collect(),
                masks: match state.masks.len() == state.active.len() {
                    true => state.masks.values().copied().collect(),
                    false => Vec::new(),
                },
                active: state.active.clone(),
                aggregate: state.outcome.as_ref().map(|outcome| outcome.aggregate),
                acks: state.acks.iter().map(narrow).collect(),
                voided: state.voided,
                removed_helper: state.voided
                    && next.is_some_and(|next| next.active.len() < state.active.len()),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use fedtally_core::keys::SigAlg;
    use fedtally_core::model::{Dtype, Layout, TensorSpec, TensorView, write_model};
    use fedtally_core::vector::LEAF_COORDS;
    use fedtally_node::local::{LocalFederation, Options};

    use super::*;

    fn federation(dir: &Path, alg: SigAlg) -> LocalFederation {
        let layout = Layout {
            tensors: vec![TensorSpec {
                name: "w".into(),
                dtype: Dtype::F32,
                shape: vec![300],
            }],
        };
        std::fs::create_dir_all(dir).unwrap();
        let init = dir.join("init.safetensors");
        write_model(&init, &layout, &[TensorView::F32(&[0.5; 300])]).unwrap();
        let mut options = Options::new(3, 1, 2);
        options.chunk_coords = LEAF_COORDS;
        options.sig_alg = alg;
        LocalFederation::create(dir, layout, &init, &options).unwrap()
    }

    #[test]
    fn a_run_starts_from_its_plan_and_reads_only_whole_lines() {
        let dir = tempfile::tempdir().unwrap();
        let fed = federation(dir.path(), SigAlg::Ed25519);
        let log = dir.path().join("log.jsonl");
        let mut run = Run::open(&fed.plan_line, &log).unwrap();
        let view = run.view().clone();
        assert_eq!(view.terms, fed.plan.ledger_terms(fed.plan_id).unwrap());
        assert_eq!(view.wallets.len(), 3);
        assert_eq!(view.publisher, anchor::wallet(&fed.plan.sponsor).unwrap());
        assert!(view.rounds.is_empty() && view.joins.is_empty());
        assert!(!run.refresh().unwrap());

        // The plan arrives in two writes. Half a line is not a line.
        let (head, tail) = fed.plan_line.split_at(40);
        let mut file = File::create(&log).unwrap();
        file.write_all(head).unwrap();
        assert!(!run.refresh().unwrap());
        file.write_all(tail).unwrap();
        file.write_all(b"\n{\"torn").unwrap();
        assert!(run.refresh().unwrap());
        assert!(!run.refresh().unwrap());
        assert_eq!(run.view(), &view);
        assert_eq!(run.replay().entries, 1);

        // Ending in a newline doesn't make it a record, but only a last line is torn.
        file.write_all(b"\": 1}\n").unwrap();
        assert!(!run.refresh().unwrap());
        assert_eq!(run.replay().entries, 1);
        file.write_all(&fed.plan_line).unwrap();
        file.write_all(b"\n").unwrap();
        assert!(matches!(run.refresh(), Err(ChainError::Fault(_))));
    }

    #[test]
    fn a_log_for_another_plan_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let fed = federation(&dir.path().join("a"), SigAlg::Ed25519);
        let other = federation(&dir.path().join("b"), SigAlg::Ed25519);
        let log = dir.path().join("log.jsonl");
        std::fs::write(&log, [other.plan_line.as_slice(), b"\n"].concat()).unwrap();
        assert!(matches!(
            Run::open(&fed.plan_line, &log),
            Err(ChainError::AnotherPlan)
        ));
        assert!(matches!(Run::open(b"{}", &log), Err(ChainError::NotAPlan)));
    }

    #[test]
    fn a_plan_whose_parties_have_no_wallets_cannot_be_anchored() {
        let dir = tempfile::tempdir().unwrap();
        let fed = federation(dir.path(), SigAlg::MlDsa65);
        let refused = Run::open(&fed.plan_line, &dir.path().join("log.jsonl"));
        assert!(matches!(refused, Err(ChainError::Terms(why)) if why.contains("Ed25519")));
    }

    #[test]
    fn a_list_keeps_its_number() {
        let number = list_number("https://issuer.example/lists/3");
        assert_eq!(number, list_number("https://issuer.example/lists/3"));
        assert_ne!(number, list_number("https://issuer.example/lists/4"));
    }
}
