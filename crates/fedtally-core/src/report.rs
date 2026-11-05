//! What a valid log says, in terms a reader can act on.

use std::io;
use std::path::Path;

use serde::Serialize;
use thiserror::Error;

use crate::admission::Admission;
use crate::log::read_lines;
use crate::replay::{Fault, Phase, Replay};
use crate::vector::Digest;

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("entry {entry}: {fault}")]
    Fault { entry: u64, fault: Fault },
    #[error("the log is empty")]
    Empty,
    #[error("the model file has digest {got}, the log says {want}")]
    Model { got: Digest, want: Digest },
}

#[derive(Debug, Serialize)]
pub struct SiteJoin {
    pub site: String,
    pub at: i64,
    pub admission: Admission,
}

#[derive(Debug, Serialize)]
pub struct Contributor {
    pub site: String,
    pub weight: u64,
}

#[derive(Debug, Serialize)]
pub struct RoundReport {
    pub round: u32,
    pub included: Vec<Contributor>,
    /// Signed a contribution but were left out of the round.
    pub excluded: Vec<String>,
    /// On the plan, but nothing from them is in the round. They may have sat
    /// out, or the aggregator may have refused to log them.
    pub absent: Vec<String>,
    /// Included, but only the aggregator and helpers speak for their fingerprints.
    pub unattested: Vec<String>,
    pub unacked: Vec<String>,
    pub total_weight: Option<u64>,
    pub model: Option<Digest>,
    pub sealed: bool,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub plan_id: Digest,
    pub purpose: String,
    pub masked: bool,
    pub helpers: Vec<String>,
    pub aggregator: String,
    pub head: Digest,
    pub entries: u64,
    pub finished: bool,
    pub aborted: Option<String>,
    pub model: Digest,
    pub joined: Vec<SiteJoin>,
    pub rounds: Vec<RoundReport>,
}

impl Report {
    /// True when every included site attested and acknowledged every round,
    /// and the run reached its end.
    pub fn complete(&self) -> bool {
        self.finished
            && self.rounds.iter().all(|r| {
                r.sealed
                    && r.excluded.is_empty()
                    && r.absent.is_empty()
                    && r.unattested.is_empty()
                    && r.unacked.is_empty()
            })
    }

    /// The rounds a site's data went into.
    pub fn rounds_with(&self, site: &str) -> Vec<u32> {
        self.rounds
            .iter()
            .filter(|r| r.included.iter().any(|c| c.site == site))
            .map(|r| r.round)
            .collect()
    }
}

impl Replay {
    pub fn report(&self) -> Option<Report> {
        let plan = self.plan()?;
        let name = |site: &usize| plan.sites[*site].clone();
        let rounds = self
            .rounds
            .iter()
            .map(|r| RoundReport {
                round: r.round,
                included: r
                    .included
                    .iter()
                    .map(|s| Contributor {
                        site: name(s),
                        weight: r.contributions[s].weight,
                    })
                    .collect(),
                excluded: if r.close_head == Digest::default() {
                    Vec::new()
                } else {
                    r.contributions
                        .keys()
                        .filter(|s| !r.included.contains(s))
                        .map(name)
                        .collect()
                },
                absent: (0..plan.sites.len())
                    .filter(|s| !r.contributions.contains_key(s))
                    .map(|s| name(&s))
                    .collect(),
                unattested: r
                    .included
                    .iter()
                    .filter(|s| !r.site_attests.contains_key(s))
                    .map(name)
                    .collect(),
                unacked: r
                    .included
                    .iter()
                    .filter(|s| !r.acks.contains(s))
                    .map(name)
                    .collect(),
                total_weight: r.outcome.as_ref().map(|o| o.total_weight),
                model: r.outcome.as_ref().map(|o| o.model),
                sealed: r.seal.is_some(),
            })
            .collect();
        Some(Report {
            plan_id: self.plan_id,
            purpose: plan.purpose.clone(),
            masked: plan.is_masked(),
            helpers: plan.helpers.iter().map(|h| h.did.clone()).collect(),
            aggregator: plan.aggregator.did.clone(),
            head: self.head,
            entries: self.entries,
            finished: self.phase == Phase::Done,
            aborted: self.abort.clone(),
            model: self.model,
            joined: self
                .joined
                .iter()
                .map(|(s, j)| SiteJoin {
                    site: name(s),
                    at: j.at,
                    admission: j.admission.clone(),
                })
                .collect(),
            rounds,
        })
    }
}

/// Replays a log file from scratch and optionally checks the final model file.
pub fn verify_log(log: &Path, model: Option<&Path>) -> Result<Report, VerifyError> {
    let mut replay = Replay::new();
    for line in read_lines(log)? {
        replay
            .apply_line(&line)
            .map_err(|fault| VerifyError::Fault {
                entry: replay.entries,
                fault,
            })?;
    }
    let report = replay.report().ok_or(VerifyError::Empty)?;
    if let Some(path) = model {
        let got = Digest::file(path)?;
        if got != report.model {
            return Err(VerifyError::Model {
                got,
                want: report.model,
            });
        }
    }
    Ok(report)
}
