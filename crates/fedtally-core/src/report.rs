//! What a valid log says, in terms a reader can act on.

use std::io;
use std::path::Path;

use serde::Serialize;
use thiserror::Error;

use crate::admission::Admission;
use crate::log::read_lines;
use crate::privacy::{self, rho_units};
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
    /// The dataset the site said it trains on.
    pub dataset: Option<String>,
    /// The credentials it was admitted on.
    pub credentials: Vec<Held>,
}

#[derive(Debug, Serialize)]
pub struct Held {
    pub kind: String,
    pub issuer: String,
    pub digest: Digest,
    pub valid_until: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct Revoked {
    pub site: String,
    pub reason: String,
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
    /// The sites drawn for the round, where the plan samples.
    pub invited: Option<Vec<String>>,
    /// Could have been in the round, but nothing from them is in it. They may
    /// have sat out, or the aggregator may have refused to log them.
    pub absent: Vec<String>,
    /// Included, but only the aggregator and helpers speak for their fingerprints.
    pub unattested: Vec<String>,
    pub unacked: Vec<String>,
    pub total_weight: Option<u64>,
    pub model: Option<Digest>,
    pub sealed: bool,
    /// The zCDP rho this round spent, in a private run.
    pub rho: Option<f64>,
}

/// What a private run has spent. Rho is what the rules enforce. The epsilons
/// are worked out from it at the plan's delta, for people.
#[derive(Debug, Serialize)]
pub struct PrivacyReport {
    pub rho_spent: f64,
    pub rho_budget: f64,
    pub epsilon_spent: f64,
    pub epsilon_budget: f64,
    /// The target the plan states. It should not be below `epsilon_budget`.
    pub epsilon_stated: f64,
    pub delta: f64,
    /// Sites per cohort the accounting assumes give their noise away.
    pub colluding: u32,
    /// Rho per round for one site if every mask were removed later.
    pub rho_unmasked: f64,
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
    pub privacy: Option<PrivacyReport>,
    /// Sites whose credentials stopped meeting the policy during the run.
    pub revoked: Vec<Revoked>,
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

    /// The rho spent on rounds this site's data went into. A site that sat
    /// some rounds out has spent less than the run as a whole.
    pub fn rho_spent_by(&self, site: &str) -> Option<f64> {
        self.privacy.as_ref()?;
        Some(
            self.rounds
                .iter()
                .filter(|r| r.included.iter().any(|c| c.site == site))
                .filter_map(|r| r.rho)
                .sum(),
        )
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
                invited: r
                    .invited
                    .as_ref()
                    .map(|sites| sites.iter().map(name).collect()),
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
                absent: if r.close_head == Digest::default() {
                    self.expected(plan)
                } else {
                    r.expected.clone()
                }
                .iter()
                .filter(|s| !r.contributions.contains_key(s))
                .map(name)
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
                rho: plan.privacy.map(|_| rho_units(r.cost)),
            })
            .collect();
        let privacy = plan.privacy.map(|p| PrivacyReport {
            rho_spent: rho_units(self.spent),
            rho_budget: rho_units(p.budget),
            epsilon_spent: privacy::epsilon(rho_units(self.spent), p.delta),
            epsilon_budget: privacy::epsilon(rho_units(p.budget), p.delta),
            epsilon_stated: p.epsilon,
            delta: p.delta,
            colluding: p.colluding,
            rho_unmasked: p
                .round_cost(plan.layout.coords(), p.colluding + 1)
                .map_or(f64::INFINITY, rho_units),
        });
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
                    dataset: j.dataset.clone(),
                    credentials: j
                        .credentials
                        .iter()
                        .map(|c| Held {
                            kind: c.types.first().cloned().unwrap_or_default(),
                            issuer: c.issuer.clone(),
                            digest: c.digest,
                            valid_until: c.valid_until,
                        })
                        .collect(),
                })
                .collect(),
            rounds,
            privacy,
            revoked: self
                .revoked
                .iter()
                .map(|(site, reason)| Revoked {
                    site: name(site),
                    reason: reason.clone(),
                })
                .collect(),
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
