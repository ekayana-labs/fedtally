//! Who a run admits: a rule over the credentials each party holds.
//!
//! A rule only ever asks for credentials, never for their absence, so adding
//! a credential can't turn an admitted party away and revoking one can't let
//! a refused party in.

use std::collections::BTreeMap;

use pq_ucan::did::Did;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::credential::Credential;
use crate::duo::{self, DataUse, Purpose};

const MAX_DEPTH: usize = 8;
const MAX_ATOMS: usize = 64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    /// What the run will do with the data. Checked against each dataset's terms.
    pub purpose: Purpose,
    /// What every site must hold.
    pub sites: Rule,
    /// What the aggregator and every helper must hold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operators: Option<Rule>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    All(Vec<Rule>),
    Any(Vec<Rule>),
    Holds(Atom),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum About {
    /// The credential is issued to the party itself.
    #[default]
    Holder,
    /// The credential is issued to the dataset the site trains on.
    Dataset,
}

/// One credential a party must hold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Atom {
    /// The credential type, such as `ControlledAccessGrants`.
    pub kind: String,
    /// The issuers the run accepts for it.
    pub issuers: Vec<String>,
    #[serde(default)]
    pub about: About,
    /// Claims that must be there with exactly these values, by dotted path.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub claims: BTreeMap<String, Value>,
    /// The claim that must name the site's dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub names_dataset: Option<String>,
    /// The claim holding data use terms, which must allow the run's purpose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_use: Option<String>,
}

/// The party a rule is asked about.
#[derive(Clone, Copy, Debug)]
pub struct Holder<'a> {
    pub did: &'a str,
    pub dataset: Option<&'a str>,
}

impl Rule {
    fn walk(&self, depth: usize, atoms: &mut usize) -> Result<(), String> {
        if depth > MAX_DEPTH {
            return Err("the rule is nested too deeply".into());
        }
        match self {
            Rule::All(rules) => rules.iter().try_for_each(|r| r.walk(depth + 1, atoms)),
            Rule::Any(rules) if rules.is_empty() => Err("an empty choice admits nobody".into()),
            Rule::Any(rules) => rules.iter().try_for_each(|r| r.walk(depth + 1, atoms)),
            Rule::Holds(atom) => {
                *atoms += 1;
                if *atoms > MAX_ATOMS {
                    return Err("the rule asks for too many credentials".into());
                }
                if atom.kind.is_empty() || atom.issuers.is_empty() {
                    return Err("a credential needs a type and at least one issuer".into());
                }
                match atom.issuers.iter().find(|i| Did::parse(i).is_err()) {
                    Some(issuer) => Err(format!("{issuer} is not a DID")),
                    None => Ok(()),
                }
            }
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.walk(0, &mut 0)
    }
}

impl Rule {
    fn names(&self, issuer: &str) -> bool {
        match self {
            Rule::All(rules) | Rule::Any(rules) => rules.iter().any(|r| r.names(issuer)),
            Rule::Holds(atom) => atom.issuers.iter().any(|i| i == issuer),
        }
    }
}

impl Policy {
    pub fn validate(&self) -> Result<(), String> {
        self.sites.validate()?;
        self.operators.as_ref().map_or(Ok(()), Rule::validate)
    }

    /// Every issuer a rule accepts credentials from.
    pub fn issuers(&self) -> Vec<&str> {
        fn collect<'a>(rule: &'a Rule, out: &mut Vec<&'a str>) {
            match rule {
                Rule::All(rules) | Rule::Any(rules) => rules.iter().for_each(|r| collect(r, out)),
                Rule::Holds(atom) => out.extend(atom.issuers.iter().map(String::as_str)),
            }
        }
        let mut out = Vec::new();
        collect(&self.sites, &mut out);
        if let Some(rule) = &self.operators {
            collect(rule, &mut out);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// True when some rule accepts credentials from `issuer`.
    pub fn relies_on(&self, issuer: &str) -> bool {
        self.sites.names(issuer) || self.operators.as_ref().is_some_and(|r| r.names(issuer))
    }
}

/// Checks one party against a rule, at one time.
pub struct Check<'a> {
    pub purpose: &'a Purpose,
    pub holder: Holder<'a>,
    pub credentials: &'a [Credential],
    pub now: i64,
    pub revoked: &'a dyn Fn(&Credential) -> bool,
}

impl Check<'_> {
    fn fits(&self, atom: &Atom, credential: &Credential) -> Result<(), String> {
        let about = match atom.about {
            About::Holder => Some(self.holder.did),
            About::Dataset => self.holder.dataset,
        };
        if about != Some(credential.subject.as_str()) {
            return Err("it is issued to someone else".into());
        }
        if !atom.issuers.contains(&credential.issuer) {
            return Err(format!("{} is not an accepted issuer", credential.issuer));
        }
        if !credential.valid_at(self.now) {
            return Err("it is not valid at this time".into());
        }
        if (self.revoked)(credential) {
            return Err("it has been revoked".into());
        }
        for (path, wanted) in &atom.claims {
            if credential.claim(path) != Some(wanted) {
                return Err(format!("its claim {path} is not {wanted}"));
            }
        }
        if let Some(path) = &atom.names_dataset {
            let named = credential.claim(path).and_then(Value::as_str);
            if named.is_none() || named != self.holder.dataset {
                return Err("it is for another dataset".into());
            }
        }
        if let Some(path) = &atom.data_use {
            let terms: DataUse = credential
                .claim(path)
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok())
                .ok_or("it carries no readable data use terms")?;
            duo::permits(&terms, self.purpose, self.now).map_err(|why| why.join("; "))?;
        }
        Ok(())
    }

    fn holds(&self, atom: &Atom) -> Result<(), String> {
        let mut closest = format!("no {} credential", atom.kind);
        for credential in self
            .credentials
            .iter()
            .filter(|c| c.types.contains(&atom.kind))
        {
            match self.fits(atom, credential) {
                Ok(()) => return Ok(()),
                Err(why) => closest = format!("{}: {why}", atom.kind),
            }
        }
        Err(closest)
    }

    /// Why the rule is not met, or nothing.
    pub fn meets(&self, rule: &Rule) -> Result<(), String> {
        match rule {
            Rule::All(rules) => rules.iter().try_for_each(|r| self.meets(r)),
            Rule::Any(rules) => {
                let mut reasons = Vec::new();
                for rule in rules {
                    match self.meets(rule) {
                        Ok(()) => return Ok(()),
                        Err(why) => reasons.push(why),
                    }
                }
                Err(format!("none of: {}", reasons.join(" | ")))
            }
            Rule::Holds(atom) => self.holds(atom),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::Draft;
    use crate::duo::{DISEASE_SPECIFIC, ETHICS_APPROVAL_REQUIRED, Research};
    use crate::keys::{Identity, SigAlg};
    use pq_ucan::crypto::Signer;
    use serde_json::{Map, json};

    const NOW: i64 = 1_750_000_000;
    const SITE: &str = "did:example:site";
    const DATASET: &str = "did:example:dataset";

    fn identity() -> Identity {
        Identity::generate(SigAlg::Ed25519).unwrap()
    }

    fn credential(issuer: &Identity, kind: &str, subject: &str, claims: Value) -> Credential {
        let draft = Draft {
            kind: kind.into(),
            subject: subject.into(),
            claims: claims.as_object().cloned().unwrap_or_default(),
            valid_from: NOW - 1000,
            valid_until: Some(NOW + 1000),
            status: None,
        };
        Credential::issue(issuer, &draft, NOW - 1000).unwrap()
    }

    fn atom(kind: &str, issuer: &Identity) -> Atom {
        Atom {
            kind: kind.into(),
            issuers: vec![issuer.did_string()],
            about: About::Holder,
            claims: BTreeMap::new(),
            names_dataset: None,
            data_use: None,
        }
    }

    fn purpose() -> Purpose {
        Purpose {
            research: Research::Disease,
            disease: vec!["MONDO:0005015".into()],
            ethics_approval: true,
            ..Purpose::default()
        }
    }

    fn meets(rule: &Rule, credentials: &[Credential]) -> Result<(), String> {
        meets_at(rule, credentials, NOW, &|_| false)
    }

    fn meets_at(
        rule: &Rule,
        credentials: &[Credential],
        now: i64,
        revoked: &dyn Fn(&Credential) -> bool,
    ) -> Result<(), String> {
        Check {
            purpose: &purpose(),
            holder: Holder {
                did: SITE,
                dataset: Some(DATASET),
            },
            credentials,
            now,
            revoked,
        }
        .meets(rule)
    }

    /// The paper's example: affiliation and ethics approval for the site, an
    /// access grant for the site and dataset, provenance for the dataset.
    struct Genomics {
        university: Identity,
        board: Identity,
        dac: Identity,
        rule: Rule,
    }

    fn genomics() -> Genomics {
        let (university, board, dac) = (identity(), identity(), identity());
        let mut grant = atom("ControlledAccessGrants", &dac);
        grant.claims.insert("ga4gh_visa_v1.by".into(), json!("dac"));
        grant.names_dataset = Some("ga4gh_visa_v1.value".into());
        grant.data_use = Some("data_use".into());
        let mut provenance = atom("Provenance", &dac);
        provenance.about = About::Dataset;
        let rule = Rule::All(vec![
            Rule::Holds(atom("AffiliationAndRole", &university)),
            Rule::Holds(atom("EthicsApproval", &board)),
            Rule::Holds(grant),
            Rule::Holds(provenance),
        ]);
        Genomics {
            university,
            board,
            dac,
            rule,
        }
    }

    fn grant_claims(dataset: &str) -> Value {
        json!({
            "ga4gh_visa_v1": {"type": "ControlledAccessGrants", "by": "dac", "value": dataset,
                               "source": "https://dac.example", "asserted": NOW - 1000},
            "data_use": {"permission": DISEASE_SPECIFIC, "disease": "MONDO:0005015",
                         "modifiers": [{"term": ETHICS_APPROVAL_REQUIRED}]},
        })
    }

    fn wallet(g: &Genomics) -> Vec<Credential> {
        vec![
            credential(&g.university, "AffiliationAndRole", SITE, json!({})),
            credential(&g.board, "EthicsApproval", SITE, json!({})),
            credential(
                &g.dac,
                "ControlledAccessGrants",
                SITE,
                grant_claims(DATASET),
            ),
            credential(&g.dac, "Provenance", DATASET, json!({})),
        ]
    }

    #[test]
    fn the_genomic_policy_admits_a_site_that_holds_all_four() {
        let g = genomics();
        g.rule.validate().unwrap();
        meets(&g.rule, &wallet(&g)).unwrap();
    }

    #[test]
    fn each_missing_credential_is_named() {
        let g = genomics();
        let names = [
            "AffiliationAndRole",
            "EthicsApproval",
            "ControlledAccessGrants",
            "Provenance",
        ];
        for (missing, name) in names.iter().enumerate() {
            let mut held = wallet(&g);
            held.remove(missing);
            let why = meets(&g.rule, &held).unwrap_err();
            assert!(why.contains(name), "{why}");
        }
    }

    #[test]
    fn a_credential_from_an_issuer_the_run_did_not_name_does_not_count() {
        let g = genomics();
        let mut held = wallet(&g);
        held[0] = credential(&identity(), "AffiliationAndRole", SITE, json!({}));
        let why = meets(&g.rule, &held).unwrap_err();
        assert!(why.contains("not an accepted issuer"), "{why}");
    }

    #[test]
    fn a_credential_for_someone_else_does_not_count() {
        let g = genomics();
        let mut held = wallet(&g);
        held[1] = credential(
            &g.board,
            "EthicsApproval",
            "did:example:another-site",
            json!({}),
        );
        assert!(meets(&g.rule, &held).unwrap_err().contains("someone else"));
        let mut held = wallet(&g);
        held[3] = credential(&g.dac, "Provenance", SITE, json!({}));
        assert!(meets(&g.rule, &held).unwrap_err().contains("someone else"));
    }

    #[test]
    fn a_grant_must_be_for_the_dataset_the_site_trains_on() {
        let g = genomics();
        let mut held = wallet(&g);
        held[2] = credential(
            &g.dac,
            "ControlledAccessGrants",
            SITE,
            grant_claims("did:example:other-data"),
        );
        assert!(
            meets(&g.rule, &held)
                .unwrap_err()
                .contains("another dataset")
        );
    }

    #[test]
    fn a_grant_must_carry_the_claims_the_rule_asks_for() {
        let g = genomics();
        let mut claims = grant_claims(DATASET);
        claims["ga4gh_visa_v1"]["by"] = json!("self");
        let mut held = wallet(&g);
        held[2] = credential(&g.dac, "ControlledAccessGrants", SITE, claims);
        assert!(
            meets(&g.rule, &held)
                .unwrap_err()
                .contains("ga4gh_visa_v1.by")
        );
    }

    #[test]
    fn the_dataset_terms_must_allow_what_the_run_will_do() {
        let g = genomics();
        let mut claims = grant_claims(DATASET);
        claims["data_use"]["disease"] = json!("MONDO:0004992");
        let mut held = wallet(&g);
        held[2] = credential(&g.dac, "ControlledAccessGrants", SITE, claims);
        let why = meets(&g.rule, &held).unwrap_err();
        assert!(why.contains("does not allow"), "{why}");

        let mut claims = grant_claims(DATASET);
        claims.as_object_mut().unwrap().remove("data_use");
        held[2] = credential(&g.dac, "ControlledAccessGrants", SITE, claims);
        assert!(
            meets(&g.rule, &held)
                .unwrap_err()
                .contains("no readable data use")
        );
    }

    #[test]
    fn expiry_and_revocation_both_end_admission() {
        let g = genomics();
        let held = wallet(&g);
        assert!(
            meets_at(&g.rule, &held, NOW + 1000, &|_| false)
                .unwrap_err()
                .contains("not valid at this time")
        );
        assert!(meets_at(&g.rule, &held, NOW - 1001, &|_| false).is_err());
        let ethics = held[1].digest;
        let why = meets_at(&g.rule, &held, NOW, &|c| c.digest == ethics).unwrap_err();
        assert!(
            why.contains("EthicsApproval") && why.contains("revoked"),
            "{why}"
        );
    }

    #[test]
    fn a_choice_is_met_by_any_branch_and_reports_them_all() {
        let (a, b) = (identity(), identity());
        let rule = Rule::Any(vec![
            Rule::Holds(atom("ResearcherStatus", &a)),
            Rule::Holds(atom("AffiliationAndRole", &b)),
        ]);
        let by_a = credential(&a, "ResearcherStatus", SITE, json!({}));
        let by_b = credential(&b, "AffiliationAndRole", SITE, json!({}));
        meets(&rule, std::slice::from_ref(&by_a)).unwrap();
        meets(&rule, std::slice::from_ref(&by_b)).unwrap();
        let why = meets(&rule, &[]).unwrap_err();
        assert!(
            why.contains("ResearcherStatus") && why.contains("AffiliationAndRole"),
            "{why}"
        );
        // Revoking one branch leaves the other standing.
        let digest = by_a.digest;
        meets_at(&rule, &[by_a, by_b], NOW, &|c| c.digest == digest).unwrap();
        meets(&Rule::All(Vec::new()), &[]).unwrap();
    }

    #[test]
    fn a_second_credential_of_the_same_type_can_meet_the_rule() {
        let g = genomics();
        let mut held = wallet(&g);
        held.insert(
            2,
            credential(
                &g.dac,
                "ControlledAccessGrants",
                SITE,
                grant_claims("did:example:other-data"),
            ),
        );
        meets(&g.rule, &held).unwrap();
    }

    #[test]
    fn a_dataset_rule_needs_a_dataset() {
        let g = genomics();
        let check = Check {
            purpose: &purpose(),
            holder: Holder {
                did: SITE,
                dataset: None,
            },
            credentials: &wallet(&g),
            now: NOW,
            revoked: &|_| false,
        };
        assert!(check.meets(&g.rule).is_err());
    }

    #[test]
    fn rules_that_cannot_work_are_refused_up_front() {
        let dac = identity();
        assert!(Rule::Any(Vec::new()).validate().is_err());
        let mut nameless = atom("X", &dac);
        nameless.kind.clear();
        assert!(Rule::Holds(nameless).validate().is_err());
        let mut unissued = atom("X", &dac);
        unissued.issuers.clear();
        assert!(Rule::Holds(unissued).validate().is_err());
        let mut odd = atom("X", &dac);
        odd.issuers.push("not a did".into());
        assert!(Rule::Holds(odd).validate().is_err());

        let mut deep = Rule::Holds(atom("X", &dac));
        for _ in 0..MAX_DEPTH {
            deep = Rule::All(vec![deep]);
        }
        deep.validate().unwrap();
        assert!(Rule::All(vec![deep]).validate().is_err());

        let many = |n: usize| Rule::All((0..n).map(|_| Rule::Holds(atom("X", &dac))).collect());
        many(MAX_ATOMS).validate().unwrap();
        assert!(many(MAX_ATOMS + 1).validate().is_err());

        let mut policy = Policy {
            purpose: purpose(),
            sites: many(1),
            operators: Some(Rule::Any(Vec::new())),
        };
        assert!(policy.validate().is_err());
        let other = identity();
        policy.operators = Some(Rule::Any(vec![Rule::Holds(atom("Y", &other))]));
        policy.validate().unwrap();
        assert!(policy.relies_on(&dac.did_string()) && policy.relies_on(&other.did_string()));
        let mut both = [dac.did_string(), other.did_string()];
        both.sort();
        assert_eq!(policy.issuers(), both);
        assert!(!policy.relies_on(&identity().did_string()));
    }

    #[test]
    fn a_policy_reads_from_json_and_writes_the_same_back() {
        let dac = identity();
        let text = format!(
            r#"{{"purpose":{{"research":"health","commercial":false,"not_for_profit":true,"genetic":false,"ethics_approval":true,"will_publish":false,"collaborates":false,"returns_results":false}},"sites":{{"all":[{{"holds":{{"kind":"ControlledAccessGrants","issuers":["{0}"],"about":"holder","names_dataset":"ga4gh_visa_v1.value"}}}},{{"any":[{{"holds":{{"kind":"Provenance","issuers":["{0}"],"about":"dataset"}}}}]}}]}}}}"#,
            dac.did().as_str()
        );
        let policy: Policy = serde_json::from_str(&text).unwrap();
        policy.validate().unwrap();
        assert_eq!(serde_json::to_string(&policy).unwrap(), text);
        let _ = Map::<String, Value>::new();
    }
}
