//! Matching what a run will do against what a dataset allows, in the terms of
//! the GA4GH Data Use Ontology.
//!
//! DUO has no normative matching algorithm. The rules here follow the term
//! definitions and the published DUOS logic: a purpose fits a dataset tagged
//! with the same term or a broader one, never a narrower one. Anything this
//! code cannot decide is refused, since a person should look at it.

use serde::{Deserialize, Serialize};

use crate::credential::parse_time;

pub const NO_RESTRICTION: &str = "DUO:0000004";
pub const GENERAL_RESEARCH: &str = "DUO:0000042";
pub const HEALTH: &str = "DUO:0000006";
pub const DISEASE_SPECIFIC: &str = "DUO:0000007";
pub const ANCESTRY_ONLY: &str = "DUO:0000011";

// The ontology's own identifier for this term has eight digits.
pub const NO_ANCESTRY: &str = "DUO:00000044";
pub const RESEARCH_SPECIFIC: &str = "DUO:0000012";
pub const NO_METHODS: &str = "DUO:0000015";
pub const GENETIC_ONLY: &str = "DUO:0000016";
pub const NOT_FOR_PROFIT_NON_COMMERCIAL: &str = "DUO:0000018";
pub const NOT_FOR_PROFIT_ORGANISATION: &str = "DUO:0000045";
pub const NON_COMMERCIAL: &str = "DUO:0000046";
pub const PUBLICATION_REQUIRED: &str = "DUO:0000019";
pub const COLLABORATION_REQUIRED: &str = "DUO:0000020";
pub const ETHICS_APPROVAL_REQUIRED: &str = "DUO:0000021";
pub const GEOGRAPHIC: &str = "DUO:0000022";
pub const PUBLICATION_MORATORIUM: &str = "DUO:0000024";
pub const TIME_LIMIT: &str = "DUO:0000025";
pub const USER_SPECIFIC: &str = "DUO:0000026";
pub const PROJECT_SPECIFIC: &str = "DUO:0000027";
pub const INSTITUTION_SPECIFIC: &str = "DUO:0000028";
pub const RETURN_TO_DATABASE: &str = "DUO:0000029";
pub const CLINICAL_CARE: &str = "DUO:0000043";

/// What a dataset allows: one permission, narrowed by modifiers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataUse {
    pub permission: String,
    /// The disease a disease-specific permission names, as an ontology term.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disease: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modifiers: Vec<Modifier>,
}

/// A DUO modifier. Some carry a value: a region, an institution, a project, a
/// user, or a date.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Modifier {
    pub term: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Research {
    #[default]
    General,
    Health,
    Disease,
    Ancestry,
    Methods,
}

/// What a run will do with the data. The sponsor signs this as part of the
/// plan, so each statement in it is the sponsor's own commitment.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Purpose {
    pub research: Research,
    /// For disease research: the disease studied, then every broader term.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub disease: Vec<String>,
    pub commercial: bool,
    pub not_for_profit: bool,
    pub genetic: bool,
    pub ethics_approval: bool,
    pub will_publish: bool,
    pub collaborates: bool,
    pub returns_results: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub institution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

fn permission_allows(data_use: &DataUse, purpose: &Purpose) -> Result<(), String> {
    use Research::*;
    let research = purpose.research;
    let allowed = match data_use.permission.as_str() {
        NO_RESTRICTION | GENERAL_RESEARCH => true,
        HEALTH => matches!(research, Health | Disease | Methods),
        DISEASE_SPECIFIC => {
            let Some(disease) = &data_use.disease else {
                return Err("a disease-specific permission names no disease".into());
            };
            match research {
                Disease => purpose.disease.contains(disease),
                Methods => true,
                _ => false,
            }
        }
        ANCESTRY_ONLY => matches!(research, Ancestry | Methods),
        other => return Err(format!("{other} is not a permission this code knows")),
    };
    if allowed {
        Ok(())
    } else {
        Err(format!(
            "{} does not allow {research:?} research",
            data_use.permission
        ))
    }
}

fn modifier_allows(modifier: &Modifier, purpose: &Purpose, now: i64) -> Result<(), String> {
    let term = modifier.term.as_str();
    let need = |met: bool, what: &str| {
        if met {
            Ok(())
        } else {
            Err(format!("{term} requires {what}"))
        }
    };
    let named = |stated: &Option<String>, what: &str| {
        let wanted = modifier
            .value
            .as_ref()
            .ok_or_else(|| format!("{term} names no {what}"))?;
        need(stated.as_ref() == Some(wanted), &format!("{what} {wanted}"))
    };
    match term {
        NO_ANCESTRY => need(
            purpose.research != Research::Ancestry,
            "no ancestry research",
        ),
        NO_METHODS => need(
            purpose.research != Research::Methods,
            "more than methods research",
        ),
        GENETIC_ONLY => need(purpose.genetic, "a genetic study"),
        NOT_FOR_PROFIT_NON_COMMERCIAL => need(
            purpose.not_for_profit && !purpose.commercial,
            "a not-for-profit organisation and non-commercial use",
        ),
        NOT_FOR_PROFIT_ORGANISATION => {
            need(purpose.not_for_profit, "a not-for-profit organisation")
        }
        NON_COMMERCIAL => need(!purpose.commercial, "non-commercial use"),
        PUBLICATION_REQUIRED => need(purpose.will_publish, "publication of results"),
        COLLABORATION_REQUIRED => need(purpose.collaborates, "collaboration with the study"),
        ETHICS_APPROVAL_REQUIRED => need(purpose.ethics_approval, "ethics approval"),
        RETURN_TO_DATABASE => need(purpose.returns_results, "results returned to the database"),
        GEOGRAPHIC => named(&purpose.region, "region"),
        INSTITUTION_SPECIFIC => named(&purpose.institution, "institution"),
        PROJECT_SPECIFIC => named(&purpose.project, "project"),
        USER_SPECIFIC => named(&purpose.user, "user"),
        TIME_LIMIT => {
            let until = modifier
                .value
                .as_deref()
                .and_then(parse_time)
                .ok_or_else(|| format!("{term} names no end date"))?;
            need(now < until, "use before its end date")
        }
        // These bind what happens after the research, not whether it may run.
        PUBLICATION_MORATORIUM | CLINICAL_CARE => Ok(()),
        RESEARCH_SPECIFIC => Err(format!("{term} needs a person to read the restriction")),
        other => Err(format!("{other} is not a modifier this code knows")),
    }
}

/// Every reason `purpose` falls outside `data_use`, or none.
pub fn permits(data_use: &DataUse, purpose: &Purpose, now: i64) -> Result<(), Vec<String>> {
    let reasons: Vec<String> = std::iter::once(permission_allows(data_use, purpose))
        .chain(
            data_use
                .modifiers
                .iter()
                .map(|m| modifier_allows(m, purpose, now)),
        )
        .filter_map(Result::err)
        .collect();
    if reasons.is_empty() {
        Ok(())
    } else {
        Err(reasons)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Research::*;

    fn data(permission: &str) -> DataUse {
        DataUse {
            permission: permission.into(),
            disease: None,
            modifiers: Vec::new(),
        }
    }

    fn with(permission: &str, term: &str, value: Option<&str>) -> DataUse {
        DataUse {
            modifiers: vec![Modifier {
                term: term.into(),
                value: value.map(str::to_owned),
            }],
            ..data(permission)
        }
    }

    fn doing(research: Research) -> Purpose {
        Purpose {
            research,
            ..Purpose::default()
        }
    }

    fn ok(data_use: &DataUse, purpose: &Purpose) -> bool {
        permits(data_use, purpose, 1_700_000_000).is_ok()
    }

    #[test]
    fn a_purpose_fits_the_same_permission_or_a_broader_one() {
        let all = [General, Health, Disease, Ancestry, Methods];
        let table: [(&str, &[Research]); 4] = [
            (NO_RESTRICTION, &all),
            (GENERAL_RESEARCH, &all),
            (HEALTH, &[Health, Disease, Methods]),
            (ANCESTRY_ONLY, &[Ancestry, Methods]),
        ];
        for (permission, allowed) in table {
            for research in all {
                let mut purpose = doing(research);
                purpose.disease = vec!["MONDO:0005015".into()];
                assert_eq!(
                    ok(&data(permission), &purpose),
                    allowed.contains(&research),
                    "{permission} for {research:?}"
                );
            }
        }
    }

    #[test]
    fn disease_data_fits_that_disease_or_a_narrower_study() {
        let diabetes = DataUse {
            disease: Some("MONDO:0005015".into()),
            ..data(DISEASE_SPECIFIC)
        };
        let study = |terms: &[&str]| Purpose {
            disease: terms.iter().map(|t| (*t).to_owned()).collect(),
            ..doing(Disease)
        };
        assert!(ok(&diabetes, &study(&["MONDO:0005015"])));
        // Type 2 diabetes is narrower, and lists diabetes above it.
        assert!(ok(&diabetes, &study(&["MONDO:0005148", "MONDO:0005015"])));
        // Cancer research is another branch.
        assert!(!ok(&diabetes, &study(&["MONDO:0004992"])));
        assert!(!ok(&diabetes, &study(&[])));
        assert!(
            !ok(&diabetes, &doing(Health)),
            "broader than the data allows"
        );
        assert!(!ok(&diabetes, &doing(General)));
        assert!(ok(&diabetes, &doing(Methods)));
        assert!(!ok(&data(DISEASE_SPECIFIC), &study(&["MONDO:0005015"])));
    }

    #[test]
    fn each_modifier_asks_for_one_commitment() {
        type Edit = fn(&mut Purpose);
        let cases: [(&str, Edit); 7] = [
            (GENETIC_ONLY, |p| p.genetic = true),
            (NOT_FOR_PROFIT_ORGANISATION, |p| p.not_for_profit = true),
            (PUBLICATION_REQUIRED, |p| p.will_publish = true),
            (COLLABORATION_REQUIRED, |p| p.collaborates = true),
            (ETHICS_APPROVAL_REQUIRED, |p| p.ethics_approval = true),
            (RETURN_TO_DATABASE, |p| p.returns_results = true),
            (NOT_FOR_PROFIT_NON_COMMERCIAL, |p| p.not_for_profit = true),
        ];
        for (term, commit) in cases {
            let data_use = with(GENERAL_RESEARCH, term, None);
            let mut purpose = doing(Health);
            assert!(!ok(&data_use, &purpose), "{term} without the commitment");
            commit(&mut purpose);
            assert!(ok(&data_use, &purpose), "{term} with the commitment");
        }
    }

    #[test]
    fn commercial_use_is_kept_out_where_the_data_says_so() {
        let mut company = doing(Health);
        company.commercial = true;
        assert!(!ok(&with(GENERAL_RESEARCH, NON_COMMERCIAL, None), &company));
        assert!(ok(
            &with(GENERAL_RESEARCH, NON_COMMERCIAL, None),
            &doing(Health)
        ));
        company.not_for_profit = true;
        assert!(!ok(
            &with(GENERAL_RESEARCH, NOT_FOR_PROFIT_NON_COMMERCIAL, None),
            &company
        ));
        assert!(ok(&data(GENERAL_RESEARCH), &company));
    }

    #[test]
    fn research_kinds_can_be_ruled_out() {
        assert!(!ok(
            &with(GENERAL_RESEARCH, NO_ANCESTRY, None),
            &doing(Ancestry)
        ));
        assert!(ok(
            &with(GENERAL_RESEARCH, NO_ANCESTRY, None),
            &doing(Health)
        ));
        assert!(!ok(
            &with(GENERAL_RESEARCH, NO_METHODS, None),
            &doing(Methods)
        ));
        assert!(ok(
            &with(GENERAL_RESEARCH, NO_METHODS, None),
            &doing(Health)
        ));
    }

    #[test]
    fn named_modifiers_need_the_same_name() {
        type Set = fn(&mut Purpose, String);
        let cases: [(&str, Set); 4] = [
            (GEOGRAPHIC, |p, v| p.region = Some(v)),
            (INSTITUTION_SPECIFIC, |p, v| p.institution = Some(v)),
            (PROJECT_SPECIFIC, |p, v| p.project = Some(v)),
            (USER_SPECIFIC, |p, v| p.user = Some(v)),
        ];
        for (term, set) in cases {
            let data_use = with(GENERAL_RESEARCH, term, Some("this one"));
            let mut purpose = doing(Health);
            assert!(!ok(&data_use, &purpose), "{term} unstated");
            set(&mut purpose, "another".into());
            assert!(!ok(&data_use, &purpose), "{term} mismatched");
            set(&mut purpose, "this one".into());
            assert!(ok(&data_use, &purpose), "{term} matched");
            assert!(
                !ok(&with(GENERAL_RESEARCH, term, None), &purpose),
                "{term} unnamed"
            );
        }
    }

    #[test]
    fn a_time_limit_ends_use_at_its_date() {
        let limited = with(GENERAL_RESEARCH, TIME_LIMIT, Some("2030-01-01T00:00:00Z"));
        let end = parse_time("2030-01-01T00:00:00Z").unwrap();
        assert!(permits(&limited, &doing(Health), end - 1).is_ok());
        assert!(permits(&limited, &doing(Health), end).is_err());
        assert!(!ok(
            &with(GENERAL_RESEARCH, TIME_LIMIT, Some("soon")),
            &doing(Health)
        ));
        assert!(!ok(
            &with(GENERAL_RESEARCH, TIME_LIMIT, None),
            &doing(Health)
        ));
    }

    #[test]
    fn obligations_after_the_research_do_not_block_it() {
        for term in [PUBLICATION_MORATORIUM, CLINICAL_CARE] {
            assert!(
                ok(&with(GENERAL_RESEARCH, term, None), &doing(Health)),
                "{term}"
            );
        }
    }

    #[test]
    fn what_cannot_be_decided_is_refused_with_every_reason() {
        assert!(!ok(
            &with(GENERAL_RESEARCH, RESEARCH_SPECIFIC, None),
            &doing(Health)
        ));
        assert!(!ok(
            &with(GENERAL_RESEARCH, "DUO:9999999", None),
            &doing(Health)
        ));
        assert!(!ok(&data("DUO:9999999"), &doing(Health)));
        let many = DataUse {
            modifiers: vec![
                Modifier {
                    term: ETHICS_APPROVAL_REQUIRED.into(),
                    value: None,
                },
                Modifier {
                    term: GENETIC_ONLY.into(),
                    value: None,
                },
            ],
            ..data(ANCESTRY_ONLY)
        };
        let reasons = permits(&many, &doing(Health), 0).unwrap_err();
        assert_eq!(reasons.len(), 3, "{reasons:?}");
    }

    #[test]
    fn terms_read_from_the_json_a_credential_carries() {
        let text = r#"{"permission":"DUO:0000007","disease":"MONDO:0005015","modifiers":[{"term":"DUO:0000021"},{"term":"DUO:0000022","value":"EU"}]}"#;
        let data_use: DataUse = serde_json::from_str(text).unwrap();
        assert_eq!(data_use.modifiers[1].value.as_deref(), Some("EU"));
        assert_eq!(serde_json::to_string(&data_use).unwrap(), text);
        let purpose: Purpose =
            serde_json::from_str(r#"{"research":"disease","disease":["MONDO:0005015"],"ethics_approval":true,"region":"EU"}"#)
                .unwrap();
        assert!(ok(&data_use, &purpose));
        assert!(serde_json::from_str::<Purpose>(r#"{"reserch":"health"}"#).is_err());
    }
}
