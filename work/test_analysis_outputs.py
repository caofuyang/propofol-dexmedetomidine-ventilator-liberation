#!/usr/bin/env python3
"""Integrity checks for generated analysis artifacts."""

from pathlib import Path

import pandas as pd
import duckdb


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "outputs" / "primary_analysis"
EXTERNAL = ROOT / "outputs" / "external_validation"
EVENT_REVIEW = ROOT / "work" / "outcome_event_review_sample_blinded.csv"
EVENT_KEY = ROOT / "work" / "outcome_event_review_sample_key.csv"
TRIAL_CACHE = ROOT / "work" / "cache" / "mimic_landmark_trials_v11_strategy_outcome_qc.parquet"
MANUSCRIPT = ROOT / "outputs" / "manuscript_draft_v2_corrected.md"
ABSTRACT = ROOT / "outputs" / "abstract_draft_v2_corrected.md"
TABLES = ROOT / "outputs" / "manuscript_tables_and_legends_v2_corrected.md"


def main() -> None:
    required = [
        PRIMARY / "table0_flow_counts.csv",
        PRIMARY / "table1_missingness.csv",
        PRIMARY / "table2_covariate_balance.csv",
        PRIMARY / "table3_sensitivity_effects.csv",
        PRIMARY / "table4_landmark_effects.csv",
        PRIMARY / "table5_secondary_outcomes.csv",
        PRIMARY / "table6_outcome_event_audit.csv",
        PRIMARY / "table7_descriptive_baseline.csv",
        PRIMARY / "table8_model_specification_sensitivity.csv",
        PRIMARY / "table9_model_specification_balance.csv",
        PRIMARY / "figure0_cohort_flow.png",
        PRIMARY / "figure1_love_plot.png",
        PRIMARY / "figure2_propensity_overlap.png",
        PRIMARY / "figure3_effect_forest.png",
        PRIMARY / "figure4_landmark_effects.png",
        EXTERNAL / "table2_external_effects.csv",
        EXTERNAL / "table4_common_support_balance.csv",
        EXTERNAL / "figure1_external_effects.png",
    ]
    for path in required:
        assert path.exists() and path.stat().st_size > 100, f"Missing or empty: {path}"

    effects = pd.read_csv(PRIMARY / "table3_sensitivity_effects.csv")
    primary = effects.iloc[0]
    assert primary.trials == 54980
    assert primary.treated_trials == 994
    assert primary.unique_subjects < primary.unique_stays < primary.trials
    assert primary.max_abs_smd_weighted < 0.10
    assert primary.rd_ci_low > 0
    assert 0 < primary.weighted_risk_control < 1
    assert 0 < primary.weighted_risk_treated < 1

    enhanced = effects.loc[effects.analysis.str.startswith("Enhanced readiness")].iloc[0]
    assert enhanced.trials == 14157
    assert enhanced.treated_trials == 328
    assert enhanced.rd_ci_low > 0
    assert enhanced.risk_difference < primary.risk_difference
    no_cvicu = effects.loc[effects.analysis.eq("Excluding cardiovascular ICU")].iloc[0]
    assert no_cvicu.treated_trials == 728
    assert no_cvicu.rd_ci_low > 0
    assert no_cvicu.risk_difference < primary.risk_difference

    flow = pd.read_csv(PRIMARY / "table0_flow_counts.csv")
    assert flow.iloc[0].trial_instances == 72504
    assert flow.loc[flow.stage.eq("Invasive ventilation covers window end"), "trial_instances"].iloc[0] == 61355
    assert flow.loc[flow.stage.eq("Strategy drug covers window end"), "trial_instances"].iloc[0] == 60431
    assert flow.loc[flow.stage.eq("Final v11 analysis cohort"), "trial_instances"].iloc[0] == 54980
    assert flow.loc[flow.stage.eq("Transition strategy"), "trial_instances"].iloc[0] == 994

    balance = pd.read_csv(PRIMARY / "table2_covariate_balance.csv")
    assert balance.smd_weighted.abs().max() < 0.10
    assert not balance.variable.str.startswith("dx_").any()

    descriptive = pd.read_csv(PRIMARY / "table7_descriptive_baseline.csv")
    binary_rows = descriptive.level.eq("Yes")
    assert descriptive.loc[binary_rows, ["smd_unweighted", "smd_weighted"]].notna().all().all()
    assert descriptive.loc[descriptive.variable.eq("sbt_successful_24h"), "smd_unweighted"].abs().iloc[0] > 0.15

    flexible = pd.read_csv(PRIMARY / "table8_model_specification_sensitivity.csv").iloc[0]
    assert flexible.trials == primary.trials
    assert flexible.treated_trials == primary.treated_trials
    assert flexible.max_abs_smd_weighted < 0.10
    assert flexible.rd_ci_low > 0
    assert abs(flexible.risk_difference - primary.risk_difference) < 0.02
    flexible_balance = pd.read_csv(PRIMARY / "table9_model_specification_balance.csv")
    assert flexible_balance.smd_weighted.abs().max() < 0.10

    cohort = pd.read_csv(PRIMARY / "analysis_cohort_weights.csv")
    assert {"subject_id", "hadm_id", "stay_id"}.issubset(cohort.columns)
    assert len(cohort) == primary.trials
    assert TRIAL_CACHE.exists()
    trial_qc = duckdb.sql(
        f"""select count(*) n,
                    min(invasive_vent_at_grace_end) min_vent_end,
                    min(prop_rate) min_prop_rate,
                    max(prop_rate) max_prop_rate,
                    min(case when treatment=1 then raw_dex_at_grace_end else 1 end) min_dex_end,
                    min(case when treatment=0 then raw_prop_continuous_window else 1 end) min_prop_continuous,
                    sum(case when successful_extubation=1 and invasive_vent_restart_72h_time is not null then 1 else 0 end) success_with_vent_restart
             from read_parquet('{TRIAL_CACHE}')"""
    ).fetchone()
    assert trial_qc[0] == primary.trials
    assert trial_qc[1] == 1
    assert trial_qc[2] > 0
    assert trial_qc[3] <= 200
    assert trial_qc[4] == 1
    assert trial_qc[5] == 1
    assert trial_qc[6] == 0

    secondary = pd.read_csv(PRIMARY / "table5_secondary_outcomes.csv")
    assert not secondary.outcome.str.contains("Stable successful planned extubation").any()
    for term in ["New bradycardia", "New hypotension", "vasopressor"]:
        assert secondary.outcome.str.contains(term, case=False).any()
    assert secondary[["weighted_risk_control", "weighted_risk_treated"]].notna().all().all()
    safety = secondary.loc[secondary.outcome.str.contains("bradycardia|hypotension", case=False)]
    assert (safety.observed_control_trials <= primary.control_trials).all()
    assert (safety.observed_treated_trials <= primary.treated_trials).all()
    assert (safety.observed_control_trials < primary.control_trials).any()
    assert (safety.observed_treated_trials < primary.treated_trials).any()
    assert (safety.events_control <= safety.observed_control_trials).all()
    assert (safety.events_treated <= safety.observed_treated_trials).all()

    event_audit = pd.read_csv(PRIMARY / "table6_outcome_event_audit.csv")
    audit = dict(zip(event_audit.audit_metric, event_audit["count"]))
    assert audit["unique_documented_extubation_events"] > 5000
    assert audit["cross_icu_stay_reintubation_within_72h"] > 0
    assert audit["stable_success_events"] <= audit["unique_documented_extubation_events"]
    assert audit["stable_success_events_with_unplanned_code"] == 0
    assert audit["stable_success_events_with_vent_restart"] == 0
    assert audit["vent_restart_without_reintubation_code"] > 0

    review = pd.read_csv(EVENT_REVIEW)
    assert len(review) == 60
    assert "successful_extubation" not in review.columns
    assert review.review_id.is_unique
    key = pd.read_csv(EVENT_KEY)
    assert len(key) == 60 and key.review_id.is_unique
    assert key.algorithmic_successful_extubation.sum() == 30
    assert key.algorithmic_successful_extubation.eq(0).sum() == 30

    external = pd.read_csv(EXTERNAL / "table2_external_effects.csv")
    common = external.loc[external.analysis.str.contains("Common-support")].iloc[0]
    assert common.treated_trials == 45
    assert common.max_abs_smd_weighted < 0.10

    landmark = pd.read_csv(PRIMARY / "table4_landmark_effects.csv")
    assert set(landmark.landmark_hour) == {6, 12, 18, 24, 30, 36, 42, 48}

    for document in [MANUSCRIPT, ABSTRACT, TABLES]:
        text = document.read_text(encoding="utf-8")
        assert "54,980" in text
        assert "994" in text
        assert "11.28" in text
        for stale in ["60,666", "60,568", "1,199", "1,131", "8.31", "3.80", "4.72", "11.70"]:
            assert stale not in text, f"Stale result {stale} in {document.name}"
    print("All analysis integrity checks passed.")


if __name__ == "__main__":
    main()
