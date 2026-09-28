# Reproducibility Guide

## Analysis environment

- Python 3.12.13
- DuckDB 1.5.0
- pandas 3.0.6
- NumPy 2.5.3
- scikit-learn 1.9.1
- Matplotlib 3.11.2
- Random seed: 20260924

The local analysis used MIMIC-IV 3.1 and eICU-CRD 2.0 under PhysioNet data-use restrictions. Patient-level data cannot be redistributed.

## Required data-root environment variables

- `MIMIC_IV_ROOT`: extracted MIMIC-IV 3.1 directory.
- `EICU_CRD_ROOT`: extracted eICU-CRD 2.0 directory.

The scripts fail immediately when the required variable is absent or is not a directory. Do not publish credentials or patient-level derived files.

## Reproduction command

From the project directory:

```bash
MIMIC_IV_ROOT=/path/to/mimic-iv-3.1 python work/build_mimic_landmark_analysis.py
EICU_CRD_ROOT=/path/to/eicu-crd-2.0 python work/build_eicu_external_validation.py
python work/run_blinded_outcome_audit.py --stage1
python work/run_blinded_outcome_audit.py --stage2
python work/test_analysis_outputs.py
```

Install the exact dependencies from the packaged `requirements.txt` before running.

## Required primary outputs

- `table0_flow_counts.csv`: cohort construction.
- `table2_covariate_balance.csv`: complete balance diagnostics.
- `table3_sensitivity_effects.csv`: primary and subgroup estimates.
- `table5_secondary_outcomes.csv`: secondary and safety outcomes.
- `table6_outcome_event_audit.csv`: admission-level outcome audit.
- `table7_descriptive_baseline.csv`: publication-ready baseline table source.
- `table8_model_specification_sensitivity.csv`: spline-logit propensity-model result.
- `table9_model_specification_balance.csv`: spline-logit balance diagnostics.
- `figure0_cohort_flow.png`, `figure1_love_plot.png`, `figure2_propensity_overlap.png`, `figure3_effect_forest.png`, and `figure4_landmark_effects.png`.

## Automated integrity checks

`work/test_analysis_outputs.py` verifies:

- required output existence;
- exact primary cohort and transition counts;
- subject/stay/trial hierarchy;
- primary and alternative-model covariate balance;
- conclusion-changing attenuation in the enhanced-readiness subgroup;
- safety-outcome observed denominators;
- hospital-admission-level event logic;
- event-review sample composition;
- landmark coverage; and
- exploratory eICU common-support counts and balance.

## Files that must not enter the evidentiary chain

Earlier cloning–censoring–weighting artifacts remain in the workspace for audit provenance but are not valid confirmatory outputs for the current manuscript. They should either be moved to an explicitly deprecated archive before public release or omitted from the release manifest. Do not cite them in the manuscript or supplementary results.

## Remaining reproducibility work before submission

1. Add a machine-readable output manifest with hashes to the eventual repository release.
2. Keep controlled patient-level intermediates separate from redistributable aggregate outputs.
3. Add a license and repository DOI after the public repository is created.
4. Record whether the analysis was prospectively registered; current materials do not support such a claim.
