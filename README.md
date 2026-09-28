# Final analysis scripts (v11)

This package contains the final code used for the manuscript's v11 MIMIC-IV primary analysis, exploratory eICU consistency analysis, locked two-stage structured outcome audit, and automated integrity checks.

## Included files

- `work/build_mimic_landmark_analysis.py`: final MIMIC-IV sequential-landmark analysis and figure/table generation.
- `work/build_eicu_external_validation.py`: exploratory eICU consistency analysis.
- `work/run_blinded_outcome_audit.py`: two-stage locked structured outcome audit.
- `work/test_analysis_outputs.py`: integrity tests for the final outputs.
- `requirements.txt`: exact direct Python dependencies used in the final run.
- `docs/reproducibility_guide.md`: environment, outputs, and data-governance notes.
- `docs/target_trial_specification_table.md`: target-trial specification and emulation mapping.

Deprecated cloning-censoring-weighting scripts and outputs are deliberately excluded because they are not part of the manuscript's evidentiary chain.

## Data prerequisites

The code requires authorized local copies of:

- MIMIC-IV 3.1
- eICU-CRD 2.0

The databases are governed by PhysioNet data-use agreements and are not included. Patient-level caches, review samples, database files, and derived patient-level outputs are also excluded.

## Expected layout

Extract this package as a project directory containing `work/`, `docs/`, and `requirements.txt`. The scripts create `outputs/primary_analysis/`, `outputs/external_validation/`, and controlled intermediate files under `work/cache/`.

## Installation

Use Python 3.12 and install the locked direct dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run order

Set the database roots to the extracted dataset directories, then run:

```bash
export MIMIC_IV_ROOT=/path/to/mimic-iv-3.1
export EICU_CRD_ROOT=/path/to/eicu-collaborative-research-database-2.0

python work/build_mimic_landmark_analysis.py
python work/build_eicu_external_validation.py
python work/run_blinded_outcome_audit.py --stage1
python work/run_blinded_outcome_audit.py --stage2
python work/test_analysis_outputs.py
```

Stage 1 of the audit must be completed and its adjudication file locked before stage 2 reads the separately generated outcome key. The audit validates structured event logic; it is not a free-text clinical chart review.

## Expected primary result

The final cohort contains 54,980 decision instances, including 994 completed transitions. The primary overlap-weighted risk difference is 11.28 percentage points. `test_analysis_outputs.py` checks these values, balance, event logic, sensitivity attenuation, safety denominators, and exploratory eICU common-support counts.

## Verification performed before packaging

- All four Python files passed Python 3.12 byte-code compilation.
- The automated integrity test completed with `All analysis integrity checks passed.`
- No raw database, patient-level cache, local database, credentials, or machine-specific data path is included.


## Reference outputs

`reference_outputs/` contains the aggregate outputs of the final v11 run
(cohort flow, balance, sensitivity, secondary/safety, landmark,
model-specification, and exploratory eICU results) plus
`OUTPUTS_MANIFEST_SHA256.txt`. These files are aggregate and contain no
patient-level data; they are included so that the reported results can be
checked without re-running the pipeline. Patient-level caches and the locked
audit review files are excluded (see `.gitignore`).
