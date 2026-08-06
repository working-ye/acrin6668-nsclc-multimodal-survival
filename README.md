# multimodal survival modeling for ACRIN 6668/RTOG 0235

[![CI](https://github.com/working-ye/acrin6668-nsclc-multimodal-survival/actions/workflows/ci.yml/badge.svg)](https://github.com/working-ye/acrin6668-nsclc-multimodal-survival/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the model-building and validation code for a secondary
analysis of the ACRIN 6668/RTOG 0235 NSCLC FDG-PET/CT study. It implements four
36-month overall-survival random survival forest (RSF) models:

| Model | Definition | Inputs |
|---|---|---:|
| CP | clinical–PET model | 3 protocol-specified clinical/PET variables |
| CPR | clinical–PET–radiomics model | CP + Top 10 CT radiomics features |
| CPRD | clinical–PET–radiomics–deep feature model | CPR + Top 10 frozen-encoder features |
| RD | CT-based radiomics–deep feature model | Top 10 radiomics + Top 10 frozen-encoder features |

The core design is intentionally strict: validation patients are not accepted
by the model-building command, and the validation workflow stops before
prediction if patient IDs or optional image fingerprints overlap with the
development cohort.

> **Release status.** This is a strict reanalysis/reference implementation and
> contains no study estimates. It becomes manuscript-generating code only after
> the complete study is rerun with this exact workflow and the manuscript
> numbers, private manifests, Git commit, and tagged release all agree. It must
> not imply reproduction of estimates from a different development/evaluation
> design.

> **Scope.** The public workflow starts from a de-identified wide table of
> precomputed clinical/PET, radiomics, and deep features. It does not distribute
> clinical records, images, ROIs, extracted patient-level features, predictions,
> trained model bundles, or neural-network weights.

## Leakage-control contract

The code enforces all of the following:

1. `build` accepts a development table only; it has no validation-data argument.
2. Imaging feature selection, median imputation, standardization, and RSF fitting
   are repeated inside each cross-validation training fold.
3. Hyperparameters are selected only from development-fold performance.
4. Final features, preprocessing objects, RSF estimators, and median-risk cutoffs
   are frozen after refitting on the complete development cohort.
5. `predict` reads only the ID, fingerprint, and frozen feature columns. Outcome
   columns are not loaded even when they are present in the same CSV file.
6. `evaluate` reads outcomes only after prediction and contains no `fit` or
   `fit_transform` operation.
7. Patient-ID, Study/Series UID, image-hash, and ROI-hash collisions cause a hard
   failure before prediction when those fields are available.
8. Early censoring is retained as censoring. It is never recoded as a negative
   36-month class for ordinary binary ROC or Hosmer–Lemeshow analysis.
9. Build-manifest, model-artifact, prediction-manifest, prediction-file, and
   configuration SHA-256 values are verified before deserialization/evaluation;
   evaluation also requires the prediction-manifest SHA-256 archived before
   outcomes are opened.
10. Every locked run requires a new empty output directory so stale results
    cannot be mixed into the analysis.

See [Validation policy](docs/VALIDATION_POLICY.md) for the complete definition
of strict held-out evaluation.

## Repository layout

```text
configs/                 Paper and synthetic-demo configurations
docs/                    Methods, schema, validation, and reproducibility notes
scripts/build_models.py  Development-only feature selection, tuning, and fitting
scripts/predict_models.py Frozen prediction without loading outcomes
scripts/evaluate_models.py Censoring-aware evaluation of saved predictions
scripts/validate_models.py Convenience wrapper for prediction + evaluation
src/acrin_survival/      Tested implementation
tests/                   Leakage, schema, and end-to-end tests
```

## Installation

The release was verified with Python 3.9.25. Create the exact environment with:

```bash
conda env create -f environment.yml
conda activate acrin-nsclc-survival
pip install -e .
```

Alternatively:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
```

## Reproducibility smoke test

The example data are fully synthetic and have no relationship to study patients.

```bash
python scripts/make_synthetic_data.py --output-dir examples/_demo_data

python scripts/build_models.py \
  --development-csv examples/_demo_data/development.csv \
  --config configs/demo.yaml \
  --output-dir examples/_demo_artifacts

python scripts/validate_models.py \
  --validation-csv examples/_demo_data/heldout_validation.csv \
  --artifacts-dir examples/_demo_artifacts \
  --config configs/demo.yaml \
  --output-dir examples/_demo_results \
  --cohort-name synthetic_heldout
```

These demo directories are ignored by Git and must not be interpreted as study
results.

## Running the study workflow

### 1. Audit the locked split

```bash
python scripts/audit_split.py \
  --development-csv /secure/path/development.csv \
  --validation-csv /secure/path/heldout_validation.csv \
  --config configs/paper.yaml
```

### 2. Build the frozen models

```bash
python scripts/build_models.py \
  --development-csv /secure/path/development.csv \
  --config configs/paper.yaml \
  --output-dir /secure/path/artifacts
```

No validation table can be supplied to this command.

### 3. Predict without outcomes

```bash
python scripts/predict_models.py \
  --features-csv /secure/path/heldout_validation.csv \
  --artifacts-dir /secure/path/artifacts \
  --config configs/paper.yaml \
  --output-dir /secure/path/private_predictions
```

Record the printed `prediction_manifest_sha256` in an immutable analysis log
before opening or joining the validation outcomes.

For an imaging-only transportability cohort, add `--models RD`.

### 4. Evaluate the frozen predictions

```bash
python scripts/evaluate_models.py \
  --predictions-csv /secure/path/private_predictions/predictions.csv \
  --outcomes-csv /secure/path/heldout_validation.csv \
  --artifacts-dir /secure/path/artifacts \
  --config configs/paper.yaml \
  --output-dir /secure/path/results \
  --cohort-name tcia_heldout \
  --expected-prediction-manifest-sha256 <PRE_OUTCOME_SHA256>
```

The convenience `validate_models.py` command performs the same two stages while
reading feature and outcome columns separately.

## Required input

Each patient must occupy one row. The default schema requires:

- `patient_id`
- `overall_survival_days`
- `death_event_overall` (`0` or `1`)
- the three configuration-locked clinical/PET columns
- radiomics columns beginning with `rad_tumor_` or `rad_peri5_`
- frozen-encoder columns beginning with `dl_swin_`

Optional `study_instance_uid`, `series_instance_uid`, `image_sha256`, and
`roi_sha256` columns strengthen the leakage audit. Detailed definitions are in
[Data dictionary](docs/DATA_DICTIONARY.md).

The three clinical/PET inputs are treated as protocol-specified predictors by
this software. If an analysis selects those predictors empirically, that
selection must be completed using development data only before `paper.yaml` is
locked.

## Statistical outputs

Model building writes a model registry, feature rankings, selected variables,
cross-validation search results, input summaries, frozen model bundles, and a
hash-based analysis manifest.

Strict evaluation writes:

- Harrell C-index with patient-bootstrap confidence intervals;
- Uno C-index;
- IPCW cumulative/dynamic AUC at 12, 24, and 36 months;
- IPCW Brier scores and integrated Brier score;
- Kaplan–Meier observed-risk calibration by predicted-risk quantile;
- frozen-cutoff Kaplan–Meier risk groups and log-rank tests;
- paired C-index differences using a shared patient-bootstrap index matrix;
- vector PDF, 600 dpi PNG, and LZW-compressed TIFF figures.

Patient-level predictions are private intermediate data. `configs/paper.yaml`
therefore sets `export_patient_level: false` by default.

IBS is integrated over a prespecified 365–1,095 day grid at 30-day intervals
plus the exact endpoint. Bootstrap intervals are conditional, pointwise
percentile intervals for a frozen development pipeline; they do not include
uncertainty from rebuilding that pipeline.

## Important interpretation boundary

This repository contains methodology and executable code, not hard-coded
manuscript estimates. Reported numbers must come from a single, locked strict
run whose input hashes, configuration hash, Git commit, environment, and release
tag match the archived analysis manifest. A cohort viewed during feature,
algorithm, region, threshold, or reporting decisions is not a pristine
confirmatory validation cohort even when patient rows are absent from final
model fitting.

Validation missingness is audited after locked encodings. Any required feature
exceeding the prespecified 20% validation missingness limit stops prediction;
an aggregate missingness table accompanies the private prediction audit.

## Data availability and privacy

The ACRIN-NSCLC-FDG-PET imaging collection is available through
[The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/)
under its applicable data-use terms and has the permanent dataset DOI
[10.7937/TCIA.2019.30ILQFCL](https://doi.org/10.7937/TCIA.2019.30ILQFCL).
This repository does not redistribute TCIA data or institution-specific patient
data. See [Data availability](DATA_AVAILABILITY.md).

Generated `.joblib` bundles contain private development identifiers and
survival outcomes for overlap auditing and censoring-aware metrics. They are
ignored by Git and must not be uploaded to GitHub or a public release.

## Testing

```bash
python -m pytest
ruff check src tests scripts
```

The test suite includes patient and image collision failures, administrative
censoring, prediction/outcome separation, frozen-artifact immutability, outcome
invariance of predictions, and a four-model synthetic end-to-end run.

## Citation

Use the repository's `CITATION.cff`. After the manuscript author list and
archive DOI are finalized, cite the tagged software release and its permanent
Zenodo DOI rather than the moving `main` branch.

## License and disclaimer

Code is released under the [MIT License](LICENSE). The software is intended for
research reproducibility only. It is not a medical device and must not be used
for diagnosis, treatment selection, or individual patient management.
