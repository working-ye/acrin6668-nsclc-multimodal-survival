# Multimodal survival model training and TCIA internal validation

[![CI](https://github.com/working-ye/acrin6668-nsclc-multimodal-survival/actions/workflows/ci.yml/badge.svg)](https://github.com/working-ye/acrin6668-nsclc-multimodal-survival/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the reproducible code accompanying the ACRIN 6668/RTOG
0235 secondary analysis of multimodal predictors of 36-month overall survival in
locally advanced NSCLC. The public release is limited to model training,
frozen prediction, and TCIA internal validation.

The workflow trains four random survival forest models:

| Model | Definition | Inputs |
|---|---|---|
| CP | clinical–PET | three protocol-configured clinical/PET variables |
| CPR | clinical–PET–radiomics | CP plus the selected CT radiomics features |
| CPRD | clinical–PET–radiomics–deep | CPR plus the selected frozen deep features |
| RD | radiomics–deep | selected CT radiomics and frozen deep features |

No clinical records, images, ROIs, extracted patient-level features, predictions,
trained bundles, or neural-network weights are distributed. The example generator
creates synthetic data only and has no relationship to study participants.

## Repository layout

```text
configs/                   Locked paper and synthetic-demo configurations
docs/                      Methods, input schema, validation, and reproducibility
scripts/build_models.py    Model training, feature selection, tuning, and fitting
scripts/predict_models.py  Frozen prediction from feature columns
scripts/evaluate_models.py Censoring-aware TCIA internal validation metrics
scripts/validate_models.py Combined prediction and validation wrapper
src/acrin_survival/        Tested implementation
tests/                     Unit and end-to-end tests
```

## Installation

The release is tested with Python 3.9–3.12. The project environment can be
created with:

```bash
conda env create -f environment.yml
conda activate acrin-nsclc-survival
python -m pip install -e ".[test]"
```

## Synthetic smoke test

```bash
python scripts/make_synthetic_data.py --output-dir examples/_demo_data

python scripts/build_models.py \
  --training-csv examples/_demo_data/training.csv \
  --config configs/demo.yaml \
  --output-dir examples/_demo_artifacts

python scripts/validate_models.py \
  --validation-csv examples/_demo_data/tcia_internal_validation.csv \
  --artifacts-dir examples/_demo_artifacts \
  --config configs/demo.yaml \
  --output-dir examples/_demo_results \
  --cohort-name synthetic_tcia_internal
```

The example directories are ignored by Git and are not study results.

## TCIA internal validation workflow

Train the frozen models from the approved training table:

```bash
python scripts/build_models.py \
  --training-csv /secure/path/tcia_training.csv \
  --config configs/paper.yaml \
  --output-dir /secure/path/model_artifacts
```

Apply the frozen models to the TCIA internal validation feature table. The
prediction stage reads only `patient_id` and the locked feature columns:

```bash
python scripts/predict_models.py \
  --features-csv /secure/path/tcia_internal_validation_features.csv \
  --artifacts-dir /secure/path/model_artifacts \
  --config configs/paper.yaml \
  --output-dir /secure/path/tcia_predictions
```

Evaluate the saved predictions with the corresponding outcome table:

```bash
python scripts/evaluate_models.py \
  --predictions-csv /secure/path/tcia_predictions/predictions.csv \
  --outcomes-csv /secure/path/tcia_internal_validation_outcomes.csv \
  --artifacts-dir /secure/path/model_artifacts \
  --config configs/paper.yaml \
  --output-dir /secure/path/tcia_results \
  --cohort-name tcia_internal_validation \
  --expected-prediction-manifest-sha256 <PREDICTION_MANIFEST_SHA256>
```

`scripts/validate_models.py` runs the same prediction and evaluation stages in
one command when a single wide validation table is available.

## Input table

Each row represents one de-identified record. The training and evaluation tables
contain:

- `patient_id`;
- `overall_survival_days` and `death_event_overall` (`0` or `1`);
- the three configured clinical/PET columns;
- radiomics columns beginning with `rad_tumor_` or `rad_peri5_`;
- frozen deep-feature columns beginning with `dl_swin_`.

The complete schema and encodings are documented in
[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md). Feature extraction is
assumed to have been completed with the locked imaging protocol.

## Methods and outputs

Feature selection, median imputation, standardization, and hyperparameter tuning
are performed within the training cross-validation procedure. The final model
freezes selected features, preprocessing, the random survival forest, and the
training risk cutoff. Validation reports censoring-aware Harrell and Uno
C-index, IPCW time-dependent AUC, IPCW Brier/IBS, Kaplan–Meier calibration,
risk-group summaries, paired model comparisons, and publication-ready figures.

Build and validation manifests record input/configuration hashes, software
versions, random seeds, artifact hashes, and the Git commit when available.
Patient-level outputs are disabled by default in `configs/paper.yaml`.

See [docs/METHODS.md](docs/METHODS.md),
[docs/VALIDATION_POLICY.md](docs/VALIDATION_POLICY.md), and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the computational
specification.

## Data availability

The ACRIN-NSCLC-FDG-PET collection is available through
[The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/)
under its applicable data-use terms and has DOI
[10.7937/TCIA.2019.30ILQFCL](https://doi.org/10.7937/TCIA.2019.30ILQFCL).
This repository does not redistribute study data.

## Testing

```bash
python -m pytest
ruff check src tests scripts
```

## Citation and license

Please cite the tagged release using [CITATION.cff](CITATION.cff). Code is
released under the [MIT License](LICENSE) for research reproducibility and is
not a medical device.
