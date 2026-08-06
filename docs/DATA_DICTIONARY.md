# Input data dictionary

The modeling input is a de-identified, one-row-per-patient CSV. Real data are
never included in this repository.

## Required identifiers and outcome

| Column | Type | Definition |
|---|---|---|
| `patient_id` | string | Unique de-identified patient key; duplicates are fatal |
| `overall_survival_days` | positive numeric | Follow-up from the protocol-defined baseline |
| `death_event_overall` | integer | `1` for observed death, `0` for censoring |

The outcome columns are required for development and evaluation. The prediction
reader deliberately excludes them.

## Protocol-configured clinical/PET inputs

| Column | Accepted representation |
|---|---|
| `le_pre_pet_total_metabolic_tumor_volume` | Numeric |
| `baseline_zubrod_performance_status` | Numeric or text beginning with an integer |
| `ss_metastatic_disease_present_on_pet_before_rt` | `yes/no` or `1/0` |

These fields are treated as locked inputs, not selected by the public script.

## Imaging feature families

- Radiomics candidates begin with `rad_tumor_` or `rad_peri5_`.
- Frozen deep-feature candidates begin with `dl_swin_`.
- Every selected feature must be numeric after coercion.
- Missing values are permitted within the configured development threshold and
  are imputed from development-fold medians.
- A selected validation column that is entirely missing causes a hard failure.

The public code assumes feature extraction has already been completed with a
locked imaging protocol. Do not combine feature tables produced with different
voxel spacing, intensity normalization, ROI definitions, software versions, or
encoder weights without a prespecified harmonization analysis.

## Strongly recommended identity fingerprints

| Column | Purpose |
|---|---|
| `study_instance_uid` | Detect the same DICOM study under a renamed patient key |
| `series_instance_uid` | Detect repeated CT series across splits |
| `image_sha256` | Detect copied or renamed image files |
| `roi_sha256` | Detect copied or renamed masks |

When present in both development and prediction tables, any collision stops the
workflow before scoring.

## Separate development and validation files

The repository expects independently materialized files rather than a mutable
`split` column. This makes it impossible for `build_models.py` to access held-out
rows accidentally. The split manifest, source file hashes, and cohort flow must
be frozen before model construction.
