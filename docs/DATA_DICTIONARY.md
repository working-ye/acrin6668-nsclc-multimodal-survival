# Input data dictionary

The modeling input is a de-identified, one-row-per-record CSV. Real data are
never included in this repository.

## Identifiers and outcome

| Column | Type | Definition |
|---|---|---|
| `patient_id` | string | De-identified record key used to join predictions and outcomes |
| `overall_survival_days` | positive numeric | Follow-up from the protocol-defined baseline |
| `death_event_overall` | integer | `1` for observed death, `0` for censoring |

Outcome columns are required for model training and metric evaluation. The
prediction reader excludes them.

## Configured clinical/PET inputs

| Column | Accepted representation |
|---|---|
| `le_pre_pet_total_metabolic_tumor_volume` | Numeric |
| `baseline_zubrod_performance_status` | Numeric or text beginning with an integer |
| `ss_metastatic_disease_present_on_pet_before_rt` | `yes/no` or `1/0` |

## Imaging feature families

- Radiomics candidates begin with `rad_tumor_` or `rad_peri5_`.
- Frozen deep-feature candidates begin with `dl_swin_`.
- Selected features must be numeric after the configured coercion.
- Missing values are handled by the training-fold median and checked against the
  configured validation threshold.
- A required feature that is entirely missing in the TCIA validation table
  stops prediction.

Feature extraction is assumed to have been completed with the locked voxel,
intensity-normalization, ROI, software, and encoder settings.
