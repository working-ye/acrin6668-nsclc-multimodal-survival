# Computational methods

## Analysis endpoint

The primary endpoint is overall survival administratively censored at 1,095
days. For a raw follow-up time `T` and death indicator `D`, the code constructs:

```text
time = min(T, 1095)
event = 1 only when D = 1 and T <= 1095; otherwise 0
```

Early loss to follow-up remains censored. It is not relabeled as survival at 36
months.

For time-dependent library calculations only, an event-free observation known
to reach the administrative boundary is represented at the next floating-point
value after day 1,095. This preserves the rule that deaths on day 1,095 are
cases while patients event-free through day 1,095 remain controls at that exact
horizon; reported follow-up remains capped at 1,095 days.

## Model definitions

All four models use a random survival forest. CP contains three
configuration-locked clinical/PET predictors. CPR adds 10 radiomics features.
CPRD adds 10 deep features to CPR. RD contains the same 10 radiomics and 10 deep
features without clinical/PET inputs. The CT region is fixed to tumor plus a
5-mm peritumoral ring before the public modeling workflow begins.

## Development-only feature selection

Within every cross-validation training fold, candidate radiomics and deep
features are processed separately:

1. remove features with more than 20% missing values;
2. remove features with variance at or below `1e-10`;
3. median-impute temporarily within that training fold for univariable ranking;
4. rank by the better-oriented univariable Harrell C-index;
5. traverse the ranking and reject a feature when its absolute Pearson
   correlation with an already retained feature is at least 0.90;
6. retain exactly 10 features per imaging modality.

The held-out cohort never enters this procedure. After hyperparameter selection,
the selector is fitted once on the complete development cohort to freeze the
final features.

## Preprocessing and model tuning

Median imputation and standardization are fitted inside each CV training fold
and then applied to that fold's internal validation partition. The public paper
configuration uses three folds and the following 12 RSF candidates:

- trees: 100 or 200;
- minimum terminal-node size: 3, 5, or 10;
- candidate features per split: square root or 30% of inputs;
- maximum depth: unrestricted.

The candidate with the highest mean internal-fold Harrell C-index is selected;
lower between-fold standard deviation and the serialized parameter tuple are
deterministic tie breakers. The selected pipeline is refitted on the complete
development cohort. The development-risk median is frozen as the high/low risk
threshold. Four explicit model seeds are stored in `configs/paper.yaml`.

Internal cross-validation is used for tuning, not as the final generalization
estimate. Final performance is computed only after the pipeline is frozen.

## Two-stage held-out validation

Prediction and outcome evaluation are separated:

1. `predict_models.py` loads only IDs, optional image fingerprints, and frozen
   feature columns. It checks for cross-split identity collisions and applies
   the saved selector outputs, imputer, scaler, RSF, and cutoff. It does not load
   survival outcomes.
2. `evaluate_models.py` joins the saved predictions to a separate outcome-only
   read. It contains no model-fitting operation, requires the prediction-manifest
   SHA-256 archived before outcome access, verifies the frozen build-manifest
   hash, and verifies that risk groups equal
   `risk_score >= frozen development cutoff`.

## Performance measures

Primary measures are Harrell C-index, Uno C-index, IPCW cumulative/dynamic AUC,
IPCW Brier score, and integrated Brier score. IBS is prespecified over 365–1,095
days on a 30-day grid with the exact endpoint appended. The development survival
outcome is used as the censoring reference for IPCW estimates. Harrell C-index and
time-dependent AUC confidence intervals use patient-level bootstrap resampling.
All pairwise C-index differences use a shared bootstrap index matrix to preserve
the paired comparison.

Uno C uses `tau` at the next representable time after day 1,095 so deaths on the
exact administrative boundary remain eligible; its effective `tau` and status
are written with every model result. AUC, Brier, IBS, and Uno calculations carry
explicit failure/insufficient-follow-up status fields rather than silently
reporting failed calculations as available.

Bootstrap intervals are conditional, pointwise percentile intervals with the
development pipeline held fixed; they quantify validation-sample uncertainty,
not model-development uncertainty. At least the configured fraction of
replicates must be valid or the interval is unavailable. Uno C-index,
Brier/IBS, and calibration are point estimates in the core workflow.

Calibration uses the RSF's direct predicted survival function. Patients are
grouped by predicted 36-month event risk; observed event risk is `1 - KM(1095)`
within each bin. Risk stratification uses only the frozen development median,
followed by Kaplan–Meier curves and the log-rank test.

Calibration uses quantiles of the original risk without splitting identical
values across bins. Bins below the prespecified minimum size or without horizon
follow-up are flagged and omitted from the figure; number at risk and
Kaplan–Meier confidence limits remain in the table.

Ordinary binary ROC, ordinary PR, Hosmer–Lemeshow tests, and classification
metrics that label early-censored patients as negative are intentionally absent.
Decision-curve and reclassification analyses are also omitted from the core
workflow until a separately tested, time-specific IPCW implementation is locked.

At each AUC time, outputs record cases, controls, earlier censoring, number at
risk, and the development censoring-survival estimate. A time with insufficient
follow-up, cases, controls, or censoring positivity is unavailable and is never
silently replaced by an earlier time.

Using the development censoring distribution for a cross-institutional cohort
assumes compatible censoring mechanisms. Such analyses require an explicit
transportability interpretation and, when material, a target-cohort censoring
sensitivity analysis.

## Reproducibility record

Each build and evaluation writes input/config SHA-256 hashes, patient-set
digests, software versions, random seeds, artifact hashes, the Git commit when
available, sample/event counts, and explicit leakage-control fields. Patient IDs
are not written to public manifests.
