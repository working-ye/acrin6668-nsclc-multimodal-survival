# Computational methods

## Endpoint

The primary endpoint is overall survival administratively censored at 1,095
days. For follow-up time `T` and death indicator `D`, the software constructs:

```text
time = min(T, 1095)
event = 1 only when D = 1 and T <= 1095; otherwise 0
```

Early loss to follow-up remains censored. For time-dependent library calls, an
event-free record observed through day 1,095 is represented at the next
floating-point value after the horizon so deaths on the exact horizon remain
events.

## Model definitions

All four models use a random survival forest. CP contains three configured
clinical/PET predictors. CPR adds the selected CT radiomics features. CPRD adds
the selected frozen deep features to CPR. RD contains the selected radiomics and
deep features without clinical/PET inputs. The imaging region is tumor plus a
5-mm peritumoral ring under the locked extraction protocol.

## Feature selection and preprocessing

Radiomics and deep features are processed separately within each training
cross-validation fold:

1. remove features exceeding the configured missing-value fraction;
2. remove features with variance at or below `1e-10`;
3. median-impute temporarily for univariable ranking;
4. rank by the better-oriented univariable Harrell C-index;
5. reject a feature when its absolute Pearson correlation with a retained
   feature is at least `0.90`;
6. retain the configured number of features per imaging modality.

Imputation and standardization are fitted on each fold's training records and
then applied to that fold's scoring records. The selected hyperparameters are
refit on the complete training table. The training-risk median is frozen as the
high/low risk cutoff.

## TCIA internal validation

`predict_models.py` applies the saved selector outputs, imputer, scaler, random
survival forest, and cutoff to the TCIA feature table. `evaluate_models.py`
loads outcomes only for metric calculation and performs no model fitting.

The censoring reference for IPCW metrics is the outcome object stored with the
trained model. Outputs include C-index, Uno C-index, time-dependent AUC, Brier
and integrated Brier scores, calibration, risk-group summaries, paired model
comparisons, missingness tables, and vector/600-dpi figure exports.

## Reproducibility fields

Each run records input and configuration SHA-256 values, package versions,
random seeds, artifact hashes, sample/event counts, metric status fields, and
the Git commit when available. Patient-level result export is disabled by
default for the paper configuration.
