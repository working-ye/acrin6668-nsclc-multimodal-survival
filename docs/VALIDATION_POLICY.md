# TCIA internal validation policy

This repository evaluates frozen multimodal survival models on the TCIA
internal validation table supplied for the locked analysis. The validation
stage does not refit an estimator, reselect variables, alter preprocessing, or
optimize the risk cutoff.

## Required conditions

1. The model artifact, configuration, prediction file, and manifests must match
   their recorded SHA-256 values.
2. Every configured model is evaluated on the same TCIA validation records.
3. The frozen feature list, imputer, scaler, random survival forest, and training
   risk cutoff are used exactly as saved.
4. Early loss to follow-up remains censoring and is not recoded as a negative
   36-month class.
5. Sample size, event count, early censoring, missingness, and metric status are
   written with the results.

## Metrics

The primary outputs are Harrell C-index, Uno C-index, IPCW cumulative/dynamic
AUC at 12, 24, and 36 months, IPCW Brier scores, integrated Brier score,
Kaplan–Meier calibration, frozen-cutoff risk groups, and paired C-index
comparisons. Bootstrap intervals are conditional, pointwise percentile
intervals with the trained pipeline held fixed.

Ordinary binary ROC, ordinary PR, Hosmer–Lemeshow tests, and other analyses that
recode censored observations as negatives are not part of this workflow.

## Confidential material

Study tables, images, ROIs, patient-level predictions, and model bundles remain
in approved local storage and must not be committed to the public repository.
