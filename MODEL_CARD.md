# Model card

## Intended use

The software supports research reproducibility for multimodal 36-month overall
survival modeling in locally advanced non-small-cell lung cancer. It is intended
for retrospective methodological evaluation and scientific review.

## Prohibited use

The models and code are not medical devices and must not be used for diagnosis,
treatment selection, prognosis communication, eligibility decisions, or
individual patient management.

## Inputs and outputs

Inputs are protocol-configured clinical/PET variables, precomputed tumor plus
5-mm peritumoral CT radiomics, and frozen deep features. The random survival
forest produces relative risk scores and survival functions. Risk groups use the
training-table median cutoff saved with each model.

## Evaluation

The release provides censoring-aware TCIA internal validation metrics, including
C-index, time-dependent AUC, Brier/IBS, calibration, and survival risk-group
summaries. No performance estimate is embedded in the repository; reported
claims must be linked to a locked manifest and tagged release.

## Limitations

Retrospective analyses may have selection bias, missingness, scanner
heterogeneity, preprocessing sensitivity, and limited event counts. The code
does not establish clinical utility or fairness across demographic, disease,
scanner, or institution subgroups.

## Privacy

Study tables, fitted bundles, and patient-level predictions contain confidential
material. Keep them in approved secure storage and exclude them from Git.
