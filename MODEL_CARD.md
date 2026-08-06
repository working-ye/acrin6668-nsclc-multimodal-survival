# Model card

## Intended use

The software supports research reproducibility for multimodal 36-month overall
survival modeling in locally advanced non-small-cell lung cancer. It is designed
for retrospective methodological evaluation and independent scientific review.

## Prohibited use

The models and code are not medical devices and must not be used for diagnosis,
treatment selection, prognosis communication, eligibility decisions, or
individual patient management.

## Inputs

The workflow accepts protocol-configured clinical/PET variables and precomputed
tumor plus 5-mm peritumoral CT radiomics and frozen-encoder features. Input
definitions and preprocessing must match the locked study protocol.

## Outputs

The RSF estimators produce relative risk scores and survival functions. Risk
groups use a development-cohort median cutoff frozen before validation.

## Evaluation

The code supports strict zero-overlap evaluation with censoring-aware C-index,
time-dependent AUC, Brier score, calibration, and survival-stratification
outputs. No performance claim is embedded in this repository; claims must be
linked to a locked manifest and tagged release.

## Important limitations

- Retrospective secondary analyses may have selection bias, missingness, scanner
  heterogeneity, and limited event counts.
- Transportability can be affected by treatment, acquisition, ROI, endpoint,
  and follow-up differences.
- High-dimensional imaging features are sensitive to preprocessing and software
  versions.
- A technically disjoint cohort is not confirmatory if it was previously used
  for model or reporting decisions.
- The code does not establish clinical utility or fairness across demographic,
  disease, scanner, or institution subgroups.

## Privacy

Generated bundles and predictions contain confidential patient-level audit
material. They must remain in approved secure storage and are excluded from Git.
